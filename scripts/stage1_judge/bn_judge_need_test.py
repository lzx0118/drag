# -*- coding: utf-8 -*-
"""
用 Qwen3-32B 判断 Bengali 句子是否需要检索 + 领域 + 歧义词
"""

import os, re, json, argparse, gc
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

# ===== 路径配置（bn 版） =====
IN_TSV      = "<PATH_TO_INPUT_TSV>"   # TSV with columns: id, <src_text_col>
OUT_TSV     = "<PATH_TO_OUTPUT_TSV>"  # output TSV with judge results
CKPT_PATH   = "<PATH_TO_CHECKPOINT_FILE>"  # for resume
RAW_LOG     = "<PATH_TO_DEBUG_LOG>"  # optional

DEFAULT_QUANT = "4bit"  # {4bit, 8bit, none}

ID_COL   = "id"
TEXT_COL = "bn"

# Source language name used in the judge prompt (edit when switching languages)
SRC_LANG_ZH = "孟加拉语"   # e.g., "乌尔都语"
SRC_LANG_EN = "Bengali"    # e.g., "Urdu"

# 需要从候选歧义词里排除的词（明显不是你要的“歧义词”）
STOP_AMB = {"no", "yes"}

# ===== Prompt 模板（语言改成孟加拉语） =====
PROMPT_TMPL = (
    f"源语言的语言是：{SRC_LANG_ZH}（{SRC_LANG_EN}）。\n"
    "请严格按以下要求，只输出纯粹 JSON：\n"
    "（1）确认这句话的源语言是孟加拉语。\n"
    "（2）根据这句话的内容，帮我给出一个最合适的领域标签 domain，"
    "用你认为最合适的一个简短英文词来概括即可（比如 medical、sports "
    "这类英文词，但不限于这些）。\n"
    "（3）根据领域知识，判断这句话是否存在歧义词，如果存在则视为需要检索，把 "
    "need_retrieval 设为 \"yes\"；如果不需要，把 need_retrieval 设为 \"no\"。\n"
    "（4）请把这些歧义词列在 ambiguous_terms 里；如果没有，就返回空列表 []。\n"
    "输出 JSON 格式如下：\n"
    "{\n"
    '  "need_retrieval": "yes 或 no",\n'
    '  "domain": "一句话英文领域标签",\n'
    '  "ambiguous_terms": ["词1", "词2", ...]\n'
    "}\n"
    "你必须严格输出合法 JSON，不要输出任何多余解释或注释。\n"
    "源语言句子（孟加拉语）：{sent}\n"
    "只输出 JSON："
)

JSON_RE = re.compile(r"\{.*\}", re.S)

# ===== 断点 =====
def load_checkpoint():
    if os.path.isfile(CKPT_PATH):
        try:
            with open(CKPT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return int(data.get("i", 0))
        except Exception:
            return 0
    return 0

def save_checkpoint(i):
    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump({"i": int(i)}, f, ensure_ascii=False)

# ===== 结果写入 =====
def append_tsv(row):
    is_new = not os.path.isfile(OUT_TSV)
    with open(OUT_TSV, "a", encoding="utf-8") as f:
        if is_new:
            f.write("id\tbn\tneed_retrieval\tdomain\tambiguous_terms\n")
        f.write(
            f"{row['id']}\t{row['bn']}\t"
            f"{row['need_retrieval']}\t{row['domain']}\t"
            f"{json.dumps(row['ambiguous_terms'], ensure_ascii=False)}\n"
        )

# ===== prompt =====
def build_prompt(sent: str) -> str:
    # 只替换 {sent}，避免 JSON 里的 { } 被当成占位符
    return PROMPT_TMPL.replace("{sent}", sent.strip())

# ===== 模型加载 / 释放 =====
def load_model(quant: str = DEFAULT_QUANT):
    print(f"[INFO] loading model... (quant={quant})")
    kwargs = {}
    if quant == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["quantization_config"] = bnb
        kwargs["device_map"] = "auto"
        torch_dtype = torch.bfloat16
    elif quant == "8bit":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["quantization_config"] = bnb
        kwargs["device_map"] = "auto"
        torch_dtype = torch.bfloat16
    else:
        kwargs["device_map"] = "auto"
        torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        **kwargs,
    )
    model.eval()
    return tokenizer, model

def free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ===== 单轮 LLM 调用 =====
def llm_once(tokenizer, model, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [
        # 禁止输出 <think> / 思考过程，只能输出 JSON
        {"role": "system", "content": "You are a careful JSON generator. 禁止输出 <think> 标签或思考过程，只能输出最终 JSON 对象。"},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# ===== 解析模型输出 =====
def safe_json(raw_text: str, sent_bn: str):
    """
    优先解析 JSON；否则从解释文本里挖 domain 和歧义词。
    最终：
      ambiguous_terms 非空 => need_retrieval = "yes"
      ambiguous_terms 为空 => need_retrieval = "no"
    完全忽略模型自己给的 need_retrieval 字段。
    """
    raw_text = raw_text or ""
    text = raw_text.strip()

    # 去掉 <think> 标签内容再找 JSON
    text_no_think = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)

    domain = None
    amb_list = []

    # ---------- 1. JSON 片段 ----------
    m = JSON_RE.search(text_no_think)
    if m:
        snippet = m.group(0).strip()
        data = {}
        try:
            data = json.loads(snippet)
        except Exception:
            import ast
            try:
                data = ast.literal_eval(snippet)
            except Exception:
                data = {}
        if isinstance(data, dict):
            d_dom = data.get("domain")
            if isinstance(d_dom, str) and d_dom.strip():
                domain = d_dom.strip()
            amb = data.get("ambiguous_terms", [])
            if isinstance(amb, str):
                amb = [w.strip() for w in re.split(r"[，,；;]", amb) if w.strip()]
            if isinstance(amb, list):
                amb_list = [str(w).strip() for w in amb if str(w).strip()]

    # ---------- 2. 从解释里挖 domain / 歧义词 ----------
    if domain is None or not amb_list:
        clean = text  # 包含 <think>

        # 2.1 领域
        if domain is None:
            m_dom = re.search(
                r"(?:领域|domain)[^\n。]*?(?:为|是|可能是|选择|select|label)[^A-Za-z]*([A-Za-z][A-Za-z_\-/]*)",
                clean,
                flags=re.I,
            )
            if m_dom:
                domain = m_dom.group(1).strip()

        # 2.2 歧义词：解释里所有英文词，筛到原句里确实出现过的
        if not amb_list:
            cand = set()
            # 引号里的内容
            for m_q in re.finditer(r'[“"](.*?)[”"]', clean):
                seg = m_q.group(1)
                for m_word in re.finditer(r"[A-Za-z][A-Za-z'\-]*", seg):
                    cand.add(m_word.group(0))
            # 所有英文单词
            for m_word in re.finditer(r"[A-Za-z][A-Za-z'\-]*", clean):
                cand.add(m_word.group(0))

            sent_lower = sent_bn.lower()
            tmp_list = []
            for w in cand:
                w_clean = w.strip().lower()
                if len(w_clean) < 2:
                    continue
                if w_clean in STOP_AMB:  # 过滤掉 no / yes 等
                    continue
                if w_clean in sent_lower:
                    tmp_list.append(w_clean)

            seen = set()
            amb_list = [x for x in tmp_list if not (x in seen or seen.add(x))]

    # ---------- 3. 收尾：根据歧义词强制 need ----------
    amb_list = [
        str(w).strip()
        for w in amb_list
        if str(w).strip() and str(w).strip().lower() not in STOP_AMB
    ]

    if amb_list:
        need = "yes"
    else:
        need = "no"

    if domain is None or not str(domain).strip():
        domain = "general"
    else:
        domain = str(domain).strip()

    return {
        "need_retrieval": need,
        "domain": domain,
        "ambiguous_terms": amb_list,
    }

# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant",
        type=str,
        default=DEFAULT_QUANT,
        choices=["4bit", "8bit", "none"],
    )
    args = parser.parse_args()

    print(f"[INFO] reading tsv from {IN_TSV}")
    df = pd.read_csv(IN_TSV, sep="\t")
    if ID_COL not in df.columns or TEXT_COL not in df.columns:
        raise ValueError(f"输入文件必须包含列: {ID_COL}, {TEXT_COL}")

    start_i = load_checkpoint()
    print(f"[INFO] resume from line index = {start_i}")

    if args.quant == "4bit":
        quant_list = ["4bit", "8bit", "none"]
    elif args.quant == "8bit":
        quant_list = ["8bit", "none"]
    else:
        quant_list = ["none"]

    q_idx = 0
    tokenizer, model = load_model(quant_list[q_idx])

    # 清空 debug 日志
    with open(RAW_LOG, "w", encoding="utf-8") as fdbg:
        fdbg.write("")

    for i in tqdm(range(start_i, len(df)), desc="Judging"):
        row = df.iloc[i]
        sent_id = row[ID_COL]
        sent_bn = str(row[TEXT_COL]).strip()

        if not sent_bn:
            out_row = {
                "id": sent_id,
                "bn": sent_bn,
                "need_retrieval": "no",
                "domain": "general",
                "ambiguous_terms": [],
            }
            append_tsv(out_row)
            save_checkpoint(i + 1)
            print(f"[DEBUG] id={sent_id}, empty sentence -> no/general")
            continue

        prompt = build_prompt(sent_bn)

        # 调用 LLM（带 OOM 回退）
        while True:
            try:
                raw = llm_once(tokenizer, model, prompt)
                break
            except torch.cuda.OutOfMemoryError:
                print(f"[WARN] OOM at i={i}, quant={quant_list[q_idx]}")
                free_model(model)
                q_idx += 1
                if q_idx >= len(quant_list):
                    raise RuntimeError("All quant modes OOM")
                tokenizer, model = load_model(quant_list[q_idx])
            except Exception as e:
                print(f"[WARN] error at i={i}: {e}")
                raw = ""
                break

        # 记录前 50 条原始输出
        if i < 50:
            with open(RAW_LOG, "a", encoding="utf-8") as fdbg:
                fdbg.write(f"===== id={sent_id} =====\n")
                fdbg.write(raw + "\n\n")

        parsed = safe_json(raw, sent_bn)

        out_row = {
            "id": sent_id,
            "bn": sent_bn,
            "need_retrieval": parsed["need_retrieval"],
            "domain": parsed["domain"],
            "ambiguous_terms": parsed["ambiguous_terms"],
        }
        append_tsv(out_row)
        save_checkpoint(i + 1)

        print(
            f"[DEBUG] id={sent_id}, need={out_row['need_retrieval']}, "
            f"domain={out_row['domain']}, amb={out_row['ambiguous_terms']}"
        )

    free_model(model)
    print("[INFO] done.")

if __name__ == "__main__":
    main()
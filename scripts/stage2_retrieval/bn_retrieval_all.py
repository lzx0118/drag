#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
bn_dual_retrieval.py

Stage1：背景检索（Qwen3-Embedding-8B + Chroma）【全库检索版】
- 不用候选池（不再 collection.query(n_results=...)）
- 直接拉取 lang=bn 全库 embeddings + docs
- streaming topK（分 chunk matmul），避免一次性 BxN 大矩阵
- 过滤：自匹配/近重复/时间碎片/过短/对白碎片
- 选 top3 并做 bg 之间去重兜底

Stage2：WordNet + Qwen3-32B 词义选择【保持你当前版本】
- WordNet：lemma -> synonyms 拆成 List[str]
- Qwen：在候选字符串列表里选一个编号
- chosen_senses：{term: chosen_string}

输出：
- 保存到 OUT_TSV（TSV）
- main() 返回 DataFrame
"""

import os
import json
import ast
import time
import re
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import chromadb
from chromadb.config import Settings

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


# ================== 路径与常量配置（按你自己的保持/修改） ==================

JUDGE_TSV = "<PATH_TO_JUDGE_TSV>"  # output of Stage-1 judge

CHROMA_DB_DIR = "<PATH_TO_CHROMA_DB_DIR>"
CHROMA_COLLECTION = "<CHROMA_COLLECTION_NAME>"
LANG_CODE = "bn"  # edit when switching languages, used to filter chroma metadata
TEXT_COL = "bn"   # source text column in TSV

EMB_MODEL_DIR = "<PATH_TO_TEXT_EMBEDDING_MODEL>"  # e.g., Qwen3-Embedding-8B
QWEN_LLM_PATH = "<PATH_TO_QWEN3_32B_OR_OTHER_LLM>"

WORDNET_TSV = "<PATH_TO_WORDNET_OR_SYNONYM_TSV>"
WN_LEMMA_COL = "lemma_bn"
WN_SYN_COL = "synonyms_bn"

# ✅ 结果保存路径
OUT_TSV = "<PATH_TO_OUTPUT_TSV>"

TOP_K_BG = 3
RETRIEVE_BATCH_SIZE = 16

# ===== Stage1 过滤参数（沿用你原逻辑）=====
BG_MIN_TOKENS = 6
BG_MAX_CHARS = 260
MIN_COS_SIM = 0.0

QUERY_NEAR_DUP_TRIGRAM = 0.95     # bg 与 query 太像就丢
DUP_BG_TRIGRAM = 0.92             # bg1/bg2/bg3 之间太像就去重

# ===== Stage1 全库检索参数（关键）=====
# 全库 topK 回溯深度：因为后面要过滤，top3 可能被过滤掉，所以需要更大的回溯。
GLOBAL_TOPK_FOR_FILTER = 5000

# KB streaming chunk size：越大越快但越吃显存/内存
KB_CHUNK_SIZE = 100_000

# collection.get 分页拉全库（若你的 Chroma 不支持 offset/limit，会自动 fallback 一次性拉取）
CHROMA_GET_LIMIT = 5000


# ================== utils ==================

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def parse_ambiguous_terms(cell: Any) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]

    s = str(cell).strip()
    if not s:
        return []
    s2 = s.replace('""', '"').strip()

    for cand in [s2, s]:
        try:
            v = json.loads(cand)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
        try:
            v = ast.literal_eval(cand)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass

    parts = re.split(r"[，,；; ]+", s2)
    return [p.strip().strip('"').strip("'") for p in parts if p.strip()]

def char_trigram_jaccard(a: str, b: str) -> float:
    a = re.sub(r"\s+", "", normalize_text(a))
    b = re.sub(r"\s+", "", normalize_text(b))
    if not a or not b:
        return 0.0
    if len(a) <= 3 or len(b) <= 3:
        return 1.0 if a == b else 0.0
    sa = {a[i:i+3] for i in range(len(a)-2)}
    sb = {b[i:i+3] for i in range(len(b)-2)}
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def looks_like_time_fragment_bn(s: str) -> bool:
    s = normalize_text(s)
    if re.search(r"[০-৯0-9]{1,2}\s*[:：]\s*[০-৯0-9]{1,2}", s):
        return True
    if re.search(r"[০-৯0-9]{1,2}\s*ট", s):
        return True
    if re.search(r"(মিনিট|ঘণ্টা|সেকেন্ড)", s):
        return True
    return False

def is_low_quality_bg(s: str) -> bool:
    """
    保持你原脚本的严格过滤：
    - 时间碎片
    - 过短/过长
    - 对白碎片（? / 引号）直接丢
    """
    s = normalize_text(s)
    if not s:
        return True
    if len(s) > BG_MAX_CHARS:
        return True
    if len(s.split()) < BG_MIN_TOKENS:
        return True
    if looks_like_time_fragment_bn(s):
        return True
    if s.count('"') >= 2 or s.count("“") >= 1 or s.count("?") >= 1:
        return True
    return False


# ================== Embedding 模型 ==================

class QwenLocalEmbedder:
    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        max_length: int = 1024,
        dtype: torch.dtype = torch.float16,
    ):
        self.model_dir = model_dir
        self.max_length = max_length
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        log(f"Loading Qwen3-Embedding from {model_dir} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model.eval()
        log("Qwen3-Embedding loaded.")

    @torch.no_grad()
    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        x = out.last_hidden_state
        mask = enc.get("attention_mask", torch.ones(x.size()[:2], device=x.device)).unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x.detach().cpu().float().tolist()

    def embed(self, texts: List[str]) -> List[List[float]]:
        texts = [normalize_text(t) for t in texts]
        out = []
        bs = RETRIEVE_BATCH_SIZE
        for i in range(0, len(texts), bs):
            out.extend(self._batch_embed(texts[i:i+bs]))
        return out


# ================== Chroma ==================

def get_chroma_collection() -> chromadb.Collection:
    log(f"Connecting to Chroma DB at {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR, settings=Settings(allow_reset=False))
    try:
        col = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        col = client.create_collection(CHROMA_COLLECTION)
    log(f"Using collection = {CHROMA_COLLECTION}")
    return col


# ================== Stage1：全库加载 + streaming topK ==================

def _try_put_on_device(t: torch.Tensor, prefer: str) -> torch.Tensor:
    if prefer != "cuda":
        return t.cpu()
    if not torch.cuda.is_available():
        return t.cpu()
    try:
        return t.cuda()
    except RuntimeError as e:
        log(f"[WARN] Move tensor to CUDA failed (OOM?). Fallback to CPU. err={e}")
        return t.cpu()

def load_full_kb_from_chroma(
    collection: chromadb.Collection,
    lang: str,
    prefer_device: str = "cuda",
) -> Tuple[List[str], torch.Tensor]:
    where = {"lang": {"$eq": lang}}
    docs: List[str] = []
    embs: List[List[float]] = []

    # 优先分页（offset）
    offset = 0
    paged_ok = True

    while True:
        try:
            got = collection.get(
                where=where,
                include=["documents", "embeddings"],
                limit=CHROMA_GET_LIMIT,
                offset=offset,
            )
        except TypeError:
            paged_ok = False
            break

        ids = got.get("ids", [])
        if not ids:
            break

        d = got.get("documents", [])
        e = got.get("embeddings", [])

        docs.extend([normalize_text(x) for x in d])
        embs.extend(e)

        offset += len(ids)
        log(f"[KB] loaded {offset} items ...")

        if len(ids) < CHROMA_GET_LIMIT:
            break

    if not paged_ok:
        log("[WARN] Chroma collection.get does NOT support offset/limit. "
            "Fallback to one-shot get (may require large RAM).")
        got = collection.get(where=where, include=["documents", "embeddings"])
        docs = [normalize_text(x) for x in got.get("documents", [])]
        embs = got.get("embeddings", [])

    if not docs:
        raise RuntimeError(f"No KB entries loaded for lang={lang}. Check collection/where filter.")

    C = torch.tensor(embs, dtype=torch.float32)
    C = torch.nn.functional.normalize(C, p=2, dim=1)
    C = _try_put_on_device(C, prefer_device)

    log(f"[KB] DONE. total={len(docs)}, dim={C.shape[1]}, device={C.device}")
    return docs, C

@torch.no_grad()
def streaming_topk_cosine_batch(
    C: torch.Tensor,  # [N, D] normalized
    Q: torch.Tensor,  # [B, D] normalized
    topk: int,
    chunk_size: int = KB_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert C.dim() == 2 and Q.dim() == 2
    N, _ = C.shape
    B = Q.shape[0]
    topk = min(topk, N)

    device = C.device
    Q = Q.to(device)

    best_scores = torch.full((B, topk), -1e9, device=device, dtype=torch.float32)
    best_idx = torch.full((B, topk), -1, device=device, dtype=torch.long)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Cc = C[start:end]  # [chunk, D]

        scores = torch.matmul(Cc, Q.transpose(0, 1))         # [chunk, B]
        scores_t = scores.transpose(0, 1).contiguous()        # [B, chunk]

        k2 = min(topk, scores_t.shape[1])
        s2, i2 = torch.topk(scores_t, k=k2, dim=1, largest=True, sorted=False)
        i2 = i2 + start

        merged_scores = torch.cat([best_scores, s2], dim=1)
        merged_idx = torch.cat([best_idx, i2], dim=1)

        s3, pos = torch.topk(merged_scores, k=topk, dim=1, largest=True, sorted=False)
        best_scores = s3
        best_idx = torch.gather(merged_idx, 1, pos)

    best_scores, order = torch.sort(best_scores, dim=1, descending=True)
    best_idx = torch.gather(best_idx, 1, order)
    return best_scores, best_idx

def retrieve_background_top3_full_kb(
    embedder: QwenLocalEmbedder,
    kb_docs: List[str],
    kb_embs: torch.Tensor,   # [N,D] normalized
    texts: List[str],
) -> Dict[str, List[Any]]:
    q_embs = embedder.embed(texts)  # list[list[float]]
    Q = torch.tensor(q_embs, dtype=torch.float32)
    Q = torch.nn.functional.normalize(Q, p=2, dim=1)
    Q = Q.to(kb_embs.device)

    scores, idxs = streaming_topk_cosine_batch(
        C=kb_embs,
        Q=Q,
        topk=GLOBAL_TOPK_FOR_FILTER,
        chunk_size=KB_CHUNK_SIZE,
    )

    out_bg1, out_bg2, out_bg3 = [], [], []
    out_s1, out_s2, out_s3 = [], [], []

    for q_text_raw, s_row, i_row in zip(texts, scores, idxs):
        q_text = normalize_text(q_text_raw)

        picked: List[Tuple[str, float]] = []
        for score, idx in zip(s_row.tolist(), i_row.tolist()):
            if idx < 0:
                continue
            d = kb_docs[idx]
            if not d:
                continue
            if d == q_text:
                continue
            if score < MIN_COS_SIM:
                continue
            if char_trigram_jaccard(d, q_text) >= QUERY_NEAR_DUP_TRIGRAM:
                continue
            if is_low_quality_bg(d):
                continue
            if any(char_trigram_jaccard(d, pd) >= DUP_BG_TRIGRAM for pd, _ in picked):
                continue

            picked.append((d, float(score)))
            if len(picked) >= TOP_K_BG:
                break

        while len(picked) < TOP_K_BG:
            picked.append(("", -999.0))

        out_bg1.append(picked[0][0]); out_s1.append(picked[0][1])
        out_bg2.append(picked[1][0]); out_s2.append(picked[1][1])
        out_bg3.append(picked[2][0]); out_s3.append(picked[2][1])

    return {
        "bg1": out_bg1, "bg2": out_bg2, "bg3": out_bg3,
        "bg1_score": out_s1, "bg2_score": out_s2, "bg3_score": out_s3,
    }


# ================== Stage2：词义选择（保持你当前版本） ==================

def split_synonyms(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    parts = re.split(r"[，,、;；\|]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if p and p not in out:
            out.append(p)
    return out

def load_wordnet_options(path: str) -> Dict[str, List[str]]:
    log(f"Loading Bengali WordNet from {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    for col in [WN_LEMMA_COL, WN_SYN_COL]:
        if col not in df.columns:
            raise ValueError(f"WordNet 文件缺少列: {col}")

    wn: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        lemma = str(row[WN_LEMMA_COL]).strip()
        if not lemma:
            continue
        syns = str(row.get(WN_SYN_COL, "") or "").strip()
        cands = split_synonyms(syns)
        if not cands:
            cands = [lemma]
        wn.setdefault(lemma, [])
        for c in cands:
            if c not in wn[lemma]:
                wn[lemma].append(c)

    log(f"WordNet loaded, unique lemmas = {len(wn)}")
    return wn

def load_qwen_llm() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    log(f"Loading Qwen3-32B from {QWEN_LLM_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_LLM_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_LLM_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model

def qwen_choose_sense_text(
    tokenizer,
    model,
    sentence: str,
    domain: str,
    term: str,
    candidates: List[str],
) -> str:
    domain = domain or "general"
    sentence = normalize_text(sentence)
    term = term.strip()

    cand_block = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

    system_prompt = (
        "你是一个严格的孟加拉语词义消歧助手。"
        "根据给定的句子与领域标签，从候选解释/同义短语中选择最合适的一个。"
    )

    user_content = f"""
/no_think
领域（英文标签）：{domain}

孟加拉语原句：
{sentence}

需要消歧的孟加拉词语：
{term}

候选解释/同义短语：
{cand_block}

请只输出一个阿拉伯数字（1开始），表示最佳候选编号，不要输出其它内容。
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"\d+", resp)
    if not m:
        return candidates[0]
    v = int(m.group(0))
    if 1 <= v <= len(candidates):
        return candidates[v - 1]
    return candidates[0]


# ================== 主流程：保存 + 返回 DataFrame ==================

def main(save_path: Optional[str] = OUT_TSV) -> pd.DataFrame:
    log(f"Reading judge TSV: {JUDGE_TSV}")
    df = pd.read_csv(JUDGE_TSV, sep="\t", dtype=str)

    required_cols = ["id", TEXT_COL, "need_retrieval", "domain", "ambiguous_terms"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"bn_judge.tsv 缺少列: {c}")

    sentences = df[TEXT_COL].fillna("").tolist()
    needs = df["need_retrieval"].fillna("").str.lower().tolist()
    domains = df["domain"].fillna("").tolist()
    amb_terms_list = [parse_ambiguous_terms(x) for x in df["ambiguous_terms"]]

    n = len(df)
    log(f"Total rows = {n}")

    # ================= Stage 1：全库检索（不使用候选池） =================
    log("=== Stage 1: Background retrieval (FULL KB; streaming topK; filter->top3) ===")
    embedder = QwenLocalEmbedder(EMB_MODEL_DIR)
    collection = get_chroma_collection()

    kb_docs, kb_embs = load_full_kb_from_chroma(collection, LANG_CODE, prefer_device="cuda")

    all_bg1, all_bg2, all_bg3 = [], [], []
    all_s1, all_s2, all_s3 = [], [], []

    bs = RETRIEVE_BATCH_SIZE
    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch = sentences[start:end]
        log(f"[BG] rows {start} ~ {end-1}")
        bg = retrieve_background_top3_full_kb(embedder, kb_docs, kb_embs, batch)
        all_bg1.extend(bg["bg1"]); all_bg2.extend(bg["bg2"]); all_bg3.extend(bg["bg3"])
        all_s1.extend(bg["bg1_score"]); all_s2.extend(bg["bg2_score"]); all_s3.extend(bg["bg3_score"])

    # 释放 Stage1 大对象，给 Stage2 腾显存
    del embedder, collection, kb_docs, kb_embs
    torch.cuda.empty_cache()
    log("Stage 1 done. Freed KB/embedding resources and cleared CUDA cache.")

    # ================= Stage 2：词义选择（保持不变） =================
    log("=== Stage 2: Sense selection (synonyms -> List[str], choose one) ===")
    wordnet = load_wordnet_options(WORDNET_TSV)
    qwen_tok, qwen_model = load_qwen_llm()

    chosen_senses_all: List[str] = []

    for i in range(n):
        if needs[i] != "yes":
            chosen_senses_all.append("{}")
            continue

        sent = sentences[i]
        domain = domains[i]
        amb_terms = amb_terms_list[i]

        clean_terms = []
        for t in amb_terms:
            t = (t or "").strip()
            if not t:
                continue
            if " " in t:
                continue
            if len(t) > 24:
                continue
            clean_terms.append(t)

        per: Dict[str, str] = {}
        for term in clean_terms:
            cands = wordnet.get(term)
            if not cands:
                per[term] = term
            elif len(cands) == 1:
                per[term] = cands[0]
            else:
                try:
                    per[term] = qwen_choose_sense_text(qwen_tok, qwen_model, sent, domain, term, cands)
                except Exception as e:
                    log(f"[WARN] sense selection failed term={term} i={i} err={e}")
                    per[term] = cands[0]

        chosen_senses_all.append(json.dumps(per, ensure_ascii=False) if per else "{}")

    del qwen_tok, qwen_model
    torch.cuda.empty_cache()
    log("Stage 2 done. Freed Qwen3-32B and cleared CUDA cache.")

    out_df = pd.DataFrame({
        "id": df["id"],
        TEXT_COL: df[TEXT_COL],
        "domain": df["domain"],
        "need_retrieval": df["need_retrieval"],
        "ambiguous_terms": df["ambiguous_terms"],
        "bg1": all_bg1, "bg1_score": all_s1,
        "bg2": all_bg2, "bg2_score": all_s2,
        "bg3": all_bg3, "bg3_score": all_s3,
        "chosen_senses": chosen_senses_all,
    })

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        log(f"Saving to: {save_path}")
        out_df.to_csv(save_path, sep="\t", index=False)

    log("Done. Returning DataFrame.")
    return out_df


if __name__ == "__main__":
    df_out = main(save_path=OUT_TSV)
    print("\n===== Preview (first 5 rows) =====")
    print(df_out.head(5).to_string(index=False))

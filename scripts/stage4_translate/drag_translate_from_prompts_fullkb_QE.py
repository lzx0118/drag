#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
drag_translate_from_prompts_fullkb_QE.py

基于 prompt JSONL 翻译（Qwen）并生成多个候选，用 COMET-QE 选择最优输出。

- 输入：{lang}.prompts.*.jsonl （每行包含 id/mode/system_prompt/user_prompt/src/src_col 等）
- 输出：TSV：id, <src_col>, translation_{mode}_en ...
- 缓存：<out_tsv>.cache.jsonl  (支持 --resume 跳过已完成 (id,mode))


LANGS="bn"   # 你要跑哪些就写哪些
MODES="drag"         # 你想同时跑就留着；只跑drag就改成 "drag"

CUDA_VISIBLE_DEVICES=$GPU python drag_translate_from_prompts_fullkb_QE.py \
  --model_path "$MODEL" --use_gpu \
  --qe_ckpt_path "$QE_CKPT" --qe_use_gpu --qe_batch_size 32 \
  --langs "$LANGS" \
  --prompt_root "$PROMPT_ROOT" \
  --prompt_pattern "{lang}_WMT/{lang}_WMT.prompts.normA.jsonl" \
  --output_root "$OUT_ROOT" \
  --output_pattern "{lang}_WMT/{lang}_WMT.translations.normA_QE.tsv" \
  --modes "$MODES" \
  --batch_size 4 --max_new_tokens 256 \
  --num_candidates 3 --temperature 0.8 --top_p 0.95 --top_k 50 \
  --seed 42 \
  --resume

离线：默认强制 OFFLINE（除非 --online）
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Set, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)

FIXED_LANGS = ["hu", "ms", "ur", "bn", "fa", "id", "hi", "gu", "ta", "mr", "ne", "vi", "uz"]


def set_offline_env():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def clean_translation(s: str) -> str:
    s = s or ""
    s = THINK_RE.sub("", s)
    s = s.replace("<think>", "").replace("</think>", "")
    s = s.strip()
    s = re.sub(r"^(translation\s*:|english\s*:)\s*", "", s, flags=re.IGNORECASE).strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    # 保持与你原脚本一致：只取第一行
    return lines[0] if lines else ""


def apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    msgs = []
    if (system_prompt or "").strip():
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, recs: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_done(cache_path: str) -> Set[Tuple[str, str]]:
    done: Set[Tuple[str, str]] = set()
    if not os.path.exists(cache_path):
        return done
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done.add((str(r.get("id", "")), str(r.get("mode", ""))))
    return done


def pivot_to_tsv(out_tsv: str, cache_path: str, modes: List[str]):
    by_id: Dict[str, Dict[str, Any]] = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rid = str(r.get("id", ""))
            mode = str(r.get("mode", ""))
            src_col = r.get("src_col", "src")
            if rid not in by_id:
                by_id[rid] = {"id": rid, src_col: r.get("src", ""), "_src_col": src_col}
            by_id[rid][f"translation_{mode}_en"] = r.get("translation", "")

    src_col = None
    for _, obj in by_id.items():
        src_col = obj.get("_src_col")
        break
    if src_col is None:
        src_col = "src"

    for rid, obj in by_id.items():
        for m in modes:
            k = f"translation_{m}_en"
            if k not in obj:
                obj[k] = ""

    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    cols = ["id", src_col] + [f"translation_{m}_en" for m in modes]
    with open(out_tsv, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(cols) + "\n")
        for rid in sorted(by_id.keys(), key=lambda x: (len(x), x)):
            obj = by_id[rid]
            f.write("\t".join([str(obj.get(c, "")) for c in cols]) + "\n")


@torch.no_grad()
def batch_generate_candidates(
    tokenizer,
    model,
    prompts: List[str],
    max_new_tokens: int,
    num_candidates: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[List[str]]:
    """
    return: List[bs][num_candidates] -> cleaned translations
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    do_sample = True if num_candidates > 1 else False

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=top_k if do_sample else None,
        num_beams=1,
        num_return_sequences=num_candidates,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    gen_ids = out[:, prompt_len:]
    texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    texts = [clean_translation(t) for t in texts]

    bs = len(prompts)
    # reshape to [bs][num_candidates]
    grouped = []
    idx = 0
    for _ in range(bs):
        grouped.append(texts[idx : idx + num_candidates])
        idx += num_candidates
    return grouped


def qe_rerank_batch(qe_model, batch_src: List[str], cand_texts: List[List[str]], qe_batch_size: int, qe_gpus: int):
    """
    batch_src: [bs]
    cand_texts: [bs][K]
    return:
      best_text: [bs]
      best_idx:  [bs]
      scores:    [bs][K]
    """
    bs = len(batch_src)
    K = len(cand_texts[0]) if bs > 0 else 0

    flat = []
    for i in range(bs):
        src = batch_src[i]
        for j in range(K):
            mt = cand_texts[i][j]
            # QE 需要 src / mt
            flat.append({"src": src, "mt": mt})

    pred = qe_model.predict(flat, batch_size=qe_batch_size, gpus=qe_gpus)
    flat_scores = list(pred.scores)

    scores = []
    best_text = []
    best_idx = []
    t = 0
    for i in range(bs):
        row = flat_scores[t : t + K]
        t += K
        scores.append(row)
        jbest = max(range(K), key=lambda j: row[j])
        best_idx.append(jbest)
        best_text.append(cand_texts[i][jbest])
    return best_text, best_idx, scores


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--use_gpu", action="store_true")

    ap.add_argument("--qe_ckpt_path", type=str, required=True, help="e.g. .../wmt21-comet-qe-da/checkpoints/model.local.ckpt")
    ap.add_argument("--qe_use_gpu", action="store_true")
    ap.add_argument("--qe_batch_size", type=int, default=32)

    ap.add_argument("--lang", type=str, default="")
    ap.add_argument("--langs", type=str, default=",".join(FIXED_LANGS))

    ap.add_argument("--prompt_root", type=str, required=True)
    ap.add_argument("--prompt_pattern", type=str, required=True)

    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--output_pattern", type=str, required=True)

    ap.add_argument("--modes", type=str, default="drag,direct,bg,bg_sense")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)

    ap.add_argument("--num_candidates", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()

    if not args.online:
        set_offline_env()

    # seed（采样可复现）
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    modes = [x.strip() for x in re.split(r"[,\s]+", args.modes) if x.strip()]
    langs = [args.lang.strip()] if args.lang.strip() else [x.strip() for x in re.split(r"[,\s]+", args.langs) if x.strip()]

    # ===== load generator =====
    if args.use_gpu and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    # ===== load QE =====
    from comet import load_from_checkpoint
    qe = load_from_checkpoint(args.qe_ckpt_path)
    qe.eval()
    qe_gpus = 1 if (args.qe_use_gpu and torch.cuda.is_available()) else 0

    for lang in langs:
        prompt_path = os.path.join(args.prompt_root, args.prompt_pattern.format(lang=lang))
        out_tsv = os.path.join(args.output_root, args.output_pattern.format(lang=lang))
        cache_path = out_tsv + ".cache.jsonl"

        if not os.path.exists(prompt_path):
            print(f"[WARN] skip {lang}: prompt not found {prompt_path}", flush=True)
            continue

        prompts = read_jsonl(prompt_path)
        if not prompts:
            print(f"[WARN] skip {lang}: empty prompt file", flush=True)
            continue

        done = load_done(cache_path) if args.resume else set()
        todo = [r for r in prompts if (str(r.get("id", "")), str(r.get("mode", ""))) not in done]
        print(f"[INFO] {lang}: total={len(prompts)} todo={len(todo)} resume={args.resume}", flush=True)

        bs = max(1, int(args.batch_size))
        K = max(1, int(args.num_candidates))

        for st in range(0, len(todo), bs):
            batch = todo[st : st + bs]

            full_prompts = [apply_chat_template(tokenizer, b.get("system_prompt", ""), b.get("user_prompt", "")) for b in batch]

            cand = batch_generate_candidates(
                tokenizer=tokenizer,
                model=model,
                prompts=full_prompts,
                max_new_tokens=args.max_new_tokens,
                num_candidates=K,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            # src 列：优先用 b["src"]，如果没有就尽量兜底
            batch_src = []
            for b in batch:
                src = b.get("src", "")
                if not src:
                    # 兜底：某些文件用 text 字段
                    src = b.get("text", "")
                batch_src.append(src)

            best_text, best_idx, scores = qe_rerank_batch(
                qe_model=qe,
                batch_src=batch_src,
                cand_texts=cand,
                qe_batch_size=args.qe_batch_size,
                qe_gpus=qe_gpus,
            )

            recs = []
            for b, bt, bi, sc, cands in zip(batch, best_text, best_idx, scores, cand):
                rec = dict(b)
                rec["translation"] = bt
                rec["num_candidates"] = K
                rec["qe_best_idx"] = bi
                rec["qe_scores"] = sc
                rec["candidates"] = cands
                recs.append(rec)

            append_jsonl(cache_path, recs)

            done_n = min(st + bs, len(todo))
            if args.log_every > 0 and (done_n % args.log_every == 0 or done_n == len(todo)):
                print(f"[INFO] {lang}: generated {done_n}/{len(todo)}", flush=True)

        pivot_to_tsv(out_tsv, cache_path, modes=modes)
        print(f"[OK] {lang} -> {out_tsv}", flush=True)


if __name__ == "__main__":
    main()
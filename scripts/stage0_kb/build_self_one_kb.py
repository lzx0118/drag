#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 Qwen3-Embedding-8B 构建多语言知识库 self_one（不依赖 LangChain）

- 输入 txt 目录：<PATH_TO_KB_TXT_DIR>
  目前文件示例：
    bn_all_merged_dedup.txt
    fa_all_merged_dedup.txt
    hu_all_merged_dedup.txt
    ms_merged_clean_dedup.txt
    ur_all_merged_dedup.txt
- 模型目录：<PATH_TO_EMB_MODEL_DIR>
- Chroma DB 目录：<PATH_TO_CHROMA_DB_DIR>
- Collection 名：self_one
- 断点文件：<db_dir>/checkpoints.json

补充知识库
python build_self_one_kb.py \
  --ingest \
  --data_dir <PATH_TO_KB_TXT_DIR> \
  --db_dir <PATH_TO_CHROMA_DB_DIR> \
  --collection self_one
用法：
  # 首次/增量构建（会自动断点续存）
  python build_self_one_kb.py --ingest

  # 指定其他目录或 DB 目录
  python build_self_one_kb.py --ingest \
      --data_dir <PATH_TO_KB_TXT_DIR> \
      --db_dir <PATH_TO_CHROMA_DB_DIR>

  # 检索示例（按语言过滤）
  python build_self_one_kb.py \
      --db_dir <PATH_TO_CHROMA_DB_DIR> \
      --query '{"lang":"bn","terms":["连衣裙","棉"]}' \
      --top_k 5
"""

import os
import re
import json
import time
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def stable_id(text: str, lang: str, lineno: int) -> str:
    h = hashlib.sha1(f"{lang}::{lineno}::{text}".encode("utf-8")).hexdigest()
    return f"{lang}_{lineno}_{h[:10]}"


class QwenLocalEmbedder:
    """本地 Qwen3-Embedding -> 句向量（平均池化 + L2）"""

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        max_length: int = 1024,
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ):
        self.model_dir = model_dir
        self.max_length = max_length
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        log(f"Loading Qwen3-Embedding from {model_dir} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
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
        if hasattr(out, "last_hidden_state"):
            x = out.last_hidden_state  # [B, L, H]
            mask = (
                enc.get(
                    "attention_mask",
                    torch.ones(x.size()[:2], device=x.device),
                )
                .unsqueeze(-1)
            )
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            x = out[0]
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x.detach().cpu().float().tolist()

    def embed(self, texts: List[str]) -> List[List[float]]:
        texts = [normalize_text(t) for t in texts]
        out, bs = [], 64
        for i in range(0, len(texts), bs):
            out.extend(self._batch_embed(texts[i : i + bs]))
        return out


def detect_lang_from_filename(fn: str) -> str:
    """根据文件名前缀猜测语言代码"""
    base = Path(fn).name.lower()
    if base.startswith("bn"):
        return "bn"
    if base.startswith("fa"):
        return "fa"
    if base.startswith("hu"):
        return "hu"
    if base.startswith("ms"):
        return "ms"
    if base.startswith("ur"):
        return "ur"
    if base.startswith("id"):
        return "id"
    # 后续你再加语言时，可以在这里继续加分支
    return "unk"


def iter_lines_with_resume(filepath: str, start_line: int = 0):
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_line:
                continue
            yield idx, line.rstrip("\n")


def save_ckpt(cp_path: Path, state: Dict[str, Any]):
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_ckpt(cp_path: Path) -> Dict[str, Any]:
    if cp_path.exists():
        with open(cp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"files": {}}


def get_collection(client: chromadb.PersistentClient, name: str):
    # 如果已存在就返回，不存在则创建
    try:
        col = client.get_collection(name)
    except Exception:
        col = client.create_collection(name)
    return col


def ingest_all(
    data_dir: str,
    db_dir: str,
    model_dir: str,
    collection: str = "self_one",
    batch_size: int = 512,
    max_lines: Optional[int] = None,
):
    embedder = QwenLocalEmbedder(model_dir=model_dir)

    os.makedirs(db_dir, exist_ok=True)
    client = chromadb.PersistentClient(
        path=db_dir, settings=Settings(allow_reset=False)
    )
    col = get_collection(client, name=collection)

    cp_path = Path(db_dir) / "checkpoints.json"
    ckpt = load_ckpt(cp_path)

    files = sorted(
        [str(p) for p in Path(data_dir).glob("*.txt")],
        key=lambda p: Path(p).name,
    )
    if not files:
        log(f"[WARN] No txt under {data_dir}")
        return

    for fp in files:
        lang = detect_lang_from_filename(fp)
        start = int(ckpt["files"].get(fp, 0))
        log(f"Ingest {fp} (lang={lang}) from line {start} ...")

        ids, docs, metas = [], [], []
        added, dropped = 0, 0

        for lineno, raw in iter_lines_with_resume(fp, start):
            text = normalize_text(raw)
            if not text:
                dropped += 1
                continue

            rid = stable_id(text, lang, lineno)
            ids.append(rid)
            docs.append(text)
            metas.append(
                {
                    "lang": lang,
                    "source_file": Path(fp).name,
                    "lineno": lineno,
                }
            )

            if len(docs) >= batch_size:
                embs = embedder.embed(docs)
                col.add(
                    ids=ids,
                    documents=docs,
                    metadatas=metas,
                    embeddings=embs,
                )
                ids, docs, metas = [], [], []
                added += len(embs)

                ckpt["files"][fp] = lineno + 1
                save_ckpt(cp_path, ckpt)
                log(f"  added={added} dropped={dropped} (last line={lineno})")

                if max_lines and added >= max_lines:
                    break

        if docs:
            embs = embedder.embed(docs)
            col.add(
                ids=ids, documents=docs, metadatas=metas, embeddings=embs
            )
            added += len(embs)
            ckpt["files"][fp] = lineno + 1 if "lineno" in locals() else start
            save_ckpt(cp_path, ckpt)
            log(f"  added={added} dropped={dropped} (final flush)")

        log(f"Done {fp}: total_added={added}, total_dropped={dropped}")

    log("All files ingested.")


def query_terms(
    db_dir: str,
    model_dir: str,
    collection: str,
    query_json: str,
    top_k: int = 5,
):
    payload = json.loads(query_json)
    terms: List[str] = payload.get("terms", [])
    lang: Optional[str] = payload.get("lang")

    if not terms:
        raise ValueError("query terms 为空")

    embedder = QwenLocalEmbedder(model_dir=model_dir)

    client = chromadb.PersistentClient(
        path=db_dir, settings=Settings(allow_reset=False)
    )
    col = get_collection(client, name=collection)

    query_text = " ".join(terms)
    qvec = embedder.embed([query_text])[0]

    where = {}
    if lang:
        where = {"lang": {"$eq": lang}}

    res = col.query(
        query_embeddings=[qvec],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    print("\n=== Top-K Results ===")
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i, (doc, md, dist) in enumerate(zip(docs, metas, dists), 1):
        print(
            f"[{i}] dist={dist:.4f} | lang={md.get('lang')} | "
            f"file={md.get('source_file')} | line={md.get('lineno')}"
        )
        print(doc)
        print("-" * 80)



def _assert_config_ok(path_value: str, name: str):
    """Fail fast if the user forgot to replace placeholder paths."""
    if isinstance(path_value, str) and path_value.strip().startswith("<PATH_TO_"):
        raise ValueError(
            f"{name} is a placeholder: {path_value}. "
            f"Please set --{name} to a real local path."
        )


def main():
    p = argparse.ArgumentParser(
        description="Build multi-lingual KB self_one with Qwen3-Embedding-8B + Chroma"
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="<PATH_TO_KB_TXT_DIR>",
        help="txt 数据所在目录",
    )
    p.add_argument(
        "--db_dir",
        type=str,
        default="<PATH_TO_CHROMA_DB_DIR>",
        help="Chroma DB 目录",
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default="<PATH_TO_EMB_MODEL_DIR>",
        help="Qwen3-Embedding-8B 模型目录",
    )
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--max_lines", type=int, default=0)
    p.add_argument(
        "--collection",
        type=str,
        default="self_one",
        help="Chroma collection 名称（知识库名）",
    )
    p.add_argument("--ingest", action="store_true")
    p.add_argument("--query", type=str, default="")
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()

    # PLACEHOLDER_GUARD
    _assert_config_ok(args.data_dir, 'data_dir')
    _assert_config_ok(args.db_dir, 'db_dir')
    _assert_config_ok(args.model_dir, 'model_dir')

    if args.ingest:
        ingest_all(
            data_dir=args.data_dir,
            db_dir=args.db_dir,
            model_dir=args.model_dir,
            collection=args.collection,
            batch_size=args.batch,
            max_lines=(args.max_lines or None),
        )

    if args.query:
        query_terms(
            db_dir=args.db_dir,
            model_dir=args.model_dir,
            collection=args.collection,
            query_json=args.query,
            top_k=args.top_k,
        )

    if not args.ingest and not args.query:
        print("Nothing to do. Use --ingest and/or --query.")


if __name__ == "__main__":
    main()

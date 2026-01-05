# DRAG / D³RAG (FullKB) — Anonymous Repro Package

This repository provides a **minimal, script-only** reproduction of the DRAG/D³RAG pipeline:
- **Judge** (LLM routing): domain + ambiguous terms + `need_retrieval`
- **Retrieval** (FullKB): background top-3 + synonym/sense selection (WordNet-style resource)
- **Prompt builder (Norm-A)**: build structured prompts (JSONL)
- **Translate + QE**: generate multiple candidates and select the best with **COMET-QE**

All user-specific absolute paths are replaced by placeholders like `<PATH_TO_...>`.
To run the code, edit the **CONFIG blocks** (Stage 1/2) and/or pass CLI args (Stage 3/4).

## Repository layout

```
scripts/
  stage0_kb/                  # (optional) KB / synonym resource preparation
  stage1_judge/               # Judge (default: bn template)
  stage2_retrieval/           # Retrieval + sense selection (default: bn template)
  stage3_prompt/              # Prompt builder (CLI)
  stage4_translate/           # Translation + QE reranking (CLI)
docs/
  PIPELINE.md                 # stage-by-stage overview
  FORMATS.md                  # input/output formats
  README_legacy.md            # original flat-file README (for reference)
tools/
  scan_anonymity.py           # helps check for accidental leaks
data/                         # (optional) place released TSV/JSONL here
outputs/                      # (optional) outputs will be written here
```

## Requirements

Install packages listed in `requirements.txt` (GPU + CUDA recommended).  
For offline environments, set (optional):

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

## Quick start (bn → en)

### Stage 1 — Judge
Edit the CONFIG block at the top of:
- `scripts/stage1_judge/bn_judge_need_test.py`

Then run:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/stage1_judge/bn_judge_need_test.py
```

### Stage 2 — Retrieval (FullKB) + sense selection
Edit the CONFIG block at the top of:
- `scripts/stage2_retrieval/bn_retrieval_all.py`

Then run:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/stage2_retrieval/bn_retrieval_all.py
```

### Stage 3 — Build prompts (Norm-A)
```bash
python scripts/stage3_prompt/drag_build_prompts_normA_fullkb.py \
  --lang bn \
  --input_tsv <PATH_TO_STAGE2_TSV> \
  --output_dir <PATH_TO_PROMPT_JSONL_DIR> \
  --modes drag
```

### Stage 4 — Translate + COMET-QE rerank
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/stage4_translate/drag_translate_from_prompts_fullkb_QE.py \
  --lang bn \
  --prompt_jsonl <PATH_TO_PROMPT_JSONL> \
  --output_root <PATH_TO_OUTPUT_DIR> \
  --num_candidates 3 \
  --resume
```

## Switching languages

Stage 1/2 scripts are **language templates**. To run another language `xx`:
1. Copy `scripts/stage1_judge/bn_judge_need_test.py` → `xx_judge_need_test.py`
2. Copy `scripts/stage2_retrieval/bn_retrieval_all.py` → `xx_retrieval_all.py`
3. Edit the CONFIG variables:
   - input/output TSV paths
   - `TEXT_COL` (source text column name)
   - language name used in prompts (if present)
   - checkpoint/model paths

Stage 3/4 are multi-language through CLI flags.

## Documentation

- Stage-by-stage pipeline: `docs/PIPELINE.md`
- File formats: `docs/FORMATS.md`
- (Reference) original release notes: `docs/README_legacy.md`

## Anonymity check

Before uploading, you can scan for common absolute paths:

```bash
python tools/scan_anonymity.py --root .
```

## License

Code is released under the MIT License (see `LICENSE`).


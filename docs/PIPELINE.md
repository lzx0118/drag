# Pipeline Overview (DRAG / D³RAG FullKB)

This package contains a **minimal, script-based** reproduction of the DRAG/D³RAG pipeline used in the paper.
All paths are anonymized using placeholders like `<PATH_TO_...>`.

## Stages

### Stage 0 (optional): Build a monolingual KB index (Chroma)
- Script: `scripts/stage0_kb/build_self_one_kb.py`
- Output: a Chroma persistent directory containing sentence embeddings and texts.

### Stage 0b (optional): Extract synonym candidates from a WordNet-style resource
- Script: `scripts/stage0_kb/build_ms_synonyms_from_wn.py`
- Output: TSV/CSV with `term`, `candidates`, `gloss` (format depends on the resource).

### Stage 1: Judge (domain + ambiguous terms + retrieval routing)
- Script: `scripts/stage1_judge/bn_judge_need_test.py`
- Input: TSV with at least `id` and a source text column (e.g., `bn`)
- Output: TSV with `domain`, `need_retrieval`, `ambiguous_terms` (and raw logs for debugging)

### Stage 2: Retrieval (FullKB) + sense selection
- Script: `scripts/stage2_retrieval/bn_retrieval_all.py`
- Input: Stage-1 TSV; Chroma KB; synonym TSV
- Output: TSV extended with:
  - background sentences: `bg1,bg1_score,bg2,bg2_score,bg3,bg3_score`
  - selected senses/gloss: `chosen_senses` (JSON string)

### Stage 3: Prompt building (Norm-A)
- Script: `scripts/stage3_prompt/drag_build_prompts_normA_fullkb.py`
- Input: Stage-2 TSV
- Output: prompt JSONL (one example per line)

### Stage 4: Translation + QE reranking (COMET-QE)
- Script: `scripts/stage4_translate/drag_translate_from_prompts_fullkb_QE.py`
- Input: prompt JSONL
- Output: TSV with candidate translations + selected best translation; cache for resume.

## Notes on reproducibility
- Stage 1/2 scripts are **language templates** (default: bn). To run another language `xx`, copy the scripts and edit the CONFIG block at the top.
- Stage 3/4 scripts are multi-language via CLI arguments.


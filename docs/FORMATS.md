# File Formats

## Stage 1 input TSV
Required columns:
- `id`: unique identifier
- `<src_text_col>`: source sentence (e.g., `bn`, `ur`, `id_text`, ...)

## Stage 1 output TSV (Judge)
Adds:
- `domain`: predicted domain label
- `need_retrieval`: `yes/no`
- `ambiguous_terms`: JSON list (string)

## Stage 2 output TSV (Retrieval + sense)
Adds:
- `bg1,bg1_score,bg2,bg2_score,bg3,bg3_score`
- `chosen_senses`: JSON object (string) mapping term -> selected sense/gloss/candidates

## Prompt JSONL (Stage 3)
Each line is a JSON dict with at least:
- `id`
- `lang`
- `mode`
- `prompt` (final prompt string sent to the translator)

## Translation output TSV (Stage 4)
Adds:
- `translation_<mode>_en` (final selected)
- optional intermediate columns for candidates and QE scores
- `*.cache.jsonl` for resume/debug

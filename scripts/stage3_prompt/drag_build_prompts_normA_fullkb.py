#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
drag_build_prompts_normA_fullkb.py

FULL-DATA prompt builder (no split), input = {lang}_retrieval_fullkb.tsv in one root dir.

TSV columns (your files):
  id, <lang_col>, domain, need_retrieval, ambiguous_terms,
  bg1,bg1_score,bg2,bg2_score,bg3,bg3_score, chosen_senses

Norm-A (aligns with translate_not_need_direct_bg_v2.py style):
- Evidence normalization uses NFC + remove ASCII/C1/bidi controls + remove U+FFFD + collapse spaces
- KEEP ZWJ/ZWNJ (U+200D/U+200C) intact
- Background lines de-duplicated

Key fix:
- If chosen_senses is empty/unparseable, fallback to ambiguous_terms as caution terms (if appears in source).

Gating (full-data main exp):
- if need_retrieval != yes:
    sense -> direct
    bg_sense -> bg
    drag -> bg
- if need_retrieval == yes:
    drag -> bg_sense
"""

import argparse
import ast
import csv
import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Tuple

# avoid csv field limit errors
csv.field_size_limit(1024 * 1024 * 50)

FIXED_LANGS = ["hu", "ms", "ur", "bn", "fa", "id", "hi", "gu", "ta", "mr", "ne", "vi", "uz"]

LANG_CONFIG: Dict[str, Dict[str, str]] = {
    "hu": {"col": "hu", "name": "Hungarian"},
    "ms": {"col": "ms", "name": "Malay"},
    "ur": {"col": "ur", "name": "Urdu"},
    "bn": {"col": "bn", "name": "Bengali"},
    "fa": {"col": "fa", "name": "Persian"},
    "id": {"col": "id_text", "name": "Indonesian"},  # will auto-fallback if not present
    "hi": {"col": "hi", "name": "Hindi"},
    "gu": {"col": "gu", "name": "Gujarati"},
    "ta": {"col": "ta", "name": "Tamil"},
    "mr": {"col": "mr", "name": "Marathi"},
    "ne": {"col": "npi", "name": "Nepali"},          # will auto-fallback if not present
    "npi": {"col": "npi", "name": "Nepali"},
    "vi": {"col": "vi", "name": "Vietnamese"},
    "uz": {"col": "uz", "name": "Uzbek"},
}

SPACE_RE = re.compile(r"\s+")
ASCII_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
C1_CTRL_RE = re.compile(r"[\u0080-\u009f]")
BIDI_CTRL_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")  # keep 200c/200d


def normA_evidence(text: str, bg_max_chars: int = 0) -> str:
    if text is None:
        return ""
    t = str(text)
    t = unicodedata.normalize("NFC", t)   # keep ZWJ/ZWNJ
    t = t.replace("\ufffd", "")
    t = t.replace("\u00a0", " ")
    t = t.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    t = ASCII_CTRL_RE.sub("", t)
    t = C1_CTRL_RE.sub("", t)
    t = BIDI_CTRL_RE.sub("", t)
    t = SPACE_RE.sub(" ", t).strip()
    if bg_max_chars and bg_max_chars > 0 and len(t) > bg_max_chars:
        t = t[:bg_max_chars].rstrip()
    return t


def normA_for_match(text: str, lang: str) -> str:
    x = normA_evidence(text)
    if lang in ("hu", "ms", "id", "vi", "uz"):
        x = x.lower()
    return x


def clean_token(x: str) -> str:
    x = (x or "").strip()
    x = SPACE_RE.sub(" ", x)
    if len(x) >= 2 and x[0] == x[-1] and x[0] in ("'", '"'):
        x = x[1:-1].strip()
    return x


def _try_parse_json_or_literal(s: str):
    s = (s or "").strip()
    if not s:
        return None
    if s in ("{}", "[]", "null", "None", "[NO_SENSE_SELECTION]", "[NO_SENSE]"):
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def parse_chosen_senses_dict(s: str, do_norm: bool) -> Dict[str, str]:
    obj = _try_parse_json_or_literal(s)
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in obj.items():
        k0 = clean_token(str(k))
        v0 = clean_token(str(v))
        if do_norm:
            k0 = normA_evidence(k0)
            v0 = normA_evidence(v0)
        if k0:
            out[k0] = v0
    return out


def parse_ambiguous_terms_list(s: str) -> List[str]:
    obj = _try_parse_json_or_literal(s)
    terms: List[str] = []
    if isinstance(obj, list):
        for x in obj:
            t = clean_token(str(x))
            if t:
                terms.append(t)
        return terms

    # fallback: split by comma/semicolon
    raw = (s or "").strip()
    if not raw:
        return []
    parts = re.split(r"[;,，；]\s*", raw)
    for p in parts:
        p = clean_token(p)
        if p:
            terms.append(p)
    return terms


def looks_noisy_value(v: str) -> bool:
    if not v:
        return True
    if len(v) > 80:
        return True
    if sum(v.count(p) for p in [",", ";", ":", "|", "/", "\\", "=", "(", ")", "[", "]", "{", "}"]) >= 10:
        return True
    return False


def extract_disambig_and_caution(
    src_raw: str,
    chosen_senses_raw: str,
    ambiguous_terms_raw: str,
    lang: str,
    do_norm: bool,
    max_items: int = 12,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    src_raw = (src_raw or "").strip()
    src_m = normA_for_match(src_raw, lang) if do_norm else src_raw

    sense_dict = parse_chosen_senses_dict(chosen_senses_raw, do_norm=do_norm)

    disambig_pairs: List[Tuple[str, str]] = []
    caution_terms: List[str] = []
    seen = set()

    # 1) from chosen_senses
    for term, sense in sense_dict.items():
        term = clean_token(term)
        sense = clean_token(sense)
        if do_norm:
            term = normA_evidence(term)
            sense = normA_evidence(sense)

        if not term:
            continue
        term_m = normA_for_match(term, lang) if do_norm else term
        if term_m and term_m in src_m:
            if term in seen:
                continue
            seen.add(term)
            if sense and (sense != term) and (not looks_noisy_value(sense)):
                disambig_pairs.append((term, sense))
            else:
                caution_terms.append(term)

        if len(disambig_pairs) + len(caution_terms) >= max_items:
            break

    # 2) fallback: ambiguous_terms -> caution (only if chosen_senses gives nothing useful)
    amb_terms = parse_ambiguous_terms_list(ambiguous_terms_raw)
    disambig_terms = {t for t, _ in disambig_pairs}
    for t in amb_terms:
        if len(disambig_pairs) + len(caution_terms) >= max_items:
            break
        if t in disambig_terms or t in caution_terms:
            continue
        tm = normA_for_match(t, lang) if do_norm else t
        if tm and tm in src_m:
            caution_terms.append(t)

    return disambig_pairs, caution_terms


def format_bg_block(bg1: str, bg2: str, bg3: str, lang: str, do_norm: bool, bg_max_chars: int = 0) -> str:
    lines = []
    for t in [bg1, bg2, bg3]:
        t = (t or "").strip()
        if not t:
            continue
        if do_norm:
            t = normA_evidence(t, bg_max_chars=bg_max_chars)
        lines.append(t)

    # de-dup
    deduped = []
    seen = set()
    for s in lines:
        if s and s not in seen:
            deduped.append(s)
            seen.add(s)

    if not deduped:
        return "[NO_BACKGROUND]"
    return "\n".join([f"{i+1}. {t}" for i, t in enumerate(deduped)])


def format_disambig_block(pairs: List[Tuple[str, str]]) -> str:
    if not pairs:
        return "[NO_DISAMBIGUATED_TERMS]"
    return "\n".join([f"- {t} => {s}" for t, s in pairs])


def format_caution_block(terms: List[str]) -> str:
    if not terms:
        return "[NO_CAUTION_TERMS]"
    return ", ".join(terms)


def build_user_prompt(
    lang_name: str,
    src_raw: str,
    mode: str,
    bg_block: str,
    disambig_block: str,
    caution_block: str,
) -> str:
    base_rules = (
        "Task: Translate the source sentence into fluent and faithful English.\n"
        "Rules:\n"
        "- Output ONLY the English translation (no explanations, no notes).\n"
        "- Do NOT add information not present in the source.\n"
        "- Do NOT copy source-language words into English.\n"
    )

    if mode == "direct":
        return f"""Source ({lang_name}):
{src_raw}

{base_rules}""".strip()

    if mode == "bg":
        return f"""Source ({lang_name}):
{src_raw}

Background (optional; may be noisy):
{bg_block}

{base_rules}
- Use background only if it helps disambiguate the meaning in the source.""".strip()

    if mode == "sense":
        return f"""Source ({lang_name}):
{src_raw}

Disambiguated terms (monolingual; apply ONLY if the source term appears in the source; do NOT copy to output):
{disambig_block}

Caution terms (ambiguous; translate carefully; do not mistranslate; do not leave untranslated):
{caution_block}

{base_rules}
- For each disambiguated term, interpret the source term according to the given monolingual sense, then translate into English.
- For caution terms, rely on context and be extra careful.""".strip()

    if mode == "bg_sense":
        return f"""Source ({lang_name}):
{src_raw}

Background (optional; may be noisy):
{bg_block}

Disambiguated terms (monolingual; apply ONLY if the source term appears in the source; do NOT copy to output):
{disambig_block}

Caution terms (ambiguous; translate carefully; do not mistranslate; do not leave untranslated):
{caution_block}

{base_rules}
- Use background only if it helps disambiguate.
- Apply disambiguated terms when relevant; be careful for caution terms.""".strip()

    return f"""Source ({lang_name}):
{src_raw}

{base_rules}""".strip()


def resolve_src_col_and_name(lang: str, header: List[str]) -> Tuple[str, str]:
    # prefer config; fallback to existing header
    if lang in LANG_CONFIG:
        preferred = LANG_CONFIG[lang]["col"]
        name = LANG_CONFIG[lang]["name"]
    else:
        preferred = lang
        name = lang

    # robust fallback for your real TSVs
    candidates = [preferred]
    if lang == "id":
        candidates += ["id_text", "id"]
    if lang in ("ne", "npi"):
        candidates += ["npi", "ne"]
    candidates += [lang]  # final fallback

    for c in candidates:
        if c in header:
            return c, name

    raise SystemExit(f"[ERROR] {lang}: cannot find source column. candidates={candidates}, header={header[:30]}")


def effective_mode(mode: str, need_yes: bool) -> str:
    mode = mode.strip()
    if mode == "drag":
        return "bg_sense" if need_yes else "bg"
    if not need_yes:
        if mode == "sense":
            return "direct"
        if mode == "bg_sense":
            return "bg"
    return mode


def read_tsv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [dict(r) for r in reader]
        return reader.fieldnames or [], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", type=str, default="")
    ap.add_argument("--langs", type=str, default=",".join(FIXED_LANGS))

    ap.add_argument("--input_root", type=str, required=True)
    ap.add_argument("--input_pattern", type=str, default="{lang}_retrieval_fullkb.tsv")

    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--output_pattern", type=str, default="{lang}/{lang}.prompts.normA.jsonl")

    ap.add_argument("--modes", type=str, default="drag,direct,bg,bg_sense")
    ap.add_argument("--lang_norm", action="store_true")
    ap.add_argument("--bg_max_chars", type=int, default=0)
    ap.add_argument("--no_think_prefix", action="store_true")
    ap.add_argument("--id_col", type=str, default="id")
    args = ap.parse_args()

    langs = [args.lang.strip()] if args.lang.strip() else [x.strip() for x in re.split(r"[,\s]+", args.langs) if x.strip()]
    modes = [x.strip() for x in re.split(r"[,\s]+", args.modes) if x.strip()]

    for lang in langs:
        in_path = os.path.join(args.input_root, args.input_pattern.format(lang=lang))
        out_path = os.path.join(args.output_root, args.output_pattern.format(lang=lang))
        if not os.path.exists(in_path):
            print(f"[WARN] skip {lang}: not found {in_path}", flush=True)
            continue

        header, rows = read_tsv(in_path)
        if not rows:
            print(f"[WARN] skip {lang}: empty {in_path}", flush=True)
            continue

        src_col, lang_name = resolve_src_col_and_name(lang, header)

        system_prompt = (
            f"You are a professional {lang_name}-to-English translator. "
            f"Output MUST be English only. Output ONLY the translation. "
            f"Do not add information not present in the source."
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        n_out = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                rid = r.get(args.id_col, "")
                src_raw = r.get(src_col, "")

                need_yes = (r.get("need_retrieval", "") or "").strip().lower() == "yes"
                chosen_raw = r.get("chosen_senses", "")  # your real column
                amb_raw = r.get("ambiguous_terms", "")

                bg_block = format_bg_block(
                    r.get("bg1", ""), r.get("bg2", ""), r.get("bg3", ""),
                    lang=lang, do_norm=args.lang_norm, bg_max_chars=args.bg_max_chars
                )

                pairs, cautions = extract_disambig_and_caution(
                    src_raw=src_raw,
                    chosen_senses_raw=chosen_raw,
                    ambiguous_terms_raw=amb_raw,
                    lang=lang,
                    do_norm=args.lang_norm,
                )
                disambig_block = format_disambig_block(pairs)
                caution_block = format_caution_block(cautions)

                for mode in modes:
                    eff = effective_mode(mode, need_yes)
                    user_prompt = build_user_prompt(
                        lang_name=lang_name,
                        src_raw=src_raw,  # NEVER normalize source
                        mode=eff,
                        bg_block=bg_block,
                        disambig_block=disambig_block,
                        caution_block=caution_block,
                    )
                    if args.no_think_prefix:
                        user_prompt = "/no_think\n" + user_prompt

                    rec = {
                        "id": rid,
                        "lang": lang,
                        "mode": mode,
                        "effective_mode": eff,
                        "src_col": src_col,
                        "src": src_raw,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_out += 1

        print(f"[OK] NormA {lang} -> {out_path} (rows={len(rows)}, prompts={n_out})", flush=True)


if __name__ == "__main__":
    main()

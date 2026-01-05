#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 wn-msa-all.tab 构造马来语(ms)单语近义词词典

输入：wn-msa-all.tab  (OMW Malay / msa WordNet)
  每行典型格式：
      synset_id \t B/I/M \t L/M/O/X/Y \t 词 或 释义

  - 以 "#" 开头的是注释行，跳过
  - 第二列:
        B 或 M  = lemma 行（近义词候选）
        I       = 释义行（不要）
  - 我们只取第二列为 B/M 的行，把第四列当作马来语词形

输出：ms_synonyms.tsv
  - 表头：lemma_ms  synonyms_ms
  - 每行：
        lemma_ms      某个马来词
        synonyms_ms   与它同 synset 出现的其它马来近义词（用英文逗号连接）
"""

import os
import csv
import argparse
from collections import defaultdict

# ===== 按实际情况改路径 =====
INPUT_TAB = "<PATH_TO_WN_TAB_FILE>"
OUTPUT_TSV = "<PATH_TO_OUTPUT_TSV>"
# ============================



def parse_args():
    p = argparse.ArgumentParser(description="Build a monolingual synonym TSV from OMW WordNet tab file (ms/msa).")
    p.add_argument("--input_tab", type=str, default=INPUT_TAB, help="Path to wn-msa-all.tab (or equivalent).")
    p.add_argument("--output_tsv", type=str, default=OUTPUT_TSV, help="Path to output TSV.")
    return p.parse_args()



def main():
    args = parse_args()
    input_tab = args.input_tab
    output_tsv = args.output_tsv

    if input_tab.strip().startswith('<PATH_TO_'):
        raise ValueError('Please set --input_tab to a real file path.')
    if output_tsv.strip().startswith('<PATH_TO_'):
        raise ValueError('Please set --output_tsv to a real file path.')

    if not os.path.isfile(input_tab):
        raise FileNotFoundError(input_tab)

    # synset_id -> set(lemmas)
    synset2lemmas = defaultdict(set)

    with open(input_tab, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            synset_id, col2, col3, lemma = (
                parts[0],
                parts[1],
                parts[2],
                parts[3].strip(),
            )

            # 只要 B/M 行，I 行是释义
            if col2 not in ("B", "M"):
                continue
            if not lemma:
                continue

            synset2lemmas[synset_id].add(lemma)

    # lemma -> set(all synonyms from all its synsets)
    lemma2syns = defaultdict(set)

    for synset_id, lemmas in synset2lemmas.items():
        if len(lemmas) <= 1:
            # 一个 synset 只有一个词就不产生近义词关系
            continue
        for w in lemmas:
            for v in lemmas:
                if v != w:
                    lemma2syns[w].add(v)

    out_dir = os.path.dirname(output_tsv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["lemma_ms", "synonyms_ms"])
        for lemma in sorted(lemma2syns.keys()):
            syns = sorted(lemma2syns[lemma])
            if not syns:
                continue
            syn_str = ", ".join(syns)  # 近义词用逗号连接
            writer.writerow([lemma, syn_str])

    print(f"Done. 写出 {len(lemma2syns)} 个 ms 词条到 {output_tsv}")


if __name__ == "__main__":
    main()

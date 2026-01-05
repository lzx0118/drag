#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan the repository for potential de-anonymizing strings or absolute paths."""

import argparse, os, re, sys

DEFAULT_PATTERNS = [
    r"/mnt/",
    r"/home/",
    r"C:\\Users\\",
    r"D:\\",
    r"\\Users\\",
    r"@.*\.edu",
]

def iter_files(root):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(('.py','.md','.txt','.tsv','.json','.jsonl','.yaml','.yml')):
                yield os.path.join(dp, fn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='root directory to scan')
    ap.add_argument('--pattern', action='append', default=[], help='extra regex pattern(s)')
    args = ap.parse_args()

    patterns = DEFAULT_PATTERNS + args.pattern
    regs = [re.compile(p) for p in patterns]

    hit = False
    for fp in iter_files(args.root):
        try:
            txt = open(fp, 'r', encoding='utf-8', errors='ignore').read()
        except Exception:
            continue
        for rg in regs:
            if rg.search(txt):
                print(f"[HIT] {fp}  pattern={rg.pattern}")
                hit = True
    if hit:
        sys.exit(1)
    print("No matches found.")

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""merge_matrix.py -- combine per-source shards into one matrix file.

The CPU sweep runs one process per source domain so that the 16 cores are used in
parallel; each shard writes its own JSON to avoid concurrent-write clobbering.

Usage:
  python scripts/experiments/merge_matrix.py ms_shard_*.json --out multisource_matrix.json
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "multisource_matrix.json"))
    args = ap.parse_args()

    files = []
    for p in args.patterns:
        hit = sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
        files += hit
    if not files:
        raise SystemExit("no shard files matched")

    merged = {}
    if os.path.exists(args.out):
        merged = json.load(open(args.out, encoding="utf-8"))
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        print("%-40s %2d cells" % (os.path.basename(f), len(d)))
        for k, v in d.items():
            if k in merged and merged[k] != v:
                print("  ! overwriting", k)
            merged[k] = v

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=1)
    print("merged -> %s  (%d cells)" % (args.out, len(merged)))


if __name__ == "__main__":
    main()

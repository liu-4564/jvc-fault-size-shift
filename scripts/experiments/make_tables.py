# -*- coding: utf-8 -*-
"""make_tables.py -- render result tables from multisource_matrix.json.

Produces tables.md with:
  T1  the source x target accuracy matrix (per backbone)
  T2  the source x target profile cosine matrix
  T3  the correlation table (pooled, per backbone, per seed)
  T4  the per-source separation gap
  T5  the label-free descriptors

Usage:
  python scripts/experiments/make_tables.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MAT = os.path.join(HERE, "multisource_matrix.json")
ANA = os.path.join(HERE, "multisource_analysis.json")
FISH = os.path.join(ROOT, "data", "fisher_12domains.json")
OUT = os.path.join(HERE, "tables.md")

F = {k: np.array(v, float) for k, v in json.load(open(FISH, encoding="utf8")).items()}
SOURCES = ["0.007@0", "0.007@1", "0.007@2", "0.007@3"]
TARGETS = ["0.007@0", "0.007@1", "0.007@2", "0.007@3",
           "0.014@0", "0.014@1", "0.014@2", "0.014@3",
           "0.021@0", "0.021@1", "0.021@2", "0.021@3"]


def cos(a, b):
    return float(F[a] @ F[b] / (np.linalg.norm(F[a]) * np.linalg.norm(F[b])))


def short(k):
    return k.replace("0.007", ".007").replace("0.014", ".014").replace("0.021", ".021")


def main():
    M = json.load(open(MAT, encoding="utf8"))
    A = json.load(open(ANA, encoding="utf8")) if os.path.exists(ANA) else {}
    L = []
    add = L.append

    cells = {}
    for key, cell in M.items():
        bb, rep, src, seed = key.split("|")
        cells[(bb, src.replace("HP", ""), int(seed))] = cell

    backs = sorted({k[0] for k in cells})
    seeds = sorted({k[2] for k in cells})
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if only is not None:
        if only not in seeds:
            raise SystemExit("seed %d not present; have %s" % (only, seeds))
        seeds = [only]

    # ---- T1 accuracy matrix, per backbone, averaged over seeds -------------
    for bb in backs:
        add("## T1 accuracy (%%), backbone = %s\n" % bb)
        add("| source | " + " | ".join(short(t) for t in TARGETS) + " |")
        add("|" + "---|" * (len(TARGETS) + 1))
        for s in SOURCES:
            row = []
            for t in TARGETS:
                if t == s:
                    row.append("—")
                    continue
                vals = [cells[(bb, s, sd)][t + "HP"]["acc"]
                        for sd in seeds if (bb, s, sd) in cells and t + "HP" in cells[(bb, s, sd)]]
                row.append("%.1f" % np.mean(vals) if vals else "·")
            add("| %s | " % short(s) + " | ".join(row) + " |")
        add("")

    # ---- T2 cosine matrix --------------------------------------------------
    add("## T2 profile cosine\n")
    add("| source | " + " | ".join(short(t) for t in TARGETS) + " |")
    add("|" + "---|" * (len(TARGETS) + 1))
    for s in SOURCES:
        row = ["%.3f" % cos(s, t) if t != s else "—" for t in TARGETS]
        add("| %s | " % short(s) + " | ".join(row) + " |")
    add("")

    # ---- T3 correlations ---------------------------------------------------
    def block(title, b):
        if not b:
            return
        add("\n**%s** (n=%d)\n" % (title, b["n"]))
        add("| metric | r | 95% CI | rho | p(perm) |")
        add("|---|---|---|---|---|")
        for k, m in sorted(b["metrics"].items(), key=lambda kv: -abs(kv[1]["pearson"])):
            add("| %s | %+.3f | [%+.3f, %+.3f] | %+.3f | %.4f |"
                % (k, m["pearson"], m["ci95"][0], m["ci95"][1], m["spearman"], m["p_perm"]))
        add("")

    add("## T3 correlation with accuracy\n")
    block("pooled", A.get("overall"))
    for bb, b in A.get("by_bb", {}).items():
        block("backbone=" + bb, b)
    for fam, b in A.get("by_family", {}).items():
        block("family=" + fam, b)
    for sd, b in A.get("by_seed", {}).items():
        block("seed=" + sd, b)

    # ---- T4 separation -----------------------------------------------------
    sep = A.get("separation", {})
    if sep:
        add("\n## T4 per-source separation of the two severity groups\n")
        add("| source | max cos(0.014) | min cos(0.021) | gap | max acc(0.014) | min acc(0.021) |")
        add("|---|---|---|---|---|---|")
        for s, v in sep.items():
            if s == "global":
                continue
            add("| %s | %.3f | %.3f | %+.3f | %.1f | %.1f |"
                % (short(s.replace("HP", "")), v["max_cos_014"], v["min_cos_021"],
                   v["gap"], v["max_acc_014"], v["min_acc_021"]))
        g = sep["global"]
        add("| **global** | %.3f | %.3f | %+.3f | | |"
            % (g["max_cos_014"], g["min_cos_021"], g["min_cos_021"] - g["max_cos_014"]))
        add("")

    # ---- T5 label-free descriptors, per backbone ---------------------------
    add("\n## T5 label-free descriptors (computed without target labels)\n")
    add("| backbone | metric | r | 95% CI | p |")
    add("|---|---|---|---|---|")
    for bb, b in A.get("by_bb", {}).items():
        for k in ("kl_hist", "mmd2", "coral", "entropy"):
            m = b["metrics"].get(k)
            if m:
                add("| %s | %s | %+.3f | [%+.3f, %+.3f] | %.4f |"
                    % (bb, k, m["pearson"], m["ci95"][0], m["ci95"][1], m["p_perm"]))
    add("")

    if "seed_sd" in A:
        add("\n**Seed spread:** %d cells with >1 seed, mean SD %.2f points, max SD %.2f points.\n"
            % (A["seed_sd"]["n_cells"], A["seed_sd"]["mean"], A["seed_sd"]["max"]))

    open(OUT, "w", encoding="utf8").write("\n".join(L))
    print("written:", OUT, "(%d lines)" % len(L))


if __name__ == "__main__":
    main()

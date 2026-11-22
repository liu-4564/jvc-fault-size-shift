# -*- coding: utf-8 -*-
"""analyze_matrix.py -- turn multisource_matrix.json into the paper's new tables.

For every (backbone, representation, source, target) cell it pairs the measured
accuracy with descriptors of the inter-race Fisher profile, and with the
label-free transferability descriptors recorded at evaluation time, then reports
correlations with a cluster bootstrap over target domains (the 4 sources of one
target are not independent) and a permutation test.

Usage:
  python scripts/experiments/analyze_matrix.py
"""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FISH = os.path.join(ROOT, "data", "fisher_12domains.json")
MAT = os.path.join(HERE, "multisource_matrix.json")
OUT = os.path.join(HERE, "multisource_analysis.json")

NBAND = 8
CENTERS = np.arange(NBAND) * 0.5 + 0.25          # band centres, kHz
F = {k: np.array(v, dtype=float) for k, v in json.load(open(FISH, encoding="utf8")).items()}


def descriptors(src, tgt):
    s, t = F[src.replace("HP", "")], F[tgt.replace("HP", "")]
    ps, pt = s / s.sum(), t / t.sum()
    si, ti = int(np.argmax(s)), int(np.argmax(t))
    return {
        "peak_band": float(CENTERS[ti]),
        "f_at_peak": float(t[ti]),
        "dpeak": float(CENTERS[ti] - CENTERS[si]),
        "src_band_ratio": float(t[si] / (s[si] + 1e-12)),
        "centroid": float((CENTERS * pt).sum() - (CENTERS * ps).sum()),
        "cosine": float(t @ s / (np.linalg.norm(t) * np.linalg.norm(s))),
        "l1": float(np.abs(pt - ps).sum()),
    }


def load_pairs(reps=None, backs=None):
    M = json.load(open(MAT, encoding="utf8"))
    rows = []
    for key, cell in M.items():
        bb, rep, src, seed = key.split("|")
        if reps and rep not in reps:
            continue
        if backs and bb not in backs:
            continue
        for tgt, v in cell.items():
            r = {"bb": bb, "rep": rep, "src": src, "seed": int(seed), "tgt": tgt}
            r.update({kk: v[kk] for kk in ("acc", "acc3", "mmd2", "coral", "pad",
                                           "entropy", "kl_hist") if kk in v})
            r.update(descriptors(src, tgt))
            # Two different mechanisms live in the 44-cell matrix.  A change of
            # fault size (0.007 -> 0.014 / 0.021) reshapes the inter-race profile;
            # a change of load at fixed size (0.007 -> 0.007) leaves it nearly
            # intact while accuracy still collapses.  Mixing them would let the
            # load-drift cells dominate the correlation, so they are kept apart.
            r["family"] = "load" if tgt.startswith("0.007") else "severity"
            rows.append(r)
    return rows


def cluster_bootstrap_ci(x, y, groups, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    out = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug), replace=True)
        ids = np.concatenate([idx[g] for g in pick])
        if len(np.unique(x[ids])) < 3 or len(np.unique(y[ids])) < 3:
            continue
        rr = np.corrcoef(x[ids], y[ids])[0, 1]
        if not np.isfinite(rr):
            continue
        out.append(rr)
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def perm_p(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    r0 = abs(np.corrcoef(x, y)[0, 1])
    cnt = sum(abs(np.corrcoef(x, rng.permutation(y))[0, 1]) >= r0 for _ in range(n))
    return float((cnt + 1) / (n + 1))


def summarise(rows, name, keys):
    acc = np.array([r["acc"] for r in rows])
    tgt = np.array([r["tgt"] for r in rows])
    out = {"n": int(len(rows)), "group": name, "metrics": {}}
    for k in keys:
        x = np.array([r[k] for r in rows], dtype=float)
        if np.std(x) < 1e-12:
            continue
        r = float(np.corrcoef(x, acc)[0, 1])
        rho = float(stats.spearmanr(x, acc).statistic)
        lo, hi = cluster_bootstrap_ci(x, acc, tgt)
        out["metrics"][k] = {"pearson": r, "spearman": rho, "ci95": [lo, hi],
                             "p_perm": perm_p(x, acc)}
    return out


def main():
    rows = load_pairs()
    if not rows:
        raise SystemExit("no rows: run multisource_matrix.py first")

    # Optional: restrict to one seed, so that averaged and single-seed cells are
    # never mixed in the same correlation.
    import sys
    if len(sys.argv) > 1:
        only = int(sys.argv[1])
        rows = [r for r in rows if r["seed"] == only]
        print("restricted to seed %d: %d rows" % (only, len(rows)), flush=True)

    # average over seeds within a cell first
    cells = {}
    for r in rows:
        cells.setdefault((r["bb"], r["rep"], r["src"], r["tgt"]), []).append(r)
    agg = []
    for k, v in cells.items():
        d = {"bb": k[0], "rep": k[1], "src": k[2], "tgt": k[3],
             "family": v[0]["family"]}
        for f in ("acc", "acc3", "mmd2", "coral", "pad", "entropy", "kl_hist",
                  "cosine", "dpeak", "src_band_ratio", "l1", "centroid", "f_at_peak"):
            d[f] = float(np.mean([x[f] for x in v]))
        d["sd"] = float(np.std([x["acc"] for x in v]))
        d["n_seed"] = len(v)
        agg.append(d)

    DESC = ["cosine", "dpeak", "src_band_ratio", "l1", "centroid", "f_at_peak"]
    FREE = ["mmd2", "coral", "pad", "entropy", "kl_hist"]

    sev = [a for a in agg if a["family"] == "severity"]
    lod = [a for a in agg if a["family"] == "load"]

    # Per-source separation: does the cosine ordering of the two severity groups
    # survive a change of source even though no single global threshold does?
    sep = {}
    for s in sorted({a["src"] for a in sev}):
        lo = [a["cosine"] for a in sev if a["src"] == s and a["tgt"].startswith("0.014")]
        hi = [a["cosine"] for a in sev if a["src"] == s and a["tgt"].startswith("0.021")]
        alo = [a["acc"] for a in sev if a["src"] == s and a["tgt"].startswith("0.014")]
        ahi = [a["acc"] for a in sev if a["src"] == s and a["tgt"].startswith("0.021")]
        if lo and hi:
            sep[s] = {"max_cos_014": float(np.max(lo)), "min_cos_021": float(np.min(hi)),
                      "gap": float(np.min(hi) - np.max(lo)),
                      "max_acc_014": float(np.max(alo)), "min_acc_021": float(np.min(ahi)),
                      "separated": bool(np.min(hi) > np.max(lo))}
    sep["global"] = {
        "max_cos_014": float(max(v["max_cos_014"] for k, v in sep.items() if k != "global")),
        "min_cos_021": float(min(v["min_cos_021"] for k, v in sep.items() if k != "global"))}
    sep["global"]["separated"] = bool(sep["global"]["min_cos_021"]
                                      > sep["global"]["max_cos_014"])

    res = {"n_cells": len(agg), "targets": sorted({a["tgt"] for a in agg}),
           "sources": sorted({a["src"] for a in agg}),
           "backbones": sorted({a["bb"] for a in agg}),
           "separation": sep,
           "overall": summarise(agg, "all", DESC + FREE),
           "by_family": {}, "by_rep": {}, "by_bb": {}}
    if sev:
        res["by_family"]["severity"] = summarise(sev, "severity", DESC + FREE)
    if lod:
        res["by_family"]["load"] = summarise(lod, "load", DESC + FREE)
    for rep in sorted({a["rep"] for a in agg}):
        res["by_rep"][rep] = summarise([a for a in agg if a["rep"] == rep], rep, DESC + FREE)
    for bb in sorted({a["bb"] for a in agg}):
        res["by_bb"][bb] = summarise([a for a in agg if a["bb"] == bb], bb, DESC + FREE)

    # Seed robustness: repeat the correlation for every seed on its own, so the
    # reported r is not an artefact of one initialisation.
    seeds = sorted({r["seed"] for r in rows})
    res["by_seed"] = {}
    if len(seeds) > 1:
        for sd in seeds:
            sub = [r for r in rows if r["seed"] == sd]
            if len(sub) >= 4:
                res["by_seed"][str(sd)] = summarise(sub, "seed=%d" % sd, DESC + FREE)
    multi = [a["sd"] for a in agg if a.get("n_seed", 1) > 1]
    if multi:
        res["seed_sd"] = {"n_cells": len(multi), "mean": float(np.mean(multi)),
                          "max": float(np.max(multi))}

    json.dump(res, open(OUT, "w", encoding="utf8"), indent=1)

    def show(tag, block):
        print("\n== %s  (n=%d) ==" % (tag, block["n"]))
        for k, m in sorted(block["metrics"].items(), key=lambda kv: -abs(kv[1]["pearson"])):
            print("  %-16s r=%+.3f  CI[%+.3f,%+.3f]  rho=%+.3f  p=%.4f"
                  % (k, m["pearson"], m["ci95"][0], m["ci95"][1], m["spearman"], m["p_perm"]))

    show("all cells", res["overall"])
    for fam, b in res["by_family"].items():
        show("family=" + fam, b)
    for rep, b in res["by_rep"].items():
        show("rep=" + rep, b)
    for bb, b in res["by_bb"].items():
        show("backbone=" + bb, b)
    for sd, b in res.get("by_seed", {}).items():
        show("seed=" + sd, b)
    if "seed_sd" in res:
        print("\n== seed spread across cells with >1 seed ==")
        print("  %d cells, mean SD %.2f points, max SD %.2f points"
              % (res["seed_sd"]["n_cells"], res["seed_sd"]["mean"],
                 res["seed_sd"]["max"]))

    print("\n== per-source separation of the severity groups ==")
    for s, v in res["separation"].items():
        if s == "global":
            continue
        print("  %-9s  max cos(0.014)=%.3f  min cos(0.021)=%.3f  gap=%+.3f  "
              "max acc(0.014)=%.1f  min acc(0.021)=%.1f  %s"
              % (s, v["max_cos_014"], v["min_cos_021"], v["gap"],
                 v["max_acc_014"], v["min_acc_021"],
                 "separated" if v["separated"] else "OVERLAP"))
    g = res["separation"]["global"]
    print("  global     max cos(0.014)=%.3f  min cos(0.021)=%.3f  -> %s"
          % (g["max_cos_014"], g["min_cos_021"],
             "single threshold exists" if g["separated"] else "no single global threshold"))
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()

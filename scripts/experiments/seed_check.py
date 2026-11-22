# -*- coding: utf-8 -*-
"""seed_check.py -- is the profile-fidelity *ranking* stable across seeds?

The correlation across cells is one thing; the paper's operational claim is that
the cosine ranks targets correctly, so the question is whether the within-source
ordering survives a change of seed.
"""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
F = {k: np.array(v, float)
     for k, v in json.load(open(os.path.join(ROOT, "data", "fisher_12domains.json"),
                                encoding="utf8")).items()}
M = json.load(open(os.path.join(HERE, "multisource_matrix.json"), encoding="utf8"))
SEV = ["0.014@0", "0.014@1", "0.014@2", "0.014@3",
       "0.021@0", "0.021@1", "0.021@2", "0.021@3"]


def cos(a, b):
    return float(F[a] @ F[b] / (np.linalg.norm(F[a]) * np.linalg.norm(F[b])))


rows = []
for key, cell in sorted(M.items()):
    bb, rep, src, seed = key.split("|")
    if bb != "b0":
        continue
    s = src.replace("HP", "")
    tg = [t.replace("HP", "") for t in cell if t.replace("HP", "") != s]
    xa = [cos(s, t) for t in tg]
    ya = [cell[t + "HP"]["acc"] for t in tg]
    sev = [t for t in tg if t.startswith("0.014") or t.startswith("0.021")]
    xs = [cos(s, t) for t in sev]
    ys = [cell[t + "HP"]["acc"] for t in sev]
    rows.append((s, int(seed), stats.spearmanr(xs, ys).statistic,
                 stats.spearmanr(xa, ya).statistic, len(ya)))

print("%-10s %6s  %10s %10s %5s" % ("source", "seed", "rho(sev 8)", "rho(all 11)", "n"))
for s, sd, r1, r2, n in rows:
    print("%-10s %6d  %+10.3f %+10.3f %5d" % (s, sd, r1, r2, n))

r1 = np.array([r[2] for r in rows])
r2 = np.array([r[3] for r in rows])
print("\nseverity-only rho: mean %+.3f  min %+.3f  max %+.3f" % (r1.mean(), r1.min(), r1.max()))
print("all-target   rho: mean %+.3f  min %+.3f  max %+.3f" % (r2.mean(), r2.min(), r2.max()))

# does the 0.014 / 0.021 grouping still separate, per seed?
print("\nper (source, seed): max acc(0.014) vs min acc(0.021) over the 8 severity targets")
bad = 0
for key, cell in sorted(M.items()):
    bb, rep, src, seed = key.split("|")
    if bb != "b0":
        continue
    s = src.replace("HP", "")
    lo = [cell[t + "HP"]["acc"] for t in SEV if t.startswith("0.014")]
    hi = [cell[t + "HP"]["acc"] for t in SEV if t.startswith("0.021")]
    ok = min(hi) > max(lo)
    bad += 0 if ok else 1
    print("  %-9s seed %5s  max(0.014)=%5.1f  min(0.021)=%5.1f  %s"
          % (s, seed, max(lo), min(hi), "separated" if ok else "OVERLAP"))
print("\noverlapping (source, seed) configurations: %d" % bad)

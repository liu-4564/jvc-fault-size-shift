# -*- coding: utf-8 -*-
"""band_fisher_12domains.py -- fills in the inner/outer Fisher discriminant ratio for
CWRU 3 damage sizes x 4 loads = 12 domains, to test whether "the discriminative band
shifts upward" only holds at 0HP.

Definition is identical to Table 6 of the paper (verified cell by cell on 6 known
domains, max|Δ| = 0.00):
  take the last 50 test windows of each class -> per-window Welch(nperseg=2048)
  -> log power spectrum; bin-wise F = (m_IR - m_OR)^2 / (v_IR + v_OR) -> average in band.
Note: no band-pass filtering here. An earlier version filtered first and then computed,
which deviated from Table 6 by up to 8 points -- do not use that variant.
"""
import json
import os

import numpy as np
import scipy.io as sio
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fisher_12domains.json")
FS, WIN, STEP, NPS = 12000, 2048, 512, 2048
BANDS = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500),
         (2500, 3000), (3000, 3500), (3500, 4000)]
DIRS = [
    os.environ.get("CWRU_SAME_LOAD_DIR", "data/cwru/same_load_same_size"),
    os.environ.get("CWRU_CROSS_LOAD_DIR", "data/cwru/same_size_cross_load"),
    os.environ.get("CWRU_CROSS_SIZE_DIR", "data/cwru/cross_fault_size"),
]
FREQ = welch(np.zeros(WIN), fs=FS, nperseg=NPS)[0]

DATA = {
    ("0.007", 0): {"inner": 105, "outer": 130},
    ("0.007", 1): {"inner": 106, "outer": 131},
    ("0.007", 2): {"inner": 107, "outer": 132},
    ("0.007", 3): {"inner": 108, "outer": 133},
    ("0.014", 0): {"inner": 169, "outer": 197},
    ("0.014", 1): {"inner": 170, "outer": 198},
    ("0.014", 2): {"inner": 171, "outer": 199},
    ("0.014", 3): {"inner": 172, "outer": 200},
    ("0.021", 0): {"inner": 209, "outer": 234},
    ("0.021", 1): {"inner": 210, "outer": 235},
    ("0.021", 2): {"inner": 211, "outer": 236},
    ("0.021", 3): {"inner": 212, "outer": 237},
}

CHECK = {
    ("0.007", 0): [5.54, 2.43, 9.94, 4.07, 2.83, 3.60, 7.74, 2.43],
    ("0.007", 3): [4.24, 3.50, 6.44, 4.78, 2.49, 4.71, 6.11, 1.42],
    ("0.014", 0): [3.87, 1.07, 1.33, 1.52, 5.25, 6.37, 1.34, 1.55],
    ("0.014", 3): [1.38, 0.64, 9.96, 0.86, 2.72, 3.76, 1.53, 0.72],
    ("0.021", 0): [4.17, 3.14, 5.22, 2.75, 1.73, 1.41, 3.58, 5.96],
    ("0.021", 3): [1.92, 3.90, 6.38, 2.24, 1.26, 2.25, 2.12, 3.88],
}


def find_mat(num):
    for d in DIRS:
        p = os.path.join(d, f"{num}.mat")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(num)


def log_specs(num):
    x = sio.loadmat(find_mat(num))
    x = x[[k for k in x.keys() if k.endswith("DE_time")][0]].flatten().astype(np.float64)
    n = (len(x) - WIN) // STEP + 1
    pool = np.stack([x[i * STEP: i * STEP + WIN] for i in range(n)])[n - 50:]
    return np.array([np.log(welch(w - w.mean(), fs=FS, nperseg=NPS)[1] + 1e-12) for w in pool])


def fisher(A, B):
    out = []
    for lo, hi in BANDS:
        m = (FREQ >= lo) & (FREQ < hi)
        f = ((A[:, m].mean(0) - B[:, m].mean(0)) ** 2
             / (A[:, m].var(0) + B[:, m].var(0) + 1e-12))
        out.append(float(f.mean()))
    return out


def main():
    res, worst = {}, 0.0
    print(f"{'domain':<12}" + "".join(f"{lo/1000:.1f}-{hi/1000:.1f}k".rjust(11) for lo, hi in BANDS))
    for (size, load), files in DATA.items():
        vals = fisher(log_specs(files["inner"]), log_specs(files["outer"]))
        res[f"{size}@{load}"] = [round(x, 2) for x in vals]
        flag, peak = "", BANDS[int(np.argmax(vals))]
        if (size, load) in CHECK:
            err = max(abs(a - b) for a, b in zip(vals, CHECK[(size, load)]))
            worst = max(worst, err)
            flag = f"   [check max|Δ|={err:.2f}]"
        print(f"{size}@{load}HP".ljust(12) + "".join(f"{x:11.2f}" for x in vals)
              + f"   peak {peak[0]/1000:.1f}-{peak[1]/1000:.1f}k{flag}")
    print(f"`nmax deviation from Table 6 = {worst:.2f} (expected < 0.05)")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()

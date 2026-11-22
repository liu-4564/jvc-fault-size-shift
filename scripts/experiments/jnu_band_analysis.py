# -*- coding: utf-8 -*-
"""jnu_band_analysis.py -- Fisher discriminant ratio and peak band of each class pair
under cross-speed conditions

Goal: explain why JNU degrades across speed, and whether the failure direction
(normal->inner, outer->inner/ball) comes with "the discriminative band drifting with
speed". Method follows 4.5:
  band-pass the window -> Welch(nperseg=WIN) -> log power spectrum -> per bin
  F = (m_a − m_b)^2 / (v_a + v_b) -> average inside the band.

For each speed the Fisher value of 6 class pairs is computed, and the per-band values
plus the peak band are reported.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "jnu", "cache")
OUT = os.path.join(HERE, "jnu_fisher_pairs.txt")

FS = 12500
WIN = 2048
CLASSES = ["normal", "inner", "outer", "ball"]
SPEEDS = [600, 800, 1000]
BANDS = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500),
         (2500, 3000), (3000, 3500), (3500, 4000), (4000, 4500), (4500, 5000),
         (5000, 5500), (5500, 6000)]
N_WIN = 20


def log(msg):
    print(msg, flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def windows(x, win=WIN, step=512):
    n = (len(x) - win) // step + 1
    return np.stack([x[i * step: i * step + win] for i in range(n)])


def band_specs(cls, speed):
    from scipy.signal import butter, sosfiltfilt, welch
    x = np.load(os.path.join(CACHE, f"{cls}_{speed}.npz"))["x"]
    pool = windows(x)
    sel = pool[len(pool) - 50: len(pool) - 50 + N_WIN]
    out = {}
    for lo, hi in BANDS:
        if lo == 0:
            sos = butter(4, hi / (FS / 2.0) * 0.999, btype="low", output="sos")
        else:
            sos = butter(4, [lo / (FS / 2.0), min(hi / (FS / 2.0), 0.999)],
                         btype="band", output="sos")
        sp = []
        for w in sel:
            seg = sosfiltfilt(sos, w - w.mean())
            _, p = welch(seg, fs=FS, nperseg=WIN)
            sp.append(np.log(p + 1e-12))
        out[(lo, hi)] = np.array(sp)
    return out


def fisher(A, B):
    vals = {}
    for b in BANDS:
        m1, m2 = A[b].mean(0), B[b].mean(0)
        v1, v2 = A[b].var(0), B[b].var(0)
        vals[b] = float(((m1 - m2) ** 2 / (v1 + v2 + 1e-12)).mean())
    return vals


def main():
    open(OUT, "w", encoding="utf-8").close()
    specs = {(c, s): band_specs(c, s) for c in CLASSES for s in SPEEDS}
    pairs = [(CLASSES[i], CLASSES[j]) for i in range(len(CLASSES)) for j in range(i + 1, len(CLASSES))]
    for a, b in pairs:
        log(f"\n===== {a} vs {b} =====")
        log(f"{'speed':<10}" + "".join(f"{lo/1000:.1f}-{hi/1000:.1f}k".rjust(11) for lo, hi in BANDS))
        for s in SPEEDS:
            v = fisher(specs[(a, s)], specs[(b, s)])
            best = max(v, key=v.get)
            log(f"{str(s) + ' rpm':<10}" + "".join(f"{v[k]:11.2f}" for k in BANDS))
            log(f"    peak {best[0]/1000:.1f}-{best[1]/1000:.1f} kHz  F={v[best]:.2f}"
                f"   (full-band mean {np.mean(list(v.values())):.2f})")


if __name__ == "__main__":
    main()

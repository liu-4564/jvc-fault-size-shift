# -*- coding: utf-8 -*-
"""paderborn_cross_size.py -- reproduce the CWRU "cross-damage-size degradation + band shift"
on the second dataset

Three control experiments (setting N15_M07_F10 = 1500 rpm / 0.7 Nm / 10 kN):
  A. Same-size upper bound: train on severity 1 -> test on the **other two bearings** of
     severity 1 (control baseline)
  B. Cross-size degradation: train on severity 1 -> test on severity 2 (core replication)
     The focus is whether "inner is predicted as outer" also appears and whether inner
     recall collapses to 0.
  C. Band shift: per-band inner/outer Fisher discriminant ratio, comparing whether the
     separable band moves up from severity 1 to severity 2; N09_M07_F10 is used as the
     cross-setting control.

The method follows CWRU exactly: 16 kHz signal + 2048-sample window + Morlet CWT grayscale
image + fine-tuned ImageNet-pretrained EfficientNet-B0, 3 random seeds.

Signal processing note: Paderborn vibration is natively 64 kHz; it is first decimated by
4x with an anti-alias filter to 16 kHz (Nyquist 8 kHz, the same order of magnitude as the
usable bandwidth of CWRU at 6 kHz).
"""
import os
import sys
import glob
import time
import importlib.util

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "paderborn", "raw")
EXT = os.path.join(HERE, "paderborn", "extracted")
CACHE = os.path.join(HERE, "paderborn", "cache")
os.makedirs(CACHE, exist_ok=True)

# reuse the reader script (avoids duplicating read_mat)
_spec = importlib.util.spec_from_file_location("pbread", os.path.join(HERE, "paderborn_reader.py"))
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)

# ---------------- global configuration ----------------
SETTING = "N15_M07_F10"          # main setting
ALT_SETTING = "N09_M07_F10"      # cross-setting control (900 rpm)
FS_RAW = 64000
DEC = 4
FS = FS_RAW // DEC               # 16000
WIN, STEP, S = 2048, 512, 224
SCALES = np.arange(1, 100)
N_TR, N_VA, N_TE = 150, 50, 50
CLASSES = ["normal", "inner", "outer"]
SEEDS = [42, 123, 2026]
LOG = os.path.join(HERE, "paderborn_result.txt")

# Bearing split (20 records per bearing under the same setting)
#   train and validation: K001/K002, KI01/KI03, KA01/KA05 (severity 1)
#   same-size test: K003, KI05, KA07 (the other bearings of severity 1)
#   cross-size test: K003, KI07/KI08, KA03/KA06/KA09 (severity 2)
GROUP = {
    "train": {"normal": ["K001", "K002"], "inner": ["KI01", "KI03"], "outer": ["KA01", "KA05"]},
    "valid": {"normal": ["K001", "K002"], "inner": ["KI01", "KI03"], "outer": ["KA01", "KA05"]},
    "same_size": {"normal": ["K003"], "inner": ["KI05"], "outer": ["KA07"]},
    "cross_size": {"normal": ["K003"], "inner": ["KI07", "KI08"], "outer": ["KA03", "KA06", "KA09"]},
}

BANDS = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500),
         (2500, 3000), (3000, 3500), (3500, 4000), (4000, 4500), (4500, 5000),
         (5000, 5500), (5500, 6000), (6000, 6500), (6500, 7000), (7000, 7500), (7500, 8000)]

_sig_cache = {}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------------- data loading ----------------
def load_bearing(code, setting=SETTING):
    """Return (20, 64206) float32 decimated to FS, ordered by the file index."""
    key = (code, setting)
    if key in _sig_cache:
        return _sig_cache[key]
    npz = os.path.join(CACHE, f"{code}_{setting}.npz")
    if os.path.exists(npz):
        segs = np.load(npz)["segs"]
    else:
        pb.extract(code)                       # returns immediately if already extracted
        files = sorted(glob.glob(os.path.join(EXT, code, "**", f"{setting}_{code}_*.mat"), recursive=True),
                       key=lambda p: int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]))
        if not files:
            raise FileNotFoundError(f"{code} @ {setting}: no .mat files, extract first")
        from scipy.signal import decimate
        raw = [decimate(pb.read_mat(p)["vibration"].astype(np.float64), DEC,
                        ftype="iir", zero_phase=True) for p in files]
        L = min(len(x) for x in raw)          # decimate may differ by 1 sample, truncate uniformly
        segs = np.stack([x[:L].astype(np.float32) for x in raw])
        np.savez(npz, segs=segs)
    _sig_cache[key] = segs
    return segs


def windows(segs, win=WIN, step=STEP):
    """Cut the (n_seg, L) continuous segments into all windows, return (n_win, win) float32."""
    out = []
    for s in range(segs.shape[0]):
        for o in range(0, segs.shape[1] - win + 1, step):
            out.append(segs[s, o:o + win])
    return np.stack(out).astype(np.float32)


def pick(codes, n, offset=0, setting=SETTING):
    """Draw n windows uniformly from the window pools of several bearings (offset keeps
    train/validation disjoint)."""
    pool = np.concatenate([windows(load_bearing(c, setting)) for c in codes], axis=0)
    idx = np.linspace(0, len(pool) - 1, n + offset, dtype=int)[offset:]
    return pool[idx]


def make_image(seg):
    import pywt
    from PIL import Image
    seg = seg - seg.mean()
    coef, _ = pywt.cwt(seg, SCALES, "morl", sampling_period=1.0 / FS)
    amp = np.log1p(np.abs(coef))
    amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8) * 255
    im = Image.fromarray(amp.astype(np.uint8)).resize((S, S), Image.BILINEAR)
    a = np.array(im)
    return np.stack([a] * 3, axis=2).astype(np.uint8)


def build_domain(name):
    """Build or reuse the (X, y) image set of one split."""
    path = os.path.join(CACHE, f"img_{name}_{SETTING}.npz")
    if os.path.exists(path):
        z = np.load(path)
        log(f"  reusing {name}: {z['X'].shape}")
        return z["X"], z["y"]
    codes = GROUP[name]
    if name == "train":
        ns = [N_TR]
    elif name == "valid":
        ns = [N_VA]
    else:
        ns = [N_TE]
    n = ns[0]
    off = N_TR if name == "valid" else 0
    imgs, ys = [], []
    for ci, cls in enumerate(CLASSES):
        segs = pick(codes[cls], n, offset=off)
        for j, w in enumerate(segs):
            imgs.append(make_image(w))
            if (j + 1) % 25 == 0:
                log(f"    {name} class {cls} {j + 1}/{n}")
        ys += [ci] * len(segs)
    X = np.stack(imgs)
    y = np.array(ys, dtype=np.int64)
    np.savez(path, X=X, y=y)
    log(f"  cached {name}: {X.shape}")
    return X, y


# ---------------- training / evaluation ----------------
MEAN = None
STD = None


def _baset(s):
    x = s.float() if isinstance(s, torch.Tensor) else torch.from_numpy(s).float()
    x = x / 255.0
    return (x - MEAN) / STD


def to_input(x_uint8):
    return _baset(x_uint8)


def train_one(seed, X, y, Xva, yva, device, epochs=30, n_cls=3):
    import torch
    import torch.nn as nn
    from torchvision import models
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_cls)
    model.to(device)
    back = [p for nm, p in model.named_parameters() if not nm.startswith("classifier")]
    head = list(model.classifier.parameters())
    opt = torch.optim.AdamW([{"params": back, "lr": 1e-4}, {"params": head, "lr": 1e-3}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    Xtr = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    ytr = torch.from_numpy(y)
    Xva_t = torch.from_numpy(Xva).permute(0, 3, 1, 2).contiguous()
    yva_t = torch.from_numpy(yva)
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(ytr))
        for i in range(0, len(ytr), 32):
            b = idx[i:i + 32]
            loss = crit(model(to_input(Xtr[b]).to(device)), ytr[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        val = (model(to_input(Xva_t).to(device)).argmax(1).cpu() == yva_t).float().mean().item()
    return model, val


def evaluate(model, X, y, device):
    import torch
    Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    yt = torch.from_numpy(y)
    preds = []
    with torch.no_grad():
        for i in range(0, len(yt), 32):
            preds += model(to_input(Xt[i:i + 32]).to(device)).argmax(1).cpu().tolist()
    return np.array(preds)


def confusion(pred, y, n_cls=3):
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    return cm


# ---------------- band Fisher ----------------
def band_specs(codes, setting=SETTING, n_win=20):
    """Take n_win windows from each bearing, band-pass per band and run Welch;
    return dict[band] = log-spectrum matrix."""
    from scipy.signal import butter, sosfiltfilt, welch
    out = {b: [] for b in BANDS}
    for c in codes:
        pool = windows(load_bearing(c, setting))
        idx = np.linspace(0, len(pool) - 1, n_win, dtype=int)
        for b in BANDS:
            lo, hi = b
            if lo == 0:
                sos = butter(4, hi / (FS / 2.0) * 0.999, btype="low", output="sos")
            else:
                sos = butter(4, [lo / (FS / 2.0), min(hi / (FS / 2.0), 0.999)],
                             btype="band", output="sos")
            for w in idx:
                seg = sosfiltfilt(sos, pool[w] - pool[w].mean())
                _, p = welch(seg, fs=FS, nperseg=WIN)
                out[b].append(np.log(p + 1e-12))
    return {b: np.array(v) for b, v in out.items()}


FISHER = {
    "severity 1 (KI01/03/05 vs KA01/05/07)":
        (["KI01", "KI03", "KI05"], ["KA01", "KA05", "KA07"]),
    "severity 2 (KI07/08 vs KA03/06/09)":
        (["KI07", "KI08"], ["KA03", "KA06", "KA09"]),
    "severity 1 @N09 setting":
        (["KI01", "KI03", "KI05"], ["KA01", "KA05", "KA07"]),
}


def fisher_analysis():
    log("\n===== C. inner vs outer Fisher discriminant ratio (window-level mean, Welch after 16 kHz band-pass) =====")
    header = f"{'group':<34}" + "".join(f"{lo/1000:.1f}-{hi/1000:.1f}k".rjust(10) for lo, hi in BANDS)
    log(header)
    best = {}
    for name, (irs, ors) in FISHER.items():
        setting = ALT_SETTING if "@N09" in name else SETTING
        A = band_specs(irs, setting)
        B = band_specs(ors, setting)
        vals = {}
        for b in BANDS:
            m1, m2 = A[b].mean(0), B[b].mean(0)
            v1, v2 = A[b].var(0), B[b].var(0)
            vals[b] = float(((m1 - m2) ** 2 / (v1 + v2 + 1e-12)).mean())
        bmax = max(vals, key=vals.get)
        best[name] = (bmax, vals[bmax])
        log(f"{name:<34}" + "".join(f"{vals[b]:10.2f}" for b in BANDS))
        log(f"    -> peak band {bmax[0]/1000:.1f}-{bmax[1]/1000:.1f} kHz (F={vals[bmax]:.2f})")
    return best


# ---------------- main pipeline ----------------
def main():
    global MEAN, STD
    import torch
    open(LOG, "w", encoding="utf-8").close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}  setting {SETTING}  FS={FS} Hz  WIN={WIN}")

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    log("\n===== building image sets =====")
    Xtr, ytr = build_domain("train")
    Xva, yva = build_domain("valid")
    Xsame, ysame = build_domain("same_size")
    Xcross, ycross = build_domain("cross_size")
    log(f"  train {Xtr.shape}  val {Xva.shape}  same-size test {Xsame.shape}  cross-size test {Xcross.shape}")

    res = {"same_size": [], "cross_size": []}
    for seed in SEEDS:
        log(f"\n--- seed {seed} ---")
        model, val = train_one(seed, Xtr, ytr, Xva, yva, device)
        log(f"  source val={val * 100:.2f}%")
        for tag, Xt, yt in [("same_size", Xsame, ysame), ("cross_size", Xcross, ycross)]:
            pred = evaluate(model, Xt, yt, device)
            acc = float((pred == yt).mean() * 100)
            rec = {cls: float((pred[yt == ci] == ci).mean() * 100) if (yt == ci).any() else float("nan")
                   for ci, cls in enumerate(CLASSES)}
            res[tag].append((acc, rec))
            log(f"  {tag}: acc={acc:6.2f}%  " +
                "  ".join(f"{k} recall {v:5.2f}%" for k, v in rec.items()))
            if seed == SEEDS[0]:
                cm = confusion(pred, yt)
                log("  confusion matrix (rows=true, cols=pred, order normal/inner/outer):")
                for r in cm:
                    log("    " + " ".join(f"{v:5d}" for v in r))
        del model
        torch.cuda.empty_cache()

    log("\n===== A/B cross-size results (mean +- std over 3 seeds, %) =====")
    for tag in ["same_size", "cross_size"]:
        arr = np.array([r[0] for r in res[tag]], dtype=float)
        log(f"  {tag}: {arr.mean():6.2f} ± {arr.std():.2f}")
        for cls in CLASSES:
            v = np.array([r[1][cls] for r in res[tag]], dtype=float)
            log(f"      {cls} recall {v.mean():6.2f} ± {v.std():.2f}")

    import json
    with open(os.path.join(HERE, "paderborn_ab.json"), "w", encoding="utf-8") as f:
        json.dump({k: [(a, r) for a, r in v] for k, v in res.items()}, f, ensure_ascii=False, indent=1)

    fisher_analysis()


if __name__ == "__main__":
    if "--fisher" in sys.argv:
        LOG = os.path.join(HERE, "paderborn_fisher.txt")
        open(LOG, "w", encoding="utf-8").close()
        fisher_analysis()
    else:
        main()

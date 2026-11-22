# -*- coding: utf-8 -*-
"""cross_severity_unified_protocol.py -- re-runs plain / env / order representations
under one and the same split, removing the protocol mismatch between Table 4 and
Table 4(b).

Protocol (aligned with the paper's §3.2/§3.5, and the one used by
multiseed_cross_severity.py):
  first 150 windows per class -> 120 for training + 30 for validation; test takes the
  last 50 windows per class; 30 epochs; seeds 42/123/2026.
  Model selection has two modes:
    default          roll back to the best validation epoch, writes unified_p3.json
    --final-epoch    keep the epoch-30 model (no rollback), writes unified_p3_final.json
  The paper reports the --final-epoch runs (§3.5: "no early stopping, no
  best-checkpoint rollback"), which is what data/unified_p3_final.json contains.

Usage:
  python cross_severity_unified_protocol.py env order                        # two reps, seed 42
  python cross_severity_unified_protocol.py plain env order --all-seeds
  python cross_severity_unified_protocol.py plain env order --all-seeds --final-epoch   # paper setting
"""
import argparse
import json
import os
import sys

import numpy as np
import pywt
import scipy.io as sio
import torch
import torch.nn as nn
from PIL import Image
from scipy.signal import hilbert
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache_uni")
os.makedirs(CACHE_DIR, exist_ok=True)
OUT = os.path.join(HERE, "unified_p3.json")

FS, WIN, STEP, S = 12000, 2048, 512, 224
SCALES = np.arange(1, 100)
N_TRAIN, N_TEST = 150, 50
RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DIRS = [
    os.environ.get("CWRU_SAME_LOAD_DIR", "data/cwru/same_load_same_size"),
    os.environ.get("CWRU_CROSS_LOAD_DIR", "data/cwru/same_size_cross_load"),
    os.environ.get("CWRU_CROSS_SIZE_DIR", "data/cwru/cross_fault_size"),
]
# Explicit order: 0=inner 1=outer 2=normal 3=ball (must not rely on sorted())
CLASSES = ["inner_fault", "outer_fault", "normal", "ball_fault"]

DATA = {
    ("0.007", 0): {"normal": 97, "inner_fault": 105, "ball_fault": 118, "outer_fault": 130},
    ("0.014", 0): {"normal": 97, "inner_fault": 169, "ball_fault": 185, "outer_fault": 197},
    ("0.014", 1): {"normal": 98, "inner_fault": 170, "ball_fault": 186, "outer_fault": 198},
    ("0.014", 2): {"normal": 99, "inner_fault": 171, "ball_fault": 187, "outer_fault": 199},
    ("0.014", 3): {"normal": 100, "inner_fault": 172, "ball_fault": 188, "outer_fault": 200},
    ("0.021", 0): {"normal": 97, "inner_fault": 209, "ball_fault": 222, "outer_fault": 234},
    ("0.021", 1): {"normal": 98, "inner_fault": 210, "ball_fault": 223, "outer_fault": 235},
    ("0.021", 2): {"normal": 99, "inner_fault": 211, "ball_fault": 224, "outer_fault": 236},
    ("0.021", 3): {"normal": 100, "inner_fault": 212, "ball_fault": 225, "outer_fault": 237},
}
_sig = {}


def get_signal(num):
    if num not in _sig:
        for d in DIRS:
            p = os.path.join(d, f"{num}.mat")
            if os.path.exists(p):
                m = sio.loadmat(p)
                _sig[num] = m[[k for k in m.keys() if k.endswith("DE_time")][0]].flatten().astype(np.float64)
                break
        else:
            raise FileNotFoundError(num)
    return _sig[num]


def order_resample(seg, rpm):
    """Resample by k = 1797/rpm linear interpolation, normalising speed to 1797 r/min (order in §3.4)."""
    k = 1797.0 / rpm
    n_new = int(round(len(seg) * k))
    return np.interp(np.arange(n_new) / k, np.arange(len(seg)), seg)


def make_image(seg, rep, rpm=1797):
    seg = seg - seg.mean()
    if rep == "order":
        seg = order_resample(seg, rpm)
    if rep in ("env", "order"):
        seg = np.abs(hilbert(seg))
    coef, _ = pywt.cwt(seg, SCALES, "morl", sampling_period=1.0 / FS)
    amp = np.log1p(np.abs(coef))
    amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8) * 255
    im = Image.fromarray(amp.astype(np.uint8)).resize((S, S), Image.BILINEAR)
    a = np.array(im)
    return np.stack([a] * 3, axis=2).astype(np.uint8)


def build_split(rep, size, load, split):
    path = os.path.join(CACHE_DIR, f"{rep}_{size}_{load}_{split}.npz")
    if os.path.exists(path):
        z = np.load(path)
        print(f"reusing {rep} {size}@{load}HP {split}: {z['X'].shape}", flush=True)
        return z["X"], z["y"]
    files = DATA[(size, load)]
    imgs, ys = [], []
    for ci, cls in enumerate(CLASSES):
        sig = get_signal(files[cls])
        n_win = (len(sig) - WIN) // STEP + 1
        rng = range(0, N_TRAIN) if split == "train" else range(n_win - N_TEST, n_win)
        for j, w in enumerate(rng):
            imgs.append(make_image(sig[w * STEP: w * STEP + WIN], rep, RPM[load]))
            ys.append(ci)
            if (j + 1) % 50 == 0:
                print(f"  {rep} {size}@{load} {split} class{ci} {j+1}/{len(rng)}", flush=True)
    X = np.stack(imgs)
    y = np.array(ys, dtype=np.int64)
    np.savez(path, X=X, y=y)
    return X, y


def to_input(x_uint8):
    x = x_uint8.float() / 255.0
    return (x - MEAN) / STD


def train_one(seed, X, y, Xva, yva, device, epochs=30, final_epoch=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model.to(device)
    back = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    head = list(model.classifier.parameters())
    opt = torch.optim.AdamW([{"params": back, "lr": 1e-4}, {"params": head, "lr": 1e-3}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(y))
        for i in range(0, len(y), 32):
            b = idx[i:i + 32]
            xb = to_input(X[b]).to(device)
            loss = crit(model(xb), y[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(to_input(Xva).to(device)).argmax(1).cpu()
        va = float((pv == yva).float().mean() * 100)
        if va > best:
            best = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if not final_epoch:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = model(to_input(Xva).to(device)).argmax(1).cpu()
    best = float((pv == yva).float().mean() * 100)
    return model, best


def evaluate(model, X, y, device):
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(y), 32):
            preds += model(to_input(X[i:i + 32]).to(device)).argmax(1).cpu().tolist()
    pred = np.array(preds)
    yt = y.numpy()
    acc = float((pred == yt).mean() * 100)
    n_idx = CLASSES.index("normal")
    m = yt != n_idx
    acc3 = float((pred[m] == yt[m]).mean() * 100)
    rec = {c: float((pred[yt == ci] == ci).mean() * 100) for ci, c in enumerate(CLASSES)}
    conf = np.zeros((4, 4), dtype=int)
    for t, p in zip(yt, pred):
        conf[t, p] += 1
    return acc, acc3, rec, conf.astype(int).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reps", nargs="+", default=["plain", "env", "order"])
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--final-epoch", action="store_true",
                    help="use the epoch-30 model (no best-validation rollback), matching the §4.6 countermeasure runs")
    args = ap.parse_args()
    seeds = [42, 123, 2026] if args.all_seeds else [42]
    out = os.path.join(HERE, "unified_p3_final.json" if args.final_epoch
                       else "unified_p3.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | output: {out}", flush=True)

    Xtr_all, ytr_all = build_split("plain", "0.007", 0, "train")  # only to warm the cache
    targets = [(s, l) for (s, l) in DATA if (s, l) != ("0.007", 0)]
    res = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else {}

    for rep in args.reps:
        Xtr, ytr = build_split(rep, "0.007", 0, "train")
        X = torch.from_numpy(Xtr).permute(0, 3, 1, 2).contiguous()
        y = torch.from_numpy(ytr)
        tr, va = [], []
        for ci in range(4):
            ids = np.where(ytr == ci)[0]
            tr += list(ids[:120])
            va += list(ids[120:150])
        tr, va = torch.tensor(tr), torch.tensor(va)
        Xtr_t, ytr_t = X[tr], y[tr]
        Xva_t, yva_t = X[va], y[va]
        for seed in seeds:
            key = f"{rep}|{seed}"
            if key in res:
                print(f"{key} already present, skipping", flush=True)
                continue
            print(f"`n--- {rep} seed {seed} training ---", flush=True)
            model, val = train_one(seed, Xtr_t, ytr_t, Xva_t, yva_t, device,
                                   final_epoch=args.final_epoch)
            print(f"--- {rep} seed {seed} | val {val:.2f}% ---", flush=True)
            rec_all = {}
            for size, load in targets:
                Xt, yt = build_split(rep, size, load, "test")
                Xt = torch.from_numpy(Xt).permute(0, 3, 1, 2).contiguous()
                a, a3, r, conf = evaluate(model, Xt, torch.from_numpy(yt), device)
                rec_all[f"{size}@{load}"] = {"acc": a, "acc3": a3, "recall": r,
                                             "conf": conf}
                print(f"  {size}@{load}HP = {a:6.2f}%  3-class {a3:6.2f}%  "
                      + "  ".join(f"{k} {v:5.2f}%" for k, v in r.items()), flush=True)
            res[key] = rec_all
            with open(out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
            del model
            torch.cuda.empty_cache()

    print("`n===== summary =====", flush=True)
    for rep in args.reps:
        mean_acc = []
        for size, load in targets:
            vals = [res[f"{rep}|{s}"][f"{size}@{load}"]["acc"] for s in seeds if f"{rep}|{s}" in res]
            mean_acc.append(np.mean(vals))
        print(f"  {rep}: mean over target domains = {np.mean(mean_acc):.2f}%", flush=True)


if __name__ == "__main__":
    main()

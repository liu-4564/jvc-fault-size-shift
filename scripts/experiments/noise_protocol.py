# -*- coding: utf-8 -*-
"""noise_protocol.py -- P2 supplementary run across signal-to-noise ratios

Protocol:
  source domain 0.007@0HP, 4 classes (normal/inner/ball/outer)
  training setup A: clean training (600 windows)
  training setup B: noisy training (2 noisy copies per window, SNR ~ U(-5, 10) dB)
  test conditions: clean / 10 dB / 0 dB / -5 dB (200 images per condition, fixed noise seed)
  representation: plain (raw CWT) / env (CWT after Hilbert envelope)
4 models in total: plain-clean, plain-noisy, env-clean, env-noisy.

Images are cached by (representation, purpose, SNR) in cache_p2/ and reused instead of
being recomputed. Each finished model appends its results to p2_log.txt and flushes,
so progress can be checked after an interruption.
"""
import os
import numpy as np
import scipy.io as sio
import pywt
import torch
import torch.nn as nn
from scipy.signal import hilbert
from torchvision import models
from PIL import Image

fs, WIN, STEP, S = 12000, 2048, 512, 224
scales = np.arange(1, 100)
N_TRAIN, N_TEST = 150, 50
N_COPY = 2                      # noisy training: 2 copies per window
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DIRS = [
    os.environ.get("CWRU_SAME_LOAD_DIR", "data/cwru/same_load_same_size"),
    os.environ.get("CWRU_CROSS_LOAD_DIR", "data/cwru/same_size_cross_load"),
    os.environ.get("CWRU_CROSS_SIZE_DIR", "data/cwru/cross_fault_size"),
]
CLASSES = ["inner_race", "outer_race", "normal", "ball"]
SRC = {"normal": 97, "inner_race": 105, "ball": 118, "outer_race": 130}
BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, "cache_p2")
LOG = os.path.join(BASE, "p2_log.txt")
os.makedirs(CACHE_DIR, exist_ok=True)

_sig = {}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "`n")


def find_mat(num):
    for d in DIRS:
        p = os.path.join(d, f"{num}.mat")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(num)


def get_signal(num):
    if num not in _sig:
        d = sio.loadmat(find_mat(num))
        _sig[num] = d[[k for k in d.keys() if k.endswith('DE_time')][0]].flatten().astype(np.float64)
    return _sig[num]


def add_noise(x, snr_db, rng):
    p = np.mean(x ** 2)
    n = rng.standard_normal(len(x))
    n *= np.sqrt(p / (10 ** (snr_db / 10.0)))
    return x + n


def make_image(seg, rep):
    seg = seg - seg.mean()
    if rep == "env":
        seg = np.abs(hilbert(seg))
    coef, _ = pywt.cwt(seg, scales, 'morl', sampling_period=1.0 / fs)
    amp = np.log1p(np.abs(coef))
    amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8) * 255
    im = Image.fromarray(amp.astype(np.uint8)).resize((S, S), Image.BILINEAR)
    a = np.array(im)
    return np.stack([a] * 3, axis=2).astype(np.uint8)


def windows(split):
    """Return [(class_id, raw window), ...]"""
    out = []
    for ci, cls in enumerate(CLASSES):
        sig = get_signal(SRC[cls])
        n_win = (len(sig) - WIN) // STEP + 1
        rng = range(0, N_TRAIN) if split == "train" else range(n_win - N_TEST, n_win)
        for w in rng:
            out.append((ci, sig[w * STEP: w * STEP + WIN]))
    return out


def build(rep, tag, snr=None, copies=1, split="train"):
    """tag: 'clean_train' / 'noisy_train' / 'test'; snr=None means clean"""
    key = f"{rep}_{tag}_{snr}_{copies}"
    path = os.path.join(CACHE_DIR, key + ".npz")
    if os.path.exists(path):
        z = np.load(path)
        log(f"reusing cache {key}: {z['X'].shape}")
        return z["X"], z["y"]

    items = windows(split)
    imgs, ys = [], []
    rng = np.random.default_rng(12345 if snr is not None else 42)
    for c in range(copies):
        for ci, seg in items:
            if tag == "noisy_train":
                s = float(rng.uniform(-5, 10))
                imgs.append(make_image(add_noise(seg, s, rng), rep))
            elif snr is not None:
                imgs.append(make_image(add_noise(seg, snr, rng), rep))
            else:
                imgs.append(make_image(seg, rep))
            ys.append(ci)
        log(f"  {key} copy {c + 1}/{copies} done, {len(imgs)} images so far")
    X = np.stack(imgs)
    y = np.array(ys, dtype=np.int64)
    np.savez(path, X=X, y=y)
    log(f"cached {key}: {X.shape}")
    return X, y


def to_input(x):
    x = x.float() / 255.0
    return (x - MEAN) / STD


def train_one(X, y, Xva, yva, device, epochs=30, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model.to(device)
    back = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.AdamW([{"params": back, "lr": 1e-4},
                             {"params": model.classifier.parameters(), "lr": 1e-3}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    Xtr = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    ytr = torch.from_numpy(y)
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(ytr))
        for i in range(0, len(ytr), 32):
            b = idx[i:i + 32]
            loss = crit(model(to_input(Xtr[b]).to(device)), ytr[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            log(f"    epoch {ep + 1}/{epochs}")
    model.eval()
    with torch.no_grad():
        acc = (model(to_input(Xva).to(device)).argmax(1).cpu() == yva).float().mean().item()
    return model, acc


def evaluate(model, X, y, device):
    Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    yt = torch.from_numpy(y)
    preds = []
    with torch.no_grad():
        for i in range(0, len(yt), 32):
            preds += model(to_input(Xt[i:i + 32]).to(device)).argmax(1).cpu().tolist()
    pred = np.array(preds)
    acc = float((pred == yt.numpy()).mean() * 100)
    ir = float((pred[yt.numpy() == 1] == 1).mean() * 100) if (yt.numpy() == 1).any() else -1
    return acc, ir


def main():
    open(LOG, "w", encoding="utf-8").close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device: " + str(device))

    # validation set: last 30 images per class of the clean source domain
    Xc, yc = build("plain", "clean_train", snr=None, split="train")
    tr_idx, va_idx = [], []
    # images are stacked in class order, 150 per class
    for ci in range(len(CLASSES)):
        ids = list(range(ci * N_TRAIN, ci * N_TRAIN + N_TRAIN))
        tr_idx += ids[:120]; va_idx += ids[120:]
    Xtr_c, ytr_c = Xc[tr_idx], yc[tr_idx]
    Xva, yva = torch.from_numpy(Xc[va_idx]).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(yc[va_idx])

    conds = [("clean", None), ("10 dB", 10), ("0 dB", 0), ("-5 dB", -5)]
    res = {}

    for rep in ["plain", "env"]:
        # clean training
        ntr, ntr_y = build(rep, "clean_train", snr=None, split="train")
        log(f"`n=== representation {rep} | training setup clean ===")
        model, val = train_one(ntr[tr_idx], ntr_y[tr_idx], Xva, yva, device)
        row = []
        for name, snr in conds:
            Xt, yt = build(rep, "test", snr=snr, split="test")
            acc, ir = evaluate(model, Xt, yt, device)
            row.append(acc)
            log(f"  {name:6s} = {acc:6.2f}%  inner recall {ir:5.2f}%")
        res[(rep, "clean training")] = row
        del model; torch.cuda.empty_cache()

        # noisy training
        ntr2, ntr2_y = build(rep, "noisy_train", snr=None, copies=N_COPY, split="train")
        log(f"`n=== representation {rep} | training setup noisy U(-5,10) dB ===")
        model, val = train_one(ntr2, ntr2_y, Xva, yva, device)
        row = []
        for name, snr in conds:
            Xt, yt = build(rep, "test", snr=snr, split="test")
            acc, ir = evaluate(model, Xt, yt, device)
            row.append(acc)
            log(f"  {name:6s} = {acc:6.2f}%  inner recall {ir:5.2f}%")
        res[(rep, "noisy training")] = row
        del model; torch.cuda.empty_cache()

    log("`n===== Table 3 summary (accuracy %) =====")
    log(f"{'rep':<6}{'training setup':<16}{'clean':>8}{'10 dB':>8}{'0 dB':>8}{'-5 dB':>8}")
    for (rep, setting), row in res.items():
        log(f"{rep:<6}{setting:<16}" + "".join(f"{v:8.2f}" for v in row))


if __name__ == "__main__":
    main()

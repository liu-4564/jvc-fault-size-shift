# -*- coding: utf-8 -*-
"""jnu_cross_speed.py -- cross-speed verification on the third dataset (JNU, Jiangnan University)

JNU layout: 4 classes (normal NO / inner IF / outer OF / ball BF),
3 speeds 600/800/1000 rpm, 50 kHz sampling, defect size fixed at 0.3 x 0.05 mm
(**no size grades**, so cross-damage-size cannot be verified here, only cross-speed).

Protocol (same backbone, same hyper-parameters and same time-frequency pipeline as
CWRU / Paderborn):
  source domain = first 150 windows/class of one speed (training)
  same-speed hold-out = last 50 windows/class of the same speed file (in-domain upper bound)
  cross-speed test = last 50 windows/class of the other two speed files
  3 source domains (600/800/1000) x 3 seeds (42/123/2026)

Signal processing: 50 kHz -> 4x anti-aliased decimation to 12.5 kHz (Nyquist 6.25 kHz,
the same order of magnitude as the usable bandwidth of CWRU at 12 kHz), window 2048
(about 164 ms, close to 171 ms on CWRU), step 512, Morlet CWT (scales 1-99) -> 224x224
three-channel grayscale.

Also included: per-band inner/outer Fisher discriminant ratio (to test whether the
separable band migrates with speed).
"""
import os
import sys
import glob
import json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "jnu", "src", "JNU-Bearing-Dataset-main")
CACHE = os.path.join(HERE, "jnu", "cache")
os.makedirs(CACHE, exist_ok=True)

# ---------------- global configuration ----------------
FS_RAW = 50000
DEC = 4
FS = FS_RAW // DEC                 # 12500
WIN, STEP, S = 2048, 512, 224
SCALES = np.arange(1, 100)
N_TR, N_VA, N_TE = 150, 50, 50
CLASSES = ["normal", "inner", "outer", "ball"]
SEEDS = [42, 123, 2026]
SPEEDS = [600, 800, 1000]
LOG = os.path.join(HERE, "jnu_result.txt")

# File names: n=normal, ib=inner, ob=outer, tb=ball (the suffix is the release batch id)
FILES = {
    "normal": {600: "n600_3_2.csv", 800: "n800_3_2.csv", 1000: "n1000_3_2.csv"},
    "inner": {600: "ib600_2.csv", 800: "ib800_2.csv", 1000: "ib1000_2.csv"},
    "outer": {600: "ob600_2.csv", 800: "ob800_2.csv", 1000: "ob1000_2.csv"},
    "ball": {600: "tb600_2.csv", 800: "tb800_2.csv", 1000: "tb1000_2.csv"},
}

BANDS = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500),
         (2500, 3000), (3000, 3500), (3500, 4000), (4000, 4500), (4500, 5000),
         (5000, 5500), (5500, 6000)]

_sig_cache = {}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------------- data loading ----------------
def load_signal(cls, speed):
    """Return the whole signal decimated to FS (float32), cached per (class, speed) as npz."""
    key = (cls, speed)
    if key in _sig_cache:
        return _sig_cache[key]
    npz = os.path.join(CACHE, f"{cls}_{speed}.npz")
    if os.path.exists(npz):
        x = np.load(npz)["x"]
    else:
        from scipy.signal import decimate
        path = os.path.join(SRC, FILES[cls][speed])
        raw = np.loadtxt(path, dtype=np.float64, ndmin=1).astype(np.float64)
        x = decimate(raw, DEC, ftype="iir", zero_phase=True).astype(np.float32)
        np.savez(npz, x=x)
        log(f"  decoded {FILES[cls][speed]}  raw {len(raw)} samples -> {len(x)} samples ({len(x)/FS:.2f} s)")
    _sig_cache[key] = x
    return x


def windows(x, win=WIN, step=STEP):
    n = (len(x) - win) // step + 1
    return np.stack([x[i * step: i * step + win] for i in range(n)]).astype(np.float32)


def pick(speed, cls, n, head=True, offset=0):
    """head=True takes n windows starting at offset from the head of the file (train/val);
    False takes the last n windows (test)."""
    pool = windows(load_signal(cls, speed))
    if head:
        return pool[offset: offset + n]
    return pool[len(pool) - n:]


def make_image(seg):
    import pywt
    from PIL import Image
    seg = seg.astype(np.float64)
    seg = seg - seg.mean()
    coef, _ = pywt.cwt(seg, SCALES, "morl", sampling_period=1.0 / FS)
    amp = np.log1p(np.abs(coef))
    amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8) * 255
    im = Image.fromarray(amp.astype(np.uint8)).resize((S, S), Image.BILINEAR)
    a = np.array(im)
    return np.stack([a] * 3, axis=2).astype(np.uint8)


def build_split(speed, n, head, tag, offset=0):
    """Build (X, y) for one speed. head=the n windows starting at offset, otherwise the last n windows."""
    path = os.path.join(CACHE, f"img_{tag}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["X"], z["y"]
    imgs, ys = [], []
    for ci, cls in enumerate(CLASSES):
        segs = pick(speed, cls, n, head=head, offset=offset)
        for j, w in enumerate(segs):
            imgs.append(make_image(w))
            if (j + 1) % 50 == 0:
                log(f"    {tag} {cls} {j + 1}/{n}")
        ys += [ci] * len(segs)
    X = np.stack(imgs)
    y = np.array(ys, dtype=np.int64)
    np.savez(path, X=X, y=y)
    log(f"  images {tag}: {X.shape}")
    return X, y


# ---------------- training / evaluation ----------------
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_input(x):
    x = x.float() if isinstance(x, torch.Tensor) else torch.from_numpy(x).float()
    x = x / 255.0
    return (x - MEAN) / STD


def train_one(seed, X, y, device, epochs=30, n_cls=4):
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
    return model


def evaluate(model, X, y, device):
    Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    yt = torch.from_numpy(y)
    preds = []
    with torch.no_grad():
        for i in range(0, len(yt), 32):
            preds += model(to_input(Xt[i:i + 32]).to(device)).argmax(1).cpu().tolist()
    return np.array(preds)


def confusion(pred, y, n_cls=4):
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    return cm


# ---------------- band Fisher ----------------
def band_specs(cls, speed, n_win=20):
    from scipy.signal import butter, sosfiltfilt, welch
    pool = windows(load_signal(cls, speed))
    sel = pool[len(pool) - N_TE: len(pool) - N_TE + n_win] if len(pool) > N_TE else pool[:n_win]
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


def fisher_analysis():
    log("\n===== inner vs outer Fisher discriminant ratio (per speed, Welch after 12.5 kHz band-pass) =====")
    log(f"{'speed':<10}" + "".join(f"{lo/1000:.1f}-{hi/1000:.1f}k".rjust(11) for lo, hi in BANDS))
    res = {}
    for speed in SPEEDS:
        A = band_specs("inner", speed)
        B = band_specs("outer", speed)
        vals = {}
        for b in BANDS:
            m1, m2 = A[b].mean(0), B[b].mean(0)
            v1, v2 = A[b].var(0), B[b].var(0)
            vals[b] = float(((m1 - m2) ** 2 / (v1 + v2 + 1e-12)).mean())
        best = max(vals, key=vals.get)
        log(f"{str(speed) + ' rpm':<10}" + "".join(f"{vals[b]:11.2f}" for b in BANDS))
        log(f"    -> peak band {best[0]/1000:.1f}-{best[1]/1000:.1f} kHz (F={vals[best]:.2f})")
        res[speed] = {f"{b[0]}-{b[1]}": vals[b] for b in BANDS}
    return res


# ---------------- main pipeline ----------------
def main():
    open(LOG, "w", encoding="utf-8").close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}  FS={FS} Hz  WIN={WIN}  window={WIN/FS*1000:.0f} ms")

    # per speed: train (first 150 windows), val (windows 151-200), test (last 50 windows)
    data = {}
    log("\n===== building image sets (per speed 4 classes x 150 train + 50 val + 50 test) =====")
    for speed in SPEEDS:
        Xtr, ytr = build_split(speed, N_TR, head=True, tag=f"tr{speed}")
        Xva, yva = build_split(speed, N_VA, head=True, tag=f"va{speed}", offset=N_TR)
        Xte, yte = build_split(speed, N_TE, head=False, tag=f"te{speed}")
        data[speed] = dict(tr=(Xtr, ytr), va=(Xva, yva), te=(Xte, yte))
        log(f"  {speed} rpm: train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    res = {}
    for source in SPEEDS:
        log(f"\n===== source domain {source} rpm =====")
        Xtr, ytr = data[source]["tr"]
        Xva, yva = data[source]["va"]
        res[source] = {t: [] for t in SPEEDS}
        for seed in SEEDS:
            model = train_one(seed, Xtr, ytr, device, n_cls=len(CLASSES))
            with torch.no_grad():
                Xv = torch.from_numpy(Xva).permute(0, 3, 1, 2).contiguous()
                yv = torch.from_numpy(yva)
                vpred = []
                for i in range(0, len(yv), 32):
                    vpred += model(to_input(Xv[i:i + 32]).to(device)).argmax(1).cpu().tolist()
                val = float((np.array(vpred) == yva).mean() * 100)
            log(f"  --- seed {seed} --- val {val:6.2f}%")
            for target in SPEEDS:
                Xt, yt = data[target]["te"]
                pred = evaluate(model, Xt, yt, device)
                acc = float((pred == yt).mean() * 100)
                rec = {cls: float((pred[yt == ci] == ci).mean() * 100)
                       for ci, cls in enumerate(CLASSES)}
                tag = "same-speed" if target == source else "cross-speed"
                res[source][target].append((acc, rec))
                log(f"    {tag} {target} rpm: acc={acc:6.2f}%  " +
                    "  ".join(f"{k} {v:5.2f}%" for k, v in rec.items()))
                if seed == SEEDS[0] and target != source:
                    cm = confusion(pred, yt)
                    log(f"      confusion matrix (rows=true, cols=pred, order {'/'.join(CLASSES)}):")
                    for r in cm:
                        log("        " + " ".join(f"{v:5d}" for v in r))
            del model
            torch.cuda.empty_cache()

    log("\n===== summary (mean +- std over 3 seeds, %) =====")
    summary = {}
    for source in SPEEDS:
        log(f"\nsource domain {source} rpm")
        summary[source] = {}
        for target in SPEEDS:
            arr = np.array([r[0] for r in res[source][target]], dtype=float)
            tag = "same-speed (in-domain upper bound)" if target == source else "cross-speed"
            log(f"  {tag} → {target}: {arr.mean():6.2f} ± {arr.std():.2f}")
            summary[source][target] = {"acc": float(arr.mean()), "std": float(arr.std()), "rec": {}}
            for cls in CLASSES:
                v = np.array([r[1][cls] for r in res[source][target]], dtype=float)
                log(f"      {cls} recall {v.mean():6.2f} ± {v.std():.2f}")
                summary[source][target]["rec"][cls] = [float(v.mean()), float(v.std())]
        cross = [np.array([r[0] for r in res[source][t]], dtype=float).mean()
                 for t in SPEEDS if t != source]
        log(f"  cross-speed mean = {np.mean(cross):.2f}")
        summary[source]["cross_mean"] = float(np.mean(cross))

    with open(os.path.join(HERE, "jnu_result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    fr = fisher_analysis()
    with open(os.path.join(HERE, "jnu_fisher.json"), "w", encoding="utf-8") as f:
        json.dump(fr, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if "--fisher" in sys.argv:
        LOG = os.path.join(HERE, "jnu_fisher.txt")
        open(LOG, "w", encoding="utf-8").close()
        fisher_analysis()
    else:
        main()

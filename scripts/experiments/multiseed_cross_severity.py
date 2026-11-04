# -*- coding: utf-8 -*-
"""multiseed_cross_severity.py -- repeat the cross-severity experiment with 3 random seeds
and report mean +- std

Only the images that are actually needed are cached:
  source domain 0.007@0HP train(600) + test(200)
  8 target domains test (200 each)  -> 2400 images in total
Each split is stored as its own npz (uncompressed, fast) and reused if present;
progress is printed every 25 images.
"""
import os
import json
import numpy as np
import scipy.io as sio
import pywt
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

fs, WIN, STEP, S = 12000, 2048, 512, 224
scales = np.arange(1, 100)
N_TRAIN, N_TEST = 150, 50
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DIRS = [
    os.environ.get("CWRU_SAME_LOAD_DIR", "data/cwru/same_load_same_size"),
    os.environ.get("CWRU_CROSS_LOAD_DIR", "data/cwru/same_size_cross_load"),
    os.environ.get("CWRU_CROSS_SIZE_DIR", "data/cwru/cross_fault_size"),
]
CLASSES = ["inner_race", "outer_race", "normal", "ball"]
# Note: this fixed order is what the cached npz labels were generated with and must not
# be changed -- index 0=inner_race, 1=outer_race, 2=normal, 3=ball.
# All statistics are aligned by class name; never use hard-coded indices.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_ms")
os.makedirs(CACHE_DIR, exist_ok=True)

DATA = {
    ("0.007", 0): {"normal": 97,  "inner_race": 105, "ball": 118, "outer_race": 130},
    ("0.014", 0): {"normal": 97,  "inner_race": 169, "ball": 185, "outer_race": 197},
    ("0.014", 1): {"normal": 98,  "inner_race": 170, "ball": 186, "outer_race": 198},
    ("0.014", 2): {"normal": 99,  "inner_race": 171, "ball": 187, "outer_race": 199},
    ("0.014", 3): {"normal": 100, "inner_race": 172, "ball": 188, "outer_race": 200},
    ("0.021", 0): {"normal": 97,  "inner_race": 209, "ball": 222, "outer_race": 234},
    ("0.021", 1): {"normal": 98,  "inner_race": 210, "ball": 223, "outer_race": 235},
    ("0.021", 2): {"normal": 99,  "inner_race": 211, "ball": 224, "outer_race": 236},
    ("0.021", 3): {"normal": 100, "inner_race": 212, "ball": 225, "outer_race": 237},
}
_sig = {}


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


def make_image(seg):
    coef, _ = pywt.cwt(seg, scales, 'morl', sampling_period=1.0 / fs)
    amp = np.abs(coef)
    amp = np.log1p(amp)
    amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8) * 255
    im = Image.fromarray(amp.astype(np.uint8)).resize((S, S), Image.BILINEAR)
    a = np.array(im)
    return np.stack([a] * 3, axis=2).astype(np.uint8)


def build_split(size, load, split):
    path = os.path.join(CACHE_DIR, f"{size}_{load}_{split}.npz")
    if os.path.exists(path):
        z = np.load(path)
        print(f"reusing {size}@{load}HP {split}: {z['X'].shape}", flush=True)
        return z["X"], z["y"]

    files = DATA[(size, load)]
    imgs, ys = [], []
    for ci, cls in enumerate(CLASSES):
        sig = get_signal(files[cls])
        n_win = (len(sig) - WIN) // STEP + 1
        rng = range(0, N_TRAIN) if split == "train" else range(n_win - N_TEST, n_win)
        for j, w in enumerate(rng):
            imgs.append(make_image(sig[w * STEP: w * STEP + WIN]))
            ys.append(ci)
            if (j + 1) % 25 == 0:
                print(f"  {size}@{load} {split} class{ci} {j + 1}/{len(rng)}", flush=True)
    X = np.stack(imgs)
    y = np.array(ys, dtype=np.int64)
    np.savez(path, X=X, y=y)
    print(f"cached {size}@{load}HP {split}: {X.shape}", flush=True)
    return X, y


def to_input(x_uint8):
    x = x_uint8.float() / 255.0
    return (x - MEAN) / STD


def train_one(seed, X, y, Xva, yva, device, epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model.to(device)
    back = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    head = list(model.classifier.parameters())
    opt = torch.optim.AdamW([{"params": back, "lr": 1e-4}, {"params": head, "lr": 1e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(y))
        for i in range(0, len(y), 32):
            b = idx[i:i + 32]
            xb = to_input(X[b]).to(device)
            loss = crit(model(xb), y[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1}/{epochs}", flush=True)

    model.eval()
    with torch.no_grad():
        acc = (model(to_input(Xva).to(device)).argmax(1).cpu() == yva).float().mean().item()
    return model, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    Xsrc, ysrc = build_split("0.007", 0, "train")
    # 150 per class -> 120 train + 30 val (stratified)
    tr_idx, va_idx = [], []
    for ci in range(len(CLASSES)):
        ids = np.where(ysrc == ci)[0]
        tr_idx += list(ids[:120]); va_idx += list(ids[120:150])
    tr_idx = torch.tensor(tr_idx, dtype=torch.long)
    va_idx = torch.tensor(va_idx, dtype=torch.long)
    X = torch.from_numpy(Xsrc).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(ysrc)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]
    n_val = int(0.2 * len(y))  # unused, kept to avoid confusion
    print(f"source train={len(ytr)} val={len(yva)}", flush=True)

    targets = [(s, l) for (s, l) in DATA if (s, l) != ("0.007", 0)]
    seeds = [42, 123, 2026]
    all_res = {}

    for seed in seeds:
        print(f"`n--- seed {seed} training ---", flush=True)
        model, val = train_one(seed, Xtr, ytr, Xva, yva, device)
        print(f"--- seed {seed} | val {val * 100:.2f}% ---", flush=True)
        for size, load in targets:
            Xt_raw, yt_raw = build_split(size, load, "test")
            Xt = torch.from_numpy(Xt_raw).permute(0, 3, 1, 2).contiguous()
            yt = torch.from_numpy(yt_raw)
            preds = []
            with torch.no_grad():
                for i in range(0, len(yt), 32):
                    preds += model(to_input(Xt[i:i + 32]).to(device)).argmax(1).cpu().tolist()
            pred = np.array(preds)
            yt_np = yt.numpy()
            acc = float((pred == yt_np).mean() * 100)
            n_idx = CLASSES.index("normal")
            mask = yt_np != n_idx
            acc3 = float((pred[mask] == yt_np[mask]).mean() * 100)
            rec = {cls: float((pred[yt_np == ci] == ci).mean() * 100)
                   for ci, cls in enumerate(CLASSES)}
            all_res.setdefault((size, load), []).append((acc, acc3, rec))
            print(f"  {size}@{load}HP = {acc:6.2f}%  three-class (no normal) {acc3:6.2f}%  "
                  + "  ".join(f"{k} recall {v:5.2f}%" for k, v in rec.items()), flush=True)
        del model
        torch.cuda.empty_cache()

    print("`n===== summary: mean +- std over 3 seeds =====", flush=True)
    accs_all, out = [], {}
    for size, load in targets:
        rows = all_res[(size, load)]
        acc_v = np.array([r[0] for r in rows])
        acc3_v = np.array([r[1] for r in rows])
        rec_v = {c: np.array([r[2][c] for r in rows]) for c in CLASSES}
        line = (f"  {size}@{load}HP  accuracy {acc_v.mean():6.2f} ± {acc_v.std():.2f}"
                f"   three-class {acc3_v.mean():6.2f} ± {acc3_v.std():.2f}")
        for c in CLASSES:
            line += f"   {c} recall {rec_v[c].mean():5.2f} ± {rec_v[c].std():.2f}"
        print(line, flush=True)
        out[f"{size}@{load}"] = {
            "acc": [round(float(x), 2) for x in acc_v],
            "acc3": [round(float(x), 2) for x in acc3_v],
            "recall": {c: [round(float(x), 2) for x in rec_v[c]] for c in CLASSES},
        }
        accs_all.append(acc_v.mean())
    print(f"`n  mean over all target domains = {np.mean(accs_all):.2f}%", flush=True)
    for sz in ["0.014", "0.021"]:
        idx = [i for i, (s, l) in enumerate(targets) if s == sz]
        print(f"  {sz} mean = {np.mean([accs_all[i] for i in idx]):.2f}%", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "multiseed_recall.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote multiseed_recall.json", flush=True)


if __name__ == "__main__":
    main()

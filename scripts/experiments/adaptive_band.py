# -*- coding: utf-8 -*-
"""adaptive_band.py -- label-free band selection (can it repair cross-size degradation?)

Approach:
  1) On the source domain 0.007@0HP, train one model for each of 8 band representations:
     full band + 7 candidate bands (0-1k, 1-1.5k, 1.5-2k, 2-2.5k, 2.5-3k, 3-4k, 4-4.8k).
     A band representation = band-pass to that band first, then CWT (everything else
     exactly as in plain).
  2) No labels are used on the target domain: pick the band by mean predictive entropy
     (lowest entropy = most confident).
  3) A labelled oracle (select by accuracy) gives the upper bound, i.e. how far
     "picking correctly" could go.
  4) Also report a non-degenerate selection (predictions must cover all 4 classes, so a
     confidently single-class degenerate solution cannot win).

Output: three rows per target domain -- full-band baseline / entropy selection / oracle.
"""
import os
import numpy as np
import scipy.io as sio
import pywt
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
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
CLASSES = ["inner_fault", "outer_fault", "normal", "ball_fault"]
BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "band_log.txt")

BANDS = {
    "full": None,
    "0-1k": (0, 1000),
    "1-1.5k": (1000, 1500),
    "1.5-2k": (1500, 2000),
    "2-2.5k": (2000, 2500),
    "2.5-3k": (2500, 3000),
    "3-4k": (3000, 4000),
    "4-4.8k": (4000, 4800),
}

DATA = {
    ("0.007", 0): {"normal": 97,  "inner_fault": 105, "ball_fault": 118, "outer_fault": 130},
    ("0.014", 0): {"normal": 97,  "inner_fault": 169, "ball_fault": 185, "outer_fault": 197},
    ("0.014", 1): {"normal": 98,  "inner_fault": 170, "ball_fault": 186, "outer_fault": 198},
    ("0.014", 2): {"normal": 99,  "inner_fault": 171, "ball_fault": 187, "outer_fault": 199},
    ("0.014", 3): {"normal": 100, "inner_fault": 172, "ball_fault": 188, "outer_fault": 200},
    ("0.021", 0): {"normal": 97,  "inner_fault": 209, "ball_fault": 222, "outer_fault": 234},
    ("0.021", 1): {"normal": 98,  "inner_fault": 210, "ball_fault": 223, "outer_fault": 235},
    ("0.021", 2): {"normal": 99,  "inner_fault": 211, "ball_fault": 224, "outer_fault": 236},
    ("0.021", 3): {"normal": 100, "inner_fault": 212, "ball_fault": 225, "outer_fault": 237},
}
SRC = ("0.007", 0)
TARGETS = [d for d in DATA if d != SRC]
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


def band_filter(x, band):
    if band is None:
        return x
    lo, hi = band
    if lo == 0:
        sos = butter(4, hi / (fs / 2.0), btype="low", output="sos")
    else:
        sos = butter(4, [lo / (fs / 2.0), hi / (fs / 2.0)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def make_image(seg, band):
    seg = band_filter(seg - seg.mean(), band)
    coef, _ = pywt.cwt(seg, scales, "morl", sampling_period=1.0 / fs)
    amp = np.log1p(np.abs(coef))
    amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8) * 255
    im = Image.fromarray(amp.astype(np.uint8)).resize((S, S), Image.BILINEAR)
    a = np.array(im)
    return np.stack([a] * 3, axis=2).astype(np.uint8)


def domain_images(dom, split, band):
    files = DATA[dom]
    imgs, ys = [], []
    for ci, cls in enumerate(CLASSES):
        sig = get_signal(files[cls])
        n_win = (len(sig) - WIN) // STEP + 1
        rng = range(0, N_TRAIN) if split == "train" else range(n_win - N_TEST, n_win)
        for w in rng:
            imgs.append(make_image(sig[w * STEP: w * STEP + WIN], band))
            ys.append(ci)
    return np.stack(imgs), np.array(ys, dtype=np.int64)


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
    model.eval()
    with torch.no_grad():
        val = (model(to_input(Xva).to(device)).argmax(1).cpu() == yva).float().mean().item()
    return model, val


def evaluate(model, X, y, device):
    Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    yt = torch.from_numpy(y)
    preds, ents = [], []
    with torch.no_grad():
        for i in range(0, len(yt), 32):
            logits = model(to_input(Xt[i:i + 32]).to(device))
            p = torch.softmax(logits, 1).cpu()
            preds += p.argmax(1).tolist()
            ents += (-(p * torch.log(p + 1e-8)).sum(1)).tolist()
    pred = np.array(preds)
    acc = float((pred == yt.numpy()).mean() * 100)
    ir = float((pred[yt.numpy() == 1] == 1).mean() * 100)
    ent = float(np.mean(ents))
    cover = len(np.unique(pred))
    return dict(acc=acc, ir=ir, ent=ent, cover=cover)


def main():
    open(LOG, "w", encoding="utf-8").close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device: " + str(device))

    results = {}   # results[band][dom] = dict
    for bi, (name, band) in enumerate(BANDS.items()):
        log(f"`n[{bi + 1}/{len(BANDS)}] band {name}: generating images ...")
        Xs, ys = domain_images(SRC, "train", band)
        tr, va = [], []
        for ci in range(len(CLASSES)):
            ids = list(range(ci * N_TRAIN, (ci + 1) * N_TRAIN))
            tr += ids[:120]; va += ids[120:]
        Xva = torch.from_numpy(Xs[va]).permute(0, 3, 1, 2).contiguous()
        yva = torch.from_numpy(ys[va])
        model, val = train_one(Xs[tr], ys[tr], Xva, yva, device)
        log(f"    source val={val * 100:.2f}%")

        results[name] = {}
        for dom in TARGETS:
            Xt, yt = domain_images(dom, "test", band)
            r = evaluate(model, Xt, yt, device)
            results[name][dom] = r
            log(f"    {dom[0]}@{dom[1]}HP  acc={r['acc']:6.2f}%  entropy={r['ent']:5.3f}  classes_covered={r['cover']}")
        del model; torch.cuda.empty_cache()

    log("`n===== per target domain: full band / entropy pick / non-degenerate pick / oracle =====")
    log(f"{'target':<14}{'full':>10}{'ent-band':>12}{'ent-acc':>12}"
        f"{'nondeg-band':>12}{'nondeg-acc':>12}{'oracle':>10}{'oracle-acc':>10}")
    sel_acc, orc_acc, base_acc, sel_nd_acc = [], [], [], []
    for dom in TARGETS:
        base = results["full"][dom]["acc"]
        ents = {b: results[b][dom]["ent"] for b in BANDS}
        b_ent = min(ents, key=ents.get)
        b_nd = min([b for b in BANDS if results[b][dom]["cover"] >= 4], key=ents.get)
        b_orc = max(BANDS, key=lambda b: results[b][dom]["acc"])
        a_ent = results[b_ent][dom]["acc"]
        a_nd = results[b_nd][dom]["acc"]
        a_orc = results[b_orc][dom]["acc"]
        log(f"{dom[0] + '@' + str(dom[1]) + 'HP':<14}{base:10.2f}{b_ent:>12}{a_ent:12.2f}"
            f"{b_nd:>12}{a_nd:12.2f}{b_orc:>10}{a_orc:10.2f}")
        base_acc.append(base); sel_acc.append(a_ent)
        sel_nd_acc.append(a_nd); orc_acc.append(a_orc)

    log("`n===== summary (mean over 8 target domains) =====")
    log(f"  full-band baseline {np.mean(base_acc):7.2f}%")
    log(f"  entropy selection {np.mean(sel_acc):7.2f}%   (Δ {np.mean(sel_acc) - np.mean(base_acc):+.2f})")
    log(f"  non-degenerate    {np.mean(sel_nd_acc):7.2f}%   (Δ {np.mean(sel_nd_acc) - np.mean(base_acc):+.2f})")
    log(f"  oracle upper bnd  {np.mean(orc_acc):7.2f}%   (Δ {np.mean(orc_acc) - np.mean(base_acc):+.2f})")
    for sz in ["0.014", "0.021"]:
        idx = [i for i, d in enumerate(TARGETS) if d[0] == sz]
        log(f"  {sz} group: baseline {np.mean([base_acc[i] for i in idx]):.2f}  "
            f"entropy-pick {np.mean([sel_acc[i] for i in idx]):.2f}  "
            f"oracle {np.mean([orc_acc[i] for i in idx]):.2f}")


if __name__ == "__main__":
    main()

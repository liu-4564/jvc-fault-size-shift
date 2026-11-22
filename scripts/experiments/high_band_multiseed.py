# -*- coding: utf-8 -*-
"""high_band_multiseed.py -- 3-seed repetition of the fixed 4-4.8 kHz band representation

Reuses the pipeline of adaptive_band.py, running only the 4-4.8k band with 3 seeds.
It also extends the Fisher discriminant analysis to 4.8 kHz (the original analysis
stopped at 4 kHz), to test whether the most transferable band is also the most
discriminative one.

Output:
  1) mean ± std over 8 target domains for the 3 seeds
  2) Fisher values of each size in 4-4.8k (against 3.5-4k)
"""
import os
import numpy as np
import scipy.io as sio
import pywt
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt, welch
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
LOG = os.path.join(BASE, "highband_log.txt")
BAND = (4000, 4800)

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


def band_filter(x):
    sos = butter(4, [BAND[0] / (fs / 2.0), BAND[1] / (fs / 2.0)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def make_image(seg):
    seg = band_filter(seg - seg.mean())
    coef, _ = pywt.cwt(seg, scales, "morl", sampling_period=1.0 / fs)
    amp = np.log1p(np.abs(coef))
    amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8) * 255
    im = Image.fromarray(amp.astype(np.uint8)).resize((S, S), Image.BILINEAR)
    a = np.array(im)
    return np.stack([a] * 3, axis=2).astype(np.uint8)


def domain_images(dom, split):
    imgs, ys = [], []
    for ci, cls in enumerate(CLASSES):
        sig = get_signal(DATA[dom][cls])
        n_win = (len(sig) - WIN) // STEP + 1
        rng = range(0, N_TRAIN) if split == "train" else range(n_win - N_TEST, n_win)
        for w in rng:
            imgs.append(make_image(sig[w * STEP: w * STEP + WIN]))
            ys.append(ci)
    return np.stack(imgs), np.array(ys, dtype=np.int64)


def to_input(x):
    x = x.float() / 255.0
    return (x - MEAN) / STD


def train_one(seed, X, y, Xva, yva, device, epochs=30):
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
    return model


def evaluate(model, X, y, device):
    Xt = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    yt = torch.from_numpy(y)
    preds = []
    with torch.no_grad():
        for i in range(0, len(yt), 32):
            preds += model(to_input(Xt[i:i + 32]).to(device)).argmax(1).cpu().tolist()
    pred = np.array(preds)
    acc = float((pred == yt.numpy()).mean() * 100)
    ir = float((pred[yt.numpy() == 1] == 1).mean() * 100)
    return acc, ir


def fisher_bands():
    """Compute the inner/outer Fisher discriminant ratio for both 3.5-4k and 4-4.8k"""
    bands = [(3500, 4000), (4000, 4800)]
    src = [SRC] + TARGETS
    log("`n===== inner vs outer Fisher discriminant ratio (4-4.8k added) =====")
    header = f"{'size@load':<12}" + "".join(f"{b[0] / 1000:.1f}-{b[1] / 1000:.1f}k".rjust(12) for b in bands)
    log(header)
    for dom in src:
        rows = {}
        for band in bands:
            sos = butter(4, [band[0] / (fs / 2.0), band[1] / (fs / 2.0)], btype="band", output="sos")
            specs = {}
            for cls in ["inner_fault", "outer_fault"]:
                sig = get_signal(DATA[dom][cls])
                n_win = (len(sig) - WIN) // STEP + 1
                ss = []
                for w in range(n_win - N_TEST, n_win):
                    seg = sosfiltfilt(sos, sig[w * STEP: w * STEP + WIN])
                    f, p = welch(seg, fs=fs, nperseg=WIN)
                    ss.append(np.log(p + 1e-12))
                specs[cls] = np.array(ss)
            A, B = specs["inner_fault"], specs["outer_fault"]
            rows[band] = float(((A.mean(0) - B.mean(0)) ** 2 / (A.var(0) + B.var(0) + 1e-12)).mean())
        log(f"{dom[0] + '@' + str(dom[1]) + 'HP':<12}" + "".join(f"{rows[b]:12.2f}" for b in bands))


def main():
    open(LOG, "w", encoding="utf-8").close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device: " + str(device))

    Xs, ys = domain_images(SRC, "train")
    tr, va = [], []
    for ci in range(len(CLASSES)):
        ids = list(range(ci * N_TRAIN, (ci + 1) * N_TRAIN))
        tr += ids[:120]; va += ids[120:]
    Xva = torch.from_numpy(Xs[va]).permute(0, 3, 1, 2).contiguous()
    yva = torch.from_numpy(ys[va])

    tests = {}
    for dom in TARGETS:
        tests[dom] = domain_images(dom, "test")
        log(f"test images ready {dom[0]}@{dom[1]}HP")

    res = {d: [] for d in TARGETS}
    for seed in [42, 123, 2026]:
        log(f"`n--- seed {seed} ---")
        model = train_one(seed, Xs[tr], ys[tr], Xva, yva, device)
        for dom in TARGETS:
            Xt, yt = tests[dom]
            acc, ir = evaluate(model, Xt, yt, device)
            res[dom].append((acc, ir))
            log(f"  {dom[0]}@{dom[1]}HP = {acc:6.2f}%  inner recall {ir:5.2f}%")
        del model; torch.cuda.empty_cache()

    log("`n===== fixed 4-4.8k band: 3 seeds, mean ± std =====")
    accs = []
    for dom in TARGETS:
        arr = np.array(res[dom])
        log(f"  {dom[0]}@{dom[1]}HP  accuracy {arr[:, 0].mean():6.2f} ± {arr[:, 0].std():.2f}"
            f"   inner recall {arr[:, 1].mean():5.2f} ± {arr[:, 1].std():.2f}")
        accs.append(arr[:, 0].mean())
    log(f"`n  mean over target domains = {np.mean(accs):.2f}%   (full-band baseline 62.25%, full-band 3-seed 62.52%)")
    for sz in ["0.014", "0.021"]:
        idx = [i for i, d in enumerate(TARGETS) if d[0] == sz]
        log(f"  {sz} group mean = {np.mean([accs[i] for i in idx]):.2f}%")


if __name__ == "__main__":
    main()

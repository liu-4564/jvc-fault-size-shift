# -*- coding: utf-8 -*-
"""bench_train.py -- measure one training epoch on GPU and on CPU.

Answers a single question: on this 4 GB T600 laptop, is the GPU actually faster
than the CPU for the paper's recipe (EfficientNet-B0, 224x224, 480 train windows,
30 epochs)?  Reports seconds per epoch and peak device memory.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_severity_unified_protocol as U          # noqa: E402

U.DATA.update({
    ("0.007", 1): {"normal": 98, "inner_fault": 106, "ball_fault": 119, "outer_fault": 131},
    ("0.007", 2): {"normal": 99, "inner_fault": 107, "ball_fault": 120, "outer_fault": 132},
    ("0.007", 3): {"normal": 100, "inner_fault": 108, "ball_fault": 121, "outer_fault": 133},
})


def build():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4)
    return m


def run(device_name, batch, epochs=2, n_train=480):
    device = torch.device(device_name)
    X, y = U.build_split("plain", "0.007", 0, "train")
    X = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(y)
    tr = []
    for ci in range(4):
        tr += list(np.where(y.numpy() == ci)[0][:120])
    tr = torch.tensor(tr)
    X, y = X[tr], y[tr]

    m = build().to(device)
    back = [p for n, p in m.named_parameters() if not n.startswith("classifier")]
    head = list(m.classifier.parameters())
    opt = torch.optim.AdamW([{"params": back, "lr": 1e-4}, {"params": head, "lr": 1e-3}],
                            weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    ts = []
    for ep in range(epochs):
        t0 = time.time()
        m.train()
        idx = torch.randperm(len(y))
        for i in range(0, len(y), batch):
            b = idx[i:i + batch]
            loss = crit(m(U.to_input(X[b]).to(device)), y[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.time() - t0)
        print("    epoch %d  batch %d  %.1f s" % (ep + 1, batch, ts[-1]), flush=True)

    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0.0
    print("  -> %s batch=%d  mean %.1f s/epoch  peak %.0f MiB  est. 30 ep = %.1f min"
          % (device_name, batch, np.mean(ts), peak, np.mean(ts) * 30 / 60), flush=True)
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(np.mean(ts))


if __name__ == "__main__":
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0),
              "| total %.0f MiB" % (torch.cuda.get_device_properties(0).total_memory / 2 ** 20),
              flush=True)
        for bs in (32, 16, 8):
            try:
                run("cuda", bs)
            except RuntimeError as e:
                print("  cuda batch=%d FAILED: %s" % (bs, str(e).split("\n")[0]), flush=True)
                torch.cuda.empty_cache()
    run("cpu", 32)

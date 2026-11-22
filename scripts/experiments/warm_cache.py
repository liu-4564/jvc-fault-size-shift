# -*- coding: utf-8 -*-
"""warm_cache.py -- build the CWT image caches without training anything.

Needed because the multi-source matrix evaluates representations and domains that
the original runs never cached (0.007 @ 1/2/3 HP, and the env / order pipelines
over all twelve domains).

Usage:
  python scripts/experiments/warm_cache.py plain 0.007@1 0.007@2 0.007@3
  python scripts/experiments/warm_cache.py env order
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cross_severity_unified_protocol as U          # noqa: E402

U.DATA.update({
    ("0.007", 1): {"normal": 98, "inner_fault": 106, "ball_fault": 119, "outer_fault": 131},
    ("0.007", 2): {"normal": 99, "inner_fault": 107, "ball_fault": 120, "outer_fault": 132},
    ("0.007", 3): {"normal": 100, "inner_fault": 108, "ball_fault": 121, "outer_fault": 133},
})


def main():
    reps = [a for a in sys.argv[1:] if not a[0].isdigit()] or ["plain", "env", "order"]
    doms = [a for a in sys.argv[1:] if a[0].isdigit()] or ["0.007@0", "0.007@1", "0.007@2",
                                                          "0.007@3"]
    t0 = time.time()
    for rep in reps:
        for d in doms:
            size, load = d.split("@")
            for split in ("train", "test"):
                t = time.time()
                X, y = U.build_split(rep, size, int(load), split)
                print("%-6s %-9s %-5s %s  (%.1f s)"
                      % (rep, d, split, X.shape, time.time() - t), flush=True)
    print("total %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()

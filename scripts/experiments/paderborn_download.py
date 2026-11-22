"""Parallel chunked downloader for the Paderborn dataset (the official server throttles
a single connection to about 19 KB/s, while 32 threads aggregate to about 550 KB/s).

Usage:
    python paderborn_download.py KA01 KI03 ...
    python paderborn_download.py --all
Files land in $PADERBORN_RAW (default scripts/experiments/paderborn/raw);
an existing file of the correct size is skipped.
"""
import os
import sys
import time
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PADERBORN_RAW", os.path.join(_HERE, "paderborn", "raw"))
CHUNK = 2 * 1024 * 1024
THREADS = 32

# Bearings needed by Section 4.x of the paper: healthy + artificial damage (severity 1/2)
# + real damage (severity 1/2/3)
BEARINGS = {
    "healthy": ["K001", "K002", "K003"],
    "outer-severity1": ["KA01", "KA05", "KA07"],
    "inner-severity1": ["KI01", "KI03", "KI05"],
    "outer-severity2": ["KA03", "KA06", "KA09"],
    "inner-severity2": ["KI07", "KI08"],
    "inner-severity3": ["KI16"],
}


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def fetch_chunk(url, start, end, path, retries=4):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)}")
            with open(path, "wb") as f:
                f.write(data)
            return len(data)
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2)


def download(code):
    url = BASE + code + ".rar"
    dst = os.path.join(OUT, code + ".rar")
    total = head_size(url)
    if os.path.exists(dst) and os.path.getsize(dst) == total:
        print(f"[skip] {code} already complete ({total/1e6:.0f} MB)")
        return
    parts_dir = os.path.join(OUT, "_parts", code)
    os.makedirs(parts_dir, exist_ok=True)
    jobs = [(i, s, min(s + CHUNK - 1, total - 1))
            for i, s in enumerate(range(0, total, CHUNK))]
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futs = {ex.submit(fetch_chunk, url, s, e, os.path.join(parts_dir, f"{i:05d}.bin")): i
                for i, s, e in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            done += f.result()
            if n % max(1, len(jobs) // 8) == 0 or n == len(jobs):
                kb = done / 1024 / (time.time() - t0)
                print(f"  {code}: chunk {n}/{len(jobs)}  {done/1e6:.0f}/{total/1e6:.0f} MB  {kb:.0f} KB/s")
    with open(dst, "wb") as out:
        for i, s, e in jobs:
            with open(os.path.join(parts_dir, f"{i:05d}.bin"), "rb") as f:
                out.write(f.read())
    for i, _, _ in jobs:
        os.remove(os.path.join(parts_dir, f"{i:05d}.bin"))
    os.rmdir(parts_dir)
    print(f"[done] {code} {os.path.getsize(dst)/1e6:.0f} MB, elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    args = sys.argv[1:]
    if not args or args[0] == "--all":
        codes = [c for g in BEARINGS.values() for c in g]
    elif args[0] == "--list":
        for g, cs in BEARINGS.items():
            print(g, cs)
        sys.exit(0)
    else:
        codes = args
    print("total", len(codes), "bearings:", codes)
    for c in codes:
        download(c)

"""Reader for the Paderborn bearing dataset.

Data layout (observed):
    one rar per bearing; after extraction the files are <code>/<setting>_<code>_<index>.mat, e.g.
        K001/N09_M07_F10_K001_1.mat
    the top level of the .mat is a struct named after the file, with fields: Info / X / Y / Description
        X  : 3 time axes  [Mech_4kHz(16008 samples) / HostService(256823 samples) / Temp_1Hz(5 samples)]
        Y  : 7 signal channels
             [0] force             4 kHz
             [1] phase_current_1  64 kHz (HostService)
             [2] phase_current_2  64 kHz
             [3] speed             4 kHz
             [4] temp_2_bearing_module 1 Hz
             [5] torque            4 kHz
             [6] vibration_1      64 kHz  <-- vibration, the main signal of this paper
    Setting codes: N09/N15 = 900/1500 rpm; M01/M07 = 0.1/0.7 Nm; F04/F10 = 4/10 kN.
    Each "setting x bearing" contains 20 records of 4 s (_1 ... _20).

Usage:
    python paderborn_reader.py K001          # print the file list and signal summary of this bearing
    python paderborn_reader.py --probe K001  # print the full nested structure of one .mat
    python paderborn_reader.py --check       # check whether downloaded/extracted bearings are readable
"""
import os
import sys
import glob
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("PADERBORN_RAW", os.path.join(_HERE, "paderborn", "raw"))
EXT = os.environ.get("PADERBORN_EXT", os.path.join(_HERE, "paderborn", "extracted"))
FS_VIB = 64000

# Damage annotations of Table 4/5 of the paper (Component: IR/OR; Extent 1/2/3;
# Method: artificial / lifetime test)
LABELS = {
    "K001": ("N", 0, "healthy"), "K002": ("N", 0, "healthy"), "K003": ("N", 0, "healthy"),
    "KA01": ("OR", 1, "artificial"), "KA05": ("OR", 1, "artificial"), "KA07": ("OR", 1, "artificial"),
    "KI01": ("IR", 1, "artificial"), "KI03": ("IR", 1, "artificial"), "KI05": ("IR", 1, "artificial"),
    "KA03": ("OR", 2, "artificial"), "KA06": ("OR", 2, "artificial"),
    "KA08": ("OR", 2, "artificial"), "KA09": ("OR", 2, "artificial"),
    "KI07": ("IR", 2, "artificial"), "KI08": ("IR", 2, "artificial"),
    "KI16": ("IR", 3, "lifetime"),
    "KA04": ("OR", 1, "lifetime"), "KA15": ("OR", 1, "lifetime"),
    "KA16": ("OR", 2, "lifetime"), "KA22": ("OR", 1, "lifetime"), "KA30": ("OR", 1, "lifetime"),
    "KB23": ("IR+OR", 2, "lifetime"), "KB24": ("IR+OR", 3, "lifetime"), "KB27": ("IR+OR", 1, "lifetime"),
    "KI04": ("IR", 1, "lifetime"), "KI14": ("IR", 1, "lifetime"),
    "KI17": ("IR", 1, "lifetime"), "KI21": ("IR", 1, "lifetime"), "KI18": ("IR", 2, "lifetime"),
}


def extract(code):
    """Extract a rar into EXT/<code>/ with the Windows built-in tar.exe (bsdtar), no 7-Zip/WinRAR needed."""
    rar = os.path.join(RAW, code + ".rar")
    dst = os.path.join(EXT, code)
    if glob.glob(os.path.join(dst, "**", "*.mat"), recursive=True):
        return dst
    if not os.path.exists(rar):
        raise FileNotFoundError(rar)
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["tar.exe", "-xf", rar, "-C", dst], check=True)
    return dst


def _unwrap(v):
    """Strip the MATLAB cell wrappers until the actual struct/array is reached."""
    a = np.asarray(v)
    while a.dtype == object and a.size == 1 and not isinstance(a.ravel()[0], str):
        a = np.asarray(a.ravel()[0])
    return a


def read_mat(path):
    """Read one .mat and return a dict: vibration / current1 / current2 / speed / force / torque / temp."""
    try:
        import scipy.io as sio
        m = sio.loadmat(path)
    except NotImplementedError:  # MATLAB v7.3 (HDF5)
        import h5py
        m = {}
        with h5py.File(path, "r") as f:
            for k, v in f.items():
                m[k] = _unwrap(v[()])
    key = [k for k in m if not k.startswith("__")][0]
    s = _unwrap(m[key])

    out = {}
    y = _unwrap(s["Y"])
    names = y["Name"][0]
    datas = y["Data"][0]
    for nm, d in zip(names, datas):
        name = str(np.asarray(nm).ravel()[0]) if np.asarray(nm).size else ""
        out[name] = np.asarray(_unwrap(d), dtype=np.float64).ravel()

    x = _unwrap(s["X"])
    out["_x_mech_len"] = int(np.asarray(_unwrap(x["Data"][0][0])).size)
    out["_x_fast_len"] = int(np.asarray(_unwrap(x["Data"][0][1])).size)

    # unified aliases
    out["vibration"] = out.get("vibration_1", np.array([]))
    out["current1"] = out.get("phase_current_1", np.array([]))
    out["current2"] = out.get("phase_current_2", np.array([]))
    out["fs_vib"] = FS_VIB
    out["fs_mech"] = 4000
    out["code"] = os.path.basename(path).split("_")[2]
    out["setting"] = "_".join(os.path.basename(path).split("_")[:3])
    out["index"] = int(os.path.basename(path).rsplit("_", 1)[1].split(".")[0])
    return out


def list_files(code):
    d = extract(code)
    return sorted(glob.glob(os.path.join(d, "**", "*.mat"), recursive=True))


def summarize(code):
    mats = list_files(code)
    label = LABELS.get(code, ("?", 0, "?"))
    print(f"\n=== {code}  class={label[0]} severity={label[1]} source={label[2]}  files={len(mats)} ===")
    settings = sorted({os.path.basename(p).split("_")[0] + "_" +
                       os.path.basename(p).split("_")[1] + "_" +
                       os.path.basename(p).split("_")[2] for p in mats})
    for st in settings:
        print(f"  setting {st}: {sum(st in os.path.basename(p) for p in mats)} records")
    v = read_mat(mats[0])
    print(f"  sample {os.path.basename(mats[0])}: vibration {v['vibration'].size} samples "
          f"({v['vibration'].size/v['fs_vib']:.2f} s @ {v['fs_vib']} Hz), "
          f"mean {v['vibration'].mean():.4g}, std {v['vibration'].std():.4g}, "
          f"speed {np.mean(v.get('speed', [np.nan])):.1f} rpm")


def probe(path):
    import scipy.io as sio
    m = sio.loadmat(path)
    key = [k for k in m if not k.startswith("__")][0]

    def walk(v, prefix="", depth=0):
        if depth > 5:
            return
        a = _unwrap(v)
        if a.dtype.names:
            print(f"  {'  '*depth}{prefix} <struct> {a.dtype.names}")
            for nm in a.dtype.names:
                walk(a[nm], f"{prefix}.{nm}" if prefix else nm, depth + 1)
            return
        extra = ""
        if a.dtype.kind in "fiu" and a.size > 10:
            extra = f" min={float(a.min()):.4g} max={float(a.max()):.4g}"
        print(f"  {'  '*depth}{prefix} shape={a.shape} {a.dtype}{extra}")

    print(f"\n--- {os.path.basename(path)} ---")
    walk(m[key], key)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
    elif args[0] == "--probe":
        probe(sorted(glob.glob(os.path.join(EXT, args[1], "**", "*.mat"), recursive=True))[0])
    elif args[0] == "--check":
        codes = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(RAW, "*.rar")))
        for c in codes:
            try:
                summarize(c)
            except Exception as e:
                print(f"[failed] {c}: {type(e).__name__}: {e}")
    else:
        for c in args:
            summarize(c)

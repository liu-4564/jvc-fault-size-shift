# jvc-fault-size-shift

Code, data and manuscript sources for a negative-result study of cross-severity
fault-size shift in rolling-bearing diagnosis on CWRU, with external checks on
the Paderborn and JNU datasets. Submitted to the *Journal of Vibration and Control*.

**Status:** manuscript **JVC-26-2391**, submitted 4 November 2026, under review.

Author: Jiarui Liu, Qingdao University of Technology

## Repository layout

```
manuscript/    Markdown sources (manuscript_main.md, cover_letter.md,
               supplementary_material.md) and the seven figure PNGs they embed
scripts/       plot_figures.py      - regenerate all figures (PNG+PDF)
               export_figure_csv.py - export the plotted data to CSV
               build_docx.py        - pandoc-based docx build driver
scripts/experiments/
               Training / evaluation code for every protocol and dataset:
               cross_severity_unified_protocol.py  P1/P2/P3 unified protocol
               noise_protocol.py                   P2 cross-SNR
               multiseed_cross_severity.py         3-seed repeats on P3
               adaptive_band.py                    the eight remedies
               high_band_multiseed.py              fixed 4-4.8 kHz band
               band_fisher_12domains.py            Welch-Fisher band analysis
               paderborn_reader.py                 Paderborn data loading
               paderborn_download.py               Paderborn download helper
               paderborn_cross_size.py             Paderborn cross-size check
               jnu_cross_speed.py                  JNU cross-speed check
               jnu_band_analysis.py                JNU band analysis
data/          unified_p3_final.json, fisher_12domains.json,
               multiseed_recall.json and figure_data_csv/ (per-figure CSVs)
tests/         test_data_consistency.py - asserts the committed JSON reproduces
               the numbers printed in the paper and the supplement
```

`requirements.txt`, `LICENSE`, `CITATION.cff` and `.github/workflows/` sit at the
top level.

## Environment

```
python -m pip install -r requirements.txt
```

Reproducing the seven figures and the CSV exports needs only `numpy`,
`matplotlib` and `scienceplots` — the narrowest subset that runs `plot_figures.py`.
The remaining packages in `requirements.txt` (`torch`, `torchvision`,
`pywavelets`, `scipy`, `Pillow`) are needed only to re-run the training and
evaluation scripts, and a CUDA GPU is required for those runs; the reference
environment was a single 4 GB NVIDIA T600, as stated in Section S8 of the
Supplementary Material.

`pandoc` is a system binary rather than a Python package, so it is not listed in
`requirements.txt`. It is needed only by `build_docx.py`; see the note under
Reproduce.

## Data access

The datasets are public but too large to redistribute here, so they are
downloaded from their official sources and placed under `data/`.

| Dataset | Source | Expected local path | Notes |
|---|---|---|---|
| CWRU | https://engineering.case.edu/bearingdatacenter/download-data-file | `data/cwru/` | 12 kHz drive-end `.mat` files; see Table 1 of the manuscript for the file numbers used |
| Paderborn | https://groups.uni-paderborn.de/kat/BearingDataCenter/ | `data/paderborn/` | registration required; `paderborn_download.py` automates the transfer once access is granted |
| JNU | https://github.com/ClarkGableWang/JNU-Bearing-Dataset | `data/jnu/` | 50 kHz, 600/800/1000 rpm, four health states |

Because directory names differ between machines, the three CWRU roots are read
from environment variables. The defaults are relative paths, so the experiment
scripts must be run from the repository root for them to resolve:

```
CWRU_SAME_LOAD_DIR   (default data/cwru/same_load_same_size)
CWRU_CROSS_LOAD_DIR  (default data/cwru/same_size_cross_load)
CWRU_CROSS_SIZE_DIR  (default data/cwru/cross_fault_size)
```

So a default run looks like this, from the repository root:

```bash
python scripts/experiments/cross_severity_unified_protocol.py plain env order
```

Set the variables only if your dataset lives outside the repository:

```bash
# Linux / macOS
export CWRU_SAME_LOAD_DIR=/path/to/same_load_same_size
export CWRU_CROSS_LOAD_DIR=/path/to/same_size_cross_load
export CWRU_CROSS_SIZE_DIR=/path/to/cross_fault_size
```

```powershell
# Windows (PowerShell)
$env:CWRU_SAME_LOAD_DIR   = "D:\data\same_load_same_size"
$env:CWRU_CROSS_LOAD_DIR  = "D:\data\same_size_cross_load"
$env:CWRU_CROSS_SIZE_DIR  = "D:\data\cross_fault_size"
```

```bat
REM Windows (cmd.exe)
set CWRU_SAME_LOAD_DIR=D:\data\same_load_same_size
```

Only the aggregated JSON/CSV results in `data/` are committed, so figure
generation and CSV export work straight after cloning, without any dataset.

## Reproduce

1. Figures (reads `data/*.json`, writes PNG+PDF into `manuscript/`):

   ```
   python scripts/plot_figures.py
   ```

   Requires Python with `matplotlib`, `numpy` and `scienceplots`. No dataset is
   needed. The data behind every panel are identical to those in the paper, but
   the PNG bytes are not guaranteed to match the committed files bit for bit,
   because matplotlib rendering varies between versions.

2. CSV exports (rewrites `data/figure_data_csv/`):

   ```
   python scripts/export_figure_csv.py
   ```

3. Docx build — **not reproducible from a clean clone.** `scripts/build_docx.py`
   is kept as the driver actually used for submission, but it depends on a
   journal-specific post-processing helper (`docx_build_helper`) that is not
   distributed with this repository. Running it after cloning exits with
   `docx_build_helper module not found`. The submitted `.docx` files are the
   artefacts of record; the Markdown sources in `manuscript/` are the editable
   form, and `pandoc` alone can convert them to `.docx` if needed.

## Data sources

All experiments use publicly available bearing datasets: CWRU (Case Western
Reserve University Bearing Data Center), Paderborn (KAt datacenter) and JNU
(Jiangnan University). The JSON files in `data/` contain only aggregated
accuracy / Fisher-ratio results produced by our evaluation protocols.

## Data consistency check

`tests/test_data_consistency.py` asserts that the committed JSON reproduces the
numbers printed in the manuscript (Tables 4 and S2, the 62.25% / 62.52% /
50.03% headline figures, and the per-domain standard deviations in the
abstract). It uses the standard library only:

```
python tests/test_data_consistency.py
```

The same check runs on every push through
`.github/workflows/data-consistency.yml`.

## Citation and license

If you use this code or these results, please cite the paper (see
`CITATION.cff`, or the "Cite this repository" button on GitHub). The code is
released under the MIT License (`LICENSE`); the manuscript text and figures are
released under CC BY 4.0.

This repository is the public companion to the submitted manuscript (JVC-26-2391)
and the URL above is cited in the paper's Data availability statement. Once the
paper is accepted it will be archived on Zenodo; the version DOI will be added
here.

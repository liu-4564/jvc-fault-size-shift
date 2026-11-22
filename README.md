# jvc-fault-size-shift

Code and aggregated results for a controlled study of cross-severity fault-size
shift in rolling-bearing diagnosis: three time–frequency representations
compared on CWRU, with external checks on the Paderborn and JNU datasets.

The repository contains experiment code, the aggregated result files those runs
produced, and the plotting code that turns them into figures. It contains no
manuscript text.

## Repository layout

```
scripts/       plot_figures.py      - regenerate all figures (PNG+PDF)
               export_figure_csv.py - export the plotted data to CSV
scripts/experiments/
               Training / evaluation code for every protocol and dataset:
               cross_severity_unified_protocol.py  P1/P2/P3 unified protocol
               multisource_matrix.py               44-pair replication matrix
               merge_matrix.py                     merge matrix shards
               analyze_matrix.py                   label-free descriptors
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
               make_tables.py                      tabulate the matrix results
               seed_check.py, warm_cache.py        utilities
data/          unified_p3_final.json, fisher_12domains.json,
               multiseed_recall.json and figure_data_csv/ (per-figure CSVs)
figures/       output directory for plot_figures.py (not tracked)
tests/         test_data_consistency.py - asserts the committed JSON matches the
               reference values
```

`requirements.txt`, `LICENSE` and `.github/workflows/` sit at the top level.

## Environment

```
python -m pip install -r requirements.txt
```

Reproducing the figures and the CSV exports needs only `numpy`, `matplotlib`
and `scienceplots` — the narrowest subset that runs `plot_figures.py`. The
remaining packages in `requirements.txt` (`torch`, `torchvision`,
`pywavelets`, `scipy`, `Pillow`) are needed only to re-run the training and
evaluation scripts, and a CUDA GPU is required for those runs; the reference
environment was a single 4 GB NVIDIA T600.

## Data access

The datasets are public but too large to redistribute here, so they are
downloaded from their official sources and placed under `data/`.

| Dataset | Source | Expected local path | Notes |
|---|---|---|---|
| CWRU | https://engineering.case.edu/bearingdatacenter/download-data-file | `data/cwru/` | 12 kHz drive-end `.mat` files; the file numbers are defined at the top of the experiment scripts |
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

Only the aggregated JSON/CSV results in `data/` and the per-run JSON files in
`scripts/experiments/` are committed, so figure generation and CSV export work
straight after cloning, without any dataset.

## Reproduce

1. Figures (reads `data/*.json`, writes PNG+PDF into `figures/`):

   ```
   python scripts/plot_figures.py
   ```

   Requires Python with `matplotlib`, `numpy` and `scienceplots`. No dataset is
   needed. The PNG bytes are not guaranteed to be identical to any previously
   generated copy, because matplotlib rendering varies between versions.

2. CSV exports (rewrites `data/figure_data_csv/`):

   ```
   python scripts/export_figure_csv.py
   ```

## Data sources

All experiments use publicly available bearing datasets: CWRU (Case Western
Reserve University Bearing Data Center), Paderborn (KAt datacenter) and JNU
(Jiangnan University). The JSON files in `data/` contain only aggregated
accuracy / Fisher-ratio results produced by the evaluation protocols above.

## Data consistency check

`tests/test_data_consistency.py` asserts that the committed JSON reproduces the
reference values (the 62.25% / 62.52% / 50.03% headline accuracies and the
per-domain standard deviations). It uses the standard library only:

```
python tests/test_data_consistency.py
```

The same check runs on every push through
`.github/workflows/data-consistency.yml`.

## License

The code is released under the MIT License (`LICENSE`).

# Supplemental Material

**Fault-size shift in rolling-bearing diagnosis: a controlled comparison of time–frequency representations and the limits of band-migration invariance**

Jiarui Liu

This supplement contains the numeric tables and protocol details that are summarised in, but not reproduced by, the main article. Table and figure numbers of the main text are prefixed "Table" and "Figure"; supplement items are prefixed "S".

---

## S1 Confusion matrices of the two representative target domains

Source domain 0.007@0HP, plain representation, 50 test windows per class. Rows are true classes, columns predicted classes.

Table S1(a) 0.014@0HP (accuracy 50.00%)

| True \ Predicted | Inner race | Outer race | Ball | Normal | Recall |
|---|---|---|---|---|---|
| Inner race | 0 | **50** | 0 | 0 | 0.0% |
| Outer race | 0 | 0 | **50** | 0 | 0.0% |
| Normal | 0 | 0 | 0 | 50 | 100% |
| Ball | 0 | 0 | 50 | 0 | 100% |

Table S1(b) 0.021@0HP (accuracy 75.00%)

| True \ Predicted | Inner race | Outer race | Ball | Normal | Recall |
|---|---|---|---|---|---|
| Inner race | 0 | 0 | **50** | 0 | 0.0% |
| Outer race | 0 | 50 | 0 | 0 | 100% |
| Normal | 0 | 0 | 0 | 50 | 100% |
| Ball | 0 | 0 | 50 | 0 | 100% |

---

## S2 Band-wise inner/outer-race Fisher discriminant ratio, all 12 size–load domains

Per-window Welch log-power spectra were pooled over the last 50 test windows of each class; the ratio `F = (m_IR − m_OR)² / (v_IR + v_OR)` was evaluated frequency-bin-wise and averaged inside each band. No band-pass filtering was applied, which is why the values differ from those of a filter-bank implementation. The 4–4.8 kHz band used as a remedy in the main text (Section 4.7, Table 6) was not included in this sweep; the adjacent 3.5–4 kHz column is used there as a proxy.

Table S2 Inner/outer-race Fisher ratio per band

| Size@Load | 0–500 | 0.5–1k | 1–1.5k | 1.5–2k | 2–2.5k | 2.5–3k | 3–3.5k | 3.5–4k | Peak |
|---|---|---|---|---|---|---|---|---|---|
| 0.007@0HP | 5.54 | 2.43 | **9.94** | 4.07 | 2.83 | 3.60 | 7.74 | 2.43 | 1–1.5 |
| 0.007@1HP | 5.20 | 3.13 | **8.36** | 6.12 | 3.76 | 4.30 | 5.87 | 2.06 | 1–1.5 |
| 0.007@2HP | 5.01 | 2.16 | **8.37** | 6.24 | 3.14 | 6.01 | 7.81 | 2.83 | 1–1.5 |
| 0.007@3HP | 4.24 | 3.50 | **6.44** | 4.78 | 2.49 | 4.71 | 6.11 | 1.42 | 1–1.5 |
| 0.014@0HP | 3.87 | 1.07 | 1.33 | 1.52 | 5.25 | **6.37** | 1.34 | 1.55 | 2.5–3 |
| 0.014@1HP | 2.22 | 0.71 | 0.93 | 0.65 | 3.48 | **4.13** | 1.03 | 1.71 | 2.5–3 |
| 0.014@2HP | 1.52 | 0.73 | 1.57 | 0.58 | 3.45 | **3.85** | 0.96 | 1.19 | 2.5–3 |
| 0.014@3HP | 1.38 | 0.64 | **9.96** | 0.86 | 2.72 | 3.76 | 1.53 | 0.72 | 1–1.5 |
| 0.021@0HP | 4.17 | 3.14 | 5.22 | 2.75 | 1.73 | 1.41 | 3.58 | **5.96** | 3.5–4 |
| 0.021@1HP | 2.78 | 4.19 | **6.45** | 2.50 | 1.26 | 1.08 | 2.54 | 4.74 | 1–1.5 |
| 0.021@2HP | 2.66 | 3.94 | **7.41** | 2.41 | 1.41 | 1.60 | 2.46 | 4.06 | 1–1.5 |
| 0.021@3HP | 1.92 | 3.90 | **6.38** | 2.24 | 1.26 | 2.25 | 2.12 | 3.88 | 1–1.5 |

---

## S3 Band-wise Fisher ratio on Paderborn

Method as in S2, with the same estimator and no band-pass filtering; 16 kHz signals.

Table S3 Inner/outer-race Fisher ratio per band on Paderborn (window-level average)

| Group | 0–0.5 | 0.5–1 | 1–1.5 | 1.5–2 | 2–2.5 | 2.5–3 | 3–3.5 | 3.5–4 | 4–4.5 | 4.5–5 | 5–5.5 | 5.5–6 | 6–6.5 | 6.5–7 | 7–7.5 | 7.5–8 kHz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Grade 1 | 0.03 | 0.04 | 0.04 | 0.05 | 0.05 | 0.05 | **0.06** | 0.04 | 0.03 | 0.04 | 0.04 | 0.03 | 0.02 | 0.02 | 0.02 | 0.03 |
| Grade 2 | 0.06 | 0.08 | **0.10** | 0.09 | 0.09 | 0.09 | 0.09 | 0.07 | 0.05 | 0.03 | 0.02 | 0.03 | 0.02 | 0.02 | 0.02 | 0.02 |
| Grade 1 (N09 condition) | 0.02 | 0.03 | **0.03** | 0.02 | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.01 |

---

## S4 Per-class recall of the six JNU transfer tasks

Table S4 Per-class recall (mean over 3 seeds, %)

| Task | Normal | Inner race | Outer race | Ball | Overall accuracy |
|---|---|---|---|---|---|
| 600 → 800 | 17.33 | 100.00 | 43.33 | 96.00 | 64.17 |
| 600 → 1000 | 3.33 | 100.00 | 48.67 | 96.00 | 62.00 |
| 800 → 600 | 54.67 | 100.00 | 54.00 | 98.67 | 76.83 |
| 800 → 1000 | 50.67 | 100.00 | 100.00 | 100.00 | 87.67 |
| 1000 → 600 | 0.00 | 99.33 | 20.00 | 100.00 | 54.83 |
| 1000 → 800 | 91.33 | 98.67 | 100.00 | 100.00 | 97.50 |

---

## S5 Peak band of the Fisher ratio for key class pairs at each JNU speed

Table S5 Peak band (peak F in parentheses)

| Class pair | 600 rpm | 800 rpm | 1000 rpm |
|---|---|---|---|
| Normal vs inner race | 2.0–2.5 kHz (0.44) | 3.0–3.5 kHz (0.34) | 2.0–2.5 kHz (0.25) |
| Inner race vs outer race | 1.0–1.5 kHz (0.22) | 4.0–4.5 kHz (0.49) | 4.0–4.5 kHz (0.49) |
| Normal vs outer race | 1.0–1.5 kHz (0.32) | 0.5–1.0 kHz (0.38) | 4.0–4.5 kHz (0.29) |

Each column is computed from its own per-speed Fisher matrix; the two 0.49 entries for inner vs outer race at 800 and 1000 rpm are therefore separate estimates that coincide to two decimals.

---

## S6 Classification-head capacity

On the 0.007 data the linear head (5,124 parameters) was replaced by an MLP head and a KAN head (Liu et al., 2024) of approximately 100,000 parameters, with the backbone fully unfrozen and all other settings identical. Cross-load accuracies are given below; the protocol is P1 (envelope representation), so the linear-head reference is 70.67%.

Table S6 Cross-load accuracy with enlarged classification heads (%)

| Head | Parameters | Cross-load mean |
|---|---|---|
| Linear (reference) | 5,124 | 70.67 |
| MLP | ~100,000 | 45.33 |
| KAN | ~100,000 | 56.83 |

Under the same protocol the plain representation reaches 99.50%, so a larger head does not help and the linear head is sufficient. Replacing the classification head is therefore not a viable remedy for severity drift.

---

## S7 Complete protocol and hyperparameters

**CWRU.** Drive-end acceleration, 12 kHz. Window 2048 points, stride 512. Morlet CWT, scales 1–99, magnitude compressed by `log1p`, min–max normalised to 0–255, resized to 224×224, replicated to three channels. Training partition: first 150 windows per class (120 optimisation + 30 convergence monitoring); test: last 50 windows per class. Source domain 0.007@0HP; eight target domains {0.014, 0.021}@{0,1,2,3}HP.

**Paderborn.** 64 kHz decimated 4-fold (anti-aliased) to 16 kHz; 2048-point window; 224×224 Morlet CWT; condition N15_M07_F10 (1500 rpm / 0.7 Nm / 10 kN); 20 segments per bearing; three classes (normal, inner race, outer race); 150/50/50 windows per class. Group A: train KI01/KI03, KA01/KA05, test KI05/KA07 (grade 1). Group B: test KI07/KI08, KA03/KA06/KA09 (grade 2). Healthy class tested on K003 in both groups.

**JNU.** 50 kHz decimated 4-fold to 12.5 kHz; 2048-point window (164 ms); stride 512; four classes; speeds 600/800/1000 rpm; first 150 windows per class for training, next 50 for validation, last 50 for test.

**Model.** EfficientNet-B0, ImageNet pretrained, four-class (three-class for Paderborn) fully connected head. Differential learning rates 1e-4 (backbone) / 1e-3 (head); AdamW, weight decay 1e-4; cosine annealing; 30 epochs; batch 32. No early stopping, no best-checkpoint rollback: the reported model is the one after epoch 30 for every run. Augmentation: RandomResizedCrop(224, scale=(0.85,1.0)) and light ColorJitter. No domain-adaptation module.

**Noise protocol (P2).** Additive Gaussian white noise with SNR = 10log10(P_s/P_n), P_s the signal power within the window; test noise added independently per condition with a fixed noise seed. Noise training: two noisy copies per window, SNR ~ U(−5,10) dB.

---

## S8 Reproducibility

All scripts, cached feature splits and the exact command lines used for the CWRU, Paderborn and JNU experiments are available from the corresponding author on reasonable request and will be deposited in a public repository with a Zenodo DOI for the frozen version on acceptance. Seeds are fixed at 42/123/2026; the software environment is Python 3.10 + PyTorch 2.x + torchvision, on a single NVIDIA T600 Laptop GPU (4 GB). Every reported number comes from the 30th epoch, and the eight target domains of P3 are evaluated in one pass after training.

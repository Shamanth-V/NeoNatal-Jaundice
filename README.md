# NeoJaundice — Smartphone-Based Screening for Neonatal Jaundice

Predicting clinically significant neonatal hyperbilirubinemia (jaundice) from smartphone skin photographs and basic newborn metadata, using classical ML, gradient boosting, and transfer-learning CNNs — with rigorous **patient-level** (not image-level) evaluation to avoid data leakage.

> ⚕️ **Disclaimer:** This is a research/educational project. It is **not** a validated medical device and must not be used for real clinical decision-making. See [Limitations](#limitations--disclaimer).

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Limitations & Disclaimer](#limitations--disclaimer)
- [Future Work](#future-work)
- [License](#license)

---

## Problem Statement

Neonatal jaundice, caused by elevated total serum bilirubin (TSB), affects a large share of newborns and can lead to severe complications (kernicterus) if undetected. The clinical gold standard is a blood test (TSB) or a dedicated transcutaneous bilirubinometer — both requiring specialized equipment often unavailable in low-resource settings.

This project explores whether a **standard smartphone photo** of a newborn's skin (captured alongside a color reference card for lighting calibration) plus basic metadata (age in days, gestational age, weight, gender) can screen for clinically significant jaundice, defined here as **TSB > 12.9 mg/dL**.

## Dataset

- **Source:** [NeoJaundice public dataset](https://github.com/) — 2,232 skin images from newborns, paired with lab-confirmed bilirubin values and metadata (`chd_jaundice_published_2.csv`).
- **Labels:** Binary classification (TSB > 12.9 mg/dL threshold) with support for a 3-class variant. Class balance ≈ 58% / 42%.
- **Grouping:** Images are grouped by patient ID — multiple photos can belong to the same infant, which is critical for correct cross-validation (see below).

## Approach

The pipeline was built and validated incrementally across four stages:

**1. Preprocessing**
- Detect the color reference card in each raw image and use it to color-correct the photo, normalizing for lighting/white-balance variation across devices and environments.
- Segment the skin region of interest (ROI) and resize to a fixed 128×128 patch.

**2. Baselines & Transfer Learning**
- Tabular baselines with **LightGBM** on metadata alone, color statistics alone, and the combination — to establish a performance floor before touching pixels.
- A **2D CNN** (EfficientNetB0, ImageNet-pretrained) with a metadata side-branch, trained with a two-phase strategy: frozen-backbone head training, followed by fine-tuning the top ~30 layers with a low learning rate (BatchNorm layers kept frozen).
- Automatic accelerator detection (TPU → GPU → CPU) via `tf.distribute`.

**3. Feature-Rich Ensemble**
- Extracted 100+ hand-crafted color/texture descriptors per image: per-channel statistics (mean, std, percentiles) across RGB, HSV, LAB, and YCrCb color spaces, chromaticity ratios, skin-pixel-only statistics (via YCrCb skin masking), and both **color-corrected** and **raw/uncorrected** pixel statistics (letting the model learn its own implicit lighting correction).
- Combined these with metadata and card-region statistics, then trained a **LightGBM + Logistic Regression** ensemble, blended with the CNN's out-of-fold predictions.
- Repeated across multiple cross-validation fold seeds and averaged for stability.

**4. Diagnostics & Honest Evaluation**
- Sanity-checked the color card detector against bilirubin (to rule out the "card" actually being skin/background).
- Broke down accuracy by distance from the classification threshold to confirm errors cluster near the clinical decision boundary (label ambiguity) rather than indicating a systematic modeling failure.
- Trained a **regression-then-threshold** model (predicting continuous TSB, then thresholding) as an alternative framing.
- Reported clinically meaningful **operating points** (specificity at fixed sensitivity targets of 90% / 95%).
- Directly compared an **image-level random split** (leaky — same patient can appear in train and test) against the **patient-level grouped split** (honest) to quantify how much leakage would have inflated the reported numbers.

## Results

All results below use **patient-level, grouped, stratified 5-fold cross-validation** (`StratifiedGroupKFold`) — no patient's images appear in both train and test for the same fold.

| Model | Per-image Accuracy | AUC | Patient-Averaged Accuracy |
|---|---|---|---|
| LightGBM — metadata only | 0.673 | 0.737 | — |
| LightGBM — color only | 0.754 | 0.830 | — |
| LightGBM — metadata + color | 0.773 | 0.852 | — |
| EfficientNetB0 + metadata (CNN) | 0.727 | 0.792 | 0.754 |
| LightGBM — meta + rich ROI features | 0.772 | 0.859 | 0.806 |
| LogReg — meta + rich ROI features | 0.774 | 0.853 | 0.819 |
| LightGBM — meta + ROI + card/raw | 0.785 | 0.867 | 0.812 |
| LogReg — meta + ROI + card/raw | 0.779 | 0.863 | 0.828 |
| **LGBM + LogReg ensemble** | 0.795 | 0.875 | 0.827 |
| **LGBM + LogReg, 3 fold-seed average (best)** | **0.797** | **0.876** | **0.831** (AUC 0.910) |

**Clinical operating points** (out-of-fold, best ensemble):
- At **90% sensitivity** → 69.3% specificity
- At **95% sensitivity** → 56.7% specificity

**Leakage check:** an image-level (leaky) random split reported 80.7% accuracy / 0.884 AUC, only marginally higher than the honest patient-level result (79.7% / 0.876) — reassuring, but a reminder that naive splits should always be checked against grouped splits when a dataset contains repeated subjects.

## Key Engineering Decisions

- **Patient-level cross-validation from the start.** Using `StratifiedGroupKFold` on patient ID prevents the same infant's images from leaking between train and test — a common and easy-to-miss source of inflated accuracy in medical imaging projects.
- **No color augmentation on medical color signal.** Since jaundice *is* a color signal, only geometric augmentations (flips, small rotations) were applied to the CNN — color jitter would have destroyed the target signal.
- **Color-card calibration + raw fallback.** Rather than trusting a single lighting-correction step, the pipeline extracts features from both the corrected and raw images, letting downstream models learn which representation is more informative.
- **Diagnostics as a first-class step.** A dedicated diagnostic stage validates that the color-card detector isn't quietly measuring skin, that errors cluster at the ambiguous clinical threshold rather than being random, and that reported metrics survive a leakage stress-test — treating model evaluation with the same rigor as model building.

## Tech Stack

`Python` · `TensorFlow / Keras` (EfficientNetB0 transfer learning) · `LightGBM` · `scikit-learn` (StratifiedGroupKFold, Logistic Regression, Ridge) · `OpenCV` (color-space conversion, ROI/card detection) · `pandas` / `NumPy` · Google Colab (TPU / GPU auto-detection)

## Repository Structure

```
.
├── notebooks/
│   └── neojaundice_pipeline.ipynb   # end-to-end pipeline (stages 1–4)
├── data/                            # images/ + labels CSV (not tracked in git)
├── results/                         # cached ROI arrays, metrics, figures
├── models/                          # saved model checkpoints
└── README.md
```

## Getting Started

1. Open `notebooks/neojaundice_pipeline.ipynb` in Google Colab.
2. Mount Google Drive and point `NEOJAUNDICE_ROOT` at your copy of the dataset (images + `chd_jaundice_published_2.csv`).
3. Run cells top to bottom:
   - Stage 1 — environment setup & accelerator detection
   - Stage 2 — tabular baselines + CNN training
   - Stage 3 — rich color features + ensemble
   - Stage 4 — diagnostics & honest evaluation
4. Dependencies (installed automatically in the notebook):
   ```bash
   pip install tensorflow numpy pandas scikit-learn opencv-python-headless tqdm lightgbm
   ```

## Limitations & Disclaimer

- Trained and evaluated on a single public dataset; performance may not generalize to different phone cameras, lighting conditions, or skin tones not well represented in the data.
- Errors concentrate near the clinical decision threshold, reflecting genuine label ambiguity in borderline cases — this is a screening aid, not a diagnostic replacement.
- This project has **not** undergone clinical validation, regulatory review, or IRB-approved prospective testing. It should be treated strictly as a research/portfolio project.

## Future Work

- Expand to multi-site data to test generalization across devices and skin tones.
- Explore lightweight on-device models for real-time mobile screening.
- Calibrate predicted probabilities and report threshold-free clinical utility (e.g., decision curve analysis).

## License

This project is released under the [MIT License](LICENSE).

---

*If you use or build on this work, a citation or link back is appreciated.*

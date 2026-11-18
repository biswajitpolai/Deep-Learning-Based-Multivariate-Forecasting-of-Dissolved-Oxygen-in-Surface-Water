# Dissolved Oxygen Forecasting with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-research-orange.svg)]()

A comparative study of five deep learning architectures for **multivariate, multi-step forecasting of dissolved oxygen (DO)** in surface water. Models are evaluated under identical data, windowing, and training conditions on water-quality records from the Xiatang monitoring station (Huangshui River).

Developed as part of a Summer Research Internship (SIRE) at IISER Berhampur, under the supervision of Dr. Pragnaditya Malakar.

---

## Table of Contents

- [Problem](#problem)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Getting Started](#getting-started)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Problem

Given the previous **24 observations** (4 days at a 4-hour interval) of **9 water-quality variables**, predict the next **6 DO values** (24 hours) in a single forward pass.

| | |
|---|---|
| **Input** | 24 × 9 window of standardised water-quality variables |
| **Output** | 6 future DO values (mg/L) |
| **Task** | Direct multi-step, multivariate-input, univariate-output forecasting |

---

## Repository Structure

```
.
├── Xiatang.csv                              # Raw station records (6,946 rows)
├── XIATANG_analysis.ipynb                   # Data diagnostics and exploratory analysis
├── lstm_xia (1).ipynb                       # LSTM
├── informer__xia.ipynb                      # Informer
├── autoformer_xia.ipynb                     # Autoformer
├── PatchTST_xia (1).ipynb                   # PatchTST
├── TimeMixer_same_setup (1).ipynb           # TimeMixer
└── README.md
```

Each model notebook is self-contained: it loads the data, applies preprocessing, trains the model, and reports test metrics.

---

## Data

**Source:** Xiatang monitoring station, Water Quality of Huangshui River dataset (IEEE DataPort).

**Period:** 17 Dec 2020 – 5 Dec 2025, 6,946 raw records.

| Column | Description |
|---|---|
| `TEMP` | Water temperature |
| `pH` | pH |
| `EC` | Electrical conductivity |
| `EUR` | See report (strongly skewed; max 9999) |
| `COMMn` | See report |
| `NH3-N` | Ammonia nitrogen |
| `TP` | Total phosphorus |
| `TN` | Total nitrogen |
| `DO` | Dissolved oxygen (**target**) |

### Preprocessing

All models share the following pipeline:

1. Sort by timestamp and remove duplicates.
2. Resample to a regular **4-hour grid** (10,884 slots).
3. Fill gaps of up to 12 consecutive slots by linear interpolation, then by forward/backward fill.
4. Split chronologically **70 / 15 / 15** (train / validation / test), without shuffling.
5. Standardise each variable using **training-set statistics only**.
6. Create sliding windows (input 24, output 6) within each split.

| Split | Rows | Windows |
|---|---:|---:|
| Train | 7,618 | 7,589 |
| Validation | 1,633 | 1,604 |
| Test | 1,633 | 1,604 |

> **Note:** `Autoformer` uses a different gap-filling strategy (4-hour averaging followed by forward fill). The `TimeMixer` notebook is the same-setup version and uses the common pipeline above.

---

## Models

| Model | Notebook | Trainable parameters |
|---|---|---:|
| LSTM | `lstm_xia (1).ipynb` | 204,038 |
| Informer | `informer__xia.ipynb` | 521,857 |
| TimeMixer | `TimeMixer_same_setup (1).ipynb` | 2,109,893 |
| PatchTST | `PatchTST_xia (1).ipynb` | 9,515,526 |
| Autoformer | `autoformer_xia.ipynb` | 10,538,497 |

### Training configuration

| Setting | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Loss | MSE on scaled DO |
| Max epochs | 100 |
| Early stopping | patience 10 on validation loss |
| Checkpoint | lowest validation loss |
| Random seed | 42 (where set in the notebook) |

Each model was trained **once**. Differences of roughly 0.01 mg/L should not be treated as significant.

---

## Results

Test set (1,604 windows), 6-step horizon. Metrics are in original DO units; MAPE is in percent.

### Overall

| Model | RMSE (mg/L) | MAE (mg/L) | MAPE (%) | R² |
|---|---:|---:|---:|---:|
| **TimeMixer** | **0.4193** | **0.2623** | **3.45** | **0.9045** |
| PatchTST | 0.5542 | 0.3972 | 5.44 | 0.8336 |
| Autoformer | 0.5646 | **0.3601** | 4.76 | 0.8330 |
| LSTM | 0.6103 | 0.3872 | 5.22 | 0.7982 |
| Informer | 0.6124 | 0.4753 | 6.11 | 0.7963 |

### RMSE by forecast step (mg/L)

| Model | t+1 | t+2 | t+3 | t+4 | t+5 | t+6 |
|---|---:|---:|---:|---:|---:|---:|
| **TimeMixer** | **0.3140** | **0.3849** | **0.4222** | **0.4383** | **0.4548** | **0.4806** |
| PatchTST | 0.4054 | 0.4660 | 0.5268 | 0.5656 | 0.6268 | 0.6862 |
| Autoformer | 0.4710 | 0.5346 | 0.5532 | 0.5856 | 0.6168 | 0.6128 |
| LSTM | 0.4569 | 0.5341 | 0.5871 | 0.6326 | 0.6829 | 0.7277 |
| Informer | 0.5859 | 0.5852 | 0.6073 | 0.6237 | 0.6170 | 0.6525 |

Full per-step results for MAE, MAPE, and R² are in the report.

---

## Getting Started

The notebooks are designed for **Google Colab** with a GPU runtime.

1. Open any model notebook in Colab.
2. When prompted, upload `Xiatang.csv` (or place it in the working directory).
3. Run all cells. The final cells print test metrics and per-step results.

**Dependencies:** PyTorch, NumPy, pandas, scikit-learn, matplotlib. All are preinstalled on Colab.

To run locally:

```bash
pip install torch numpy pandas scikit-learn matplotlib jupyter
jupyter notebook
```

---

## Reproducibility

- All models use the same data, split, windows, and metric definitions.
- Random seed is fixed to 42 where the notebook sets one.
- Results come from a **single run per model**. Run-to-run variance has not been measured.

To assess stability, rerun each notebook with several seeds and report the mean and standard deviation.

---

## Limitations

- **Single station, single split.** Conclusions may not generalise to other sites or periods.
- **Uneven sampling history.** Only 18% and 17% of 4-hour slots in 2020 and 2021 hold real measurements; the rest are interpolated.
- **Distribution shift.** DO in 2025 is about 2 mg/L lower than in earlier years at comparable temperatures.
- **Unhandled outliers.** The `EUR` column contains extreme values (up to 9999) that were not treated.
- **Single run.** Model rankings, especially between PatchTST and Autoformer, should be confirmed with repeated runs.

---

## Citation

If you use this work, please cite the accompanying SIRE report:

> B. Polai, *Deep Learning Based Multivariate Forecasting of Dissolved Oxygen in Surface Water*, SIRE Report, IISER Berhampur, 2026.

---

## Acknowledgements

Supervised by **Dr. Pragnaditya Malakar**, Department of Earth and Environmental Sciences, IISER Berhampur.

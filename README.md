# Deep Learning-Based Multivariate Forecasting of Dissolved Oxygen in Surface Water

This repository compares five deep learning models for forecasting dissolved oxygen (DO) in surface water. The study uses water quality records from the Xiatang monitoring station, part of the Huangshui River dataset. It was carried out as a Summer Research Internship (SIRE) at IISER Berhampur.

## Task

- **Type:** multivariate, single-target, direct multi-step time series forecasting
- **Input:** the previous 24 observations (4 days, one per 4-hour step) of 9 variables
- **Target:** dissolved oxygen (DO, mg/L)
- **Output:** the next 6 DO values (24 hours)

### Input variables

| Column | Description |
|---|---|
| TEMP | Water temperature |
| pH | pH |
| EC | Electrical conductivity |
| EUR | See note in the report (strongly skewed, max 9999) |
| COMMn | See note in the report |
| NH3-N | Ammonia nitrogen |
| TP | Total phosphorus |
| TN | Total nitrogen |
| DO | Dissolved oxygen (target, placed last) |

## Data

`Xiatang.csv` holds 6,946 raw records from 17 December 2020 (20:00) to 5 December 2025 (16:00), with a `time` column and the nine variables above.

Preprocessing, as described in the report:

1. Sort by time and drop duplicate timestamps.
2. Place the records on a regular 4-hour grid from the first to the last timestamp (10,884 slots).
3. Fill gaps of up to 12 consecutive slots by linear interpolation, then fill any remaining gaps by forward and backward fill.
4. Split chronologically: 70% training (7,618 rows), 15% validation (1,633 rows), 15% test (1,633 rows). No shuffling.
5. Standardise each variable with the mean and standard deviation of the training segment only.
6. Build sliding windows of 24 input steps and 6 output steps inside each segment. This gives 7,589 training, 1,604 validation, and 1,604 test windows.

The Autoformer notebook differs in step 2 and 3: it averages records within each 4-hour interval and fills gaps by forward fill.

## Models

| Model | Notebook | Parameters |
|---|---|---|
| LSTM | `lstm_xia (1).ipynb` | 204,038 |
| Informer | `informer__xia.ipynb` | 521,857 |
| Autoformer | `autoformer_xia.ipynb` | 10,538,497 |
| PatchTST | `PatchTST_xia (1).ipynb` | 9,515,526 |
| TimeMixer | `TimeMixer_same_setup (1).ipynb` | 2,109,893 |

Each model is trained with Adam (learning rate 0.001, batch size 32), an MSE loss on scaled DO, and up to 100 epochs. Early stopping uses a patience of 10 epochs on validation loss, and the checkpoint with the lowest validation loss is used for testing. Each model was trained once with seed 42 where the notebook sets one, so the results come from a single run.

The TimeMixer notebook is the same-setup version: it uses the same 4-hour grid, split, and windows as the other models.

## Other files

- `XIATANG_analysis.ipynb`: data diagnostics for the station (share of measured slots per year, monthly DO, DO by water temperature).

## Results

Test set, 6-step (24-hour) forecast, 1,604 windows. Values are in original DO units (mg/L; MAPE in %).

| Model | RMSE | MAE | MAPE (%) | R² |
|---|---:|---:|---:|---:|
| TimeMixer | **0.4193** | **0.2623** | **3.45** | **0.9045** |
| PatchTST | 0.5542 | 0.3972 | 5.44 | 0.8336 |
| Autoformer | 0.5646 | **0.3601** | 4.76 | 0.8330 |
| LSTM | 0.6103 | 0.3872 | 5.22 | 0.7982 |
| Informer | 0.6124 | 0.4753 | 6.11 | 0.7963 |

Per-step RMSE (mg/L) for each forecast step t+1 to t+6:

| Model | t+1 | t+2 | t+3 | t+4 | t+5 | t+6 |
|---|---:|---:|---:|---:|---:|---:|
| TimeMixer | 0.3140 | 0.3849 | 0.4222 | 0.4383 | 0.4548 | 0.4806 |
| PatchTST | 0.4054 | 0.4660 | 0.5268 | 0.5656 | 0.6268 | 0.6862 |
| Autoformer | 0.4710 | 0.5346 | 0.5532 | 0.5856 | 0.6168 | 0.6128 |
| LSTM | 0.4569 | 0.5341 | 0.5871 | 0.6326 | 0.6829 | 0.7277 |
| Informer | 0.5859 | 0.5852 | 0.6073 | 0.6237 | 0.6170 | 0.6525 |

Notes:
- All five models are compared on the same grid and test windows, so the overall table is a fair comparison.
- The differences between PatchTST and Autoformer (about 0.01 mg/L) are within run-to-run variation for a single training run.
- TimeMixer's lead should be confirmed over several random seeds before drawing firm conclusions.

## How to run

The notebooks are written for Google Colab with a GPU.

1. Open a notebook in Colab.
2. Upload `Xiatang.csv` when prompted (the first data cell asks for it), or place it in the working directory.
3. Run all cells. Each notebook prints the training log and the test metrics.

Required packages: PyTorch, NumPy, pandas, scikit-learn, and matplotlib. All of these are preinstalled in Colab.

## Requirements

- Python 3.11 (tested in Google Colab)
- PyTorch 2.x (CUDA optional; training was run on a GPU)

## Acknowledgements

This work was carried out as part of the Summer Research Internship at the Indian Institute of Science Education and Research (IISER) Berhampur, under the supervision of Dr. Pragnaditya Malakar.

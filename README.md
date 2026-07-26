# Deep Learning-Based Multivariate Forecasting of Dissolved Oxygen in Surface Water

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)]()

## Overview

This repository presents a comprehensive comparative study of deep learning models for **multivariate single-target time series forecasting** of **Dissolved Oxygen (DO)** in surface water.

The objective is to accurately predict future dissolved oxygen concentrations using historical multivariate water quality observations. Dissolved Oxygen is one of the most important indicators of aquatic ecosystem health and water quality.

Seven deep learning models are implemented and evaluated under identical experimental settings to provide a fair comparison.

---

## Research Objective

To develop an accurate deep learning framework for forecasting Dissolved Oxygen (DO) using multivariate water quality data and compare the performance of state-of-the-art forecasting architectures.

---

## Forecasting Task

**Forecasting Type**

> Multivariate Single-Target Time Series Forecasting

**Input**

Historical water quality observations

**Target**

Dissolved Oxygen (DO)

---

## Input Features

The model uses the following nine water quality variables:

- Water Temperature
- pH
- Conductivity
- Biochemical Oxygen Demand (BOD)
- Nitrate
- Fecal Coliform
- Total Coliform
- Turbidity
- Dissolved Oxygen (DO)

Total Input Features = **9**

---

## Forecasting Configuration

| Parameter | Value |
|------------|-------|
| Input Sequence Length | 24 |
| Forecast Horizon | 6 |
| Input Features | 9 |
| Target | Dissolved Oxygen |
| Batch Size | 32 |
| Epochs | 100 |
| Early Stopping | 10 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Mean Squared Error (MSE) |

---

# Deep Learning Models

The repository includes implementations of:

- Long Short-Term Memory (LSTM)
- Informer
- Autoformer
- PatchTST
- TimeMixer

---

# Results

## Overall Model Performance

| Model | RMSE | MAE | MAPE (%) | R² |
|--------|-----:|----:|---------:|----:|
| TimeMixer | **0.4941** | **0.3114** | **4.20** | 0.6941* |
| PatchTST | 0.5542 | 0.3972 | 5.44 | **0.8336** |
| Autoformer | 0.5646 | **0.3601** | **4.76** | 0.8330 |
| LSTM | 0.6103 | 0.3872 | 5.22 | 0.7982 |
| Informer | 0.6124 | 0.4753 | 6.11 | 0.7963 |

---

## Per-Step Forecast Results

### Long Short-Term Memory (LSTM)

| Step | RMSE | MAE | MAPE (%) | R² |
|-----:|-----:|----:|---------:|----:|
| 1 | 0.4569 | 0.2777 | 3.73 | 0.8876 |
| 2 | 0.5341 | 0.3392 | 4.56 | 0.8460 |
| 3 | 0.5871 | 0.3729 | 4.99 | 0.8134 |
| 4 | 0.6326 | 0.4087 | 5.50 | 0.7828 |
| 5 | 0.6829 | 0.4467 | 6.04 | 0.7464 |
| 6 | 0.7277 | 0.4782 | 6.48 | 0.7115 |

---

### Informer

| Step | RMSE | MAE | MAPE (%) | R² |
|-----:|-----:|----:|---------:|----:|
| 1 | 0.5859 | 0.4528 | 5.89 | 0.8146 |
| 2 | 0.5852 | 0.4549 | 5.86 | 0.8147 |
| 3 | 0.6073 | 0.4704 | 6.03 | 0.7999 |
| 4 | 0.6237 | 0.4836 | 6.19 | 0.7885 |
| 5 | 0.6170 | 0.4805 | 6.16 | 0.7926 |
| 6 | 0.6525 | 0.5098 | 6.52 | 0.7675 |

---

### Autoformer

| Step | RMSE | MAE | MAPE (%) | R² |
|-----:|-----:|----:|---------:|----:|
| 1 | 0.4710 | 0.3037 | 4.01 | 0.8844 |
| 2 | 0.5346 | 0.3419 | 4.51 | 0.8508 |
| 3 | 0.5532 | 0.3518 | 4.65 | 0.8398 |
| 4 | 0.5856 | 0.3718 | 4.92 | 0.8201 |
| 5 | 0.6168 | 0.3934 | 5.20 | 0.8000 |
| 6 | 0.6128 | 0.3979 | 5.25 | 0.8022 |

---

### PatchTST

| Step | RMSE | MAE | MAPE (%) | R² |
|-----:|-----:|----:|---------:|----:|
| 1 | 0.4054 | 0.2825 | 3.84 | 0.9115 |
| 2 | 0.4660 | 0.3304 | 4.47 | 0.8827 |
| 3 | 0.5268 | 0.3776 | 5.17 | 0.8498 |
| 4 | 0.5656 | 0.4069 | 5.57 | 0.8264 |
| 5 | 0.6268 | 0.4671 | 6.44 | 0.7864 |
| 6 | 0.6862 | 0.5188 | 7.16 | 0.7434 |

---

### TimeMixer

| Step | RMSE | MAE | MAPE (%) | R² |
|-----:|-----:|----:|---------:|----:|
| 1 | 0.4021 | 0.2466 | 3.32 | 0.7964 |
| 2 | 0.4612 | 0.2939 | 3.94 | 0.7327 |
| 3 | 0.4984 | 0.3171 | 4.27 | 0.6884 |
| 4 | 0.5122 | 0.3247 | 4.39 | 0.6715 |
| 5 | 0.5365 | 0.3417 | 4.62 | 0.6403 |
| 6 | 0.5399 | 0.3444 | 4.66 | 0.6363 |
---

## Key Findings

- TimeMixer achieved the lowest RMSE, MAE, and MAPE among the evaluated models.
- LSTM served as a strong baseline for multivariate dissolved oxygen forecasting.



# Acknowledgements

This work was carried out as part of the Summer Research Internship at **Indian Institute of Science Education and Research (IISER) Berhampur**.


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

Current overall model performance:

| Model | RMSE | MAE | MAPE (%) | R² |
|-------|------:|------:|---------:|------:|
| TimeMixer | **0.4941** | **0.3114** | **4.20** | 0.6941* |
| PatchTST | 0.5542 | 0.3972 | 5.44 | **0.8336** |
| Autoformer | 0.5646 | **0.3601** | **4.76** | 0.8330 |
| LSTM | 0.6103 | 0.3872 | 5.22 | 0.7982 |
| Informer | 0.6124 | 0.4753 | 6.11 | 0.7963 |


---




# Acknowledgements

This work was carried out as part of the Summer Research Internship at **Indian Institute of Science Education and Research (IISER) Berhampur**.


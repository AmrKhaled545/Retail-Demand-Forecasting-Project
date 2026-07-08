# Project Description — Retail Demand Forecasting

**Digital Egypt Pioneers Initiative (DEPI)** · Team 5 · GIZ4_AIS4_S4
**Instructor:** Eng. Ahmed Ayman

**Team Members:**
- Amr Khaled Mohamed
- Ahmed Mostafa Elembaby
- Omar Abdelkhaleq Ahmed
- Youssef Gomaa Hassan Mohamed

---

## Summary

Retail Demand Forecasting is an end-to-end machine learning project that
predicts daily unit sales for Walmart retail products over a **28-day
horizon**, using the M5 Forecasting (Walmart) dataset. The project covers
the full lifecycle of a demand-forecasting solution — data engineering,
exploratory analysis, model development and comparison, and deployment
through an interactive web application — with a specific engineering focus
on running reliably within limited compute and memory environments (e.g. a
free-tier Google Colab session).

## Problem Statement

Retailers need accurate, per-product, per-store demand forecasts to make
informed decisions about inventory levels, replenishment timing, and
promotional planning. Doing this at scale is a data engineering challenge as
much as a modeling one: a single retail dataset can span thousands of
products across dozens of stores and years of daily history, which quickly
exceeds the memory available on standard or free-tier compute.

## Objectives

- Build a data pipeline that cleans and transforms raw sales, pricing, and
  calendar data into a model-ready feature set **without requiring the full
  dataset to be held in memory at once**.
- Ensure the selected product sample is **representative across product
  categories**, rather than dominated by the highest-volume category.
- Engineer time-series features (lags, rolling statistics, calendar effects)
  known to be predictive of retail demand.
- Train and fairly compare multiple forecasting models (XGBoost, LightGBM,
  and optionally an LSTM) using consistent evaluation metrics.
- Produce a genuine 28-day-ahead forecast, not just a same-day prediction.
- Deliver the results through an interactive application usable by a
  non-technical stakeholder.

## Dataset

The project uses the **M5 Forecasting Accuracy** dataset (Walmart), which
includes:
- `sales_train_validation.csv` — daily unit sales history per product per
  store.
- `calendar.csv` — dates, weekdays, special events, and SNAP (food
  assistance) indicators per state.
- `sell_prices.csv` — weekly selling price per product per store.

The data spans multiple U.S. states, stores, and product categories
(`FOODS`, `HOUSEHOLD`, `HOBBIES`).

## Key Capabilities

- **Configurable, memory-bounded pipeline** — the number of products used
  (`TOP_N`) and the processing batch size (`CHUNK_SIZE`) are independent
  parameters; peak memory usage depends only on `CHUNK_SIZE`.
- **Category-balanced product selection** — products are chosen
  proportionally across categories so no category is under-represented in
  training.
- **Rich exploratory analysis** — sales trends, category and store
  performance, event impact, and price–demand relationships, all computed
  from incrementally accumulated aggregates.
- **Model comparison, not a single fixed choice** — XGBoost and LightGBM are
  trained and scored on identical features and metrics (RMSE, MAE, MAPE,
  R²); an optional LSTM baseline is included for completeness.
- **True multi-day forecasting** — a recursive walk-forward process
  generates a genuine 28-day-ahead forecast per product–store combination.
- **Interactive deployment** — a Streamlit application lets a user explore
  forecasts by state/store/product or upload a bulk list of products and
  receive a forecast summary table, all without needing to touch the
  notebook.

## Target Users / Use Cases

- **Inventory and demand planners** who need a quick, product-level 28-day
  outlook to inform replenishment decisions.
- **Data science students and practitioners** looking for a reference
  implementation of a memory-efficient, scalable forecasting pipeline built
  on a well-known public dataset.

## Technology Stack

Python · Pandas · NumPy · XGBoost · LightGBM · scikit-learn ·
Matplotlib/Seaborn/Plotly (analysis) · Streamlit + Plotly (deployment) ·
TensorFlow/Keras (optional LSTM comparison)

## Project Context

This project was developed as a graduation project, using the M5
Forecasting (Walmart) competition dataset as its foundation, and is designed
to run end-to-end on Google Colab's free tier.

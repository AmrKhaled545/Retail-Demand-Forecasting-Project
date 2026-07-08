# Retail Demand Forecasting 🛒📊

🔗 **Live App:** [retail-demand-forecasting-project-depi2026.streamlit.app](https://retail-demand-forecasting-project-depi2026.streamlit.app/)

## Project Overview

This project predicts daily sales of Walmart retail products over a **28-day forecasting horizon**, using the [M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset — historical sales, item prices, and calendar/event data across multiple stores and U.S. states. The goal is to turn that raw data into a memory-efficient, end-to-end pipeline that trains a demand forecasting model and serves its predictions through an interactive app, to support inventory and demand planning decisions.

## 🚀 Current Progress

The pipeline is complete end-to-end: data engineering, EDA, feature engineering, model training and comparison, 28-day recursive forecasting, and an interactive deployment app.

### 1. Scalable Data Engineering

- **Chunked processing:** Products are split into batches (`CHUNK_SIZE`) and melted, merged, cleaned, and feature-engineered one batch at a time — so memory use stays bounded no matter how many products (`TOP_N`) are included. Intermediate frames are deleted and garbage-collected after every batch.
- **Category-balanced product selection:** Rather than picking the overall top-selling products (which would be dominated by the largest category), `TOP_N` is allocated proportionally across categories based on each category's share of unique items — guaranteeing that `FOODS`, `HOUSEHOLD`, and `HOBBIES` are all fairly represented in training.
- **Memory optimization:** Dynamic downcasting (`reduce_mem_usage`) shrinks numeric and categorical columns to the smallest safe dtype at every stage, cutting memory footprint substantially versus the raw data.
- **Data sanitization:** Missing calendar events are labeled explicitly, missing prices are forward/backward-filled within each store–item series, exact duplicates are dropped, and extreme sales outliers are capped at the 99th percentile.

### 2. Feature Engineering

- Lag features at multiple horizons (`lag_7`, `lag_14`, `lag_28`) and rolling averages (`rolling_mean_7`, `rolling_mean_28`) to capture recent demand trends.
- Calendar features: day of week, month, weekend flag, and SNAP (food-assistance) indicators per state.
- All engineered features are computed per-chunk and downcast to `float32`/`int16` before being kept, so the final modeling table stays compact.

### 3. Exploratory Data Analysis

- Category-level and state-level sales/revenue breakdowns.
- Weekday and event-type sales patterns (e.g. the impact of holidays on demand).
- A category × weekday heatmap of average sales.
- All EDA aggregates are accumulated incrementally per chunk (never by holding the full merged dataset in memory at once).

### 4. Model Training & Comparison

- **XGBoost** and **LightGBM** regressors are trained on the same feature set and compared on held-out data (the most recent 28 days) using RMSE, MAE, MAPE, and R².
- An optional **LSTM** deep-learning comparison is included behind a `RUN_DEEP_LEARNING` flag, for use on a GPU runtime — skipped by default since the tree-based models are the primary, CPU-friendly baseline.
- The best model (by RMSE) is selected automatically for deployment.

### 5. Recursive 28-Day Forecasting

- A recursive walk-forward routine generates predictions one day at a time for every product–store series, feeding each day's prediction back in as history for the next day's lag/rolling features — producing a genuine 28-day-ahead forecast rather than a single-step approximation.

### 6. Interactive Deployment (Streamlit)

A companion `streamlit_app.py` reads the notebook's exported artifacts and provides:

- **Explore & Predict tab:** cascading filters (state → store → product), an adjustable forecast horizon (1–28 days), a "Predict" button, KPI cards (recent average vs. forecast average, % change, total forecasted units), an interactive actual-vs-forecast chart, and a day-by-day forecast table.
- **Bulk Upload Predict tab:** upload a CSV/Excel file with a list of product IDs and get a summary table of each product's forecasted demand (with automatic column detection and fuzzy-matching suggestions for unrecognized IDs), plus a downloadable CSV of the results.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Modeling:** XGBoost, LightGBM, scikit-learn (TensorFlow/Keras optional, for the LSTM comparison)
- **Deployment:** Streamlit

## 📁 Repository Structure

```
├── Retail_Demand_Forecasting.ipynb        # Main notebook: data pipeline, EDA, modeling, forecasting
├── app.py                                 # Interactive deployment app
├── requirements.txt                       # Python dependencies
├── PROJECT_DESCRIPTION.txt                # Project Description
├── PROJECT_DOCUMENTATION.txt              # Python Documentaion
└── README.md
```

**Generated artifacts** (created after running the notebook — not tracked in the repo, see `.gitignore`):

| File pattern                             | Contents                                                         |
| ---------------------------------------- | ---------------------------------------------------------------- |
| `m5_sales_features_<N>products.parquet`  | Cleaned, feature-engineered modeling table                       |
| `demand_forecast_28days_<N>products.csv` | 28-day forecast per product–store series                         |
| `sales_history_90days_<N>products.csv`   | Last 90 days of actuals per product–store series                 |
| `demand_forecast_model_<model>.pkl`      | The best-performing trained model                                |
| `run_metadata_<N>products.json`          | Run configuration (`TOP_N`, `CHUNK_SIZE`, chosen model, metrics) |

## ⚙️ Setup & Usage

1. **Get the data:** download the M5 Forecasting dataset from Kaggle (`calendar.csv`, `sell_prices.csv`, `sales_train_validation.csv`) and place the files in a folder of your choice.
2. **Configure the path:** in the notebook, set `DATA_DIR` to point to that folder (the default assumes Google Drive on Colab).
3. **Run the notebook** top to bottom. Adjust `TOP_N` (how many products to include) and `CHUNK_SIZE` (how many products are processed per batch) as needed — lower `CHUNK_SIZE` if you hit memory limits; `TOP_N` can be raised freely since it doesn't affect peak memory.
4. **Launch the app:**
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```
   Make sure `streamlit_app.py` sits in the same folder as the generated `demand_forecast_28days_*products.csv` and `sales_history_90days_*products.csv` files — the app discovers them automatically.

   Or just try the already-deployed version: **[retail-demand-forecasting-project-depi2026.streamlit.app](https://retail-demand-forecasting-project-depi2026.streamlit.app/)**

## ⏭️ Next Steps (Upcoming Commits)

- Richer lag features (`lag_1`–`lag_3`), rolling standard deviation, and price-momentum features.
- Early stopping for XGBoost/LightGBM using a dedicated validation split, for faster and more robust training.
- A Tweedie-loss LightGBM variant, better suited to intermittent, count-based demand data.
- A vectorized recursive forecasting routine (batched per-day predictions instead of per-series predictions) for a significant speed-up with no added memory cost.
- Prediction intervals (quantile regression) around the point forecast in the Streamlit app.

## 📊 Model Performance

| Model    | RMSE     | MAE      | MAPE      | R²       |
| -------- | -------- | -------- | --------- | -------- |
| XGBoost  | 2.669628 | 1.604892 | 76.643463 | 0.721522 |
| LightGBM | 2.671962 | 1.608828 | 76.948829 | 0.721035 |

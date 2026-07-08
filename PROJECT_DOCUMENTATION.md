# Project Documentation — Retail Demand Forecasting

**Digital Egypt Pioneers Initiative (DEPI)** · Team 5 · GIZ4_AIS4_S4
**Instructor:** Eng. Ahmed Ayman

**Team Members:**
- Amr Khaled Mohamed
- Ahmed Mostafa Elembaby
- Omar Abdelkhaleq Ahmed
- Youssef Gomaa Hassan Mohamed

---

## 1. Overview

This document describes the technical implementation of the Retail Demand
Forecasting pipeline: a notebook that transforms raw M5 (Walmart) sales
data into a 28-day demand forecast, and a companion Streamlit app that
serves that forecast interactively. It is intended as a reference for
understanding, reproducing, or extending the project.

The pipeline is designed around one constraint in particular: it must run
on a **free-tier Google Colab runtime** (roughly 12 GB RAM), regardless of
how many products (`TOP_N`) are included. This shapes almost every design
decision described below.

---

## 2. Repository Structure

```
├── Retail_Demand_Forecasting.ipynb   # Main notebook: pipeline, EDA, modeling, forecasting
├── app.py                            # Streamlit deployment app
├── requirements.txt                  # Python dependencies
├── README.md
├── PROJECT_DESCRIPTION.md
└── PROJECT_DOCUMENTATION.md          # This file
```

**Generated artifacts** (produced by running the notebook; not committed to
the repository — see Section 10):

| File pattern | Contents |
|---|---|
| `m5_sales_features_<N>products.parquet` | Cleaned, feature-engineered modeling table |
| `demand_forecast_28days_<N>products.csv` | 28-day forecast per product–store series |
| `sales_history_90days_<N>products.csv` | Last 90 days of actuals per product–store series |
| `demand_forecast_model_<model>.pkl` | The best-performing trained model |

---

## 3. Data Sources

The project uses the **M5 Forecasting Accuracy** (Walmart) dataset:

| File | Description |
|---|---|
| `sales_train_validation.csv` | Wide-format daily unit sales: one row per product–store combination, one column per day (`d_1`...`d_1913`). |
| `calendar.csv` | One row per date, with weekday, month, event names/types (up to two per day), and SNAP indicators for California, Texas, and Wisconsin. |
| `sell_prices.csv` | Weekly selling price per `(store_id, item_id, wm_yr_wk)`. |

Data is not included in the repository; it must be downloaded separately
(see Section 12) and referenced via the `DATA_DIR` variable in the
notebook.

---

## 4. Pipeline Architecture

### 4.1 Design parameters

Two independent parameters control the pipeline's scope and memory profile:

- **`TOP_N`** — total number of distinct products to include.
- **`CHUNK_SIZE`** — number of products processed together in a single
  batch.

Peak memory usage is a function of `CHUNK_SIZE` only. `TOP_N` can be
increased freely (e.g. from 15 to 1,500+) without changing the pipeline's
memory profile — only its total runtime.

### 4.2 Product selection

Rather than ranking all products by total units sold (which biases
selection toward the category with the most SKUs — `FOODS` in this
dataset), `TOP_N` is allocated **proportionally to each category's share of
unique items**. The best-selling products are then drawn from *within* each
category, so the final selection is category-balanced by construction. The
resulting split is printed and visualized as a bar chart comparing selected
vs. available products per category.

### 4.3 Chunked processing

`TARGET_ITEMS` is split into batches of `CHUNK_SIZE` products
(`item_chunks`). Each batch independently goes through:

1. **Melt** — reshape from wide format (`d_1 ... d_1913`) to long format:
   one row per item–store–day.
2. **Merge** — join with `calendar` on `d`, and with `sell_prices` (filtered
   to the same chunk) on the composite key
   `['store_id', 'item_id', 'wm_yr_wk']`. Both merges are one-to-one, with
   no row duplication.
3. **Clean** — see Section 5.
4. **Accumulate EDA aggregates** — sums/counts only, appended to running
   lists (see Section 7).
5. **Feature-engineer** — see Section 6.
6. **Retain only the compact, feature-ready result** for that chunk;
   intermediate frames (`chunk_wide`, `chunk_long`, `chunk_prices`) are
   explicitly deleted and garbage-collected before the next chunk begins.

After all chunks are processed, the small per-chunk results are
concatenated into the final modeling table (`model_df`), and the
accumulated EDA aggregates are combined into their final form.

### 4.4 Memory utilities

- `reduce_mem_usage(df)` — downcasts numeric columns to the smallest dtype
  that safely fits the data's range, applied after every merge.
- `print_memory_usage(label)` — best-effort process RAM reporting (via
  `psutil`, if installed) printed after each chunk, used to monitor the
  pipeline's actual memory footprint during development.

### 4.5 Practical guidance

On a free-tier Colab runtime, `CHUNK_SIZE` values of 150–250 are safe even
when `TOP_N` exceeds 1,000. If memory pressure occurs, lower `CHUNK_SIZE`
rather than `TOP_N`.

---

## 5. Data Cleaning & Preprocessing

Performed per-chunk (Section 4.3, step 3), via dedicated functions:

- **`fill_missing_events(df)`** — labels missing `event_name_*` /
  `event_type_*` values explicitly as `'No_Event'` rather than leaving them
  null, preserving them as a valid category rather than missing data.
- **`impute_prices(df)`** — forward/backward-fills missing `sell_price`
  values within each `(store_id, item_id)` series, since short gaps
  typically reflect reporting gaps rather than a genuine absence of a
  price.
- **Duplicate removal** — exact duplicate rows are dropped
  (`drop_duplicates()`), and the number removed is tracked across all
  chunks (`eda_duplicates_total`).
- **`handle_outliers_and_scale(df)`** — caps sales at the 99th percentile
  to limit the influence of extreme spikes (e.g. bulk purchases, data
  entry errors) on model training, and derives a `sales_log` column
  (`log1p` transform) as an alternative target representation.
- **Missing-value tracking** — the count of nulls per column is
  accumulated across chunks (`eda_missing_counts`) *before* cleaning fills
  them, so the notebook can report exactly how much missing data existed
  in the raw merged data.

---

## 6. Feature Engineering

Implemented in `add_time_series_features(df, group_col='id', target='sales')`,
applied per chunk after cleaning:

| Feature | Description |
|---|---|
| `lag_7`, `lag_14`, `lag_28` | Sales value 7/14/28 days prior, grouped by `id` (the item–store pair). |
| `rolling_mean_7`, `rolling_mean_28` | Rolling average sales over the trailing 7/28 days (computed on the series shifted by 1 day, so the current day's value isn't included in its own average). |
| `day_of_week`, `month`, `is_weekend` | Calendar-derived features from the transaction date. |

Rows without a full 28-day lag history (the first 28 rows of each series)
are dropped, since `lag_28` cannot be computed for them.

**Grouping key.** Features are computed per `id` — the item *and* store
combination — rather than per `store_id` alone, since a single store sells
many different products and each product–store pairing is its own
independent time series.

---

## 7. Exploratory Data Analysis

All EDA statistics are computed from **incrementally accumulated
aggregates** (sums and counts appended per chunk, then combined once at the
end) rather than from a fully materialized table — so this section's
memory footprint does not grow with `TOP_N`.

| Chart | Purpose |
|---|---|
| Selected vs. Available Products by Category | Confirms the category-balanced selection described in Section 4.2. |
| Distribution of Daily Unit Sales (log scale) | Shows the raw sales distribution *before* outlier capping, motivating the 99th-percentile cap in Section 5. |
| Total Sales Volume by Category | Compares overall unit-sales volume across `FOODS`, `HOUSEHOLD`, `HOBBIES`. |
| Average Daily Sales by Day of the Week | Reveals weekly seasonality in demand. |
| Interactive Daily Sales Trend | A Plotly line chart with a range slider, showing the overall sales trend across the full date range. |
| Average Daily Sales by Event Type | Quantifies how calendar events (sporting, cultural, national, religious) affect average demand relative to non-event days. |
| Total Revenue by State | Geographic revenue comparison (CA, TX, WI). |
| Total Revenue by Store | Finer-grained, store-level revenue comparison within the states above. |
| Average Daily Units Sold by Price Range | A simple price-elasticity view — whether lower-priced items move more units per day on average. |
| Average Sales by Category and Weekday (heatmap) | Combines category and weekday effects into a single view. |

---

## 8. Modeling

### 8.1 Data splitting

A **chronological** (not random) split is used: the most recent 28 days
form the test set, and everything before that is training data. A single
global cutoff date applies to all product–store series, since they share
the same calendar.

### 8.2 Feature set

```python
cat_features = ['item_id', 'store_id', 'state_id', 'cat_id', 'dept_id']
num_features = ['lag_7', 'lag_14', 'lag_28', 'rolling_mean_7', 'rolling_mean_28',
                 'sell_price', 'day_of_week', 'month', 'is_weekend', 'wday',
                 'snap_CA', 'snap_TX', 'snap_WI']
```

Categorical columns are cast to pandas `category` dtype, allowing both
XGBoost (`enable_categorical=True`) and LightGBM to consume them natively,
without manual one-hot or label encoding.

### 8.3 Evaluation metrics

Every model is scored with the same four metrics via a shared
`evaluate_model()` helper:

- **RMSE** — penalizes large misses heavily; sensitive to spikes/promotions.
- **MAE** — average error in the same units as sales; easy to interpret.
- **MAPE** — error as a percentage, comparable across products with very
  different sales volumes.
- **R²** — share of variance in daily sales explained by the model.

### 8.4 Models compared

| Model | Configuration | Notes |
|---|---|---|
| **XGBoost** (baseline) | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method='hist'`, `enable_categorical=True` | Strong, fast, memory-light default for tabular retail data. |
| **LightGBM** | Same hyperparameter values as XGBoost, for a like-for-like comparison | Histogram-based, leaf-wise tree growth; typically faster and lighter on memory than XGBoost at comparable accuracy. |
| **LSTM** (optional) | 2-layer LSTM (64 → 32 units) + dense head, `SEQ_LEN=28` | Disabled by default (`RUN_DEEP_LEARNING = False`); requires a GPU runtime. Included for completeness — see the notebook's discussion of why gradient boosting is the stronger default for this dataset shape. |

### 8.5 Model selection

The model with the lowest RMSE on the test set is selected automatically
(`comparison['rmse'].idxmin()`) and used for the recursive forecast and
deployment artifacts.

---

## 9. Recursive Forecasting

The train/test split above only measures accuracy on days actuals already
exist for. To forecast genuinely future days, `recursive_forecast()`:

1. For each product–store series (`id`) with at least 28 days of history,
   takes the most recent static feature values (price, category, etc.).
2. Predicts one day at a time, up to `horizon` (28) days ahead.
3. Feeds each day's prediction back into the series so that the next day's
   `lag_7`/`lag_14`/`lag_28` and rolling means are computed correctly —
   this step-by-step feedback is required because those lag values for
   future days depend on predictions the model itself has just produced.

**Current implementation note.** The function iterates per series and, for
each series, per forecast day — meaning it calls `model.predict()` once per
series per day rather than in a single batched call across all series for
a given day. This is functionally correct but is the primary target for the
performance improvement described in Section 13.

---

## 10. Deployment Artifacts

At the end of the notebook, the following are exported with descriptive,
self-documenting filenames (rather than bare numeric identifiers):

| Artifact | Filename pattern | Produced by |
|---|---|---|
| Cleaned modeling table | `m5_sales_features_<N>products.parquet` | `verify_and_export()` |
| 28-day forecast | `demand_forecast_28days_<N>products.csv` | Final export cell |
| Last 90 days of actuals | `sales_history_90days_<N>products.csv` | Final export cell |
| Best model | `demand_forecast_model_<model>.pkl` | `joblib.dump()` |

These files are intentionally excluded from version control (see
`.gitignore`) since they are regenerated by running the notebook and can be
large depending on `TOP_N`.

---

## 11. Streamlit Application

`app.py` reads the forecast and history CSVs described above (auto-detected
via `glob`, using the most recently modified matching file) and provides
two modes of interaction:

### 11.1 Explore & Predict tab

- Cascading filters: **State → Store → Product**.
- An adjustable forecast horizon (1–28 days) in the sidebar.
- A **Predict** button; results (KPIs, chart, table) persist across
  sidebar changes via `st.session_state` until a new prediction is made.
- KPI cards: average daily sales over the last 28 days, average daily
  forecast over the chosen horizon (with percentage change), and total
  forecasted units.
- An interactive Plotly chart combining recent actuals and the forecast,
  with a marker at the actual/forecast boundary.
- A day-by-day forecast table, downloadable as CSV.

### 11.2 Bulk Upload Predict tab

- Accepts a CSV or Excel file containing a list of product IDs (column
  name auto-detected from common variants — `item_id`, `item`, `product`,
  `sku` — with a manual column picker as a fallback).
- A preview of the first rows of the uploaded file.
- On **Predict**, produces a summary table per item: number of stores
  selling it, total forecasted units across all stores for the chosen
  horizon, and average units per store per day.
- A bar chart comparing forecasted volume across the uploaded items.
- Fuzzy-matching suggestions (via `difflib`) for item IDs that aren't found
  in the dataset, and a downloadable CSV of the results.

### 11.3 Robustness features

- Guards against missing forecast/history files, with an explanatory error
  message.
- Guards against files that are present but missing expected columns.
- A sidebar **Reload data files** button that clears the Streamlit cache,
  for picking up newly generated CSVs without restarting the app.
- Optional display of run metadata (best model, RMSE, MAPE) in the
  sidebar, if a `run_metadata_*products.json` file is present — this file
  is not currently generated by the notebook (see Section 13); the app
  simply omits this section gracefully when it's absent.

---

## 12. Setup & Reproduction

1. **Download the data.** Obtain `calendar.csv`, `sell_prices.csv`, and
   `sales_train_validation.csv` from the M5 Forecasting Accuracy
   competition and place them in a folder of your choice.
2. **Set `DATA_DIR`** in the notebook to point to that folder.
3. **Run the notebook top to bottom.** Adjust `TOP_N` and `CHUNK_SIZE` in
   Step 1 as needed (see Section 4.5 for guidance).
4. **Install app dependencies and launch the app:**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
   Ensure `app.py` sits in the same working directory as the generated
   `demand_forecast_28days_*products.csv` and
   `sales_history_90days_*products.csv` files.

---

## 13. Known Limitations & Future Work

- **Recursive forecasting performance.** The current implementation issues
  one `model.predict()` call per product–store series per forecast day.
  Batching predictions across all series for a given day (28 calls total
  instead of thousands) would substantially reduce runtime with no change
  in output or memory footprint.
- **Early stopping.** Both XGBoost and LightGBM currently train for a fixed
  number of estimators. Adding a dedicated validation split and early
  stopping would likely improve both training time and generalization.
- **Loss function for count data.** Daily unit sales are non-negative,
  often intermittent count data. A Tweedie-objective LightGBM variant
  (as used in top M5 competition solutions) is a promising accuracy
  improvement not yet implemented here.
- **Additional lag/volatility features.** Short-term lags (`lag_1`–`lag_3`),
  rolling standard deviation, and price-momentum features are natural
  extensions not yet included in the feature set.
- **Run metadata export.** The Streamlit app optionally displays run
  metadata (model choice, RMSE, MAPE) from a `run_metadata_*products.json`
  file, but the notebook does not yet generate this file — this is a
  small, low-risk addition for future work.
- **Prediction intervals.** The forecast is currently a point estimate.
  Quantile regression (e.g. with LightGBM) could provide an uncertainty
  band around each forecast for the Streamlit app.
- **Segment-specific models.** Accuracy may vary meaningfully by category
  or store; per-segment models are a reasonable next experiment once a
  baseline is established.

---

## 14. Requirements

See `requirements.txt` for pinned package versions. Summary:

- **Core:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `plotly`
- **Modeling:** `xgboost`, `lightgbm`, `scikit-learn`, `joblib`
- **Deployment:** `streamlit`, `openpyxl` (for `.xlsx` uploads)
- **Optional:** `psutil` (memory diagnostics), `tensorflow` (LSTM
  comparison only, requires `RUN_DEEP_LEARNING = True`)

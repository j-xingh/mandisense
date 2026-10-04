# MandiSense

**Onion price discovery and forecasting for Maharashtra's mandis**

MandiSense is an end-to-end machine learning system that forecasts onion modal prices across 8 major Maharashtra APMC markets, flags prices that look abnormally low compared to a market's own history and its regional peers, and serves both through a live API and a web interface.

The project is framed against a real government problem statement — **SIH26132: "Strengthening market linkages and price discovery for farmers"** (Government of Maharashtra, Smart India Hackathon 2026) — which centers on giving farmers better visibility into fair pricing and reducing information asymmetry with intermediaries.

---

## What it does

1. **Forecasts onion prices** for a chosen market and date, using lag, rolling-average, seasonal, and cross-market features.
2. **Flags potentially unfair prices** — a two-signal anomaly detector checks whether a price is abnormally low both against a market's own recent history *and* against what other markets are getting the same day.
3. **Explains its own predictions** using SHAP, so the forecast isn't a black box.
4. **Serves live predictions** through a FastAPI backend and a glassmorphism web UI, with an honest confidence indicator that reflects how much time has passed since the last reliable data point for that market.

---

## Data

- **Source:** AGMARKNET (Directorate of Marketing & Inspection, Ministry of Agriculture), accessed via data.gov.in's "Variety-wise Daily Market Prices" dataset.
- **Scope:** Onion, 8 markets — Chandvad, Devala, Lasalgaon, Manmad, Nasik, Pimpalgaon, Pune, Yeola — chosen as the highest-volume, most historically complete markets in the raw data (~4,000–5,600 rows each).
- **Span:** ~23 years of daily price reports (2002–2025), ~141,000 raw rows before cleaning.

### Cleaning and validation

- Parsed dates explicitly as `DD/MM/YYYY` (pandas' default guesser silently mis-parsed ambiguous dates like `01/02/2009`, producing logically impossible rows where `max < min`).
- Filtered to `Grade == 'FAQ'` (Fair Average Quality) — the dominant, most consistently reported grade (~88% of rows). `Variety` was dropped as a grouping feature since ~65% of entries were generically labeled "Other," making it an unreliable signal.
- Collapsed duplicate market-day entries (~8% of rows, from residual variety mixing even within one grade) using the **median** price rather than the mean, since median is robust to a single outlier variety skewing the day's reported price.
- Built a rolling-median deviation + next-day-reversion rule to detect likely data-entry errors (e.g., a single-day 8–10x price spike that fully reverts the next reading) versus genuine market events (e.g., the real December 2019 national onion price crisis, which persisted for weeks and moved across multiple markets simultaneously). This distinguished and removed 4 confirmed data-entry errors while correctly preserving real volatility.
- **Key finding:** reporting frequency for these 8 markets dropped sharply after January 2025 — from near-daily to a handful of rows every few weeks. This is a real, verified limitation of the data source itself (not a bug introduced during cleaning), and it directly informs the forecast confidence indicator described below.

---

## Features

All features are built with explicit **data-leakage prevention** — every lag and rolling feature is shifted by one period *before* aggregation, so no feature for a given day ever includes that day's own price.

| Feature | Description |
|---|---|
| `Price_Lag_1 / 7 / 30` | Price from 1, 7, and 30 readings prior (same market) |
| `Rolling_Mean_7 / 30` | Rolling average of the prior 7/30 readings, excluding the current one |
| `Month` | Calendar month, to capture onion's strong seasonal cycle |
| `Cross_Market_Avg` | Same-day average price across all 8 markets, capturing regional co-movement (markets are 0.90–0.98 correlated with each other) |

---

## Modeling

Three models were compared on a **time-based train/test split** (not random — random splitting would leak future information into training via the lag features):

| Model | MAE (₹/quintal) | Notes |
|---|---|---|
| Naive baseline (today = yesterday) | 122.4 | ~6% error; a strong baseline given how autocorrelated onion prices are |
| Linear Regression | **109.7** | Best performer |
| XGBoost (default) | 110.3 | Essentially tied with linear |
| XGBoost (tuned, higher capacity) | 118.8 | Worse — overfit on training noise |

**Finding:** linear regression matched or beat XGBoost, indicating the price relationship in this feature set is largely linear/autoregressive. This is a real, reportable result, not a failure to tune XGBoost correctly — the added model complexity had nothing to catch in this feature set.

### Explainability (SHAP)

`Cross_Market_Avg` and `Price_Lag_1` dominate the model's predictions, consistent with the strong regional correlation found in EDA. `Cross_Market_Avg` carries a small self-referential component (it averages across all 8 markets including the target market itself) — a known, documented simplification; a stricter leave-one-out version is a natural next iteration.

---

## Anomaly detection

Flags a price as a potential "unfair price" signal only when it is simultaneously:
- **< 0.7×** that market's own trailing 30-day median, **and**
- **< 0.7×** the same-day average across all 8 markets

Requiring both conditions avoids false flags during genuine seasonal lows (when every market's price drops together) and only surfaces cases where one market/farmer is getting a meaningfully worse deal than both their own recent history and their neighbors on the same day. This flagged 168 of 33,895 rows (~0.5%), clustering around documented real volatility periods (e.g., late 2009, late 2011).

---

## API (FastAPI)

- `GET /predict_future?market=<name>&target_date=<date>` — returns a prediction, routed one of two ways:
  - **Historical date** (on or before the market's last known data): looks up the real row and returns predicted vs. actual price, for validation.
  - **Future date:** constructs features live from the most recent available data for that market and forecasts forward. Returns a `confidence` level (`high` / `medium` / `low`) and `days_since_last_data`, since forecast reliability degrades the further the target date is from the last real data point — a known, standard property of any forecasting system, made visible here rather than hidden.
- `GET /history?market=<name>` — returns the last 60 price readings for charting.

## Frontend

A single-page glassmorphism UI (`static/index.html`) built with plain HTML/CSS/JS and Chart.js — no framework needed, since the backend already owns all the logic. The input panel is shown alone; submitting a prediction reveals a result panel with the price, confidence indicator, and a price-history chart together.

---

## Known limitations

- **Data staleness:** reliable reporting for these 8 markets effectively ends around January 2025. Forecasts for dates far beyond this are projected from the last known data point and explicitly flagged with lower confidence — this is disclosed in both the API response and the UI, not hidden.
- **Single-step forecast reused across horizons:** the current model predicts one step ahead; all future dates share the same base features except seasonality (`Month`). A recursive forecasting approach (feeding each day's prediction into the next day's input) is the natural next step for longer horizons.
- **`Cross_Market_Avg` self-inclusion:** as noted above, a minor, documented simplification.
- **8 markets, onion only:** the model only knows the markets and commodity it was trained on. Expanding to the broader "TOP" basket (Tomato-Onion-Potato) or additional markets would require retraining, not just a config change.

---

## Running it locally

```bash
# 1. Clone and set up
git clone https://github.com/j-xingh/mandisense.git
cd mandisense
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Start the backend
uvicorn app:app --reload

# 3. Open the frontend
# Open static/index.html directly in a browser
```

---

## Project structure

```
mandisense/
├── data/
│   ├── raw/              # original AGMARKNET CSV exports
│   └── processed/        # cleaned data at each pipeline stage
├── models/
│   └── onion_price_model.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_anomaly_detection.ipynb
├── static/
│   └── index.html
├── app.py
├── requirements.txt
└── README.md
```

---

## Why this project

Built as a resume/portfolio project to demonstrate a complete ML workflow — real messy data, honest validation, explainability, and a working deployed system — rather than a notebook trained on a pre-cleaned Kaggle dataset. Framed against a real, current government problem statement (SIH26132) to ground the work in an actual stated need rather than an invented one.
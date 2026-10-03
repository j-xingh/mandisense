from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='MandiSense API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/onion_price_model.pkl")
history = pd.read_csv("data/processed/onion_model_ready.csv", parse_dates=["Arrival_Date"])

@app.get("/")
def read_root():
    return {"Message": "Mandisense API is running"}


@app.get("/predict_future")
def predict_future(market: str, target_date: str):
    target = pd.to_datetime(target_date)
    market_history = history[history["Market"] == market].sort_values("Arrival_Date")
    
    if market_history.empty:
        raise HTTPException(status_code=404, detail="No history for this market")
    
    last_known_date = market_history["Arrival_Date"].max()

    if target <= last_known_date:
        # Historical date — look up the real row instead of forecasting
        row = market_history[market_history["Arrival_Date"] == target]
        if row.empty:
            raise HTTPException(status_code=404, detail="No data found for this exact date")
        feature_cols = ['Price_Lag_1','Price_Lag_7','Price_Lag_30','Rolling_Mean_7','Rolling_Mean_30','Month','Cross_Market_Avg']
        prediction = model.predict(row[feature_cols])[0]
        return {
            "market": market, "target_date": target_date,
            "predicted_price": round(float(prediction), 2),
            "actual_price": float(row["Modal_Price"].values[0]),
            "mode": "historical_validation"
        }

    # Future date — use most recent data as forecast base
    recent_7 = market_history.tail(7)["Modal_Price"]
    recent_30 = market_history.tail(30)["Modal_Price"]
    last_known = market_history.iloc[-1]
    same_day_regional = history[history["Arrival_Date"] == last_known_date]["Modal_Price"].mean()

    features = pd.DataFrame([{
        "Price_Lag_1": last_known["Modal_Price"],
        "Price_Lag_7": recent_7.iloc[0] if len(recent_7) >= 7 else recent_7.mean(),
        "Price_Lag_30": recent_30.iloc[0] if len(recent_30) >= 30 else recent_30.mean(),
        "Rolling_Mean_7": recent_7.mean(),
        "Rolling_Mean_30": recent_30.mean(),
        "Month": target.month,
        "Cross_Market_Avg": same_day_regional
    }])
    prediction = model.predict(features)[0]
    gap_days = (target - last_known_date).days
    confidence = "high" if gap_days <= 14 else "medium" if gap_days <= 60 else "low"

    return {
        "market": market, "target_date": target_date,
        "last_known_date": last_known_date.strftime("%Y-%m-%d"),
        "days_since_last_data": gap_days, "confidence": confidence,
        "predicted_price": round(float(prediction), 2),
        "mode": "future_forecast",
        "note": "Projected from most recent available data; all future dates currently share the same base inputs except seasonality (month) — a known limitation of this v1 approach."
    }

@app.get("/history")
def get_history(market: str):
    subset = history[history["Market"] == market].sort_values("Arrival_Date").tail(60)
    return {
        "dates": subset["Arrival_Date"].dt.strftime("%Y-%m-%d").tolist(),
        "prices": subset["Modal_Price"].tolist()
    }
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI(title='MandiSense API')

model = joblib.load("models/onion_price_model.pkl")
history = pd.read_csv("data/processed/onion_model_ready.csv", parse_dates=["Arrival_Date"])

@app.get("/")
def read_root():
    return {"Message": "Mandisense API is running"}

@app.get("/predict")
def predict_price(market: str, date: str):
    target_date = pd.to_datetime(date)
    row = history[
        (history["Market"] == market) & (history["Arrival_Date"] == target_date)
    ]
    if row.empty:
        raise HTTPException(status_code=404, detail="No data found for this market/date combination")

    feature_cols = ['Price_Lag_1', 'Price_Lag_7', 'Price_Lag_30', 'Rolling_Mean_7', 'Rolling_Mean_30', 'Month', 'Cross_Market_Avg']
    X = row[feature_cols]
    prediction = model.predict(X)[0]

    return {
        "market": market,
        "date": date,
        "predicted_price": round(float(prediction), 2),
        "actual_price": float(row["Modal_Price"].values[0])
    }
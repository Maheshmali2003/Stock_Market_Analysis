# Stock_Market_Analysis# 📈 Stock Market Forecasting with ARIMA, LSTM & Prophet

This project analyzes and forecasts TCS stock prices using three different models: ARIMA, LSTM, and Facebook Prophet.

## 📊 Problem Statement
To predict future stock prices using time series forecasting techniques and compare different models on performance.

---

## 📁 Dataset
- File: `TCS_Stock_Data.csv`
- Source: Yahoo Finance or similar
- Fields: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`

---

## 🔧 Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras (for LSTM)
- Statsmodels (for ARIMA)
- Prophet (for trend forecasting)

---

## 🧠 Models Used

| Model     | Description                               |
|-----------|-------------------------------------------|
| ARIMA     | Traditional statistical time series model |
| LSTM      | Deep learning RNN-based model             |
| Prophet   | Additive model by Facebook (trend/season) |

---

## 📈 Results

| Model   | RMSE (example) |
|---------|----------------|
| ARIMA   | 29.65          |
| LSTM    | 21.34          |
| Prophet | 23.50          |

> Lower RMSE indicates better performance.

---

## 🖼 Visuals
- Historical stock prices
- Predicted prices (30-day forecast)
- Actual vs Predicted graphs for all models

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
python app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime

st.set_page_config(page_title="📈 Stock Forecast App", layout="centered")
st.title("📊 Stock Market Forecasting")
st.markdown("Upload a CSV file with historical stock data (Date, Close) to forecast prices using ARIMA and LSTM.")

uploaded_file = st.file_uploader("Upload your stock CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("CSV must contain 'Date' and 'Close' columns.")
    else:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        st.subheader("📉 Historical Stock Prices")
        st.line_chart(df['Close'])

        # ARIMA
        st.subheader("🔁 ARIMA Forecast")
        try:
            model_arima = ARIMA(df['Close'], order=(5,1,0))
            model_fit_arima = model_arima.fit()
            forecast_arima = model_fit_arima.forecast(steps=30)

            st.line_chart(forecast_arima)
            st.success("ARIMA forecast completed.")
        except Exception as e:
            st.warning(f"ARIMA model failed: {e}")

        # LSTM
        st.subheader("🧠 LSTM Forecast")
        data = df[['Close']].values
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        train_data = data_scaled
        X, y = [], []
        for i in range(60, len(train_data)):
            X.append(train_data[i-60:i, 0])
            y.append(train_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model_lstm = Sequential()
        model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model_lstm.add(LSTM(units=50))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        model_lstm.fit(X, y, epochs=5, batch_size=32, verbose=0)

        last_60_days = data_scaled[-60:]
        lstm_input = last_60_days.reshape(1, 60, 1)
        predicted = []
        for _ in range(30):
            next_price = model_lstm.predict(lstm_input, verbose=0)[0][0]
            predicted.append(next_price)
            lstm_input = np.append(lstm_input[:, 1:, :], [[[next_price]]], axis=1)

        forecast_lstm = scaler.inverse_transform(np.array(predicted).reshape(-1, 1))
        st.line_chart(forecast_lstm)
        st.success("LSTM forecast completed.")

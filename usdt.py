import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Step 1: Fetch USDT Price Data from Binance API
def fetch_usdt_data():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=500"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch data from Binance API")
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '_', '_', '_', '_', '_', '_'])
    df = df[['timestamp', 'close']].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Step 2: Preprocess Data
def preprocess_data(df):
    if df.empty:
        raise ValueError("DataFrame is empty. Check data source.")
    df.dropna(subset=['close'], inplace=True)
    scaler = MinMaxScaler(feature_range=(0,1))
    df['scaled_close'] = scaler.fit_transform(df[['close']])
    return df, scaler

# Step 3: Prepare LSTM Data
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Step 4: Train LSTM Model
def train_lstm(X_train, y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)
    return model

# Step 5: Detect Anomalies
def detect_anomalies(df):
    model = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly_score'] = model.fit_predict(df[['close']].values.reshape(-1, 1))  # Ensure correct shape
    anomalies = df[df['anomaly_score'] == -1]
    return anomalies

# Step 6: Run the Full Pipeline
df = fetch_usdt_data()
df, scaler = preprocess_data(df)
X, y = create_sequences(df['scaled_close'].values)
if len(X) == 0 or len(y) == 0:
    raise ValueError("Not enough data to create sequences. Adjust the sequence length.")
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
model = train_lstm(X, y)

df['prediction'] = np.nan
df.iloc[-len(y):, df.columns.get_loc('prediction')] = scaler.inverse_transform(model.predict(X)).flatten()

anomalies = detect_anomalies(df)

# Step 7: Plot Results
plt.figure(figsize=(14,6))
plt.plot(df.index, df['close'], label='Actual-Price', color='green')
plt.plot(df.index, df['prediction'], label='Predicted-Price', color='red')
plt.scatter(anomalies.index, anomalies['close'], color='blue', label='Anomalies', marker='X')
plt.legend()
plt.show()

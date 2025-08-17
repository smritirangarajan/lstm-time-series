# Imports
from tensorflow import keras 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------- paths ----------
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Load Data
data = pd.read_csv("meta.csv")

# Clean columns
for col in ["Close/Last","Open","High","Low"]:
    data[col] = data[col].replace({'\$':''}, regex=True).astype(float)

# Convert date
data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y")
data = data.sort_values("Date").reset_index(drop=True)

print(data.head())
print(data.info())

# =============== 📊 EDA & Visualization (SAVED) ===============

# 1) Closing price over time
plt.figure(figsize=(12,6))
plt.plot(data["Date"], data["Close/Last"], label="Close Price", color="blue")
plt.title("Meta Stock Closing Price Over Time")
plt.xlabel("Date"); plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_close_price.png", dpi=150)
plt.close()

# 2) Moving averages
data["MA50"]  = data["Close/Last"].rolling(50).mean()
data["MA200"] = data["Close/Last"].rolling(200).mean()
plt.figure(figsize=(12,6))
plt.plot(data["Date"], data["Close/Last"], label="Close Price", alpha=0.7)
plt.plot(data["Date"], data["MA50"],  label="50-Day MA", color="red")
plt.plot(data["Date"], data["MA200"], label="200-Day MA", color="green")
plt.title("Meta Stock with Moving Averages")
plt.xlabel("Date"); plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_moving_averages.png", dpi=150)
plt.close()

# 3) Daily returns distribution
data["Returns"] = data["Close/Last"].pct_change()
plt.figure(figsize=(10,5))
sns.histplot(data["Returns"].dropna(), bins=50, kde=True, color="purple")
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return"); plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_returns_hist.png", dpi=150)
plt.close()

# 4) Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data[["Open","High","Low","Close/Last","Volume"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_corr_heatmap.png", dpi=150)
plt.close()

# 5) Trading volume over time
plt.figure(figsize=(12,6))
plt.bar(data["Date"], data["Volume"], color="orange")
plt.title("Trading Volume Over Time")
plt.xlabel("Date"); plt.ylabel("Volume")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_volume.png", dpi=150)
plt.close()

# 6) Candlestick chart (uses OHLC + Volume)
#   mplfinance expects columns named: Open, High, Low, Close, and a DatetimeIndex.
ohlc_df = data.rename(columns={"Close/Last": "Close"}).set_index("Date")[["Open","High","Low","Close","Volume"]]
mpf.plot(
    ohlc_df,
    type="candle",
    volume=True,
    style="yahoo",
    figsize=(12,6),
    tight_layout=True,
    savefig=dict(fname=f"{FIG_DIR}/06_candlestick.png", dpi=150)
)

# =============================================================

# =============== LSTM (your original approach) ===============
# Use only Close prices for LSTM
close_series = data[["Close/Last"]]
dataset = close_series.values

# Train-test split
training_data_len = int(np.ceil(len(dataset) * 0.95))

# Scale (you can also fit on train only if you prefer)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[:training_data_len]

X_train, y_train = [], []

# Sliding window (60 days)
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build model
model = keras.models.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],1)),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Test set
test_data = scaled_data[training_data_len-60:]
X_test, y_test = [], dataset[training_data_len:]
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test).reshape((-1, 60, 1))

# Predictions
predictions = model.predict(X_test, verbose=0)
predictions = scaler.inverse_transform(predictions)

# Create dataframe for plotting
train = data[:training_data_len]
test = data[training_data_len:].copy()
test = test.iloc[:len(predictions)].copy()   # align lengths just in case
test["Predictions"] = predictions.ravel()

# Plot predictions (SAVED)
plt.figure(figsize=(12,8))
plt.plot(train["Date"], train["Close/Last"], label="Train (Actual)", color="blue")
plt.plot(test["Date"],  test["Close/Last"],  label="Test (Actual)",  color="orange")
plt.plot(test["Date"],  test["Predictions"], label="Predictions",    color="red")
plt.title("Meta Stock Price Prediction")
plt.xlabel("Date"); plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_predictions.png", dpi=150)
plt.close()

# Meta (META) Stock — LSTM Time Series Forecasting

This project explores Meta's historical stock data and builds a simple LSTM-based model to forecast the closing price.  
It includes quick EDA, a candlestick chart, and a baseline LSTM with sliding windows.

## Dataset
- File: `meta.csv`
- Columns: `Date, Close/Last, Volume, Open, High, Low`
- Notes: Prices contain `$` and are cleaned to floats; dates are parsed as `%m/%d/%Y`.

## What’s inside
- **EDA**: trends, moving averages, returns distribution, correlations, and volume.
- **Candlestick**: OHLC + volume using `mplfinance`.
- **Model**: 2-layer LSTM with a dense head; 60-day lookback; `StandardScaler`.

## Key Visuals

### 1) Closing Price
![Close Price](figures/01_close_price.png)

### 2) Moving Averages (50/200)
![Moving Averages](figures/02_moving_averages.png)

### 3) Daily Returns Distribution
![Returns Dist](figures/03_returns_hist.png)

### 4) Feature Correlation Heatmap
![Correlation Heatmap](figures/04_corr_heatmap.png)

### 5) Trading Volume Over Time
![Volume](figures/05_volume.png)

### 6) Candlestick (OHLC + Volume)
![Candlestick](figures/06_candlestick.png)

### 7) LSTM Predictions vs. Actuals
![Predictions](figures/07_predictions.png)

## How to run

```bash
pip install -r requirements.txt  # or install the libs below
python main.py

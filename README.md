# Global Stock Forecasting Dissertation

MSc Data Science dissertation project using Bidirectional LSTM (BiLSTM), K-Means volatility regimes, Tanh-based post-processing, and an automated five-day stock forecasting dashboard for equities from the United States, United Kingdom, and Canada.

## Project Overview

The aim of this project is to build an end-to-end stock forecasting pipeline that combines machine learning and data engineering.

The pipeline:

- downloads historical market data from Yahoo Finance
- creates Daily Percentage Return and 21-Day Rolling Volatility features
- applies Min-Max scaling
- creates a volatility regime feature using K-Means clustering
- converts the data into 60-day historical sequences
- trains a stacked BiLSTM model
- predicts the following five normalized closing-price values
- applies Tanh-based post-processing to control large forecast movements
- anchors the final trajectory to the latest available market price
- exports the results into a wide-format CSV dashboard

## Dataset

The initial stock universe contained 150 equities from:

- United States
- United Kingdom
- Canada

Seven tickers did not provide usable historical data due to listing, ticker-symbol, or data-availability issues.

Final usable dataset:

- US: 50 stocks
- UK: 44 stocks
- Canada: 49 stocks
- Total: 143 stocks

Historical data covers approximately 2014 to 2023.

## Model Inputs

The BiLSTM model uses eight features:

1. Open
2. High
3. Low
4. Close
5. Volume
6. Daily Percentage Return
7. 21-Day Rolling Volatility
8. Volatility Regime

Each model input contains the previous 60 trading days.

The model predicts the next five normalized closing-price values.

## Model Architecture

The forecasting model uses:

- Bidirectional LSTM with 128 units
- Dropout (0.2)
- Bidirectional LSTM with 64 units
- Dropout (0.2)
- Dense layer with 32 neurons and ReLU activation
- Dense output layer with five neurons

The model is trained using:

- Adam optimizer
- learning rate: 0.0005
- Mean Squared Error loss
- batch size: 32
- maximum epochs: 15
- Early Stopping with patience of 5

## Model Results

The final model achieved:

- MAE: 0.0029
- RMSE: 0.0038
- First-day Directional Accuracy: 51.43%

MAE and RMSE are calculated in normalized closing-price space.

## Forecast Stabilisation

The raw BiLSTM output is processed using a Tanh-based percentage-bounding method.

A fixed movement limit of approximately 2.5% is applied to each forecast step before the trajectory is anchored to the latest available stock price.

This step is used to control large forecast movements and improve trajectory stability. It does not guarantee improved forecasting accuracy.

## Dashboard Output

The final dashboard contains:

- Ticker
- Market
- Currency
- Current Price
- Day 1 to Day 5 forecast prices
- 5-Day Change
- 5-Day Change (%)
- Bullish or Bearish trajectory

The final dashboard contains 143 successfully processed stocks.

During the final run:

- 108 stocks were classified as Bullish
- 35 stocks were classified as Bearish

These are model-generated signals and should not be interpreted as confirmed future market outcomes.

## Files

- `Global_Stock_Forecasting_BiLSTM.ipynb` — complete forecasting notebook
- `Global_150_Wide_Forecast_Validated.csv` — final dashboard output

## Running the Project

The notebook is designed to run in Google Colab, Jupyter Notebook, and standard Python environments.

The code uses relative file paths and does not depend on machine-specific Windows directories.

Install the required libraries using:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib seaborn tensorflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
from time import time

def load_data(file):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df['SE3'].asfreq('D').interpolate()

def plot(data, plt_title):
    data.plot(figsize=(10,5))
    plt.title(plt_title)
    plt.show()

def plot_acf_pacf(data, plt_title):
    plot_acf(data, lags=50)
    plt.title("ACF: " + plt_title)
    plt.show()

    plot_pacf(data, lags=50)
    plt.title("PACF: " + plt_title)
    plt.show()

def adfuller_test(data, title):
    result = adfuller(data)
    print(title, "p-value:", result[1])

def main():
    SE3_consumption = {}
    SE3_consumption_raw = load_data("se_d_data.csv")
    SE3_consumption_log = np.log(SE3_consumption_raw)
    SE3_consumption_diff = SE3_consumption_log.diff().dropna() # trend removal
    SE3_consumption_seasonal = SE3_consumption_log.diff(7).dropna() # weakly seasonal removale
    SE3_consumption_combination = SE3_consumption_log.diff().diff(7).dropna() # combo

    series_list = [
        (SE3_consumption_raw, "SE3 - Energy consumption (Raw)"),
        (SE3_consumption_log, "SE3 - Energy consumption (Log)"),
        (SE3_consumption_diff, "SE3 - Energy consumption (Log + Trend Removal)"),
        (SE3_consumption_seasonal, "SE3 - Energy consumption (Log + Seasonal Weekly Removal)"),
        (SE3_consumption_combination, "SE3 - Energy consumption (Log + Trend and Seasonality Removal)")
    ]

    # Regular Plot
    for data, title in series_list:
        plot(data, title)

    # Adfuller - test
    for data, title in series_list:
        adfuller_test(data, title)
    """
    Although the Augmented Dickey-Fuller test suggests that the raw series is stationary (p < 0.05), 
    visual inspection reveals strong seasonal patterns. Since the ADF test does not account for 
    deterministic seasonality, we apply seasonal differencing to remove the weekly component and 
    obtain a series suitable for modeling. 
    """
    
    # Plot ACF and PACF
    for data, title in series_list:
        plot_acf_pacf(data, title)

    train = SE3_consumption_log[:'2025-12-31'] 
    test = SE3_consumption_log['2026-01-01':]
    
    p, d, q = 0, 1, 1
    P, D, Q, m = 1, 1, 1, 7

    my_order=(p, d, q)
    my_seasonal_order=(P, D, Q, m)

    # Fit the initial model on the training set
    model = SARIMAX(train, order=my_order, seasonal_order=my_seasonal_order)
    model_fit = model.fit(disp=False)

    # Rolling forecast loop
    predictions = []
    # We use the test index to iterate
    for date in test.index:
        # Forecast the next step
        forecast = model_fit.forecast()
        predictions.append(forecast.iloc[0])
        
        # IMPORTANT: Update the model with the new observation
        # This keeps the model "fresh" without re-fitting from scratch
        model_fit = model_fit.append(test[date:date], refit=False)

    # Convert to Series for easy plotting/metrics
    rolling_predictions = pd.Series(predictions, index=test.index)

    # Convert both back to original scale
    actual_original = np.exp(test)
    predictions_original = np.exp(rolling_predictions)

    # Now plot these instead
    plt.figure(figsize=(12, 6))
    plt.plot(actual_original, label='Actual Energy Consumption')
    plt.plot(predictions_original, label='SARIMA Forecast', color='red', linestyle='--')
    plt.title('Actual vs. Forecasted Energy Consumption (Original Scale)')
    plt.legend()
    plt.show()

    # And calculate RMSE on the original scale
    rmse_original = np.sqrt(((actual_original - predictions_original) ** 2).mean())
    print(f"RMSE in original units: {rmse_original:.2f} MW")


if __name__ == "__main__":
    main()
    
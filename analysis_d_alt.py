import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
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

def plot_prediction_vs_reality(actual_original, predictions_original):
    # Now plot these instead
    plt.figure(figsize=(12, 6))
    plt.plot(actual_original, label='Actual Energy Consumption')
    plt.plot(predictions_original, label='SARIMA Forecast', color='red', linestyle='--')
    plt.title('Actual vs. Forecasted Energy Consumption (Original Scale)')
    plt.legend()
    plt.show()

def adfuller_test(data, title):
    result = adfuller(data)
    print(title, "p-value:", result[1])

def residuals_tests(residuals, title):
    fig, axes = plt.subplots(2, 2, figsize=(12,8))

    # ACF
    plot_acf(residuals, lags=50, ax=axes[0,0])
    axes[0,0].set_title("Residual ACF " + title)

    # PACF
    plot_pacf(residuals, lags=50, ax=axes[0,1])
    axes[0,1].set_title("Residual PACF " + title)

    # Residual time series
    axes[1,0].plot(residuals)
    axes[1,0].set_title("Residuals " + title)

    # Histogram
    axes[1,1].hist(residuals, bins=100)
    axes[1,1].set_title("Resuduals distribution " + title)

    plt.tight_layout()
    plt.show()

    # Ljung-Box (separate, since it's numeric)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_test = acorr_ljungbox(residuals, lags=[10,20,30], return_df=True)
    print("\nLjung-Box test " + title + ":" )
    print(lb_test)


def main():
    SE3_consumption = {}
    SE3_consumption_raw = load_data("se_d_data.csv")
    SE3_consumption_log = np.log(SE3_consumption_raw)
    SE3_consumption_diff = SE3_consumption_log.diff().dropna() # trend removal
    SE3_consumption_seasonal = SE3_consumption_log.diff(7).dropna() # weakly seasonal removale
    SE3_consumption_combination = SE3_consumption_log.diff().diff(7).dropna() # combo

    """series_list = [
        (SE3_consumption_raw, "SE3 - Energy consumption (Raw)"),
        (SE3_consumption_log, "SE3 - Energy consumption (Log)"),
        (SE3_consumption_diff, "SE3 - Energy consumption (Log + Trend Removal)"),
        (SE3_consumption_seasonal, "SE3 - Energy consumption (Log + Seasonal Weekly Removal)"),
        (SE3_consumption_combination, "SE3 - Energy consumption (Log + Trend and Seasonality Removal)")
    ]"""

    series_list = [
        (SE3_consumption_raw, "SE3 - Energy consumption (Raw)"),
        (SE3_consumption_log, "SE3 - Energy consumption (Log)"),
        (SE3_consumption_diff, "SE3 - Energy consumption (Log + Trend Removal)")
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
    test = SE3_consumption_log['2026-01-01':'2026-04-11']
    """
    SARIMA_list = [
        ((1,1,1), (1,1,1,7), "SARIMA - (1,1,1)(1,1,1,7)"), # # old first model
        ((1,0,1), (1,1,1,7), "SARIMA - (1,0,1)(1,1,1,7)"), # Option A: reduce differencing
        ((2,0,1), (1,1,1,7), "SARIMA - (2,0,1)(1,1,1,7)"),
        ((1,1,1), (2,1,2,7), "SARIMA - (1,1,1)(2,1,2,7)"), # Option B: increase seasonal AR/MA strength
        ((2,1,2), (2,1,2,7), "SARIMA - (2,1,2)(2,1,2,7)"),
        ((1,1,1), (1,0,1,7), "SARIMA - (1,1,1)(1,0,1,7)") # Option C (VERY important): try WITHOUT seasonal differencing
    ]"""

    SARIMA_list = [
        ((1,1,1), (1,0,1,7), "SARIMA - (1,1,1)(1,0,1,7)") # Option C (VERY important): try WITHOUT seasonal differencing
    ]

    for order, seasonal_order, model_name in SARIMA_list:

        # Fit the initial model on the training set
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        residuals = model_fit.resid
        residuals_tests(residuals, model_name)
        print(f"{model_name} AIC: {model_fit.aic}")
    
        # Rolling forecast loop
        predictions = []
        # We use the test index to iterate
        for date in test.index:
            #print(date)
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
        plot_prediction_vs_reality(actual_original, predictions_original)

        # And calculate RMSE on the original scale
        rmse_original = np.sqrt(((actual_original - predictions_original) ** 2).mean())
        print(f"RMSE in original units: {rmse_original:.2f} MW")


if __name__ == "__main__":
    main()
    
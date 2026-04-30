import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import os
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path='data/se_d_data.csv', target_col='SE3'):
    """Load daily data and return target series up to end of March 2026."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    df = df[df.index <= '2026-03-31']
    df.index.freq = 'D'
    return df[target_col]

def fit_deterministic(x_t):
    """Fit OLS with quadratic trend and Fourier yearly/weekly components."""
    t = (x_t.index - x_t.index[0]).days.values
    X = pd.DataFrame(index=x_t.index)
    X['const'] = 1
    X['t'] = t
    X['t2'] = t**2
    
    # Yearly Fourier (K=7, P=365.25)
    for k in range(1, 8):
        X[f'cos_y_{k}'] = np.cos(2 * np.pi * k * t / 365.25)
        X[f'sin_y_{k}'] = np.sin(2 * np.pi * k * t / 365.25)
        
    # Weekly Fourier (K=3, P=7)
    for k in range(1, 4):
        X[f'cos_w_{k}'] = np.cos(2 * np.pi * k * t / 7)
        X[f'sin_w_{k}'] = np.sin(2 * np.pi * k * t / 7)
        
    ols_model = sm.OLS(x_t, X).fit()
    z_t = x_t - ols_model.fittedvalues
    z_t.name = 'Residuals_Zt'
    return ols_model, z_t

def fit_stochastic(z_t):
    """Fit AR(8) model to the stationary residuals."""
    print("Fitting AR(8) model to residuals...")
    ar_model = ARIMA(z_t, order=(8, 0, 0), trend='n').fit()
    return ar_model

def generate_forecast(ols_model, ar_model, start_date, last_date, h=365, alpha=0.05):
    """Generate linear forecast for the next h days with confidence intervals."""
    print(f"Generating {h}-day forecast...")
    
    # 1. Forecast Stochastic Component (AR8)
    forecast_obj = ar_model.get_forecast(steps=h)
    z_hat = forecast_obj.predicted_mean
    z_ci = forecast_obj.conf_int(alpha=alpha)
    
    # 2. Forecast Deterministic Component
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=h)
    future_t = (future_dates - start_date).days.values
    
    X_future = pd.DataFrame(index=future_dates)
    X_future['const'] = 1
    X_future['t'] = future_t
    X_future['t2'] = future_t**2
    
    for k in range(1, 10):
        X_future[f'cos_y_{k}'] = np.cos(2 * np.pi * k * future_t / 365.25)
        X_future[f'sin_y_{k}'] = np.sin(2 * np.pi * k * future_t / 365.25)
        
    for k in range(1, 4):
        X_future[f'cos_w_{k}'] = np.cos(2 * np.pi * k * future_t / 7)
        X_future[f'sin_w_{k}'] = np.sin(2 * np.pi * k * future_t / 7)
        
    deterministic_hat = ols_model.predict(X_future)
    
    # 3. Combine in Log domain (Additive)
    x_hat = deterministic_hat + z_hat.values
    x_lower = deterministic_hat + z_ci.iloc[:, 0].values
    x_upper = deterministic_hat + z_ci.iloc[:, 1].values
    
    # 4. Inverse Transform (Exponentiation) to get back to Original Scale
    y_hat = np.exp(x_hat)
    y_lower = np.exp(x_lower)
    y_upper = np.exp(x_upper)
    
    # Return both original scale and Z-domain values for analysis
    forecast_results = {
        'y_hat': pd.Series(y_hat, index=future_dates),
        'y_lower': pd.Series(y_lower, index=future_dates),
        'y_upper': pd.Series(y_upper, index=future_dates),
        'z_hat': pd.Series(z_hat.values, index=future_dates),
        'z_lower': pd.Series(z_ci.iloc[:, 0].values, index=future_dates),
        'z_upper': pd.Series(z_ci.iloc[:, 1].values, index=future_dates),
        'deterministic_hat': pd.Series(deterministic_hat, index=future_dates)
    }
    return forecast_results

def plot_forecast(y_train, y_hat, y_lower, y_upper, y_test ):
    """Plot training data, actual data for forecast period, and the forecast with CI."""
    os.makedirs('figures', exist_ok=True)
    
    plt.figure(figsize=(14, 7))
    
    # Focus plot from 2025-11-01
    plot_start = '2025-11-01'
    y_train_narrow = y_train[y_train.index >= plot_start]
    
    # Plot Training Data
    plt.plot(y_train_narrow.index, y_train_narrow, 
             label='Training Data', color='blue', alpha=0.6)
    
    # Plot Actual Data for the Forecast Period
    plt.plot(y_test.index, y_test, label='Actual Data', color='black', alpha=0.8, linewidth=1.5)
    
    # Plot Forecast
    plt.plot(y_hat.index, y_hat, label='Total Forecast', color='red', linestyle='--')
    
    # Plot Confidence Intervals
    plt.fill_between(y_hat.index, y_lower, y_upper, color='red', alpha=0.15, 
                     label='95% Confidence Interval')
    
    plt.title('Out-of-Sample Forecast Evaluation: Original Scale')
    plt.ylabel('Electricity Consumption (SE3)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = 'figures/forecasting_evaluation_2026.png'
    plt.savefig(plot_path)
    print(f"Forecast evaluation plot saved to {plot_path}")

def plot_residuals_forecast(z_train, z_hat, z_lower, z_upper, z_test):
    """Plot stationary residuals vs forecast to evaluate the stochastic component."""
    plt.figure(figsize=(14, 6))
    
    # Focus from 2025-11-01
    plot_start = '2025-11-01'
    z_train_narrow = z_train[z_train.index >= plot_start]
    
    plt.plot(z_train_narrow.index, z_train_narrow, label='Actual Residuals (Train)', color='blue', alpha=0.6)
    plt.plot(z_test.index, z_test, label='Actual Residuals (Test)', color='black', alpha=0.8)
    plt.plot(z_hat.index, z_hat, label='AR(8) Residual Forecast', color='red', linestyle='--')
    
    plt.fill_between(z_hat.index, z_lower, z_upper, color='red', alpha=0.15, label='95% CI')
    
    plt.title('Stochastic Component Evaluation: AR(8) Forecast of Residuals (Z_t)')
    plt.ylabel('Stationary Residuals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = 'figures/residuals_forecast_evaluation.png'
    plt.savefig(plot_path)
    print(f"Residuals forecast evaluation plot saved to {plot_path}")

def main():
    # 1. Load Data
    print("Loading historical data...")
    y_full = load_data()
    
    # Split into train and test (for evaluation)
    # Using the split date defined in the script
    split_date = '2026-02-15' 
    y_train = y_full[y_full.index <= split_date]
    y_test = y_full[y_full.index > split_date]
    
    start_date = y_train.index[0]
    last_date = y_train.index[-1]
    
    # 2. Transformation
    print(f"Applying Log Transformation to Training Data (up to {split_date})...")
    x_train = np.log(y_train)
    
    # 3. Deterministic Decomposition
    print("Fitting Deterministic Decomposition...")
    ols_model, z_train = fit_deterministic(x_train)
    
    # 4. Stochastic Modeling
    ar_model = fit_stochastic(z_train)
    
    # 5. Forecasting
    h_days = len(y_test)
    res = generate_forecast(ols_model, ar_model, start_date, last_date, h=h_days)
    
    # 6. Calculate test Residuals for the evaluation period
    # (Log test - Deterministic Prediction for that period)
    z_test = np.log(y_test) - res['deterministic_hat']
    
    # 7. Output Generation
    plot_forecast(y_train, res['y_hat'], res['y_lower'], res['y_upper'], y_test)
    plot_residuals_forecast(z_train, res['z_hat'], res['z_lower'], res['z_upper'], z_test)
    
    print("\nForecast evaluation completed successfully.")

if __name__ == '__main__':
    main()

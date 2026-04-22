import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import inv_boxcox
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

def load_daily_data(file_path='data/se_d_data.csv'):
    """
    Load the daily electricity consumption data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}.")
    
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    
    # Only include data points up until end of January 2026
    df = df[df.index <= '2026-01-31']
    
    # Explicitly set the frequency to Daily to silence ValueWarnings
    df.index.freq = 'D'
    
    print(f"Loaded data from {file_path}. Shape: {df.shape}")
    return df

def apply_box_cox(series):
    """
    Apply Box-Cox transformation to the series.
    Returns the transformed series (X_t) and the optimal lambda.
    """
    print("Applying Box-Cox transformation...")
    
    # stats.boxcox returns (transformed_data, optimal_lambda)
    x_t_values, lmbda = stats.boxcox(series)
    
    x_t = pd.Series(x_t_values, index=series.index, name='BoxCox_' + series.name)
    
    print(f"Optimal Lambda found: {lmbda:.4f}")
    return x_t, lmbda

def decompose_series(series, plotting=True):
    """
    Decompose the series X_t into trend (m_t), seasonality (s_t), and residuals (Z_t).
    Uses a quadratic trend and Fourier terms for yearly (365.25) and weekly (7) seasonality.
    """
    print("Decomposing series (Trend + Yearly/Weekly Seasonality)...")
    
    # Time index in days since start
    t = (series.index - series.index[0]).days.values
    
    # Create feature matrix X for OLS
    X = pd.DataFrame(index=series.index)
    X['const'] = 1
    X['t'] = t
    X['t2'] = t**2
    
    year_cols = []
    # Yearly Seasonality (Fourier terms, Period = 365.25)
    for k in range(1, 10):
        c, s = f'cos_year_{k}', f'sin_year_{k}'
        X[c] = np.cos(2 * np.pi * k * t / 365.25)
        X[s] = np.sin(2 * np.pi * k * t / 365.25)
        year_cols.extend([c, s])
        
    week_cols = []
    # Weekly Seasonality (Fourier terms, Period = 7)
    for k in range(1, 4):
        c, s = f'cos_week_{k}', f'sin_week_{k}'
        X[c] = np.cos(2 * np.pi * k * t / 7)
        X[s] = np.sin(2 * np.pi * k * t / 7)
        week_cols.extend([c, s])
        
    # Fit the model
    ols_model = sm.OLS(series, X).fit()
    
    # Calculate components
    # Trend: const + t + t^2
    m_t_params = ols_model.params[['const', 't', 't2']]
    m_t = X[['const', 't', 't2']] @ m_t_params
    m_t = pd.Series(m_t, index=series.index, name='Trend')
    
    # Seasonal components
    s_year = X[year_cols] @ ols_model.params[year_cols]
    s_year = pd.Series(s_year, index=series.index, name='Yearly Seasonality')
    
    s_week = X[week_cols] @ ols_model.params[week_cols]
    s_week = pd.Series(s_week, index=series.index, name='Weekly Seasonality')
    
    s_t = s_year + s_week
    s_t = pd.Series(s_t, index=series.index, name='Total Seasonality')
    
    # Residuals: Z_t = X_t - m_t - s_t
    z_t = series - (m_t + s_t)
    z_t = pd.Series(z_t, index=series.index, name='Residuals')
    
    if plotting:
        plt.figure(figsize=(12, 6))
        # Plot last 2 years for visibility of weekly components
        plot_range = slice(-730, None) 
        plt.plot(s_year.index[plot_range], s_year.iloc[plot_range], label='Yearly Component', color='green', alpha=0.7)
        plt.plot(s_week.index[plot_range], s_week.iloc[plot_range], label='Weekly Component', color='orange', alpha=0.7)
        plt.plot(s_t.index[plot_range], s_t.iloc[plot_range], label='Total Seasonality', color='blue', alpha=0.5, linewidth=1)
        plt.title('Superimposed Seasonal Components (Fourier)')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/seasonal_components_fourier.png')
        print("Seasonal components plot saved to figures/seasonal_components_fourier.png")
    
    return m_t, s_t, z_t, ols_model

def decompose_dummy(series, plotting=True):
    """
    Decompose the series X_t into trend (m_t), seasonality (s_t), and residuals (Z_t).
    Uses a quadratic trend, dummy variables for weekly seasonality, and Fourier
    terms for yearly (365.25) seasonality.
    """
    print("Decomposing series (Trend + Yearly Fourier + Weekly Dummies)...")
    
    # Time index in days since start
    t = (series.index - series.index[0]).days.values
    
    # Create feature matrix X for OLS
    X = pd.DataFrame(index=series.index)
    X['const'] = 1
    X['t'] = t
    X['t2'] = t**2
    
    year_cols = []
    # Yearly Seasonality (Fourier terms, Period = 365.25)
    for k in range(1, 10):
        c, s = f'cos_year_{k}', f'sin_year_{k}'
        X[c] = np.cos(2 * np.pi * k * t / 365.25)
        X[s] = np.sin(2 * np.pi * k * t / 365.25)
        year_cols.extend([c, s])
        
    dow_cols = []
    # Weekly Seasonality (Dummy variables)
    # Day 0=Monday, ..., 6=Sunday. Drop Day 0 to avoid dummy variable trap with constant.
    dow = series.index.dayofweek
    for i in range(1, 7):
        col = f'dow_{i}'
        X[col] = (dow == i).astype(int)
        dow_cols.append(col)
        
    # Fit the model
    ols_model = sm.OLS(series, X).fit()
    
    # Calculate components
    # Trend: const + t + t^2
    m_t_params = ols_model.params[['const', 't', 't2']]
    m_t = X[['const', 't', 't2']] @ m_t_params
    m_t = pd.Series(m_t, index=series.index, name='Trend')
    
    # Seasonal components
    s_year = X[year_cols] @ ols_model.params[year_cols]
    s_year = pd.Series(s_year, index=series.index, name='Yearly Seasonality')
    
    s_week = X[dow_cols] @ ols_model.params[dow_cols]
    s_week = pd.Series(s_week, index=series.index, name='Weekly Seasonality')
    
    s_t = s_year + s_week
    s_t = pd.Series(s_t, index=series.index, name='Total Seasonality')
    
    # Residuals: Z_t = X_t - m_t - s_t
    z_t = series - (m_t + s_t)
    z_t = pd.Series(z_t, index=series.index, name='Residuals')
    
    if plotting:
        plt.figure(figsize=(12, 6))
        # Plot last 2 years for visibility
        plot_range = slice(-730, None)
        plt.plot(s_year.index[plot_range], s_year.iloc[plot_range], label='Yearly Component', color='green', alpha=0.7)
        plt.plot(s_week.index[plot_range], s_week.iloc[plot_range], label='Weekly Component', color='orange', alpha=0.7)
        plt.plot(s_t.index[plot_range], s_t.iloc[plot_range], label='Total Seasonality', color='blue', alpha=0.5, linewidth=1)
        plt.title('Superimposed Seasonal Components (Fourier + Dummies)')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/seasonal_components_dummy.png')
        print("Seasonal components plot saved to figures/seasonal_components_dummy.png")
    
    return m_t, s_t, z_t, ols_model

def test_stationarity(series):
    """
    Perform the Augmented Dickey-Fuller (ADF) test to check for stationarity.
    """
    print("\n--- Stationarity Test (ADF) ---")
    result = adfuller(series.dropna())
    
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    if result[1] < 0.05:
        print("Result: The series is stationary (reject H0 at 5% level).")
    else:
        print("Result: The series is NOT stationary (fail to reject H0).")
    
    return result

def plot_acf_series(series, target_col, lags=100):
    """
    Plot the Autocorrelation Function (ACF) for the series.
    """
    print(f"Generating ACF plot for {target_col}...")
    plt.figure(figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=plt.gca())
    plt.title(f'ACF: {target_col} Residuals (Z_t)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = f'figures/acf_d_{target_col.lower()}.png'
    plt.savefig(plot_path)
    print(f"ACF plot saved to {plot_path}")

def plot_pacf_series(series, target_col, lags=100):
    """
    Plot the Partial Autocorrelation Function (PACF) for the series.
    """
    print(f"Generating PACF plot for {target_col}...")
    plt.figure(figsize=(12, 4))
    plot_pacf(series.dropna(), lags=lags, ax=plt.gca(), method='ywm')
    plt.title(f'PACF: {target_col} Residuals (Z_t)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = f'figures/pacf_d_{target_col.lower()}.png'
    plt.savefig(plot_path)
    print(f"PACF plot saved to {plot_path}")

def plot_arma_diagnostics(model, p, q, target_col):
    """
    Plot residuals and ACF/PACF of residuals for a fitted ARMA model.
    """
    resid = model.resid
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Residuals
    axes[0].plot(resid)
    axes[0].set_title(f'ARMA({p},{q}) Residuals: {target_col}')
    axes[0].grid(True, alpha=0.3)
    
    # 2. ACF of residuals
    plot_acf(resid, lags=50, ax=axes[1])
    axes[1].set_title('ACF of Residuals')
    axes[1].grid(True, alpha=0.3)
    
    # 3. PACF of residuals
    plot_pacf(resid, lags=50, ax=axes[2], method='ywm')
    axes[2].set_title('PACF of Residuals')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'figures/arma_{target_col.lower()}_{p}_{q}.png'
    plt.savefig(plot_path)
    plt.close() # Close to save memory during grid search
    print(f"ARMA({p},{q}) diagnostics saved to {plot_path}")

def fit_arma_grid_search(series, target_col, max_p=3, max_q=3):
    """
    Perform a grid search for ARMA(p, q) models and plot diagnostics for each.
    """
    print(f"\n--- ARMA Grid Search (p: 1-{max_p}, q: 1-{max_q}) ---")
    results = []
    
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                # Fit ARMA(p,q) model
                # trend='n' because we already removed trend m_t
                model = ARIMA(series, order=(p, 0, q), trend='n').fit(method_kwargs={'maxiter': 500})
                
                aic = model.aic
                bic = model.bic
                results.append({'p': p, 'q': q, 'AIC': aic, 'BIC': bic})
                
                # Plot diagnostics for this p, q
                plot_arma_diagnostics(model, p, q, target_col)
                
                print(f"ARMA({p},{q}): AIC = {aic:.2f}, BIC = {bic:.2f}")
                
            except Exception as e:
                print(f"Failed to fit ARMA({p},{q}): {e}")
                
    # Create summary table
    df_results = pd.DataFrame(results)
    print("\nGrid Search Summary:")
    print(df_results.to_string(index=False))
    
    # Identify best models
    best_aic = df_results.loc[df_results['AIC'].idxmin()]
    best_bic = df_results.loc[df_results['BIC'].idxmin()]
    
    print(f"\nBest model by AIC: ARMA({int(best_aic['p'])}, {int(best_aic['q'])}), AIC = {best_aic['AIC']:.2f}")
    print(f"Best model by BIC: ARMA({int(best_bic['p'])}, {int(best_bic['q'])}), BIC = {best_bic['BIC']:.2f}")
    
    return df_results

def generate_forecast(arma_model, ols_model, lmbda, last_date, start_date, h=30):
    """
    Generate h-step ahead forecast and transform back to original scale.
    """
    print(f"\n--- Generating {h}-day Forecast ---")
    
    # 1. Forecast ARMA component (Z_t)
    forecast_obj = arma_model.get_forecast(h)
    z_hat = forecast_obj.predicted_mean
    
    # 2. Forecast Deterministic component (m_t + s_t)
    # Create future time index
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=h)
    future_t = (future_dates - start_date).days.values
    
    X_future = pd.DataFrame(index=future_dates)
    X_future['const'] = 1
    X_future['t'] = future_t
    X_future['t2'] = future_t**2
    
    # Yearly Seasonality (Harmonics 1-9)
    for k in range(1, 10):
        X_future[f'cos_year_{k}'] = np.cos(2 * np.pi * k * future_t / 365.25)
        X_future[f'sin_year_{k}'] = np.sin(2 * np.pi * k * future_t / 365.25)
        
    # Weekly Seasonality (Harmonics 1-3)
    for k in range(1, 4):
        X_future[f'cos_week_{k}'] = np.cos(2 * np.pi * k * future_t / 7)
        X_future[f'sin_week_{k}'] = np.sin(2 * np.pi * k * future_t / 7)
        
    deterministic_hat = ols_model.predict(X_future)
    
    # 3. Combine components in Box-Cox domain (X_t)
    x_hat = deterministic_hat + z_hat.values
    
    # 4. Inverse Box-Cox to get Y_t
    y_hat = inv_boxcox(x_hat, lmbda)
    
    return pd.Series(y_hat, index=future_dates, name='Forecast')

def generate_forecast_with_ci(arma_model, ols_model, lmbda, last_date, start_date, h=30, alpha=0.05):
    """
    Generate h-step ahead forecast with confidence intervals.
    """
    print(f"\n--- Generating {h}-day Forecast with Confidence Intervals (alpha={alpha}) ---")
    
    # 1. Forecast ARMA component (Z_t) with confidence intervals
    forecast_obj = arma_model.get_forecast(h)
    z_hat = forecast_obj.predicted_mean
    z_ci = forecast_obj.conf_int(alpha=alpha)
    
    # 2. Forecast Deterministic component (m_t + s_t)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=h)
    future_t = (future_dates - start_date).days.values
    
    X_future = pd.DataFrame(index=future_dates)
    X_future['const'] = 1
    X_future['t'] = future_t
    X_future['t2'] = future_t**2
    
    for k in range(1, 10):
        X_future[f'cos_year_{k}'] = np.cos(2 * np.pi * k * future_t / 365.25)
        X_future[f'sin_year_{k}'] = np.sin(2 * np.pi * k * future_t / 365.25)
    for k in range(1, 4):
        X_future[f'cos_week_{k}'] = np.cos(2 * np.pi * k * future_t / 7)
        X_future[f'sin_week_{k}'] = np.sin(2 * np.pi * k * future_t / 7)
        
    deterministic_hat = ols_model.predict(X_future)
    
    # 3. Combine components and inverse transform
    # Mean Forecast
    x_hat = deterministic_hat + z_hat.values
    y_hat = inv_boxcox(x_hat, lmbda)
    
    # Lower and Upper Bounds
    # Note: These bounds are calculated in the Box-Cox domain and then transformed
    x_lower = deterministic_hat + z_ci.iloc[:, 0].values
    x_upper = deterministic_hat + z_ci.iloc[:, 1].values
    
    y_lower = inv_boxcox(x_lower, lmbda)
    y_upper = inv_boxcox(x_upper, lmbda)
    
    return (pd.Series(y_hat, index=future_dates, name='Forecast'),
            pd.Series(y_lower, index=future_dates, name='Lower CI'),
            pd.Series(y_upper, index=future_dates, name='Upper CI'))

def main():
    # 0. Setup directories
    os.makedirs('figures', exist_ok=True)
    
    # 1. Load Data
    data = load_daily_data()
    
    start_date = data.index[0]
    last_date = data.index[-1]
    
    # Target Variable
    target_col = 'SE3'
    y_t = data[target_col]
    
    print(f"\nTarget Variable (Y_t): {target_col}")
    
    # 2. Apply Box-Cox Transformation to acquire X_t
    x_t, lmbda = apply_box_cox(y_t)
    
    # 3. Decompose X_t (Remove Trend m_t and Seasonality s_t)
    #m_t, s_t, z_t, ols_model = decompose_series(x_t, plotting=True)
    m_t, s_t, z_t, ols_model = decompose_dummy(x_t, plotting=True)
    
    # 4. Stationarity Test
    test_stationarity(z_t)
    plot_acf_series(z_t, target_col)
    plot_pacf_series(z_t, target_col)
    
    # 5. ARMA Modeling
    # You can perform a grid search to find optimal p, q
    fit_arma_grid_search(z_t, target_col, max_p=3, max_q=3)
    
    # Manually selected p, q based on grid search or ACF/PACF analysis
    p_selected = 2
    q_selected = 2
    print(f"\nFitting selected model: ARMA({p_selected}, {q_selected}) with increased iterations...")
    arma_model = ARIMA(z_t, order=(p_selected, 0, q_selected), trend='n').fit(method_kwargs={'maxiter': 500})
    
    # 6. Forecasting
    h_days = 365 # Forecast one year ahead
    y_forecast = generate_forecast(arma_model, ols_model, lmbda, last_date, start_date, h=h_days)
    
    # 6b. Forecasting with Confidence Intervals
    y_f_ci, y_low, y_high = generate_forecast_with_ci(arma_model, ols_model, lmbda, last_date, start_date, h=h_days)
    
    # 7. Visualization (Point Forecast)
    plt.figure(figsize=(12, 6))
    plt.plot(y_t.index[-730:], y_t.iloc[-730:], label='Historical Data', color='blue', alpha=0.6)
    plt.plot(y_forecast.index, y_forecast, label=f'Forecast (h={h_days})', color='red', linestyle='--')
    plt.title(f'Forecast ({target_col}): ARMA({p_selected},{q_selected})')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f'figures/forecast_d_{target_col.lower()}.png')
    
    # 8. Visualization (Forecast with CI)
    plt.figure(figsize=(12, 6))
    plt.plot(y_t.index[-730:], y_t.iloc[-730:], label='Historical Data', color='blue', alpha=0.6)
    plt.plot(y_f_ci.index, y_f_ci, label='Forecast', color='red')
    plt.fill_between(y_f_ci.index, y_low, y_high, color='red', alpha=0.2, label='95% Confidence Interval')
    plt.title(f'Forecast with Uncertainty ({target_col}): ARMA({p_selected},{q_selected})')
    plt.legend(); plt.grid(True, alpha=0.3)
    
    plot_path_ci = f'figures/forecast_ci_d_{target_col.lower()}.png'
    plt.savefig(plot_path_ci)
    print(f"\nConfidence Interval forecast plot saved to {plot_path_ci}")
    
    # Secondary plot: Decomposition components for checking
    # Specify the desired range (e.g., last 730 days). Comment out/change to slice(None) for full range.
    plot_range = slice(-500, None)
    # plot_range = slice(None)

    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(y_t.index[plot_range], y_t.iloc[plot_range], color='blue', alpha=0.6)
    plt.title(f'Original Series (Y_t): {target_col}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 2)
    plt.plot(x_t.index[plot_range], x_t.iloc[plot_range], color='green', alpha=0.4, label='X_t (Box-Cox)')
    plt.plot(m_t.index[plot_range], m_t.iloc[plot_range], color='red', linewidth=2, label='Trend (m_t)')
    plt.title(f'Transformed Series (X_t) and Trend, Lambda = {lmbda:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 3)
    plt.plot(s_t.index[plot_range], s_t.iloc[plot_range], color='orange')
    plt.title('Seasonality (s_t)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 4)
    plt.plot(z_t.index[plot_range], z_t.iloc[plot_range], color='black')
    plt.title('Residuals (Z_t)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    decomp_plot_path = f'figures/decomposition_d_{target_col.lower()}.png'
    plt.savefig(decomp_plot_path)
    print(f"Decomposition plots updated (range: {plot_range}) in {decomp_plot_path}")

if __name__ == "__main__":
    main()

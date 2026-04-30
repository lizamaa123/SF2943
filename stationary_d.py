import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path='data/se_d_data.csv', target_col='SE3'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
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

def run_stationarity_tests(z_t):
    """Run formal statistical tests for stationarity: ADF and KPSS."""
    print("\n" + "="*50)
    print("STATIONARITY TESTS")
    print("="*50)
    
    # 1. Augmented Dickey-Fuller Test
    print("\n--- 1. Augmented Dickey-Fuller (ADF) Test ---")
    print("H0: The series has a unit root (non-stationary)")
    print("H1: The series is strictly stationary")
    adf_result = adfuller(z_t.dropna())
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4e}")
    for key, value in adf_result[4].items():
        print(f"Critical Value ({key}): {value:.4f}")
    if adf_result[1] < 0.05:
        print("--> Conclusion: Reject H0. The series is STATIONARY.")
    else:
        print("--> Conclusion: Fail to reject H0. The series is NON-STATIONARY.")

    # 2. KPSS Test
    print("\n--- 2. Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test ---")
    print("H0: The series is level or trend stationary")
    print("H1: The series has a unit root (non-stationary)")
    kpss_result = kpss(z_t.dropna(), regression='c', nlags="auto")
    print(f"KPSS Statistic: {kpss_result[0]:.4f}")
    print(f"p-value: {kpss_result[1]:.4f}")
    for key, value in kpss_result[3].items():
        print(f"Critical Value ({key}): {value:.4f}")
    if kpss_result[1] < 0.05:
        print("--> Conclusion: Reject H0. The series is NON-STATIONARY.")
    else:
        print("--> Conclusion: Fail to reject H0. The series is STATIONARY.")
        
    print("\n" + "-"*50)
    print("Overall Robust Conclusion:")
    if adf_result[1] < 0.05 and kpss_result[1] >= 0.05:
        print("Both tests agree: The series is strictly STATIONARY.")
    elif adf_result[1] >= 0.05 and kpss_result[1] < 0.05:
        print("Both tests agree: The series is NON-STATIONARY.")
    else:
        print("Tests conflict. The series may be fractionally integrated or have structural breaks.")

def plot_diagnostics(z_t):
    """Plot time series, ACF, and PACF for visual stationarity inspection."""
    os.makedirs('figures', exist_ok=True)
    
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Time Series Plot
    ax1 = plt.subplot(211)
    ax1.plot(z_t.index, z_t.values, color='blue', alpha=0.7, linewidth=0.5)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_title('Residuals $Z_t$', fontsize=14)
    ax1.set_ylabel('Residual Value')
    
    # 2. ACF Plot
    ax2 = plt.subplot(223)
    plot_acf(z_t.dropna(), lags=40, ax=ax2, alpha=0.05)
    ax2.set_title('ACF', fontsize=12)
    
    # 3. PACF Plot
    ax3 = plt.subplot(224)
    plot_pacf(z_t.dropna(), lags=40, ax=ax3, method='ywm', alpha=0.05)
    ax3.set_title('PACF', fontsize=12)
    
    plt.tight_layout()
    plot_path = 'figures/stationary_d_diagnostics.png'
    plt.savefig(plot_path)
    print(f"\nDiagnostic plots saved to {plot_path}")

def main():
    print("Loading historical data...")
    y_full = load_data()
    
    # Truncate at 2026-04-01
    y_t = y_full[y_full.index <= '2026-04-01']
    
    print(f"Data truncated at 2026-04-01. Total observations: {len(y_t)}")
    
    print("Applying Log Transformation...")
    x_t = np.log(y_t)
    
    print("Fitting Deterministic Decomposition (Quadratic, Yearly K=7, Weekly K=3)...")
    _, z_t = fit_deterministic(x_t)
    
    run_stationarity_tests(z_t)
    plot_diagnostics(z_t)

if __name__ == '__main__':
    main()

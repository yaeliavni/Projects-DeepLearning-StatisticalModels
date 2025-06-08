import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pmdarima.arima import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit

# --- ADF Test Function ---
def adf_test(ts, label):
    result = adfuller(ts.dropna())
    print(f"\nADF Test on {label}:")
    print(f"  Test Statistic = {result[0]:.4f}")
    print(f"  p-value        = {result[1]:.4f}")
    crit = result[4] if len(result) > 4 and isinstance(result[4], dict) else {}
    if crit:
        print("  Critical Values:")
        for k, v in crit.items():
            print(f"     {k}: {v:.4f}")
    print("  => Stationary" if result[1] < 0.05 else "  => Not stationary")

# --- 1. Download & prepare raw Close price as a Series ---
goog   = yf.download("GOOG", start="2015-01-01", end="2016-01-01")
series = goog["Close"].dropna().asfreq("B").ffill()

# --- 2. ACF/PACF diagnostics ---
for title, data in [
    ("Google Close Price", series),
    ("1st Difference",     series.diff().dropna()),
    ("Log-Difference",     np.log(series).diff().dropna()),
]:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].plot(data); axes[0].set_title(title); axes[0].grid(True)
    plot_acf(data,  lags=30, ax=axes[1]); axes[1].set_title(f"ACF ({title})")
    plot_pacf(data, lags=30, method="ywm", ax=axes[2]); axes[2].set_title(f"PACF ({title})")
    plt.tight_layout(); plt.show()

# --- 3. TimeSeriesSplit + per-day MAPE via pandas ---
tscv      = TimeSeriesSplit(n_splits=5)
mape_list = []
series = goog["Close"].dropna().asfreq("B").ffill()

for fold, (tr_idx, te_idx) in enumerate(tscv.split(series), start=1):
    train = series.iloc[tr_idx]
    test  = series.iloc[te_idx]

    # If test is accidentally a DataFrame, make it a Series
    if isinstance(test, pd.DataFrame):
        test = test.iloc[:, 0]

    model = ARIMA(
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        suppress_warnings=True,
        with_intercept=True,
    ).fit(train)

    fc_vals, _ = model.predict(n_periods=len(test), return_conf_int=True)
    forecast    = pd.Series(fc_vals, index=test.index)

    mape = (test - forecast).abs().div(test).mul(100).dropna()
    mape_list.append(mape)

    if not mape.empty:
        mean_mape = mape.mean()
        print(f"Fold {fold} mean MAPE: {mean_mape:.2f}%")
    else:
        print(f"Fold {fold} mean MAPE: NaN (no valid forecasts for this fold)")

# --- 4. Concatenate & plot daily MAPE ---
all_mape = pd.concat(mape_list).sort_index()
plt.figure(figsize=(12, 5))
plt.plot(all_mape.index, all_mape.values, marker='o', linestyle='-')
plt.title("Daily MAPE Across All CV Folds (ARIMA(1,1,1))")
plt.xlabel("Date"); plt.ylabel("MAPE (%)")
plt.grid(True); plt.tight_layout(); plt.show()
# --- 4b. ARIMA(1,1,1) Forecast Plot with 95% CI ---

# Pick a test window for holdout forecast (e.g. last 30 business days)
n_test = 30
train = series.iloc[:-n_test]
test  = series.iloc[-n_test:]

model = ARIMA(
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),
    suppress_warnings=True,
    with_intercept=True,
).fit(train)

# Forecast with 95% confidence interval
forecast, conf_int = model.predict(n_periods=n_test, return_conf_int=True, alpha=0.05)
forecast = pd.Series(forecast, index=test.index)
conf_int = pd.DataFrame(conf_int, index=test.index, columns=["lower", "upper"])

plt.figure(figsize=(12, 5))
plt.plot(train, label="Train", color='tab:blue')
plt.plot(test, label="Test", color='orange')
plt.plot(forecast, "--", label="Forecast", color='green')
plt.fill_between(
    test.index,
    conf_int["lower"],
    conf_int["upper"],
    color="pink",
    alpha=0.4,
    label="95% CI"
)
plt.title("ARIMA(1,1,1) Forecast with 95% CI")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 5. Fit full-series ARIMA + residual diagnostics ---
full_model = ARIMA(
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),
    suppress_warnings=True,
    with_intercept=True,
).fit(series)

resid = full_model.arima_res_.resid
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].plot(resid);           axes[0,0].set_title("Residuals");   axes[0,0].grid(True)
axes[0,1].hist(resid, bins=20);  axes[0,1].set_title("Histogram")
plot_acf(resid, lags=30, ax=axes[1,1]); axes[1,1].set_title("ACF (Residuals)")
plt.tight_layout(); plt.show()

# --- 6. ADF Tests on full series ---
adf_test(series,               "Original Series")
adf_test(np.log(series).diff(), "Log-Differenced Series")

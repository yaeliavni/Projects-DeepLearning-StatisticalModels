import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import zscore, probplot
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

data_path = r"C:\Users\avni1\Documents\SeniorYear\ML\midtermproject\pythonProject\Nowcasting\MadadMardData.csv"
raw_data = pd.read_csv(data_path)

#check current columns
print("Columns loaded:", raw_data.columns)

#using only the relevant columns for our models to predict
raw_data = raw_data[['credit', 'credit_time', 'madad']]


raw_data['credit_time'] = pd.to_numeric(raw_data['credit_time'], errors='coerce')
raw_data['credit_time'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(raw_data['credit_time'], unit='D')
credit_series = raw_data[['credit', 'credit_time']].dropna()
credit_series.set_index('credit_time', inplace=True)

madad_series = raw_data[['madad']].dropna().reset_index(drop=True)
madad_start = pd.to_datetime('2004-01-01')
madad_index = pd.date_range(start=madad_start, periods=len(madad_series), freq='MS')
madad_series.index = madad_index

# Differencing Madad
diff_madad = madad_series.diff().dropna()
def log_diff(df):
    return np.log(df).diff().dropna()

def zscore_diff(df):
    diffed = df.diff().dropna()
    try:
        return diffed.apply(zscore)
    except ValueError as e:
        print(f"Z-score computation failed: {e}")
        return pd.DataFrame(columns=df.columns, index=df.index)

# Apply normalizations
log_diff_credit = log_diff(credit_series)
zscore_credit = zscore_diff(credit_series)
log_diff_madad = log_diff(madad_series)
zscore_madad = zscore_diff(madad_series)

# ============== SCATTER PLOTS ============== #

def scatter_plot(x, y, title):
    plt.figure(figsize=(12,6))
    sns.scatterplot(x=x, y=y)
    plt.title(title)
    plt.xlabel('Credit')
    plt.ylabel('Madad')
    plt.show()

# Scatter plots for each normalization
scatter_plot(credit_series['credit'], madad_series['madad'], "Original Data Scatterplot")
scatter_plot(log_diff_credit['credit'], log_diff_madad['madad'], "Log Differenced Data Scatterplot")
scatter_plot(zscore_credit['credit'], zscore_madad['madad'], "Z-score Differenced Data Scatterplot")

# ---------------------------- Stationarity Tests ---------------------------- #

def adf_test(series, title='ADF Test'):
    print(f"\n{title}")
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    for key, value in result[4].items():
        print('Critical Value (%s): %.3f' % (key, value))

def kpss_test(series, title='KPSS Test'):
    print(f"\n{title}")
    result = kpss(series.dropna(), regression='c', nlags='auto')
    print(f"KPSS Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    for key, value in result[3].items():
        print('Critical Value (%s): %.3f' % (key, value))

# Run Stationarity Tests
adf_test(madad_series['madad'], title='ADF Test on Madad')
kpss_test(madad_series['madad'], title='KPSS Test on Madad')
adf_test(diff_madad['madad'], title='ADF Test on Differenced Madad')
kpss_test(diff_madad['madad'], title='KPSS Test on Differenced Madad')

# ---------------------------- Seasonal Decomposition ---------------------------- #

if madad_series['madad'].dropna().shape[0] >= 24:
    result = seasonal_decompose(madad_series['madad'].dropna(), model='additive', period=12)
    result.plot()
    plt.show()
else:
    print("Not enough data points for seasonal decomposition.")

# ---------------------------- SARIMAX Modeling ---------------------------- #

model = sm.tsa.statespace.SARIMAX(madad_series['madad'], order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)
print(model_fit.summary())

# Residuals plots
residuals = model_fit.resid
plt.figure(figsize=(10,5))
plt.plot(residuals)
plt.title('Residuals over Time')
plt.show()

plot_acf(residuals.dropna(), lags=40)
plt.title('ACF of Residuals')
plt.show()

plot_pacf(residuals.dropna(), lags=40)
plt.title('PACF of Residuals')
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(residuals.dropna(), kde=True)
plt.title('Histogram of Residuals')
plt.show()

plt.figure(figsize=(8,8))
probplot(residuals.dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()

# ---------------------------- Prophet Modeling ---------------------------- #

prophet_df = madad_series.reset_index()
prophet_df.columns = ['ds', 'y']
prophet = Prophet()
prophet.fit(prophet_df)
future = prophet.make_future_dataframe(periods=12, freq='MS')
forecast = prophet.predict(future)
prophet.plot(forecast)
plt.title('Prophet Forecast')
plt.show()

# ---------------------------- XGBoost Modeling ---------------------------- #

# Create lag features
xgb_data = madad_series.copy()
xgb_data['lag1'] = xgb_data['madad'].shift(1)
xgb_data['lag2'] = xgb_data['madad'].shift(2)
xgb_data.dropna(inplace=True)

X = xgb_data[['lag1', 'lag2']]
y = xgb_data['madad']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model_xgb = XGBRegressor()
model_xgb.fit(X_train, y_train)

preds = model_xgb.predict(X_test)
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label='True')
plt.plot(y_test.index, preds, label='Predicted')
plt.legend()
plt.title('XGBoost Predictions')
plt.show()

# ---------------------------- LSTM Modeling ---------------------------- #

scaler = MinMaxScaler()
madad_scaled = scaler.fit_transform(madad_series.values)

X_lstm = []
y_lstm = []
for i in range(3, len(madad_scaled)):
    X_lstm.append(madad_scaled[i-3:i, 0])
    y_lstm.append(madad_scaled[i, 0])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(X_lstm.shape[1], 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

model_lstm.fit(X_lstm, y_lstm, epochs=50, batch_size=16, verbose=1)

lstm_preds = model_lstm.predict(X_lstm)

plt.figure(figsize=(10,5))
plt.plot(madad_series.index[3:], scaler.inverse_transform(lstm_preds), label='Predicted')
plt.plot(madad_series.index[3:], scaler.inverse_transform(y_lstm.reshape(-1,1)), label='True')
plt.legend()
plt.title('LSTM Predictions')
plt.show()
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --- Prophet Evaluation ---
actual_prophet = prophet_df['y']
predicted_prophet = forecast.loc[forecast['ds'].isin(prophet_df['ds']), 'yhat']

rmse_prophet = np.sqrt(mean_squared_error(actual_prophet, predicted_prophet))
mae_prophet = mean_absolute_error(actual_prophet, predicted_prophet)
mape_prophet = mean_absolute_percentage_error(actual_prophet, predicted_prophet)

print("\n📈 Prophet Performance:")
print(f"RMSE: {rmse_prophet:.3f}")
print(f"MAE:  {mae_prophet:.3f}")
print(f"MAPE: {mape_prophet:.2f}%")

# --- XGBoost Evaluation ---
rmse_xgb = np.sqrt(mean_squared_error(y_test, preds))
mae_xgb = mean_absolute_error(y_test, preds)
mape_xgb = mean_absolute_percentage_error(y_test, preds)

print("\n📉 XGBoost Performance:")
print(f"RMSE: {rmse_xgb:.3f}")
print(f"MAE:  {mae_xgb:.3f}")
print(f"MAPE: {mape_xgb:.2f}%")

# --- LSTM Evaluation ---
# Note: y_lstm and lstm_preds are scaled, so we inverse transform them for proper evaluation
y_true_lstm = scaler.inverse_transform(y_lstm.reshape(-1,1))
y_pred_lstm = scaler.inverse_transform(lstm_preds)

rmse_lstm = np.sqrt(mean_squared_error(y_true_lstm, y_pred_lstm))
mae_lstm = mean_absolute_error(y_true_lstm, y_pred_lstm)
mape_lstm = mean_absolute_percentage_error(y_true_lstm, y_pred_lstm)

print("\n🤖 LSTM Performance:")
print(f"RMSE: {rmse_lstm:.3f}")
print(f"MAE:  {mae_lstm:.3f}")
print(f"MAPE: {mape_lstm:.2f}%")

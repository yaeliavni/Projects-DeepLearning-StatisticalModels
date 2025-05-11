""" Python pipeline to compare four forecasting methods on the monthly “madad” series:
- SARIMAX (seasonal ARIMA)
- Prophet (Facebook’s decomposable time-series model)
- LSTM (deep-learning recurrent network)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prophet import Prophet
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class DataHandler:
    def __init__(self, path):
        self.raw = pd.read_csv(path)

    def load(self):
        df = self.raw[['credit','credit_time','madad']].copy()
        df['credit_time'] = pd.to_numeric(df['credit_time'], errors='coerce')
        df['credit_time'] = pd.to_datetime('1899-12-30') + \
                            pd.to_timedelta(df['credit_time'], unit='D')
        credit = df[['credit','credit_time']].dropna().set_index('credit_time')
        madad = df[['madad']].dropna().reset_index(drop=True)
        madad.index = pd.date_range('2004-01-01', periods=len(madad), freq='MS')
        return credit, madad['madad']


class SarimaxModel:
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.order, self.sorder = order, seasonal_order
        self.fitted = None

    def fit(self, series):
        m = sm.tsa.statespace.SARIMAX(series,
                                      order=self.order,
                                      seasonal_order=self.sorder)
        self.fitted = m.fit(disp=False)
        return self.fitted

    def summary(self):
        return self.fitted.summary()

class ProphetModel:
    def __init__(self,
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0):
        self.model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                             seasonality_prior_scale=seasonality_prior_scale)
        self.forecast = None

    def fit(self, series, periods: int = 12):
        df = series.reset_index()
        df.columns = ['ds','y']
        self.model.fit(df)
        future = self.model.make_future_dataframe(periods=periods, freq='MS')
        self.forecast = self.model.predict(future)
        return self.forecast

    def plot(self):
        fig = self.model.plot(self.forecast)
        plt.title('Prophet Forecast')
        plt.show()

class TimeSeriesPipeline:
    def __init__(self, path):
        self.credit, self.madad = DataHandler(path).load()

    def run(self):
        # SARIMAX
        sar = SarimaxModel()
        sar.fit(self.madad)
        print(sar.summary())
        plt.figure()
        plt.plot(self.madad.index, self.madad.values, label='Actual')
        plt.plot(self.madad.index, sar.fitted.fittedvalues, label='Fitted')
        plt.title('SARIMAX In-Sample Fit')
        plt.legend()
        plt.show()

        # PROPHET
        prop = ProphetModel()
        fc   = prop.fit(self.madad)
        prop.plot()



        # LSTM (standalone snippet)
        madad_series = self.madad
        scaler = MinMaxScaler()
        madad_scaled = scaler.fit_transform(madad_series.values.reshape(-1,1))

        X_lstm, y_lstm = [], []
        for i in range(3, len(madad_scaled)):
            X_lstm.append(madad_scaled[i-3:i, 0])
            y_lstm.append(madad_scaled[i, 0])
        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

        model_lstm = Sequential([
            LSTM(50, activation='relu', input_shape=(X_lstm.shape[1], 1)),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mse')
        model_lstm.fit(X_lstm, y_lstm, epochs=50, batch_size=16, verbose=1)

        lstm_preds = model_lstm.predict(X_lstm)
        lstm_preds_inv = scaler.inverse_transform(lstm_preds)
        y_lstm_inv    = scaler.inverse_transform(y_lstm.reshape(-1,1))

        print(">>> [LSTM] first 5 y_true:",   y_lstm_inv[:5].flatten().tolist())
        print(">>> [LSTM] first 5 preds:",    lstm_preds_inv[:5].flatten().tolist())

        plt.figure(figsize=(10,5))
        plt.plot(madad_series.index[3:], lstm_preds_inv, marker='.', label='Predicted')
        plt.plot(madad_series.index[3:], y_lstm_inv,    marker='.', label='True')
        plt.title('LSTM Predictions')
        plt.legend()
        plt.show()

        # EVALUATION
        def evaluate(y_true, y_pred, label):
            rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
            mae      = mean_absolute_error(y_true, y_pred)
            mape_val = mean_absolute_percentage_error(y_true, y_pred)
            print(f"\n{label} Performance:")
            print(f"  RMSE:  {rmse:.3f}")
            print(f"  MAE:   {mae:.3f}")
            print(f"  MAPE:  {mape_val:.2f}%")

        evaluate(y_lstm_inv.flatten(), lstm_preds_inv.flatten(), "LSTM")

if __name__ == "__main__":
    pipeline = TimeSeriesPipeline(
        r"C:\Users\avni1\Documents\SeniorYear\ML\midtermproject\pythonProject\Nowcasting\MadadMardData.csv"
    )
    pipeline.run()

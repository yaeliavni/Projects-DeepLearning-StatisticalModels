# Time Series Forecasting Pipeline Nowcasting

A unified Python pipeline to compare four forecasting methods on the monthly **madad** series:

- **SARIMAX** (Seasonal ARIMA)  
- **Prophet** (Facebook’s decomposable time-series model)  
- **LSTM** (Deep-learning recurrent network)

---

##  Prerequisites

- Python 3.7+  
- Install required packages:
  ```bash
  pip install pandas numpy matplotlib
  pip install statsmodels prophet
  pip install scikit-learn xgboost
  pip install tensorflow keras
  ```

---

##  Repository Structure

```
.
├── README.md
└── MadadPrediction.py    
```

- **MadadPrediction.py**  
  Contains all data-loading, modeling and evaluation code.

---

##  Getting Started

1. **Clone** the repository:
   ```
   https://github.com/yaeliavni/Madad-Prediction/tree/master/Nowcasting
   ```

2. **Install** dependencies (see Prerequisites).

3. **Run** the pipeline:
   ```bash
   python MadadPrediction.py
   ```
   - **SARIMAX** summary and in-sample fit plot  
   - **Prophet** forecast plot  
   - **LSTM** predictions vs. true values plot  
   - Printed evaluation metrics (RMSE, MAE, MAPE)

---

##  Code Overview

### DataHandler  
- Loads `MadadMardData.csv`  
- Converts Excel-style dates to pandas `DatetimeIndex`  
- Returns two series:  
  - `credit` (indexed by date)  
  - `madad` (monthly target series)

### SarimaxModel  
- Parameters: `order=(p,d,q)`, `seasonal_order=(P,D,Q,12)`  
- Fits a seasonal ARIMA using `statsmodels`  
- Provides `.summary()` and in-sample fit plotting  

### ProphetModel  
- Configurable changepoint and seasonality priors  
- Fits Facebook Prophet on the `madad` series  
- Forecasts 12 months ahead by default  
- Built-in `.plot()` for decomposed forecast

### LSTM (Standalone Snippet)  
- Scales data with `MinMaxScaler`  
- Builds a sliding window of 3 timesteps → 1 prediction  
- 50-unit LSTM + dense output, trained for 50 epochs  
- Inverse-scales predictions for plotting  

### Evaluation  
- **RMSE**, **MAE**, **MAPE** computed and printed for each model  

---

## Results & Interpretation

- **SARIMAX** captures trend + seasonality in-sample  
- **Prophet** handles changepoints and missing data robustly  
- **LSTM** learns nonlinear temporal patterns but requires tuning  

---

---


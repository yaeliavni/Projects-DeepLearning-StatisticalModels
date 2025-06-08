#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as sp
from itertools import cycle
import warnings

# modeling
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

warnings.filterwarnings("ignore")

# ─── 1) LOAD RAW DATA ─────────────────────────────────────────────────────────
df = pd.read_csv('price_data_multicurrency.csv')
df['TRADE_TIMESTAMP'] = pd.to_datetime(df['TRADE_TIMESTAMP'], errors='coerce')

# ─── 2) RANDOM BID TIME & PRICE ────────────────────────────────────────────────
random_price = round(random.uniform(0.02, 0.1), 4)
t0, t1 = datetime(2018,1,1), datetime(2018,12,31)
rand_secs   = random.randint(0, int((t1 - t0).total_seconds()))
random_time = t0 + timedelta(seconds=rand_secs)
print("random cutoff time:", random_time)

# ─── 3) SPLIT PAST vs FUTURE ───────────────────────────────────────────────────
df_past   = df[df.TRADE_TIMESTAMP <  random_time].copy()
df_future = df[df.TRADE_TIMESTAMP >= random_time].copy()

# ─── 4) FEATURE ENGINEERING (SEPARATELY) ───────────────────────────────────────
def engineer_features(df_sub):
    df_sub = df_sub.sort_values('TRADE_TIMESTAMP').copy()
    df_sub['PRICE_LOG']  = np.log(df_sub['PRICE'])
    df_sub['PRICE_DIFF'] = df_sub['PRICE'].diff().fillna(0)
    df_sub['HOUR']       = df_sub['TRADE_TIMESTAMP'].dt.hour
    df_sub['DAY_OF_WEEK']= df_sub['TRADE_TIMESTAMP'].dt.dayofweek
    df_sub['MONTH']      = df_sub['TRADE_TIMESTAMP'].dt.month
    return df_sub

df_past   = engineer_features(df_past)
df_future = engineer_features(df_future)

# ─── 5) PREPARE MATRICES ───────────────────────────────────────────────────────
features = ['PRICE','PRICE_LOG','PRICE_DIFF','HOUR','DAY_OF_WEEK','MONTH']
le       = LabelEncoder()

X_past = df_past[features]
y_past = le.fit_transform(df_past['CURRENCY'])

X_test = df_future[features]
y_test = le.transform(df_future['CURRENCY'])

# ─── 6) CHRONOLOGICAL TRAIN/VAL SPLIT ON PAST ─────────────────────────────────
split_idx = int(len(X_past) * 0.8)
X_train   = X_past.iloc[:split_idx].reset_index(drop=True)
y_train   = y_past[:split_idx]
X_val     = X_past.iloc[split_idx:].reset_index(drop=True)
y_val     = y_past[split_idx:]

# ─── 7) SCALE FEATURES ─────────────────────────────────────────────────────────
scaler           = StandardScaler()
X_train_scaled   = scaler.fit_transform(X_train)
X_val_scaled     = scaler.transform(X_val)
X_test_scaled    = scaler.transform(X_test)

# ─── 8) BASELINE MODEL ─────────────────────────────────────────────────────────
clf_base = GradientBoostingClassifier(random_state=42)
clf_base.fit(X_train_scaled, y_train)

y_val_prob = clf_base.predict_proba(X_val_scaled)
y_val_bin  = label_binarize(y_val, classes=np.unique(y_past))
print("Baseline Val AUC:", roc_auc_score(y_val_bin, y_val_prob, multi_class='ovr'))

# ─── 9) HYPERPARAMETER TUNING ─────────────────────────────────────────────────
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth':    [None, 10, 20]
}
from sklearn.pipeline import Pipeline
search = GridSearchCV(
    Pipeline([('clf', GradientBoostingClassifier(random_state=42))]),
    param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='roc_auc_ovr',
    n_jobs=-1
)
search.fit(X_train_scaled, y_train)

best_clf = search.best_estimator_
print("Tuned params:", search.best_params_)

y_val_prob_tuned = best_clf.predict_proba(X_val_scaled)
print("Tuned Val AUC:", roc_auc_score(y_val_bin, y_val_prob_tuned, multi_class='ovr'))

# ─── 10) FINAL EVAL ON TEST ────────────────────────────────────────────────────
y_test_prob = best_clf.predict_proba(X_test_scaled)
y_test_bin  = label_binarize(y_test, classes=np.unique(y_test))
print("Test AUC:", roc_auc_score(y_test_bin, y_test_prob, multi_class='ovr'))

y_test_pred = best_clf.predict(X_test_scaled)
print("Classification Report:\n", classification_report(y_test, y_test_pred, target_names=le.classes_))

# ─── 11) ROC PLOT ──────────────────────────────────────────────────────────────
plt.figure(figsize=(7,5))
for i, lbl in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:,i], y_test_prob[:,i])
    plt.plot(fpr, tpr, label=f"{lbl} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curves — Test Set")
plt.xlabel("False Pos Rate"); plt.ylabel("True Pos Rate")
plt.legend(); plt.show()

# ─── 12) PREDICT FUNCTION ─────────────────────────────────────────────────────
def predict_currency(price_val, trade_time_str):
    """
    price_val      : float or int
    trade_time_str : string or datetime-like
    Returns (currency, confidence)
    """
    tm = pd.to_datetime(trade_time_str)
    df_feat = pd.DataFrame([{
        'PRICE'     : price_val,
        'PRICE_LOG' : np.log(price_val),
        'PRICE_DIFF': 0.0,
        'HOUR'      : tm.hour,
        'DAY_OF_WEEK':tm.dayofweek,
        'MONTH'     : tm.month
    }])
    fs = scaler.transform(df_feat)
    idx = best_clf.predict(fs)[0]
    proba = best_clf.predict_proba(fs)[0][idx]
    return le.inverse_transform([idx])[0], proba

# ─── 13) TEST PREDICTION ───────────────────────────────────────────────────────
print(f"random time: {random_time}, random_price: {random_price}")
pred_cur, pred_conf = predict_currency(random_price, random_time)
print(f"Predicted: {pred_cur}, confidence: {pred_conf:.2%}")

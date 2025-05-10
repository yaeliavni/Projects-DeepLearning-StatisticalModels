import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
import shap
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.base import clone  #for cloning the tuned RF in cross-validation
from itertools import cycle

warnings.filterwarnings("ignore")

df = pd.read_csv(r"C:\Users\avni1\Documents\SeniorYear\ML\midtermproject\pythonProject\Project\price_data_multicurrency.csv")
# invalid parses become NaT
df['TRADE_TIMESTAMP'] = pd.to_datetime(df['TRADE_TIMESTAMP'], errors='coerce')
df['PRICE_LOG']   = np.log(df['PRICE'])                     # log of price
df['PRICE_DIFF']  = df['PRICE'].diff().fillna(0)             # difference from previous price
df['HOUR']        = df['TRADE_TIMESTAMP'].dt.hour            # hour of trade
df['DAY_OF_WEEK'] = df['TRADE_TIMESTAMP'].dt.dayofweek       # 0=Monday … 6=Sunday
df['MONTH']       = df['TRADE_TIMESTAMP'].dt.month           # month number

print("Shape:", df.shape)
print(df.describe())
print("Null counts:\n", df.isnull().sum())
sb.pairplot(df[['PRICE','PRICE_LOG','PRICE_DIFF','HOUR','DAY_OF_WEEK','MONTH']])
plt.show()

""" From this plot we can already see several things:
1. Price vs. log-price:
    -No simple linear relationship between raw price and its logarithm.
    -Price differences and log-price are strongly related.
2. Time features:
    -No strong overall correlation between time-of-day (or day-of-week/month) and price.
    -Repeated peaks at certain hours (likely market open/close) suggest a possible cyclical effect.
3. Price differences:
    -Roughly symmetric around zero and stationary (mean and variance stay constant over time).
    -Ideal for many machine-learning models because:
        Centered data helps algorithms converge faster and improves numerical stability,
        Symmetric, Gaussian-like features work well with methods that assume normality (e.g., SVM, logistic regression,
        Stationarity speeds up training and makes results more reliable.
4. Encoding time:
    -Hour/day/month are categorical, so you’ll need to one-hot-encode them or use a model that handles categoricals directly 
        (e.g., Random Forest).
5. Trading hours:
    -Certain hours show especially high activity, so later we can analyze how price-move probabilities vary by hour."""

# Class balance
sb.countplot(x='CURRENCY', data=df)
plt.title('Class Balance')
plt.show()

#here we see that most samples come from EUR so probably EUR will dominate initially

#feature prep
features = ['PRICE','PRICE_LOG','PRICE_DIFF','HOUR','DAY_OF_WEEK','MONTH']
X = df[features]
y = df['CURRENCY']

#string lables into int
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# binarize for multiclass ROC
y_bin = label_binarize(y_enc, classes=np.unique(y_enc))

# splitting our data, preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# z score scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# using IQR for outliers handleing on numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# calc z-scores
z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))

# setting threshold and find outliers
threshold = 3
outlier_mask = (z_scores > threshold)
outliers = numeric_df[(outlier_mask).any(axis=1)]

# if we wish to remove outliers just in case, but we don't
#df_cleaned = df.loc[~outlier_mask.any(axis=1)].copy()

# Summary of outliers
print("Total rows in dataset:", len(df))
print("Rows with at least one outlier:", len(outliers))
print(f"Percentage of outlier rows: {100 * len(outliers) / len(df):.2f}%")

# Show the outliers
print("\nSample of outliers:")
print(outliers.head())
# we see that there aren't any outliers in the data that can actually impact our results since they don't appear.
# boxplots before/after
fig, axs = plt.subplots(1, 2, figsize=(12,5))
sb.boxplot(y=df['PRICE'], ax=axs[0]); axs[0].set_title("Before Outlier Removal")
sb.boxplot(y=df['PRICE'], ax=axs[1]); axs[1].set_title("After Outlier Removal")
plt.tight_layout(); plt.show()

#histograms of all numeric cols
numeric_cols = df.select_dtypes(include='number').columns
plt.figure(figsize=(15,10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot((len(numeric_cols)+1)//2, 2, i)
    sb.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col); plt.ylabel("Frequency")
plt.tight_layout(); plt.show()

"""we see that the price distribution is slightly skewed, which makes sense to normalize it using a log transform (which we already did)
The log-price and the price differences both look roughly normal (gussian), which is ideal for models that assume normality (we will look more deeply on this later)
Price differences are symmetric and stationary (their mean and variance stay constant over time),
 making them especially well-suited for ML or time-series forecasting like what we are doing
The time features (hour, day of week, month) show clear patterns, so we can analyze market behavior by time:
Hour of day : shows spikes at certain times, indicating active trading hours that we can leverage in our model.
Day of week : highlights days when the market is closed, which can serve as a filter or interesting feature.
Month (and other time variables) are essentially categorical; some models require one-hot encoding, 
while others (like Random Forests) can handle them directly and discover month-specific patterns."""
# comparing multiple models
print("\n=== Test Set Evaluation ===")
models = {
    'RandomForest':       RandomForestClassifier(random_state=42),
    'GradientBoosting':   GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
    'KNeighbors':         KNeighborsClassifier(),
    'SVM':                SVC(probability=True, random_state=42)
}

for name, model in models.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\n-- {name} --")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # our confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix'); plt.show()

    # ROC curves (multiclass)
    y_score = pipe.predict_proba(X_test) if hasattr(model, 'predict_proba') else pipe.decision_function(X_test)
    n_classes = y_bin.shape[1]
    y_test_bin = label_binarize(y_test, classes=np.unique(y_enc))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Micro-average
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'], tpr['macro'] = all_fpr, mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    plt.figure()
    plt.plot(fpr['micro'], tpr['micro'], label=f"micro (AUC={roc_auc['micro']:.2f})", linestyle=':')
    plt.plot(fpr['macro'], tpr['macro'], label=f"macro (AUC={roc_auc['macro']:.2f})", linestyle=':')
    for i, color in zip(range(n_classes), cycle(['aqua','darkorange','cornflowerblue'])):
        plt.plot(fpr[i], tpr[i], color=color, label=f'class {i} (AUC={roc_auc[i]:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.title(f'{name} ROC Curves (Test Set)')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.show()

# hyper params tuning for random forest
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth':    [None, 10, 20]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rfc_grid = GridSearchCV(
    Pipeline([('scaler', StandardScaler()),
              ('clf', RandomForestClassifier(random_state=42))]),
    param_grid,
    cv=cv,
    scoring='accuracy'
)
rfc_grid.fit(X_train, y_train)
best_model = rfc_grid.best_estimator_
print("Best RF parameters:", rfc_grid.best_params_)

best_rf = best_model.named_steps['clf']

y_pred_best = best_model.predict(X_test)
print("Accuracy (Best RF):", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))
cm_best = confusion_matrix(y_test, y_pred_best)
sb.heatmap(cm_best, annot=True, fmt='d',
           xticklabels=label_encoder.classes_,
           yticklabels=label_encoder.classes_,
           cmap='Blues')
plt.title("Confusion Matrix (Best RF)"); plt.show()

#cross-validation ROC for best RF
mean_fpr = np.linspace(0,1,100)
tprs, aucs = [], []
plt.figure()
for i, (train_idx, test_idx) in enumerate(cv.split(X, y_enc)):
    # Retrain fold model with cloned best RF
    fold_model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clone(best_rf))
    ])
    fold_model.fit(X.iloc[train_idx], y_enc[train_idx])
    y_proba = fold_model.predict_proba(X.iloc[test_idx])
    y_test_bin_fold = label_binarize(y_enc[test_idx], classes=np.unique(y_enc))
    fpr_i, tpr_i, _ = roc_curve(y_test_bin_fold.ravel(), y_proba.ravel())
    tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)
    aucs.append(auc(fpr_i, tpr_i))
    plt.plot(fpr_i, tpr_i, alpha=0.3, label=f'Fold {i+1} AUC={aucs[-1]:.2f}')

mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
plt.plot(mean_fpr, mean_tpr, color='b', lw=2, label=f'Mean ROC (AUC={np.mean(aucs):.2f})')
plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('Cross-Validation ROC (Best RF)'); plt.legend(); plt.show()

# Interpretation and notes:
"""like we said before, Random Forest turned out to be the best model for this dataset, likely for several reasons:
The underlying patterns are relatively easy for this model to pick up,
Random Forest handles non-linear relationships very well, and is robust to noise and outliers,
The data distribution is consistent, with little impact from random noise or anomalies (like appeared above beforehand),
It works especially effectively with our price-based features—like price differences—which suit its tree-based structure
"""

def predict_currency(trade_time_str, price_val, prev_price_val=0.0):
    """
    Given a timestamp, a current price, and the previous price,
    computes engineered features, scales them, and predicts the currency with confidence.
    """
    trade_time = pd.to_datetime(trade_time_str)
    price_log  = np.log(price_val)
    price_diff = price_val - prev_price_val   # must supply previous price for accurate diff
    feat = pd.DataFrame([{
        'PRICE': price_val,
        'PRICE_LOG': price_log,
        'PRICE_DIFF': price_diff,
        'HOUR': trade_time.hour,
        'DAY_OF_WEEK': trade_time.dayofweek,
        'MONTH': trade_time.month
    }])
    #scale with the trained scaler
    feat_scaled = scaler.transform(feat)
    pred_idx    = best_model.predict(feat_scaled)[0]      # ERROR FIX: use best_model pipeline
    proba       = best_model.predict_proba(feat_scaled)[0][pred_idx]
    curr        = label_encoder.inverse_transform([pred_idx])[0]
    print(f"Predicted Currency: {curr}")
    print(f"Prediction Confidence: {proba:.2%}")

#SHAP summarization
explainer   = shap.TreeExplainer(best_model.named_steps['clf'])
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test, feature_names=features)

"""The SHAP summary plot for PRICE_DIFF reveals that this feature strongly influences the model’s decision
boundary. Specifically, higher values of PRICE_DIFF tend to push predictions toward a certain currency class,
while lower or negative differences shift predictions toward another. This suggests recent price changes are
a key determinant for currency prediction."""



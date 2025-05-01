#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    true_positive_rate
)
import matplotlib.pyplot as plt
import seaborn as sns


#%%
# load and subset data to recent years since 2020
# Load datasets
frs_df = pd.read_csv("FRS_FACILITIES.csv", dtype=str)
demo_df = pd.read_csv("ECHO_DEMOGRAPHICS.csv", dtype=str)
icis_air = pd.read_csv("ICIS-AIR_FACILITIES.csv", dtype=str)

# Keep only relevant ICIS-Air columns
icis_air = icis_air[['REGISTRY_ID', 'CURRENT_HPV', 'AIR_POLLUTANT_CLASS_DESC', 'AIR_OPERATING_STATUS_DESC']]

#%%
# Define binary violation label based on CURRENT_HPV
violation_keywords = [
    'Violation Identified',
    'Violation-Unresolved',
    'Violation w/in 1 Year',
    'Unaddressed-State',
    'Unaddressed-Local',
    'Unaddressed-EPA'
]
icis_air['has_hpv'] = icis_air['CURRENT_HPV'].isin(violation_keywords).astype(int)
icis_air['has_hpv'] = icis_air['has_hpv'].astype(int)

#%%
# Merge datasets
frs_demo = pd.merge(frs_df, demo_df, on='REGISTRY_ID', how='left')
merged = pd.merge(frs_demo, icis_air, on='REGISTRY_ID', how='left')

merged = merged.dropna(subset=['has_hpv', 'MINORITY_POPULATION', 'LOWINCOME'])

#%%
# Create binary features for fairness analysis
merged['pct_minority'] = (
    merged['MINORITY_POPULATION'].astype(float) / merged['ACS_POPULATION'].astype(float) * 100
)
merged['pct_low_income'] = (
    merged['LOWINCOME'].astype(float) / merged['ACS_POPULATION'].astype(float) * 100
)
merged['high_minority'] = (merged['pct_minority'] > 50).astype(int)
merged['low_income'] = (merged['pct_low_income'] > 50).astype(int)

#%%
# Save final data
merged.to_csv("final_facility_data.csv", index=False)

# %% Load cleaned data
# df = pd.read_csv("final_facility_data.csv")

X = pd.get_dummies(merged[['AIR_POLLUTANT_CLASS_DESC', 'AIR_OPERATING_STATUS_DESC']], drop_first=True)


# Outcome variable
y = merged['has_hpv']

#%%
# Sensitive features
sensitive_features = merged[['high_minority', 'low_income']]

#%% train-test split and model training
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_features, test_size=0.3, random_state=42)

X = X.astype(float)
y = y.astype(int)

#%%
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)
#%%
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall (TPR):", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#%%
# fairness metrics w.r.t minority status
# Create a MetricFrame
metric_frame = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'TPR': true_positive_rate,
        'FPR': false_positive_rate
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test['high_minority']
)

print("=== Metrics by high_minority ===")
print(metric_frame.by_group)

# Compute disparities
print("Demographic Parity Difference:", demographic_parity_difference(y_test, y_pred, sensitive_features=sens_test['high_minority']))
print("Equal Opportunity Difference:", equalized_odds_difference(y_test, y_pred, sensitive_features=sens_test['high_minority']))

#%%
# fairness metrics w.r.t low income percentage
# Create a MetricFrame
metric_frame = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'TPR': true_positive_rate,
        'FPR': false_positive_rate
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test['low_income']
)

print("=== Metrics by low income ===")
print(metric_frame.by_group)

#%%
metric_frame.by_group[['TPR', 'FPR']].plot(kind='bar', title='TPR and FPR by Minority Status')
plt.ylabel('Rate')
plt.xticks(rotation=0)
plt.show()
#%%
# This should launch a browser widget 
# FairlearnDashboard(sensitive_features=sens_test['high_minority'], y_true=y_test, y_pred=y_pred)

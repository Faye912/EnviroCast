#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# load datasets
frs_df = pd.read_csv("FRS_FACILITIES.csv", dtype=str)
demo_df = pd.read_csv("ECHO_DEMOGRAPHICS.csv", dtype=str)
icis_air = pd.read_csv("ICIS-AIR_FACILITIES.csv", dtype=str)

# Keep only relevant ICIS-Air columns
icis_air = icis_air[['PGM_SYS_ID','REGISTRY_ID', 'CURRENT_HPV', 'AIR_POLLUTANT_CLASS_DESC', 'AIR_OPERATING_STATUS_DESC']]

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
# create binary features for fairness loss analysis
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

# %% create inputs for model
# df = pd.read_csv("final_facility_data.csv")
X = pd.get_dummies(merged[['AIR_POLLUTANT_CLASS_DESC', 'AIR_OPERATING_STATUS_DESC']], drop_first=True)
# Outcome variable
y = merged['has_hpv']
# sensitive features
sensitive_features = merged[['high_minority', 'low_income']]

#%% train-test split and model training
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_features, test_size=0.3, random_state=42)

# scale and convert data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# convert to torch tensors
X_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
y_torch = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

#%%
# group tensors for sensitive attributes 
group_minority = torch.tensor(sens_train['high_minority'].values, dtype=torch.float32).view(-1, 1)
group_low_income = torch.tensor(sens_train['low_income'].values, dtype=torch.float32).view(-1, 1)

group_minority_test = torch.tensor(sens_test['high_minority'].values, dtype=torch.float32).view(-1, 1)
group_low_income_test = torch.tensor(sens_test['low_income'].values, dtype=torch.float32).view(-1, 1)

#%%
class LogisticRegressionFair(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionFair(X_torch.shape[1])

#%%
def demographic_parity_gap(preds, group):
    return (preds[group == 1].mean() - preds[group == 0].mean()) ** 2

def equal_opportunity_gap(y_true, preds, group):
    mask1 = (y_true == 1) & (group == 1)
    mask0 = (y_true == 1) & (group == 0)
    tpr1 = preds[mask1].mean() if mask1.any() else torch.tensor(0.0)
    tpr0 = preds[mask0].mean() if mask0.any() else torch.tensor(0.0)
    return (tpr1 - tpr0) ** 2
#%%
lambda_grid = [(0,1), (0.1, 0.1), (1,0.01), (0.05, 0.1), (0.25, 0.5)]

results = []

for lambda_min, lambda_low in lambda_grid:
    print(f"\nTraining with λ_minority={lambda_min}, λ_low_income={lambda_low}")

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    bce_loss = nn.BCELoss()

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_torch)

        loss_pred = bce_loss(y_pred, y_torch)

        # Demographic parity penalties
        minority_gap = (y_pred[group_minority == 1].mean() - y_pred[group_minority == 0].mean()) ** 2
        low_income_gap = (y_pred[group_low_income == 1].mean() - y_pred[group_low_income == 0].mean()) ** 2

        fairness_penalty = lambda_min * minority_gap + lambda_low * low_income_gap
        total_loss = loss_pred + fairness_penalty

        total_loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_torch)
        y_test_bin = (y_test_pred > 0.5).float()

        acc = accuracy_score(y_test_torch.numpy(), y_test_bin.numpy())

        dp_gap_min = abs(y_test_pred[group_minority_test == 1].mean().item() - y_test_pred[group_minority_test == 0].mean().item())
        dp_gap_low = abs(y_test_pred[group_low_income_test == 1].mean().item() - y_test_pred[group_low_income_test == 0].mean().item())

        results.append({
            'lambda_minority': lambda_min,
            'lambda_low_income': lambda_low,
            'accuracy': acc,
            'dp_gap_minority': dp_gap_min,
            'dp_gap_low_income': dp_gap_low
        })

#%%
# merge ICIS-Air facilities with violation history
icis_air_violations = pd.read_csv("ICIS-AIR_VIOLATION_HISTORY.csv", dtype=str)

#%%
icis_violations_merged = pd.merge(
    icis_air,  # contains REGISTRY_ID and PGM_SYS_ID
    icis_air_violations,  # contains PGM_SYS_ID
    on='PGM_SYS_ID', how='left'
)

#%%
# Step 2: Merge the result with your main facility-demographics data
full_merged = pd.merge(
    merged,        # your current training dataset
    icis_violations_merged,  # includes REGISTRY_ID and violation history
    on='REGISTRY_ID', how='left')

#%%
columns_to_keep = [
    'REGISTRY_ID', 'PGM_SYS_ID_x', 'FAC_STATE', 'FAC_COUNTY',
    'LATITUDE_MEASURE', 'LONGITUDE_MEASURE',
    'LOWINCOME', 'MINORITY_POPULATION', 'pct_low_income', 'pct_minority',
    'high_minority', 'low_income',
    'CURRENT_HPV_x', 'AIR_POLLUTANT_CLASS_DESC_x', 'AIR_OPERATING_STATUS_DESC_x',
    'has_hpv_x', 'POLLUTANT_DESCS',
    'EARLIEST_FRV_DETERM_DATE', 'HPV_DAYZERO_DATE', 'HPV_RESOLVED_DATE'
]

final_df = full_merged[columns_to_keep].copy()
# %%
date_cols = [
    'EARLIEST_FRV_DETERM_DATE',
    'HPV_DAYZERO_DATE',
    'HPV_RESOLVED_DATE'
]

for col in date_cols:
    final_df[col] = pd.to_datetime(final_df[col], errors='coerce')

#%%
# create temporal features
from datetime import datetime
#%%
# days from violation to now
today = pd.to_datetime("today")
final_df['HPV_DAYZERO_DATE'] = pd.to_datetime(final_df['HPV_DAYZERO_DATE'], errors='coerce')
final_df['days_since_dayzero'] = (today - final_df['HPV_DAYZERO_DATE']).dt.days
#%%
# duration of violation
final_df['HPV_RESOLVED_DATE'] = pd.to_datetime(final_df['HPV_RESOLVED_DATE'], errors='coerce')
final_df['violation_duration'] = (
    final_df['HPV_RESOLVED_DATE'] - final_df['HPV_DAYZERO_DATE']
).dt.days
#%%
# time gaps between events
final_df['EARLIEST_FRV_DETERM_DATE'] = pd.to_datetime(final_df['EARLIEST_FRV_DETERM_DATE'], errors='coerce')

final_df['days_to_resolution'] = (
    final_df['HPV_RESOLVED_DATE'] - final_df['EARLIEST_FRV_DETERM_DATE']
).dt.days


#%%
final_df['has_dayzero'] = final_df['HPV_DAYZERO_DATE'].notna().astype(int)
final_df['has_resolved'] = final_df['HPV_RESOLVED_DATE'].notna().astype(int)

#%%
# impute missing values
temporal_cols = [
    'days_since_dayzero',
    'violation_duration',
    'days_to_resolution'
]

# Fill missing with -1 (or use median if you prefer)
final_df[temporal_cols] = final_df[temporal_cols].fillna(-1)

# %%
# combine existing features and temporal features
X = pd.concat([X, temporal_cols], axis=1)

#%%
from sklearn.model_selection import train_test_split

#%% train-test split and model training
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_features, test_size=0.3, random_state=42)

# scale and convert data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# convert to torch tensors
X_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
y_torch = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

#%%
input_dim = X_torch.shape[1]

model = LogisticRegressionFair(input_dim) 
#%%
lambda_grid = [(0,1), (0.1, 0.1), (1,0.01), (0.05, 0.1), (0.25, 0.5)]

results = []

for lambda_min, lambda_low in lambda_grid:
    print(f"\nTraining with λ_minority={lambda_min}, λ_low_income={lambda_low}")

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    bce_loss = nn.BCELoss()

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_torch)

        loss_pred = bce_loss(y_pred, y_torch)

        # Demographic parity penalties
        minority_gap = (y_pred[group_minority == 1].mean() - y_pred[group_minority == 0].mean()) ** 2
        low_income_gap = (y_pred[group_low_income == 1].mean() - y_pred[group_low_income == 0].mean()) ** 2

        fairness_penalty = lambda_min * minority_gap + lambda_low * low_income_gap
        total_loss = loss_pred + fairness_penalty

        total_loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_torch)
        y_test_bin = (y_test_pred > 0.5).float()

        acc = accuracy_score(y_test_torch.numpy(), y_test_bin.numpy())

        dp_gap_min = abs(y_test_pred[group_minority_test == 1].mean().item() - y_test_pred[group_minority_test == 0].mean().item())
        dp_gap_low = abs(y_test_pred[group_low_income_test == 1].mean().item() - y_test_pred[group_low_income_test == 0].mean().item())

        results.append({
            'lambda_minority': lambda_min,
            'lambda_low_income': lambda_low,
            'accuracy': acc,
            'dp_gap_minority': dp_gap_min,
            'dp_gap_low_income': dp_gap_low
        })

# %%
# CLASS IMBALANCE!!
y.value_counts(normalize=True)

# %%
class LogisticRegressionFair(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  

#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

bce_results = []

# Compute pos_weight for class imbalance
pos_weight = torch.tensor([y_torch.eq(0).sum() / y_torch.eq(1).sum()]).float().to(X_torch.device)

for lambda_min, lambda_low in lambda_grid:
    print(f"\nTraining with λ_minority={lambda_min}, λ_low_income={lambda_low}")

    model = LogisticRegressionFair(X_torch.shape[1]).to(X_torch.device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X_torch)
        y_pred_probs = torch.sigmoid(logits)  # needed for fairness penalties
        
        loss_pred = loss_fn(logits, y_torch)

        # Fairness penalties (demographic parity)
        minority_gap = (y_pred_probs[group_minority == 1].mean() - y_pred_probs[group_minority == 0].mean()) ** 2
        low_income_gap = (y_pred_probs[group_low_income == 1].mean() - y_pred_probs[group_low_income == 0].mean()) ** 2

        fairness_penalty = lambda_min * minority_gap + lambda_low * low_income_gap
        total_loss = loss_pred + fairness_penalty

        total_loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_torch)
        probs_test = torch.sigmoid(logits_test)
        preds_test = (probs_test > 0.5).float()

        acc = accuracy_score(y_test_torch.cpu(), preds_test.cpu())
        precision = precision_score(y_test_torch.cpu(), preds_test.cpu(), zero_division=0)
        recall = recall_score(y_test_torch.cpu(), preds_test.cpu(), zero_division=0)
        f1 = f1_score(y_test_torch.cpu(), preds_test.cpu(), zero_division=0)
        auc = roc_auc_score(y_test_torch.cpu(), probs_test.cpu())

        dp_gap_min = abs(probs_test[group_minority_test == 1].mean().item() - probs_test[group_minority_test == 0].mean().item())
        dp_gap_low = abs(probs_test[group_low_income_test == 1].mean().item() - probs_test[group_low_income_test == 0].mean().item())

        bce_results.append({
            'lambda_minority': lambda_min,
            'lambda_low_income': lambda_low,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'dp_gap_minority': dp_gap_min,
            'dp_gap_low_income': dp_gap_low,
            'y_probs': probs_test
        })
# %%
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

for result in bce_results:
    y_probs = result['y_probs']
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    
    recall = recall[recall > 0]
    precision = precision[-len(recall):]
    
    label = f"λ_min={result['lambda_minority']}, λ_low={result['lambda_low_income']}"
    plt.plot(recall, precision, label=label)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves for Different Fairness Weights")
plt.legend(loc="lower left", fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# Compute F1 for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

# Find threshold that maximizes F1
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
best_f1 = f1_scores[best_index]

print(f"Best threshold: {best_threshold:.4f} with F1 score: {best_f1:.4f}")


# %%
# RETRAINING after adjusting evaluation metrics using best threshold
retrain_results = []

# Compute pos_weight for class imbalance
pos_weight = torch.tensor([y_torch.eq(0).sum() / y_torch.eq(1).sum()]).float().to(X_torch.device)

for lambda_min, lambda_low in lambda_grid:
    print(f"\nTraining with λ_minority={lambda_min}, λ_low_income={lambda_low}")

    model = LogisticRegressionFair(X_torch.shape[1]).to(X_torch.device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        
        logits = model(X_torch)
        y_pred_probs = torch.sigmoid(logits)  # needed for fairness penalties
        
        loss_pred = loss_fn(logits, y_torch)

        # Fairness penalties (demographic parity)
        minority_gap = (y_pred_probs[group_minority == 1].mean() - y_pred_probs[group_minority == 0].mean()) ** 2
        low_income_gap = (y_pred_probs[group_low_income == 1].mean() - y_pred_probs[group_low_income == 0].mean()) ** 2

        fairness_penalty = lambda_min * minority_gap + lambda_low * low_income_gap
        total_loss = loss_pred + fairness_penalty

        total_loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_torch)
        probs_test = torch.sigmoid(logits_test)
        preds_test_adjusted = (probs_test >= 0.4).float()

        acc = accuracy_score(y_test_torch.cpu(), preds_test_adjusted.cpu())
        precision = precision_score(y_test_torch.cpu(), preds_test_adjusted.cpu(), zero_division=0)
        recall = recall_score(y_test_torch.cpu(), preds_test_adjusted.cpu(), zero_division=0)
        f1 = f1_score(y_test_torch.cpu(), preds_test_adjusted.cpu(), zero_division=0)
        auc = roc_auc_score(y_test_torch.cpu(), probs_test.cpu())

        dp_gap_min = abs(probs_test[group_minority_test == 1].mean().item() - probs_test[group_minority_test == 0].mean().item())
        dp_gap_low = abs(probs_test[group_low_income_test == 1].mean().item() - probs_test[group_low_income_test == 0].mean().item())

        retrain_results.append({
            'lambda_minority': lambda_min,
            'lambda_low_income': lambda_low,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'dp_gap_minority': dp_gap_min,
            'dp_gap_low_income': dp_gap_low,
            'y_probs': probs_test
        })
# %%
# checking if there are differentiated predicted probabilities
unique_probs = np.unique(y_probs)
print(f"Unique probs: {len(unique_probs)}")
# %%
# checking for a spike at predicted prob around 0.2
import matplotlib.pyplot as plt
plt.hist(y_probs, bins=50)
plt.title("Histogram of predicted probabilities")
plt.show()

# %%
# baseline model without sensitive features
baseline = []
model = LogisticRegressionFair(X_torch.shape[1]).to(X_torch.device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    
    logits = model(X_torch)
    y_pred_probs = torch.sigmoid(logits)  # needed for fairness penalties
    
    loss_pred = loss_fn(logits, y_torch)

    # Fairness penalties (demographic parity)
    minority_gap = (y_pred_probs[group_minority == 1].mean() - y_pred_probs[group_minority == 0].mean()) ** 2
    low_income_gap = (y_pred_probs[group_low_income == 1].mean() - y_pred_probs[group_low_income == 0].mean()) ** 2

    fairness_penalty = lambda_min * minority_gap + lambda_low * low_income_gap
    total_loss = loss_pred + fairness_penalty

    total_loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    logits_test = model(X_test_torch)
    probs_test = torch.sigmoid(logits_test)
    preds_test = (probs_test > 0.5).float()

    acc = accuracy_score(y_test_torch.cpu(), preds_test.cpu())
    precision = precision_score(y_test_torch.cpu(), preds_test.cpu(), zero_division=0)
    recall = recall_score(y_test_torch.cpu(), preds_test.cpu(), zero_division=0)
    f1 = f1_score(y_test_torch.cpu(), preds_test.cpu(), zero_division=0)
    auc = roc_auc_score(y_test_torch.cpu(), probs_test.cpu())

    dp_gap_min = abs(probs_test[group_minority_test == 1].mean().item() - probs_test[group_minority_test == 0].mean().item())
    dp_gap_low = abs(probs_test[group_low_income_test == 1].mean().item() - probs_test[group_low_income_test == 0].mean().item())

    baseline.append({
        'lambda_minority': 0,
        'lambda_low_income': 0,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'dp_gap_minority': dp_gap_min,
        'dp_gap_low_income': dp_gap_low,
        'y_probs': probs_test
    })

# %%
# logits distribution
plt.hist(logits.detach().cpu().numpy(), bins=50)
plt.title("Logits distribution (before sigmoid)")
plt.show()

#%%


#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
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
# load datasets
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
best_lambda = None
best_acc = 0
results = []

# Training loop
for lambda_dp in [0, 0.1, 0.5, 1.0, 5.0]:
    lambda_eo = 1.0  # fix EO penalty for now
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(100):
        model.train()
        logits = model(X_torch)
        probs = torch.sigmoid(logits)

        loss_pred = loss_fn(logits, y_torch)
        loss_dp = demographic_parity_gap(probs, group_minority)
        loss_eo = equal_opportunity_gap(y_torch, probs, group_low_income)

        total_loss = loss_pred + lambda_dp * loss_dp + lambda_eo * loss_eo

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_preds = torch.sigmoid(model(torch.tensor(X_test_torch, dtype=torch.float32)))
            test_preds_bin = (test_preds > 0.5).float().numpy()
            acc = accuracy_score(y_test, test_preds_bin)
            results.append((lambda_dp, acc))
            if acc > best_acc:
                best_acc = acc
                best_lambda = lambda_dp
                
                
# Show results
print("\n=== Grid Search Results ===")
for res in results:
    print(
        f"λ_min={res['lambda_dp']} "
        f"Acc={res['acc']:.4f} "
    )

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

#%%

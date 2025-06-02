#%%
import pandas as pd
survival_data = pd.read_csv("final_df.csv")

#%%
import numpy as np
from lifelines import CoxPHFitter
#%%
# Drop rows with missing duration or event flag
df = survival_data.dropna(subset=["duration", "event_observed"])
#%%
# drop other rows with missing values in modeling variables
model_features = [
    "REGISTRY_ID",
    "duration", "event_observed",
    "log_penalty_amt", "voluntary_flag",
    "pct_minority", "pct_low_income",
    "high_minority", "low_income","REGION_CODE",
    "STATE_CODE_x"
]

df = df[model_features].dropna()

#%%
df = pd.get_dummies(df, columns=["STATE_CODE_x", "REGION_CODE"], drop_first=True)

# %%
# hazards modeling
cph = CoxPHFitter()
cph.fit(df, duration_col="duration", event_col="event_observed")
cph.print_summary()  # Shows coefficients, p-values, and confidence intervals

# %%
cph.plot()

# %%
# SCORING
# Predict partial hazard (relative risk) for each facility
df["risk_score"] = cph.predict_partial_hazard(df)

# Sort to see highest-risk facilities
df_sorted = df.sort_values("risk_score", ascending=False)

# check top 10
df_sorted[["risk_score", "duration", "event_observed"]].head(10)

# %%
# normalized scores to between 0-1
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df["normalized_risk_score"] = scaler.fit_transform(df[["risk_score"]])

#%%
df["risk_tier"] = pd.qcut(df["normalized_risk_score"], q=3, labels=["Low", "Medium", "High"])

# %%
cols_to_export = ["REGISTRY_ID", "risk_score", "normalized_risk_score", "risk_tier", "event_observed"]
df[cols_to_export].to_csv("facility_risk_scores.csv", index=False)

# %%
import plotly.express as px

# %%
facility_data = pd.read_csv("FRS_FACILITIES.csv")
#%%
geo_cols = facility_data[["REGISTRY_ID","LATITUDE_MEASURE","LONGITUDE_MEASURE"]]
# %%
plot_df = df[["REGISTRY_ID", "normalized_risk_score", "risk_tier"]].dropna()
plot_df = plot_df.merge(geo_cols, on="REGISTRY_ID", how="inner")

# %%
fig = px.scatter_mapbox(
    plot_df,
    lat="LATITUDE_MEASURE",
    lon="LONGITUDE_MEASURE",
    color="normalized_risk_score",  # Color = risk
    size="normalized_risk_score",   # Bubble size = risk
    hover_name="REGISTRY_ID",
    hover_data=["risk_tier"],
    color_continuous_scale="Reds",  # Or "Viridis", "YlOrRd"
    zoom=3,
    height=600
)

fig.update_layout(
    mapbox_style="carto-positron",  # or "open-street-map"
    margin={"r":0,"t":0,"l":0,"b":0}
)

fig.show(renderer="browser")


# %%
# SHAP analysis
# Convert to binary classification: True if event occurred within 3 years
survival_data["violation_within_3y"] = (
    (survival_data["event_observed"] == 1) & 
    (survival_data["duration"] <= 1000)
)

# %%
features = [
    "log_penalty_amt",
    "voluntary_flag",
    "pct_minority",
    "pct_low_income",
    "high_minority",
    "low_income",
    "STATE_CODE_x",  
    "REGION_CODE"
]

# Drop NA rows for this model
model_df = survival_data[features + ["violation_within_3y"]].dropna()

#%%
X = model_df[features]
X = pd.get_dummies(X, columns=["STATE_CODE_x", "REGION_CODE"], drop_first=True,dtype=int)

y = model_df["violation_within_3y"]

# %%
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# %%
import shap

# SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# %%
# Global feature importance
shap.plots.bar(shap_values)

# Summary beeswarm plot
shap.plots.beeswarm(shap_values)

# Individual facility example
shap.plots.waterfall(shap_values[0])

# %%

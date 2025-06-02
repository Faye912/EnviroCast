#%%
import pandas as pd
from datetime import datetime

# Load core files
enforcement_data = pd.read_csv("case_downloads/CASE_ENFORCEMENTS.csv", parse_dates=["ACTIVITY_STATUS_DATE", "CASE_STATUS_DATE"])
facility_data = pd.read_csv("case_downloads/CASE_FACILITIES.csv")  # includes REGISTRY_ID
violation_data = pd.read_csv("case_downloads/CASE_VIOLATIONS.csv")
inspection_data = pd.read_csv("case_downloads/ICIS_FEC_EPA_INSPECTIONS.csv")

#%%
violation_data = violation_data.merge(
    enforcement_data[["ACTIVITY_ID", "ACTIVITY_STATUS_DATE"]],
    on="ACTIVITY_ID",
    how="left"
)

case_df = violation_data.merge(facility_data, on="ACTIVITY_ID", how="left")  # adds REGISTRY_ID

#%%
# set start time as inspection start date
import pandas as pd

# Step 1: Ensure datetime conversion
inspection_data["ACTUAL_BEGIN_DATE"] = pd.to_datetime(
    inspection_data["ACTUAL_BEGIN_DATE"], errors="coerce"
)
# # Step 2: Clean and standardize REGISTRY_IDs in both dataframes
# def clean_registry_id(df, col="REGISTRY_ID"):
#     df[col] = df[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
#     return df

# case_df = clean_registry_id(case_df, "REGISTRY_ID")
# inspection_data = clean_registry_id(inspection_data, "REGISTRY_ID")

# Step 3: Extract earliest inspection date per facility
inspection_dates = (
    inspection_data
    .groupby("REGISTRY_ID", as_index=False)["ACTUAL_BEGIN_DATE"]
    .min()
    .rename(columns={"ACTUAL_BEGIN_DATE": "start_time"})
)
#%%
# Step 4: Merge inspection start_time into case_df
case_df = case_df.merge(inspection_dates, on="REGISTRY_ID", how="left")
case_df = case_df.dropna(subset=["start_time"])

#%%
# Define Event Time and Event Observed Indicator
case_df["event_time"] = case_df["ACTIVITY_STATUS_DATE"]
case_df = case_df.dropna(subset=["event_time"])

#%%
case_df = case_df.merge(enforcement_data,
    on="ACTIVITY_ID")

#%%
# Consider 'Closed' with non-null ENF_OUTCOME_DESC as an observed event
case_df["event_observed"] = ((case_df["ACTIVITY_STATUS_DESC"] == "Closed") &
                             case_df["ENF_OUTCOME_DESC"].notnull()).astype(int)

# %%
case_df.to_csv("case_data.csv", index=False)
# %%
# drop missing rows and weird values
survival_df = case_df.dropna(subset=["start_time", "event_time", "REGISTRY_ID"]).copy()
#%%
# added feature columns
import numpy as np
# Penalty amount (log-transformed optional)
survival_df["log_penalty_amt"] = survival_df["TOTAL_PENALTY_ASSESSED_AMT"].apply(lambda x: np.log1p(x) if pd.notnull(x) else 0)

# Voluntary self-disclosure flag
survival_df["voluntary_flag"] = survival_df["VOLUNTARY_SELF_DISCLOSURE_FLAG"].map({"Y": 1, "N": 0})

#%%
# Penalty amount (log-transformed optional)
survival_df["log_penalty_amt"] = survival_df["TOTAL_PENALTY_ASSESSED_AMT"].apply(lambda x: np.log1p(x) if pd.notnull(x) else 0)

# Voluntary self-disclosure flag
survival_df["voluntary_flag"] = survival_df["VOLUNTARY_SELF_DISCLOSURE_FLAG"].map({"Y": 1, "N": 0})

# %%
final_df = survival_df[[
    "REGISTRY_ID", "ACTIVITY_ID", "start_time", "event_time", "event_observed",
    "log_penalty_amt", "voluntary_flag", "STATE_CODE_x", "REGION_CODE"
]]
# %%
# time-to-event 
final_df["duration"] = (final_df["event_time"] - final_df["start_time"]).dt.days

#%%

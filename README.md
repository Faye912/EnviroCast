# EchoRisk

**Forecasting Time-Varying Environmental Compliance Risk Using EPA ECHO Data**

This project analyzes dynamic compliance behavior among industrial facilities in response to environmental inspections and policy changes, using data from the U.S. EPA’s Enforcement and Compliance History Online (ECHO) system.

## Project Goals

- Forecast future violation risk at the facility level.
- Detect strategic noncompliance timing.
- Evaluate behavioral responses to new policies.
- Track transitions across violation types.
- Identify clustering behavior in facilities with shared characteristics.

## Data

- **Source**: [EPA ECHO Data](https://echo.epa.gov/tools/data-downloads)
- **Scope**: Inspections, violations, enforcement actions, and metadata from U.S. facilities over time.
https://echo.epa.gov/tools/data-downloads/demographic-download-summary 
https://echo.epa.gov/tools/data-downloads/frs-download-summary 
https://echo.epa.gov/tools/data-downloads/icis-air-download-summary 

## Methodology

- **Model Ensemble**: Gradient Boosting, Recurrent Neural Networks, Survival Models, etc.
- **Temporal Modeling**: Time-series and lag-based features.
- **Incremental Learning**: Fine-tuning models over time as new data arrives.

## Key Research Themes

- **Strategic Noncompliance Timing**  
- **Policy Reaction Behavior**  
- **Violation Type Transition**  
- **Behavioral Clustering Across Facilities**

## Project Structure
```
  echorisk/ │
            ├── data/ # Cleaned and raw data 
            ├── notebooks/ # Exploratory and modeling notebooks 
            ├── src/ # Scripts and pipeline modules 
            ├── models/ # Saved models and training logs 
            ├── results/ # Forecasts and visualizations 
            ├── README.md 
            └── proposal.md # Project concept + methodology
```

## Status

In initial development. Currently conducting data exploration.

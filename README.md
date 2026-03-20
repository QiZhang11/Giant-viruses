# Giant Viruses Amplify Eukaryotic Pathogen Accumulation in Urban Soils
This repository contains machine-learning scripts used to predict the abundance of Urban Interface Pathogens (UIPs) associated with nucleocytoplasmic large DNA viruses (NCLDVs) under urbanization scenarios.
## Overview
This study develops a machine-learning framework to quantify and predict UIP dynamics across soil–fauna interfaces. The workflow integrates environmental and socioeconomic variables to model pathogen abundance and project future trends under Shared Socioeconomic Pathways (SSPs).
Supported models:
- Random Forest (RF)
- K-Nearest Neighbors (KNN)
- XGBoost
## Requirements
- Python ≥ 3.8
- pandas
- scikit-learn
- xgboost
- joblib
- numpy
## Train models
python scripts/train_ml_models.py \
    --input data/training_data.csv \
    --model_dir results/models \
    --metrics_dir results/metrics \
    --importance_dir results/importance
## Predict future UIP abundance 
python scripts/predict_future_uip.py \
    --input_dir data/future_ssp_data \
    --model_dir results/models \
    --output_dir results/predictions
## train_ml_models.py
## predict_future_uip.py


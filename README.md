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
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from joblib import dump


def main(args):
    data = pd.read_csv(args.input)

    X = data.iloc[:, 1:10]
    y_columns = data.columns[10:]

    models = {
        'xgboost': XGBRegressor(),
        'randomforest': RandomForestRegressor(),
        'kneighbors': KNeighborsRegressor()
    }

    param_grids = {
        'xgboost': {
            'xgboost__n_estimators': [100, 300],
            'xgboost__max_depth': [3, 7],
            'xgboost__learning_rate': [0.01, 0.1]
        },
        'randomforest': {
            'randomforest__n_estimators': [100, 300],
            'randomforest__max_depth': [None, 20]
        },
        'kneighbors': {
            'kneighbors__n_neighbors': [3, 5, 7]
        }
    }

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)
    os.makedirs(args.importance_dir, exist_ok=True)

    results = []
    feature_importances = []

    for y_column in y_columns:
        print(f"Training for: {y_column}")

        y = data[y_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )

        for model_name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                (model_name, model)
            ])

            grid = GridSearchCV(
                pipeline,
                param_grids[model_name],
                cv=5,
                scoring='r2',
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            y_pred = grid.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            model_path = os.path.join(
                args.model_dir,
                f"{y_column}_{model_name}_best_model.pkl"
            )
            dump(grid.best_estimator_, model_path)

            if model_name == 'randomforest':
                rf = grid.best_estimator_.named_steps['randomforest']
                for f, imp in zip(X.columns, rf.feature_importances_):
                    feature_importances.append({
                        'target': y_column,
                        'feature': f,
                        'importance': imp
                    })

            results.append({
                'target': y_column,
                'model': model_name,
                'r2': r2,
                'rmse': rmse
            })

    pd.DataFrame(results).to_csv(
        os.path.join(args.metrics_dir, "model_performance.csv"),
        index=False
    )

    pd.DataFrame(feature_importances).to_csv(
        os.path.join(args.importance_dir, "feature_importance.csv"),
        index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--importance_dir", required=True)

    args = parser.parse_args()
    main(args)
## predict_future_uip.py
import os
import argparse
import pandas as pd
from joblib import load


def main(args):
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"{args.input_dir} not found")

    os.makedirs(args.output_dir, exist_ok=True)

    for file in os.listdir(args.input_dir):
        if file.endswith(".csv"):
            input_file = os.path.join(args.input_dir, file)
            output_file = os.path.join(args.output_dir, file)

            print(f"Processing: {input_file}")

            data = pd.read_csv(input_file)
            X = data.iloc[:, 1:10]

            predictions = data.copy()

            for model_file in os.listdir(args.model_dir):
                if model_file.endswith("_best_model.pkl"):
                    model_path = os.path.join(args.model_dir, model_file)

                    model = load(model_path)

                    var_name = "_".join(model_file.split("_")[:2])

                    predictions[f"{var_name}_pred"] = model.predict(X)

            predictions.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()
    main(args)

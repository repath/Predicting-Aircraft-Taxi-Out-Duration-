"""
train_models.py
Implements Step 4: Model Training & Evaluation for Taxi-Out Prediction.

Guidelines:
1. Model 1: Baseline Linear Regression
   - Standardize continuous features with StandardScaler.
   - One-Hot Encode low-cardinality categorical variables.
2. Model 2: Random Forest Regressor
   - n_estimators=100, max_depth=15, min_samples_split=10, random_state=42.
3. Model 3: XGBoost Regressor
   - n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42.
4. Evaluation Metrics:
   - MAE (Mean Absolute Error, in minutes)
   - RMSE (Root Mean Squared Error)
   - R^2 Score (Explained Variance)
   - Operational Accuracy: % within +/- 5 minutes and +/- 3 minutes
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Feature specifications
CONTINUOUS_FEATURES = [
    'DEP_DELAY',
    'DISTANCE',
    'CONGESTION_1H',
    'CONGESTION_30M',
    'SIN_HOUR',
    'COS_HOUR',
    'ORIGIN_TAXI_MEAN',
    'CARRIER_TAXI_MEAN'
]

CATEGORICAL_FEATURES = [
    'OP_UNIQUE_CARRIER',
    'ORIGIN',
    'DAY_OF_WEEK'
]

TARGET_COL = 'TAXI_OUT'

def build_preprocessor():
    """Builds the scikit-learn ColumnTransformer for feature transformations."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), CONTINUOUS_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )

def evaluate_predictions(y_true, y_pred, split_name="Test"):
    """Computes academic and operational evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    within_3 = float(np.mean(np.abs(y_true - y_pred) <= 3.0) * 100)
    within_5 = float(np.mean(np.abs(y_true - y_pred) <= 5.0) * 100)
    residuals = y_true - y_pred
    
    return {
        "MAE_minutes": round(float(mae), 3),
        "RMSE_minutes": round(float(rmse), 3),
        "R2_Score": round(float(r2), 4),
        "Pct_Within_3_Min": round(within_3, 2),
        "Pct_Within_5_Min": round(within_5, 2),
        "Mean_Residual_Bias": round(float(np.mean(residuals)), 3),
        "Std_Residual": round(float(np.std(residuals)), 3)
    }

def train_and_evaluate(
    train_path="data/processed/train_features.csv",
    test_path="data/processed/test_features.csv",
    models_dir="models",
    dashboard_data_dir="dashboard"
):
    print("=" * 70)
    print("STEP 4: MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)
    
    # 1. Load Train and Test sets
    print(f"Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    feature_cols = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]
    
    print(f"Train samples (Months 1-10): {len(X_train):,}")
    print(f"Test samples (Months 11-12):  {len(X_test):,}")
    print(f"Continuous features ({len(CONTINUOUS_FEATURES)}): {CONTINUOUS_FEATURES}")
    print(f"Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(dashboard_data_dir, exist_ok=True)
    
    # 2. Build Preprocessor
    preprocessor = build_preprocessor()
    
    # 3. Model 1: Baseline Linear Regression
    print("\n" + "-" * 50)
    print("Training Model 1: Baseline Linear Regression...")
    pipeline_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    pipeline_lr.fit(X_train, y_train)
    print("Model 1 trained successfully.")
    
    y_train_pred_lr = pipeline_lr.predict(X_train)
    y_test_pred_lr = pipeline_lr.predict(X_test)
    
    metrics_lr_train = evaluate_predictions(y_train, y_train_pred_lr, "Train")
    metrics_lr_test = evaluate_predictions(y_test, y_test_pred_lr, "Test")
    
    print(f"[Model 1 Test Metrics] MAE: {metrics_lr_test['MAE_minutes']:.2f} min | "
          f"RMSE: {metrics_lr_test['RMSE_minutes']:.2f} min | "
          f"R²: {metrics_lr_test['R2_Score']:.4f} | "
          f"Within ±5m: {metrics_lr_test['Pct_Within_5_Min']:.1f}%")
    
    # 4. Model 2: Random Forest Regressor
    print("\n" + "-" * 50)
    print("Training Model 2: Random Forest Regressor...")
    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipeline_rf.fit(X_train, y_train)
    print("Model 2 trained successfully.")
    
    y_train_pred_rf = pipeline_rf.predict(X_train)
    y_test_pred_rf = pipeline_rf.predict(X_test)
    
    metrics_rf_train = evaluate_predictions(y_train, y_train_pred_rf, "Train")
    metrics_rf_test = evaluate_predictions(y_test, y_test_pred_rf, "Test")
    
    print(f"[Model 2 Test Metrics] MAE: {metrics_rf_test['MAE_minutes']:.2f} min | "
          f"RMSE: {metrics_rf_test['RMSE_minutes']:.2f} min | "
          f"R²: {metrics_rf_test['R2_Score']:.4f} | "
          f"Within ±5m: {metrics_rf_test['Pct_Within_5_Min']:.1f}%")

    # 5. Model 3: XGBoost Regressor
    print("\n" + "-" * 50)
    print("Training Model 3: XGBoost Regressor (n_estimators=100, max_depth=6, learning_rate=0.1)...")
    pipeline_xgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipeline_xgb.fit(X_train, y_train)
    print("Model 3 trained successfully.")
    
    y_train_pred_xgb = pipeline_xgb.predict(X_train)
    y_test_pred_xgb = pipeline_xgb.predict(X_test)
    
    metrics_xgb_train = evaluate_predictions(y_train, y_train_pred_xgb, "Train")
    metrics_xgb_test = evaluate_predictions(y_test, y_test_pred_xgb, "Test")
    
    print(f"[Model 3 Test Metrics] MAE: {metrics_xgb_test['MAE_minutes']:.2f} min | "
          f"RMSE: {metrics_xgb_test['RMSE_minutes']:.2f} min | "
          f"R²: {metrics_xgb_test['R2_Score']:.4f} | "
          f"Within ±5m: {metrics_xgb_test['Pct_Within_5_Min']:.1f}%")
    
    # 6. Extract Feature Importances and Transformed Feature Names
    fitted_preprocessor = pipeline_rf.named_steps['preprocessor']
    cat_feature_names = fitted_preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = CONTINUOUS_FEATURES + cat_feature_names
    
    rf_importances = pipeline_rf.named_steps['regressor'].feature_importances_
    feat_importance_dict = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(all_feature_names, rf_importances)],
        key=lambda x: x["importance"],
        reverse=True
    )
    
    xgb_importances = pipeline_xgb.named_steps['regressor'].feature_importances_
    xgb_importance_dict = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(all_feature_names, xgb_importances)],
        key=lambda x: x["importance"],
        reverse=True
    )
    
    lr_coefficients = pipeline_lr.named_steps['regressor'].coef_
    feat_coefficients_dict = sorted(
        [{"feature": f, "coefficient": round(float(coef), 4)} for f, coef in zip(all_feature_names, lr_coefficients)],
        key=lambda x: abs(x["coefficient"]),
        reverse=True
    )
    
    # 7. Save Models
    lr_model_file = os.path.join(models_dir, "linear_regression_baseline.joblib")
    rf_model_file = os.path.join(models_dir, "random_forest_regressor.joblib")
    xgb_model_file = os.path.join(models_dir, "xgboost_regressor.joblib")
    
    joblib.dump(pipeline_lr, lr_model_file)
    joblib.dump(pipeline_rf, rf_model_file)
    joblib.dump(pipeline_xgb, xgb_model_file)
    
    print(f"\nSaved models to:")
    print(f" - {lr_model_file}")
    print(f" - {rf_model_file}")
    print(f" - {xgb_model_file}")
    
    # 8. Model Performance Comparison Dictionary
    comparison = {
        "models": {
            "Linear_Regression_Baseline": {
                "train": metrics_lr_train,
                "test": metrics_lr_test,
                "top_coefficients": feat_coefficients_dict[:10]
            },
            "Random_Forest_Regressor": {
                "train": metrics_rf_train,
                "test": metrics_rf_test,
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 15,
                    "min_samples_split": 10,
                    "random_state": 42
                },
                "top_importances": feat_importance_dict[:10]
            },
            "XGBoost_Regressor": {
                "train": metrics_xgb_train,
                "test": metrics_xgb_test,
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42
                },
                "top_importances": xgb_importance_dict[:10]
            }
        },
        "all_feature_names": all_feature_names,
        "feature_importances_rf": feat_importance_dict,
        "feature_importances_xgb": xgb_importance_dict,
        "coefficients_lr": feat_coefficients_dict
    }
    
    metrics_path = os.path.join(models_dir, "model_evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")
    
    # 9. Export Test Samples for Visual Dashboard
    test_sample = test_df.sample(n=min(500, len(test_df)), random_state=42).copy()
    test_sample['PRED_TAXI_OUT_LR'] = pipeline_lr.predict(test_sample[feature_cols]).round(1)
    test_sample['PRED_TAXI_OUT_RF'] = pipeline_rf.predict(test_sample[feature_cols]).round(1)
    test_sample['PRED_TAXI_OUT_XGB'] = pipeline_xgb.predict(test_sample[feature_cols]).round(1)
    test_sample['ERROR_RF'] = (test_sample['PRED_TAXI_OUT_RF'] - test_sample['TAXI_OUT']).round(1)
    test_sample['ABS_ERROR_RF'] = np.abs(test_sample['ERROR_RF']).round(1)
    
    dashboard_payload = {
        "metrics": comparison["models"],
        "top_features": feat_importance_dict[:8],
        "sample_flights": test_sample[[
            'FL_DATE', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'CRS_DEP_TIME',
            'DEP_TIME', 'DEP_DELAY', 'DISTANCE', 'CONGESTION_1H',
            'TAXI_OUT', 'PRED_TAXI_OUT_LR', 'PRED_TAXI_OUT_RF', 'PRED_TAXI_OUT_XGB', 'ERROR_RF', 'ABS_ERROR_RF'
        ]].to_dict(orient="records"),
        "airport_congestion_stats": {
            "mean_congestion_1h": float(test_df['CONGESTION_1H'].mean()),
            "max_congestion_1h": float(test_df['CONGESTION_1H'].max())
        }
    }
    
    dashboard_json_path = os.path.join(dashboard_data_dir, "evaluation_data.json")
    with open(dashboard_json_path, "w") as f:
        json.dump(dashboard_payload, f, indent=2)
    print(f"Saved dashboard evaluation data to: {dashboard_json_path}")
    
    print("\nPipeline execution completed successfully!")
    return comparison

if __name__ == "__main__":
    train_and_evaluate()

"""
Iris ETL Pipeline + Model Training DAG
=======================================
This Airflow DAG performs:
  1. EXTRACT  — Read raw JSON iris data
  2. TRANSFORM — Clean, validate, normalize, feature-engineer
  3. LOAD — Save clean CSV to processed folder
  4. TRAIN MODEL — Train a Random Forest classifier
  5. EVALUATE — Evaluate model and log metrics

Author : Your Name
Created: 2025-01-28
"""

from datetime import datetime, timedelta
import json
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

from airflow import DAG
from airflow.operators.python import PythonOperator


# ──────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "iris_raw.json")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "iris_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "iris_rf_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

default_args = {
    "owner": "data_engineer",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


# ──────────────────────────────────────────────────────────
#  TASK 1: EXTRACT — Read raw JSON
# ──────────────────────────────────────────────────────────
def extract_data(**context):
    """Read the raw JSON file and push to XCom."""
    print(f"[EXTRACT] Reading raw data from: {RAW_DATA_PATH}")

    with open(RAW_DATA_PATH, "r") as f:
        raw_data = json.load(f)

    record_count = len(raw_data)
    print(f"[EXTRACT] Loaded {record_count} records")

    # Push raw data to XCom for next task
    context["ti"].xcom_push(key="raw_data", value=raw_data)
    context["ti"].xcom_push(key="raw_record_count", value=record_count)

    return f"Extracted {record_count} records"


# ──────────────────────────────────────────────────────────
#  TASK 2: TRANSFORM — Clean, validate & feature-engineer
# ──────────────────────────────────────────────────────────
def transform_data(**context):
    """
    Transformations applied:
      1. Convert JSON → DataFrame
      2. Standardize species names (case, prefix)
      3. Handle null/missing values (median imputation)
      4. Remove negative values (invalid measurements)
      5. Remove duplicate records
      6. Add engineered features (ratios, area estimates)
      7. Encode species labels
    """
    raw_data = context["ti"].xcom_pull(key="raw_data", task_ids="extract")
    print(f"[TRANSFORM] Starting with {len(raw_data)} records")

    # --- Step 1: JSON → DataFrame ---
    df = pd.DataFrame(raw_data)
    print(f"[TRANSFORM] Columns: {list(df.columns)}")
    print(f"[TRANSFORM] Shape: {df.shape}")

    # --- Step 2: Standardize species names ---
    # Fix inconsistent casing and missing "Iris-" prefix
    df["species"] = df["species"].str.strip().str.lower()

    species_mapping = {
        "iris-setosa": "setosa",
        "setosa": "setosa",
        "iris-versicolor": "setosa" if False else "versicolor",  # keep readable
        "iris versicolor": "versicolor",
        "versicolor": "versicolor",
        "iris-virginica": "virginica",
        "virginica": "virginica",
    }

    # Apply mapping — handle any format
    def normalize_species(s):
        s_clean = s.strip().lower().replace(" ", "-")
        # Remove 'iris-' prefix if present, then map
        base = s_clean.replace("iris-", "")
        if base in ["setosa", "versicolor", "virginica"]:
            return base
        return s_clean  # fallback

    df["species"] = df["species"].apply(normalize_species)
    print(f"[TRANSFORM] Species distribution:\n{df['species'].value_counts()}")

    # --- Step 3: Handle nulls with median imputation ---
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    null_counts = df[numeric_cols].isnull().sum()
    print(f"[TRANSFORM] Null counts before imputation:\n{null_counts}")

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[TRANSFORM] Filled {col} nulls with median: {median_val}")

    # --- Step 4: Remove invalid values (negatives) ---
    invalid_mask = (df[numeric_cols] < 0).any(axis=1)
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        print(f"[TRANSFORM] Removing {invalid_count} rows with negative values")
        df = df[~invalid_mask]

    # --- Step 5: Remove duplicates ---
    before_dedup = len(df)
    df = df.drop_duplicates(subset=numeric_cols + ["species"], keep="first")
    after_dedup = len(df)
    print(f"[TRANSFORM] Removed {before_dedup - after_dedup} duplicate rows")

    # --- Step 6: Feature engineering ---
    df["sepal_ratio"] = round(df["sepal_length"] / df["sepal_width"], 4)
    df["petal_ratio"] = round(df["petal_length"] / df["petal_width"], 4)
    df["sepal_area"] = round(df["sepal_length"] * df["sepal_width"], 4)
    df["petal_area"] = round(df["petal_length"] * df["petal_width"], 4)

    print(f"[TRANSFORM] Added 4 engineered features")
    print(f"[TRANSFORM] Final shape: {df.shape}")
    print(f"[TRANSFORM] Final columns: {list(df.columns)}")

    # --- Step 7: Encode species labels ---
    le = LabelEncoder()
    df["species_encoded"] = le.fit_transform(df["species"])

    # Drop the 'id' column (not needed for modeling)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Push cleaned data as dict
    context["ti"].xcom_push(key="clean_data", value=df.to_dict(orient="records"))
    context["ti"].xcom_push(key="clean_record_count", value=len(df))
    context["ti"].xcom_push(key="species_classes", value=list(le.classes_))

    return f"Transformed data: {len(df)} clean records with {len(df.columns)} columns"


# ──────────────────────────────────────────────────────────
#  TASK 3: LOAD — Save cleaned data as CSV
# ──────────────────────────────────────────────────────────
def load_data(**context):
    """Save the cleaned DataFrame to CSV."""
    clean_data = context["ti"].xcom_pull(key="clean_data", task_ids="transform")

    df = pd.DataFrame(clean_data)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[LOAD] Saved {len(df)} records to: {PROCESSED_DATA_PATH}")
    print(f"[LOAD] File size: {os.path.getsize(PROCESSED_DATA_PATH)} bytes")
    print(f"[LOAD] Preview:\n{df.head()}")

    return f"Loaded {len(df)} records to CSV"


# ──────────────────────────────────────────────────────────
#  TASK 4: TRAIN MODEL — Random Forest Classifier
# ──────────────────────────────────────────────────────────
def train_model(**context):
    """Train a Random Forest classifier on the cleaned iris data."""
    clean_data = context["ti"].xcom_pull(key="clean_data", task_ids="transform")
    df = pd.DataFrame(clean_data)

    # Define features and target
    feature_cols = [
        "sepal_length", "sepal_width", "petal_length", "petal_width",
        "sepal_ratio", "petal_ratio", "sepal_area", "petal_area",
    ]
    target_col = "species_encoded"

    X = df[feature_cols]
    y = df[target_col]

    # Split: 80% train, 20% test (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[TRAIN] Training set: {X_train.shape[0]} samples")
    print(f"[TRAIN] Test set:     {X_test.shape[0]} samples")

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[TRAIN] Model saved to: {MODEL_PATH}")

    # Push test data for evaluation
    context["ti"].xcom_push(key="X_test", value=X_test.to_dict(orient="records"))
    context["ti"].xcom_push(key="y_test", value=y_test.tolist())
    context["ti"].xcom_push(key="feature_cols", value=feature_cols)

    return "Model training complete"


# ──────────────────────────────────────────────────────────
#  TASK 5: EVALUATE — Score model & log metrics
# ──────────────────────────────────────────────────────────
def evaluate_model(**context):
    """Evaluate the trained model and save metrics."""
    species_classes = context["ti"].xcom_pull(key="species_classes", task_ids="transform")
    X_test_dict = context["ti"].xcom_pull(key="X_test", task_ids="train_model")
    y_test = context["ti"].xcom_pull(key="y_test", task_ids="train_model")

    X_test = pd.DataFrame(X_test_dict)
    y_test = np.array(y_test)

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=species_classes, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    print(f"[EVALUATE] Accuracy: {accuracy:.4f}")
    print(f"[EVALUATE] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=species_classes))
    print(f"[EVALUATE] Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Feature importance
    feature_cols = context["ti"].xcom_pull(key="feature_cols", task_ids="train_model")
    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print(f"[EVALUATE] Feature Importances:")
    for feat, imp in sorted_features:
        print(f"  {feat}: {imp:.4f}")

    # Save metrics to JSON
    metrics = {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "feature_importances": importances,
        "model_params": model.get_params(),
        "evaluated_at": datetime.now().isoformat(),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"[EVALUATE] Metrics saved to: {METRICS_PATH}")

    return f"Model accuracy: {accuracy:.4f}"


# ──────────────────────────────────────────────────────────
#  DAG DEFINITION
# ──────────────────────────────────────────────────────────
with DAG(
    dag_id="iris_etl_ml_pipeline",
    default_args=default_args,
    description="ETL pipeline for Iris dataset with ML model training",
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["etl", "ml", "iris"],
) as dag:

    # Task instances
    t_extract = PythonOperator(
        task_id="extract",
        python_callable=extract_data,
    )

    t_transform = PythonOperator(
        task_id="transform",
        python_callable=transform_data,
    )

    t_load = PythonOperator(
        task_id="load",
        python_callable=load_data,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    # ──────────────────────────────────────────────────────
    #  PIPELINE FLOW
    # ──────────────────────────────────────────────────────
    #
    #   extract → transform → load → train_model → evaluate_model
    #
    t_extract >> t_transform >> t_load >> t_train >> t_evaluate

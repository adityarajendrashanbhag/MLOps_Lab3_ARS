# Iris ETL Pipeline with Apache Airflow (Lab 3)

An end-to-end ETL pipeline that processes the Iris dataset from raw JSON to a trained ML model, orchestrated with Apache Airflow.

## Project Structure

```
iris_airflow_project/
├── dags/
│   └── iris_etl_pipeline.py    # Airflow DAG definition
├── data/
│   ├── raw/
│   │   └── iris_raw.json       # Raw data (messy, with quality issues)
│   └── processed/
│       └── iris_clean.csv      # Output: cleaned data (generated)
├── models/
│   ├── iris_rf_model.pkl       # Output: trained model (generated)
│   └── metrics.json            # Output: evaluation metrics (generated)
├── logs/                       # Airflow logs
├── docker-compose.yml          # Airflow local setup
├── requirements.txt            # Python dependencies
├── test_pipeline_local.py      # Run pipeline without Airflow
└── README.md
```

## Pipeline DAG Flow

```
extract → transform → load → train_model → evaluate_model
```

| Task | Description |
|------|-------------|
| **extract** | Reads raw JSON file (153 records with data quality issues) |
| **transform** | Cleans nulls, fixes casing, removes negatives/duplicates, adds features |
| **load** | Saves cleaned DataFrame to CSV |
| **train_model** | Trains a Random Forest classifier (80/20 stratified split) |
| **evaluate_model** | Computes accuracy, classification report, confusion matrix |

## Data Quality Issues (Intentional)

The raw JSON includes these issues for ETL practice:

- **Inconsistent casing**: `"iris-setosa"`, `"IRIS-SETOSA"`, `"Iris-setosa"`
- **Missing prefix**: `"setosa"` instead of `"Iris-setosa"`
- **Null values**: Missing `sepal_width`, `petal_width`, `sepal_length`
- **Negative values**: Invalid measurement (`-5.8`)
- **Duplicate records**: IDs 151, 152 duplicate earlier rows
- **Inconsistent delimiter**: `"Iris versicolor"` (space instead of hyphen)

## Quick Start

### Option A: Test Locally (No Airflow)

```bash
pip install pandas scikit-learn joblib
python test_pipeline_local.py
```

### Option B: Run with Airflow (Docker)

```bash
# Start Airflow
docker-compose up -d

# Wait ~60 seconds for initialization, then open:
# http://localhost:8080
# Login: admin / admin

# Find "iris_etl_ml_pipeline" DAG and trigger it

# Stop when done
docker-compose down
```

### Option C: Run with Airflow (Standalone)

```bash
# Install
pip install apache-airflow pandas scikit-learn joblib

# Initialize
export AIRFLOW_HOME=$(pwd)
airflow db init
airflow users create --username admin --password admin \
  --firstname Admin --lastname User --role Admin --email admin@example.com

# Copy DAG
cp dags/iris_etl_pipeline.py $AIRFLOW_HOME/dags/

# Start (in separate terminals)
airflow webserver --port 8080
airflow scheduler
```

## Engineered Features

The transform step adds these derived columns:

| Feature | Formula |
|---------|---------|
| `sepal_ratio` | sepal_length / sepal_width |
| `petal_ratio` | petal_length / petal_width |
| `sepal_area` | sepal_length × sepal_width |
| `petal_area` | petal_length × petal_width |

## Screenshots


<img width="888" height="623" alt="image" src="https://github.com/user-attachments/assets/41e335f8-bd91-439f-a34b-d56c13d8fcce" />


<img width="867" height="887" alt="image" src="https://github.com/user-attachments/assets/666b8830-f858-4b9a-af3c-587ebc26a7b7" />


<img width="1910" height="872" alt="image" src="https://github.com/user-attachments/assets/d50a57f4-0429-49e2-8a43-53484e664776" />


## Expected Results

After running the pipeline:

- **Cleaned CSV**: ~148 records (after removing invalids and duplicates)
- **Model accuracy**: ~96–98% on test set
- **Top features**: `petal_length`, `petal_width`, `petal_area`

# IBM Telco Customer Churn Prediction

End-to-end ML pipeline for predicting customer churn using the IBM Telco dataset. Built with **scikit-learn**, **MLflow**, and **pandas** — with 14 passing tests and full experiment tracking.

```python
python main.py  # Full pipeline: load → EDA → preprocess → train → evaluate → serialize
```

## Results

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.8488** |
| Precision | 0.68 |
| Recall | 0.52 |
| F1 | 0.59 |

> **Class imbalance** handled via `class_weight='balanced'` (26.5% churn rate).

## Technology Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Runtime |
| **pandas** | Data loading and manipulation |
| **scikit-learn** | Preprocessing, model training, evaluation |
| **MLflow** | Experiment tracking |
| **joblib** | Model serialization |
| **openpyxl** | Reading Excel files |
| **pytest** | Testing |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

## Project Structure

```
ibm-telco-churn-prediction/
├── main.py                    # Pipeline orchestrator
├── requirements.txt
├── Telco_customer_churn.xlsx  # Source dataset
├── data/raw/                  # Raw data
├── src/
│   ├── data/
│   │   ├── load_data.py       # load_telco_data()
│   │   └── eda.py             # generate_eda_report(), print_eda_report()
│   ├── features/
│   │   └── preprocess.py       # build_preprocessor(), configure_preprocessor()
│   └── models/
│       ├── train.py           # train_and_log() — MLflow training
│       ├── evaluate.py        # evaluate() — metrics
│       └── serialize.py        # serialize_artifacts(), load_artifacts()
├── models/                    # Saved artifacts after running pipeline
│   ├── best_model.joblib       # Fitted best pipeline (preprocessor + classifier)
│   └── preprocessor.joblib     # Fitted ColumnTransformer
├── mlruns/                    # MLflow experiment tracking
└── tests/                     # 14 unit + integration tests
```

## Pipeline Overview

| Step | Description |
|------|-------------|
| **Load** | Reads `Telco_customer_churn.xlsx` via pandas + openpyxl |
| **EDA** | Missing values and class distribution report |
| **Preprocess** | ColumnTransformer: median+StandardScaler (numerical) / mode+OHE (categorical) |
| **Train** | Logistic Regression + Random Forest with `class_weight='balanced'` |
| **Log** | All runs recorded to MLflow (metrics, ROC curves, confusion matrices) |
| **Evaluate** | ROC-AUC, precision, recall, F1 on holdout (20%) |
| **Serialize** | `best_model.joblib` and `preprocessor.joblib` to `models/` |

### Excluded columns (target leakage or non-predictive)

- `Churn Label`, `Churn Score`, `Churn Reason` — target leakage
- `CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code` — identifiers
- `CLTV` — not relevant to churn

### Conventions

- **Target**: `Churn Value` (0 = No Churn, 1 = Churn)
- **Split**: 80/20 with stratification, `random_state=42`
- **Selection**: Highest ROC-AUC on holdout set
- **Serialization**: `joblib` (avoids MLflow security warning from raw pickle)

## Running Tests

```bash
pytest tests/ -v
```

```
tests/test_load_data.py ......                                    [ 35%]
tests/test_preprocess.py ....                                     [ 57%]
tests/test_evaluate.py ....                                      [ 71%]
tests/test_pipeline_integration.py .                              [100%]

14 passed in 2.5s
```

## Loading a Saved Model

```python
import joblib

# Load the fitted pipeline
pipeline = joblib.load("models/best_model.joblib")

# Load just the preprocessor
preprocessor = joblib.load("models/preprocessor.joblib")

# Make predictions on new data
import pandas as pd
df_new = pd.read_excel("new_customers.xlsx")
predictions = pipeline.predict(df_new)
probabilities = pipeline.predict_proba(df_new)[:, 1]
```

## MLflow UI

View experiment tracking in your browser:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open [http://localhost:5000](http://localhost:5000). Compare Random Forest vs Logistic Regression runs, view ROC curves, confusion matrices, and feature importance.
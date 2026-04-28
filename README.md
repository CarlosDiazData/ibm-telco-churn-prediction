# IBM Telco Customer Churn Prediction

End-to-end ML pipeline for predicting customer churn using the IBM Telco dataset. Built with **scikit-learn**, **MLflow**, and **pandas** — with **18 passing tests**, **CLTV-weighted training**, and full experiment tracking.

```bash
python main.py  # Full pipeline: load → EDA → preprocess → train (CLTV-weighted) → evaluate → serialize
```

## Results

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.8514** |
| Precision | 0.6461 |
| Recall | 0.5321 |
| F1 | 0.5836 |
| Best Model | GradientBoosting |

> **CLTV-weighted training**: Model prioritizes accuracy on high-value customers. Class imbalance handled via `class_weight='balanced'` (26.5% churn rate).

## Technology Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Runtime |
| **pandas** | Data loading and manipulation |
| **scikit-learn** | Preprocessing, model training, evaluation |
| **MLflow** | Experiment tracking |
| **joblib** | Model serialization |
| **openpyxl** | Reading Excel files |
| **pytest** | Testing (18 tests) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python3 main.py
```

## Project Structure

```
ibm-telco-churn-prediction/
├── main.py                    # Pipeline orchestrator
├── requirements.txt
├── Telco_customer_churn.xlsx  # Source dataset
├── src/
│   ├── data/
│   │   ├── load_data.py      # load_telco_data()
│   │   └── eda.py            # generate_eda_report(), print_eda_report()
│   ├── features/
│   │   └── preprocess.py     # build_preprocessor(), configure_preprocessor()
│   └── models/
│       ├── train.py          # train_and_log() — MLflow training with sample_weight
│       ├── evaluate.py       # evaluate() — metrics
│       └── serialize.py      # serialize_artifacts(), load_artifacts()
├── models/                   # Saved artifacts after running pipeline
│   ├── best_model.joblib      # Fitted best pipeline (preprocessor + classifier)
│   └── preprocessor.joblib   # Fitted ColumnTransformer
├── mlruns/                   # MLflow experiment tracking
└── tests/                    # 18 unit + integration tests
```

## Pipeline Overview

| Step | Description |
|------|-------------|
| **Load** | Reads `Telco_customer_churn.xlsx` via pandas + openpyxl |
| **EDA** | Missing values and class distribution report |
| **Preprocess** | ColumnTransformer: median+StandardScaler (numerical) / mode+OHE (categorical) |
| **Train** | CLTV-weighted training: LogisticRegression, RandomForest, GradientBoosting, SVM |
| **Log** | All runs recorded to MLflow (metrics, ROC curves, confusion matrices) |
| **Evaluate** | ROC-AUC, precision, recall, F1 on holdout (20%) |
| **Serialize** | `best_model.joblib` and `preprocessor.joblib` to `models/` |

### CLTV-Weighted Training

Sample weights are computed as `CLTV / mean(CLTV)`, giving high-value customers more influence during training. This shifts the model to prioritize correctly predicting churn for valuable customers.

```python
weights_train = df.loc[X_train.index, "CLTV"] / df["CLTV"].mean()
# Weights: min=0.455, max=1.477, mean=1.001
```

### Excluded columns (target leakage or non-predictive)

- `Churn Label`, `Churn Score`, `Churn Reason` — target leakage
- `CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code` — identifiers
- `Latitude`, `Longitude`, `Lat Long` — geographic metadata

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
tests/test_load_data.py ......                              [ 33%]
tests/test_preprocess.py ....                               [ 55%]
tests/test_evaluate.py ....                                [ 77%]
tests/test_pipeline_integration.py ...                     [100%]
tests/test_train.py ....                                   [100%]

18 passed in 3.1s
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

Then open [http://localhost:5000](http://localhost:5000). Compare all model runs, view ROC curves, confusion matrices, and feature importance.

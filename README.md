# IBM Telco Customer Churn Prediction

End-to-end ML pipeline for predicting customer churn using the IBM Telco dataset. Built with scikit-learn, MLflow, and pandas.

## Results

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.8488** |
| Precision | 0.68 |
| Recall | 0.52 |
| F1 | 0.59 |
| Dataset size | 7,043 customers |
| Churn rate | 26.5% (handled with `class_weight='balanced'`) |

## Project Structure

```
ibm-telco-churn-prediction/
├── main.py                        # Pipeline orchestrator (run this)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── Telco_customer_churn.xlsx     # Source dataset (also in data/raw/)
├── data/
│   └── raw/
│       └── Telco_customer_churn.xlsx
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py           # load_telco_data()
│   │   └── eda.py                # generate_eda_report(), print_eda_report()
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocess.py         # build_preprocessor(), configure_preprocessor()
│   └── models/
│       ├── __init__.py
│       ├── train.py              # train_and_log() — MLflow training
│       ├── evaluate.py           # evaluate() — metrics
│       └── serialize.py          # serialize_artifacts(), load_artifacts()
├── models/                        # Created after running pipeline/
│   ├── best_model.joblib          # Fitted best pipeline (preprocessor + classifier)
│   └── preprocessor.joblib         # Fitted ColumnTransformer
├── mlruns/                        # Created by MLflow after training
│   └── <experiment_id>/
│       └── <run_id>/
│           ├── artifacts/         # ROC curve, confusion matrix plots
│           ├── metrics/           # training metrics
│           └── tags/
└── tests/                         # Unit + integration tests
    ├── __init__.py
    ├── conftest.py               # Pytest fixtures
    ├── test_load_data.py
    ├── test_preprocess.py
    ├── test_evaluate.py
    └── test_pipeline_integration.py
```

## Prerequisites

- **Python** 3.10 or higher
- **pip** (Python package manager)
- **Git** (to clone the repo)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/CarlosDiazData/ibm-telco-churn-prediction.git
cd ibm-telco-churn-prediction
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `pandas` — data loading and manipulation
- `scikit-learn` — preprocessing, model training, evaluation
- `mlflow` — experiment tracking
- `joblib` — model serialization
- `openpyxl` — reading Excel files

### 4. Verify Installation

```bash
python -c "import pandas, sklearn, mlflow, joblib, openpyxl; print('All packages OK')"
```

## Running the Pipeline

### Full Pipeline

```bash
python main.py
```

**Expected output:**

```
Loading data...
Loaded 7043 rows × 33 columns

Running EDA...
[EDA report: missing values, class distribution]

Column types — Numerical: 12, Categorical: 16
Train size: 5634, Test size: 1409

Training models (with MLflow tracking)...
MLflow run IDs: {...}

LogisticRegression — ROC-AUC: 0.82..., F1: 0.57...
RandomForest — ROC-AUC: 0.8488, F1: 0.59

Best model: RandomForest (ROC-AUC: 0.8488)

============================================================
PIPELINE COMPLETE
============================================================
Best model: RandomForest
ROC-AUC: 0.8488
F1: 0.59
Precision: 0.68
Recall: 0.52

Artifacts saved to: models/
MLflow tracking: mlruns/
============================================================
```

### What the Pipeline Does

1. **Load** — Reads `Telco_customer_churn.xlsx` via pandas + openpyxl
2. **EDA** — Reports missing values and class imbalance (73.5% / 26.5%)
3. **Preprocess** — ColumnTransformer: median+StandardScaler (numerical) / mode+OHE (categorical)
4. **Train** — Trains Logistic Regression + Random Forest with `class_weight='balanced'`
5. **Log** — Records all runs to MLflow (metrics, ROC curves, confusion matrices)
6. **Evaluate** — Computes ROC-AUC, precision, recall, F1 on holdout (20%)
7. **Serialize** — Saves `best_model.joblib` and `preprocessor.joblib` to `models/`

## Running Tests

```bash
pytest tests/ -v
```

**Expected output:**
```
tests/test_load_data.py ......                                         [ 35%]
tests/test_preprocess.py ....                                          [ 57%]
tests/test_evaluate.py ....                                           [ 71%]
tests/test_pipeline_integration.py .                                   [100%]

14 passed in 2.5s
```

## MLflow UI — Viewing Experiment Tracking

MLflow logs every training run with metrics, parameters, and artifact plots.

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open http://localhost:5000 in your browser.

You can compare the Random Forest vs Logistic Regression runs, view ROC curves, confusion matrices, and feature importance.

## Loading a Saved Model

After running the pipeline, use the saved artifacts for predictions:

```python
import joblib

# Load the fitted pipeline
pipeline = joblib.load("models/best_model.joblib")

# Or load just the preprocessor (for feature engineering)
preprocessor = joblib.load("models/preprocessor.joblib")

# Make predictions on new data
import pandas as pd
df_new = pd.read_excel("new_customers.xlsx")
predictions = pipeline.predict(df_new)
probabilities = pipeline.predict_proba(df_new)[:, 1]
```

## Preprocessing Details

| Column Type | Imputation | Transformation |
|-------------|-----------|----------------|
| Numerical (e.g. Tenure, Monthly Charges) | Median | StandardScaler |
| Categorical (e.g. Contract, PaymentMethod) | Mode | OneHotEncoder |

**Excluded columns** (not predictive features):
- `Churn Label`, `Churn Score`, `Churn Reason` — target leakage
- `CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code` — identifiers
- `CLTV` — not relevant to churn

**Class imbalance**: Handled via `class_weight='balanced'` in both Logistic Regression and Random Forest.

## Project Conventions

- Target column: `Churn Value` (0 = No Churn, 1 = Churn)
- Train/Test split: 80/20 with stratification on `y`
- Random state: 42 (reproducible)
- Best model selection: highest ROC-AUC on holdout set
- Serialization: `joblib` (not pickle, to avoid MLflow security warning — but still joblib which wraps pickle)

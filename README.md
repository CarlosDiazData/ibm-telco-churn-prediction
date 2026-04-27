# IBM Telco Churn Prediction

End-to-end ML pipeline for predicting customer churn using the IBM Telco dataset.

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.8488** |
| Precision | 0.68 |
| Recall | 0.52 |
| F1 | 0.59 |
| Churn Rate | 26.5% (balanced with `class_weight='balanced'`) |

## Project Structure

```
├── main.py                  # Pipeline orchestrator
├── requirements.txt         # Python dependencies
├── data/
│   └── raw/
│       └── Telco_customer_churn.xlsx   # Source dataset
├── src/
│   ├── data/
│   │   ├── load_data.py     # Data ingestion
│   │   └── eda.py           # Exploratory data analysis
│   ├── features/
│   │   └── preprocess.py    # ColumnTransformer preprocessing
│   └── models/
│       ├── train.py         # Model training + MLflow logging
│       ├── evaluate.py      # Metrics evaluation
│       └── serialize.py     # Model/preprocessor serialization
├── models/                  # Saved artifacts (after running)
│   ├── best_model.joblib
│   └── preprocessor.joblib
├── mlruns/                  # MLflow experiment tracking
└── tests/                  # Unit + integration tests
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Test

```bash
pytest tests/ -v
```

## MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Models

Trained with `class_weight='balanced'` to handle class imbalance:

- **Logistic Regression** (baseline)
- **Random Forest** (best model, ROC-AUC 0.8488)

## Preprocessing

- **Numerical**: Median imputation → StandardScaler
- **Categorical**: Mode imputation → OneHotEncoder

Excludes churn metadata columns (`Churn Label`, `Churn Score`, `Churn Reason`) from features.

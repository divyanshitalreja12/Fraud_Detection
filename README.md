# Intelligent Financial Fraud Detection using a Deep MLP

A Python project for the IEEE-CIS Fraud Detection dataset. It joins transaction and identity records, performs time-ordered train/validation/test splitting, fits preprocessing on training data only, trains multiple MLP configurations, selects a model using validation PR-AUC, selects a decision threshold on validation data, and evaluates once on the held-out test set.

## 1. Problem

Predict `isFraud` for each labelled transaction:

- `0`: legitimate transaction
- `1`: fraudulent transaction

This is binary classification. The dataset is imbalanced, so PR-AUC, recall, precision and F1 are important alongside ROC-AUC and accuracy.

Dataset source: [IEEE-CIS Fraud Detection on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data).

Kaggle describes the transaction and identity files as joined by `TransactionID`; identity data is not available for every transaction. The Kaggle competition rules must be accepted before downloading the files.

## 2. Project structure

```text
financial_fraud_deep_mlp/
├── config.py
├── run_project.py
├── eda.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   ├── train_transaction.csv
│   │   └── train_identity.csv
│   └── processed/
├── outputs/
└── src/
    ├── __init__.py
    ├── data.py
    ├── preprocessing.py
    ├── model.py
    ├── metrics.py
    └── plots.py
```

## 3. Environment setup

Recommended: Python 3.10 or 3.11 in a virtual environment.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation fails, use a supported Python version and consult the current TensorFlow installation guide.

## 4. Download the data

1. Sign in to Kaggle.
2. Open the [competition data page](https://www.kaggle.com/competitions/ieee-fraud-detection/data).
3. Accept the competition rules.
4. Download `train_transaction.csv` and `train_identity.csv`.
5. Put both files directly inside `data/raw/`.

The project only needs the two labelled training CSVs for its own train/validation/test experiment. Do not rename them.

Expected paths:

```text
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

You can use the Kaggle CLI instead, after configuring Kaggle authentication locally:

```bash
pip install --upgrade kaggle
kaggle competitions download -c ieee-fraud-detection -p data/raw
```

Then extract the ZIP contents into `data/raw/`. Follow the official [Kaggle API authentication guide](https://www.kaggle.com/docs/api). Never paste an API token into source code or commit credentials.

## 5. Run EDA

From the project root:

```bash
python eda.py
```

It prints dataset dimensions, class distribution and missingness, and saves plots in `outputs/`.

## 6. Train and evaluate

```bash
python run_project.py
```

The first run can take a while and requires substantial RAM and disk space. Close memory-heavy applications. If the process runs out of memory, do not silently reduce the dataset; first try a machine with more RAM or ask to adapt the pipeline to chunked/Parquet processing.

The training script:
- Sorts transactions by `TransactionDT` (a relative time offset, not a real timestamp).
- Uses the earliest 70% for training, next 15% for validation and final 15% for test.
- Fits imputation, frequency encoding and scaling on training data only.
- Uses class weights calculated from the training labels.
- Compares six controlled MLP configurations.
- Selects the best model using validation PR-AUC.
- Selects an F1 threshold using validation predictions only.
- Evaluates the selected model once on the held-out test set.

## 7. Outputs

The `outputs/` folder will contain:
- `experiment_comparison.csv`: validation metrics for each experiment.
- `best_model.keras`: selected MLP.
- `preprocessor.joblib`: fitted preprocessing pipeline.
- `model_selection.json`: chosen experiment and validation threshold.
- `final_test_metrics.json`: final held-out test metrics.
- Training curves for each experiment.
- `test_roc_curve.png`, `test_pr_curve.png`, `test_confusion_matrix.png`.

Actual metrics are generated when you run the project; do not write expected or invented scores in your report.

## 8. Important methodological notes

- `TransactionID` is excluded from model features.
- The identity table is left-joined so transactions without identity data remain.
- `TransactionDT` is used for chronological ordering and derived time features; it is not interpreted as a calendar date.
- Frequency encoding is fitted only on training data. Unseen validation/test categories map to zero.
- Numerical medians and scaling parameters are learned only from training data.
- The test split is not used for architecture selection or threshold tuning.
- The Kaggle competition test files have no `isFraud` labels; this project uses a chronological holdout carved from the labelled training data.
- The chronological split can have different fraud rates across periods. Report each split's class counts and discuss this limitation.
- Frequency encoding is a practical baseline for this MLP, but it does not learn category embeddings. That can be explored as a future improvement.
- The dataset is subject to Kaggle competition terms. Do not redistribute the raw data.



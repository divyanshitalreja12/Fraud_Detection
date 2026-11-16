from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_TRANSACTION = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY = RAW_DATA_DIR / "train_identity.csv"

TARGET = "isFraud"
ID_COLUMN = "TransactionID"
TIME_COLUMN = "TransactionDT"

RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

EPOCHS = 30
BATCH_SIZE = 2048
PATIENCE = 5

for folder in (RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

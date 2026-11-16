import numpy as np
import pandas as pd
from config import (
    TRAIN_TRANSACTION, TRAIN_IDENTITY, TARGET, ID_COLUMN, TIME_COLUMN,
    TRAIN_FRACTION, VALIDATION_FRACTION
)

def load_and_merge_data():
    """Load labelled transaction and identity files; retain every transaction."""
    if not TRAIN_TRANSACTION.exists():
        raise FileNotFoundError(
            f"Missing {TRAIN_TRANSACTION}. Download train_transaction.csv "
            "from the Kaggle competition and place it in data/raw/."
        )
    if not TRAIN_IDENTITY.exists():
        raise FileNotFoundError(
            f"Missing {TRAIN_IDENTITY}. Download train_identity.csv "
            "from the Kaggle competition and place it in data/raw/."
        )

    print("Loading transaction data...")
    transactions = pd.read_csv(TRAIN_TRANSACTION, low_memory=False)
    print("Loading identity data...")
    identity = pd.read_csv(TRAIN_IDENTITY, low_memory=False)

    if transactions[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate TransactionID values in transaction file.")
    if identity[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate TransactionID values in identity file.")

    df = transactions.merge(
        identity, on=ID_COLUMN, how="left", validate="one_to_one"
    )
    del transactions, identity

    if TARGET not in df.columns or TIME_COLUMN not in df.columns:
        raise ValueError(f"Expected columns {TARGET!r} and {TIME_COLUMN!r}.")

    print(f"Merged data shape: {df.shape}")
    return df


def add_features(df):
    """Create simple features available at transaction-scoring time."""
    data = df.copy()
    amount = pd.to_numeric(data["TransactionAmt"], errors="coerce")
    data["TransactionAmt_log1p"] = np.log1p(amount.clip(lower=0))
    # TransactionDT is a relative time offset, not a calendar timestamp.
    seconds = pd.to_numeric(data[TIME_COLUMN], errors="coerce")
    data["Transaction_day_index"] = np.floor(seconds / 86400)
    data["Transaction_hour"] = np.floor(seconds / 3600) % 24
    data["has_identity"] = data.filter(regex=r"^(id_|Device)").notna().any(axis=1).astype("int8")
    return data


def chronological_split(df):
    """Sort by relative transaction time and split 70/15/15."""
    data = df.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)
    n = len(data)
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))

    train = data.iloc[:train_end].copy()
    validation = data.iloc[train_end:val_end].copy()
    test = data.iloc[val_end:].copy()

    for name, part in (("train", train), ("validation", validation), ("test", test)):
        counts = part[TARGET].value_counts().to_dict()
        print(f"{name}: {part.shape}; class counts={counts}")
        if part[TARGET].nunique() < 2:
            raise ValueError(
                f"{name} split contains only one target class. "
                "Review the time split before training."
            )
    return train, validation, test


def make_xy(part):
    y = part[TARGET].astype("int8").to_numpy()
    X = part.drop(columns=[TARGET, ID_COLUMN], errors="ignore").copy()
    return X, y

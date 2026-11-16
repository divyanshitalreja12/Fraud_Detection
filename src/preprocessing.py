import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Integer-coded categories in this dataset should not be treated as quantities.
CATEGORICAL_NAMES = {
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo"
}
CATEGORICAL_NAMES.update(f"M{i}" for i in range(1, 10))
CATEGORICAL_NAMES.update(f"id_{i}" for i in range(12, 39))


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Map each category to its training-set frequency; unseen values map to 0."""
    def fit(self, X, y=None):
        frame = pd.DataFrame(X).copy()
        self.columns_ = list(frame.columns)
        self.maps_ = {}
        for col in self.columns_:
            values = frame[col].astype("string").fillna("__MISSING__")
            self.maps_[col] = values.value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.columns_).copy()
        output = []
        for col in self.columns_:
            values = frame[col].astype("string").fillna("__MISSING__")
            encoded = values.map(self.maps_[col]).fillna(0.0)
            output.append(encoded.to_numpy(dtype=np.float32))
        if not output:
            return np.empty((len(frame), 0), dtype=np.float32)
        return np.column_stack(output).astype(np.float32)


def identify_columns(X):
    categorical = []
    for col in X.columns:
        if col in CATEGORICAL_NAMES or (
            pd.api.types.is_object_dtype(X[col]) or
            isinstance(X[col].dtype, pd.StringDtype) or
            pd.api.types.is_categorical_dtype(X[col])
        ):
            categorical.append(col)
    numerical = [c for c in X.columns if c not in categorical]
    return numerical, categorical


def build_preprocessor(X_train):
    numerical, categorical = identify_columns(X_train)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("frequency", FrequencyEncoder())
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical),
            ("categorical", categorical_pipeline, categorical)
        ],
        remainder="drop",
        sparse_threshold=0.0
    )
    return transformer, numerical, categorical

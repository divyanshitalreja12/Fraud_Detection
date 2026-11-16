import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype("int8")
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "tn": int(confusion_matrix(y_true, predictions, labels=[0, 1])[0, 0]),
        "fp": int(confusion_matrix(y_true, predictions, labels=[0, 1])[0, 1]),
        "fn": int(confusion_matrix(y_true, predictions, labels=[0, 1])[1, 0]),
        "tp": int(confusion_matrix(y_true, predictions, labels=[0, 1])[1, 1])
    }


def best_f1_threshold(y_true, probabilities):
    # Evaluate candidate thresholds from the validation predictions only.
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.01, 0.99, 199)))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        metrics = calculate_metrics(y_true, probabilities, float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold, best_f1

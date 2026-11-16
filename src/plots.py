from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay


def plot_history(history, output_dir, name):
    output_dir = Path(output_dir)
    hist = history.history

    for metric, title, filename in [
        ("loss", "Training and validation loss", f"{name}_loss.png"),
        ("pr_auc", "Training and validation PR-AUC", f"{name}_pr_auc.png"),
        ("roc_auc", "Training and validation ROC-AUC", f"{name}_roc_auc.png")
    ]:
        val_metric = f"val_{metric}"
        if metric not in hist or val_metric not in hist:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(hist[metric], label=f"Training {metric}")
        plt.plot(hist[val_metric], label=f"Validation {metric}")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()


def plot_final_curves(y_true, probabilities, output_dir):
    output_dir = Path(output_dir)

    plt.figure(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, probabilities)
    plt.title("Final test ROC curve")
    plt.tight_layout()
    plt.savefig(output_dir / "test_roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, probabilities)
    plt.title("Final test precision-recall curve")
    plt.tight_layout()
    plt.savefig(output_dir / "test_pr_curve.png", dpi=150)
    plt.close()


def plot_confusion(y_true, probabilities, threshold, output_dir):
    predictions = (probabilities >= threshold).astype("int8")
    cm = confusion_matrix(y_true, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Final test confusion matrix")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "test_confusion_matrix.png", dpi=150)
    plt.close()

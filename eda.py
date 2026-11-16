import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.data import load_and_merge_data, add_features
from config import OUTPUT_DIR, TARGET

def main():
    df = add_features(load_and_merge_data())

    print("\\nShape:", df.shape)
    print("\\nTarget counts:")
    print(df[TARGET].value_counts())
    print("\\nTarget percentages:")
    print((df[TARGET].value_counts(normalize=True) * 100).round(3))
    print("\\nTop missing-value columns:")
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    print(missing.head(25).round(2))

    missing.head(25).sort_values().plot(
        kind="barh", figsize=(10, 8), title="Top 25 columns by missing percentage"
    )
    plt.xlabel("Missing values (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_missing_values.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Fraud target distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_target_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df, x="TransactionAmt", hue=TARGET,
        bins=80, log_scale=(False, True), element="step",
        stat="density", common_norm=False
    )
    plt.title("Transaction amount distribution by target")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_transaction_amount.png", dpi=150)
    plt.close()

    print("\\nEDA plots saved in outputs/")

if __name__ == "__main__":
    main()

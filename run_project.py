import gc
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from config import OUTPUT_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, EPOCHS, BATCH_SIZE, PATIENCE
from src.data import load_and_merge_data, add_features, chronological_split, make_xy
from src.preprocessing import build_preprocessor
from src.model import build_mlp
from src.metrics import calculate_metrics, best_f1_threshold
from src.plots import plot_history, plot_final_curves, plot_confusion


def main():
    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print("\\n[1/7] Loading and merging data")
    df = add_features(load_and_merge_data())

    print("\\n[2/7] Chronological train/validation/test split")
    train_df, val_df, test_df = chronological_split(df)
    X_train, y_train = make_xy(train_df)
    X_val, y_val = make_xy(val_df)
    X_test, y_test = make_xy(test_df)

    # Free the large merged frames as soon as possible.
    del df, train_df, val_df, test_df
    gc.collect()

    print("\\n[3/7] Fit preprocessing on training data only")
    preprocessor, numerical_cols, categorical_cols = build_preprocessor(X_train)
    X_train_np = preprocessor.fit_transform(X_train).astype("float32")
    X_val_np = preprocessor.transform(X_val).astype("float32")
    X_test_np = preprocessor.transform(X_test).astype("float32")
    print("Processed shapes:", X_train_np.shape, X_val_np.shape, X_test_np.shape)
    print("Numerical columns:", len(numerical_cols))
    print("Categorical columns:", len(categorical_cols))

    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")
    del X_train, X_val, X_test
    gc.collect()

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Training class weights:", class_weights)

    experiments = [
        {
            "name": "baseline",
            "hidden_layers": (128, 64, 32),
            "optimizer_name": "adam",
            "learning_rate": 1e-3,
            "dropout_rate": 0.0,
            "l2_strength": 0.0,
            "batch_norm": False
        },
        {
            "name": "larger_mlp",
            "hidden_layers": (256, 128, 64),
            "optimizer_name": "adam",
            "learning_rate": 1e-3,
            "dropout_rate": 0.0,
            "l2_strength": 0.0,
            "batch_norm": False
        },
        {
            "name": "rmsprop",
            "hidden_layers": (256, 128, 64),
            "optimizer_name": "rmsprop",
            "learning_rate": 1e-3,
            "dropout_rate": 0.0,
            "l2_strength": 0.0,
            "batch_norm": False
        },
        {
            "name": "dropout_l2_adamw",
            "hidden_layers": (256, 128, 64),
            "optimizer_name": "adamw",
            "learning_rate": 1e-3,
            "dropout_rate": 0.30,
            "l2_strength": 1e-4,
            "batch_norm": False
        },
        {
            "name": "batchnorm_dropout",
            "hidden_layers": (256, 128, 64),
            "optimizer_name": "adam",
            "learning_rate": 1e-3,
            "dropout_rate": 0.20,
            "l2_strength": 0.0,
            "batch_norm": True
        },
        {
            "name": "lower_lr_scheduler",
            "hidden_layers": (256, 128, 64),
            "optimizer_name": "adam",
            "learning_rate": 1e-4,
            "dropout_rate": 0.20,
            "l2_strength": 0.0,
            "batch_norm": True
        }
    ]

    print("\\n[4/7] Training controlled MLP experiments")
    comparison = []
    best_name, best_pr_auc, best_model = None, -1.0, None

    for exp in experiments:
        name = exp["name"]
        print(f"\\n--- Experiment: {name} ---")
        model = build_mlp(
            input_dim=X_train_np.shape[1],
            hidden_layers=exp["hidden_layers"],
            optimizer_name=exp["optimizer_name"],
            learning_rate=exp["learning_rate"],
            dropout_rate=exp["dropout_rate"],
            l2_strength=exp["l2_strength"],
            batch_norm=exp["batch_norm"]
        )

        checkpoint_path = OUTPUT_DIR / f"{name}_best.keras"
        callbacks = [
            EarlyStopping(
                monitor="val_pr_auc", mode="max",
                patience=PATIENCE, restore_best_weights=True
            ),
            ModelCheckpoint(
                checkpoint_path, monitor="val_pr_auc",
                mode="max", save_best_only=True, verbose=0
            )
        ]
        if name == "lower_lr_scheduler":
            callbacks.append(ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, min_lr=1e-6, verbose=1
            ))

        history = model.fit(
            X_train_np, y_train,
            validation_data=(X_val_np, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2
        )

        # Reload best checkpoint so evaluation uses the best validation epoch.
        model = tf.keras.models.load_model(checkpoint_path)
        val_prob = model.predict(X_val_np, batch_size=4096, verbose=0).ravel()
        metrics = calculate_metrics(y_val, val_prob, threshold=0.5)
        metrics["experiment"] = name
        metrics["best_val_pr_auc_during_training"] = float(max(history.history["val_pr_auc"]))
        metrics["epochs_trained"] = len(history.history["loss"])
        comparison.append(metrics)

        plot_history(history, OUTPUT_DIR, name)

        print(
            f"Validation PR-AUC={metrics['pr_auc']:.5f}; "
            f"ROC-AUC={metrics['roc_auc']:.5f}; "
            f"F1@0.5={metrics['f1']:.5f}"
        )

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_name = name
            best_model = model
            best_model.save(OUTPUT_DIR / "best_model.keras")

        del model, history
        gc.collect()
        tf.keras.backend.clear_session()

    comparison_df = pd.DataFrame(comparison).sort_values(
        "pr_auc", ascending=False
    )
    comparison_df.to_csv(OUTPUT_DIR / "experiment_comparison.csv", index=False)
    print("\\nExperiment comparison:")
    print(comparison_df.to_string(index=False))

    print("\\n[5/7] Select threshold on validation set only")
    best_model = tf.keras.models.load_model(OUTPUT_DIR / "best_model.keras")
    val_prob = best_model.predict(X_val_np, batch_size=4096, verbose=0).ravel()
    threshold, val_best_f1 = best_f1_threshold(y_val, val_prob)
    print(f"Selected model: {best_name}")
    print(f"Validation-selected threshold: {threshold:.5f}")
    print(f"Validation F1 at selected threshold: {val_best_f1:.5f}")

    with open(OUTPUT_DIR / "model_selection.json", "w") as f:
        json.dump({
            "best_experiment": best_name,
            "selection_metric": "validation PR-AUC",
            "validation_selected_threshold": threshold,
            "validation_best_f1": val_best_f1
        }, f, indent=2)

    print("\\n[6/7] One-time final test evaluation")
    test_prob = best_model.predict(X_test_np, batch_size=4096, verbose=0).ravel()
    test_metrics = calculate_metrics(y_test, test_prob, threshold=threshold)
    with open(OUTPUT_DIR / "final_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("Final test metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

    plot_final_curves(y_test, test_prob, OUTPUT_DIR)
    plot_confusion(y_test, test_prob, threshold, OUTPUT_DIR)

    print("\\n[7/7] Saving processed artifacts")
    np.savez_compressed(
        PROCESSED_DATA_DIR / "test_predictions.npz",
        y_true=y_test,
        probabilities=test_prob
    )
    print(f"Done. Models, plots, tables and metrics are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_mlp(
    input_dim,
    hidden_layers=(128, 64, 32),
    activation="relu",
    optimizer_name="adam",
    learning_rate=1e-3,
    dropout_rate=0.0,
    l2_strength=0.0,
    batch_norm=False
):
    model = keras.Sequential(name="financial_fraud_deep_mlp")
    model.add(layers.Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(layers.Dense(
            units,
            kernel_initializer="he_normal",
            kernel_regularizer=(
                regularizers.l2(l2_strength) if l2_strength > 0 else None
            )
        ))
        if batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activation="sigmoid"))

    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=1e-4
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )
    return model

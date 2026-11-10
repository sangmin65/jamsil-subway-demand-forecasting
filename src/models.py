"""Approved architectures. Import TensorFlow only when building neural networks."""
import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def build_rf(config):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**config["rf"], random_state=config["seed"])


def build_neural(name, window, context_size, config):
    import tensorflow as tf
    from tensorflow import keras
    keras.backend.clear_session()
    keras.utils.set_random_seed(config["seed"])
    tf.config.experimental.enable_op_determinism()
    keras.mixed_precision.set_global_policy("float32")

    def hidden(x, units):
        return keras.layers.Dense(units, activation="relu", kernel_initializer="he_normal", bias_initializer="zeros")(x)

    if name == "mlp":
        model_input = keras.Input(shape=(window * 36 + context_size,), name="flat")
        x = model_input
        for units in config["mlp"]["hidden_units"]:
            x = hidden(x, units)
    elif name == "lstm":
        history = keras.Input(shape=(window * 18, 2), name="history")
        context = keras.Input(shape=(context_size,), name="context")
        depth = config["lstm"].get("layers", 1)
        if depth < 1:
            raise ValueError("LSTM depth must be positive")
        x = history
        for level in range(depth):
            x = keras.layers.LSTM(config["lstm"]["units"], activation="tanh", recurrent_activation="sigmoid",
                                  kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal",
                                  bias_initializer="zeros", unit_forget_bias=True, dropout=0.0,
                                  recurrent_dropout=0.0, stateful=False, return_sequences=level + 1 < depth,
                                  go_backwards=False, unroll=False)(x)
        x = hidden(keras.layers.Concatenate()([x, context]), config["lstm"]["dense_units"])
        model_input = {"history": history, "context": context}
    elif name == "cnn":
        settings = config["cnn"]
        history = keras.Input(shape=(window * 18, 2), name="history")
        context = keras.Input(shape=(context_size,), name="context")
        days = keras.layers.Reshape((window, 18, 2), name="split_days")(history)
        # The same filters operate independently within each day; never across nights.
        depth = settings.get("layers", 1)
        if depth < 1:
            raise ValueError("CNN depth must be positive")
        daily = days
        for level in range(depth):
            name = "daily_convolution" if level == 0 else f"daily_convolution_{level + 1}"
            daily = keras.layers.TimeDistributed(keras.layers.Conv1D(
                settings["filters"], settings["kernel_size"], padding="same",
                activation="relu", kernel_initializer="he_normal", bias_initializer="zeros"),
                name=name)(daily)
        # Retain both day order and hour position rather than averaging away timing.
        x = keras.layers.Flatten(name="ordered_day_hour_features")(daily)
        x = hidden(keras.layers.Concatenate()([x, context]), settings["dense_units"])
        model_input = {"history": history, "context": context}
    else:
        raise ValueError(name)
    rate = float(config.get("head_dropout", 0.0))
    if not 0.0 <= rate < 1.0:
        raise ValueError("head_dropout must be in [0, 1)")
    if rate:
        x = keras.layers.Dropout(rate, seed=config["seed"], name="head_dropout")(x)
    output = keras.layers.Dense(36, activation="linear", kernel_initializer="glorot_uniform", bias_initializer="zeros")(x)
    model = keras.Model(model_input, output, name=name)
    optimizer_name = config.get("optimizer", "adam")
    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(**config["adam"])
    elif optimizer_name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(**config["rmsprop"])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    model.compile(optimizer=optimizer, loss="mean_absolute_error", jit_compile=False)
    return model

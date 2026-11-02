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
        x = keras.layers.LSTM(config["lstm"]["units"], activation="tanh", recurrent_activation="sigmoid",
                              kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal",
                              bias_initializer="zeros", unit_forget_bias=True, dropout=0.0,
                              recurrent_dropout=0.0, stateful=False, return_sequences=False,
                              go_backwards=False, unroll=False)(history)
        x = hidden(keras.layers.Concatenate()([x, context]), config["lstm"]["dense_units"])
        model_input = {"history": history, "context": context}
    else:
        raise ValueError(name)
    output = keras.layers.Dense(36, activation="linear", kernel_initializer="glorot_uniform", bias_initializer="zeros")(x)
    model = keras.Model(model_input, output, name=name)
    model.compile(optimizer=keras.optimizers.Adam(**config["adam"]), loss="mean_absolute_error", jit_compile=False)
    return model

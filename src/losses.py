"""Preserve Huber delta when saving with the installed Keras version."""
from tensorflow import keras

@keras.utils.register_keras_serializable(package="jamsil")
class ConfiguredHuber(keras.losses.Huber):
    def __init__(self, delta=800.0, **kwargs):
        self.configured_delta = float(delta)
        super().__init__(delta=self.configured_delta, **kwargs)

    def get_config(self):
        return {**super().get_config(), "delta": self.configured_delta}

"""CNN day-boundary isolation and ordered, shared feature extraction."""
import unittest
import numpy as np
from src.dataset import read_json
from src.experiment import ROOT
from src.models import build_neural

class DailyCNNTests(unittest.TestCase):
    def test_day_isolation_shared_weights_and_hour_positions(self):
        from tensorflow import keras
        cfg = read_json(ROOT / "configs/cnn_daily.json")
        model = build_neural("cnn", 3, 51, cfg)
        layer = model.get_layer("daily_convolution")
        encoder = keras.Model(model.get_layer("history").output, layer.output)
        day = np.arange(36, dtype="float32").reshape(1,18,2) / 36
        history = np.tile(day, (1,3,1))
        before = encoder(history).numpy()
        np.testing.assert_allclose(before[:,0], before[:,1])
        changed = history.copy(); changed[:,17,:] += 100
        after = encoder(changed).numpy()
        self.assertFalse(np.allclose(before[:,0], after[:,0]))
        np.testing.assert_array_equal(before[:,1:], after[:,1:])
        flat_encoder = keras.Model(model.get_layer("history").output,
                                   model.get_layer("ordered_day_hour_features").output)
        np.testing.assert_array_equal(flat_encoder(history).numpy(), before.reshape(1,-1))
        output = model({"history":history,"context":np.zeros((1,51),"float32")}).numpy()
        self.assertEqual(output.shape, (1,36))
        self.assertTrue(np.isfinite(output).all())
        self.assertEqual(layer.count_params(), (3*2+1)*8)

if __name__ == "__main__":
    unittest.main()

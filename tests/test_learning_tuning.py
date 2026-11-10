"""Synthetic checks for LR scheduling, stopping, optimizer compatibility and restore."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.models import build_neural
from src.experiment import train_neural, predict_neural, ROOT
from src.dataset import read_json, inputs


class LearningTuningTests(unittest.TestCase):
    def config(self):
        cfg = read_json(ROOT/'configs/initial.json')
        cfg['training'].update(epochs=8, batch_size=2, patience=4, min_delta=0)
        cfg['adam']['learning_rate'] = 0.003
        return cfg

    def data(self):
        rng = np.random.default_rng(9)
        return dict(history=rng.normal(size=(4,18,2)).astype('float32'),
                    context=rng.normal(size=(4,3)).astype('float32'),
                    y=np.ones((4,36),dtype='float32')*10)

    def test_scheduler_runs_before_stop_and_keeps_best_checkpoint(self):
        cfg = self.config()
        cfg['lr_plateau'] = dict(factor=.5, patience=2, min_delta=0., cooldown=0, min_lr=.001)
        data = self.data()
        model = build_neural('mlp',1,3,cfg)
        # Fixed validation score forces reductions on epochs 3 and 5, stopping on 5.
        with tempfile.TemporaryDirectory() as temp, patch('src.experiment.metrics',return_value={'mae':10.}):
            out=Path(temp)
            pred,details=train_neural(model,data,data,'mlp',cfg,out)
            curve=read_json(out/'learning_curve.json')
            self.assertEqual((details['best_epoch'],details['stopped_epoch']),(1,5))
            self.assertEqual([r['epoch'] for r in curve if r['lr_reduced']],[3,5])
            np.testing.assert_allclose([r['learning_rate'] for r in curve],[.003,.003,.003,.0015,.0015])
            self.assertAlmostEqual(curve[-1]['next_learning_rate'],.001)
            self.assertEqual(details['total_optimizer_steps'],10)
            from tensorflow import keras
            loaded=keras.models.load_model(out/'model.keras',compile=False)
            np.testing.assert_allclose(pred,predict_neural(loaded,inputs(data,'mlp'),2),rtol=1e-5,atol=1e-4)

    def test_default_adam_unchanged_and_rmsprop_updates_restore(self):
        from tensorflow import keras
        cfg=self.config();data=self.data();x=inputs(data,'lstm')
        default=build_neural('lstm',1,3,cfg)
        initial=[v.numpy().copy() for v in default.trainable_weights]
        expected=default(x,training=False).numpy()
        explicit=copy.deepcopy(cfg);explicit['optimizer']='adam'
        model=build_neural('lstm',1,3,explicit)
        for a,b in zip(initial,model.trainable_weights):np.testing.assert_array_equal(a,b.numpy())
        np.testing.assert_array_equal(expected,model(x,training=False).numpy())
        self.assertEqual(default.optimizer.get_config(),model.optimizer.get_config())
        rms=copy.deepcopy(cfg);rms['optimizer']='rmsprop'
        rms['rmsprop']=dict(learning_rate=.001,rho=.9,momentum=0.,epsilon=1e-7,centered=False,weight_decay=None)
        model=build_neural('lstm',1,3,rms)
        for a,b in zip(initial,model.trainable_weights):np.testing.assert_array_equal(a,b.numpy())
        self.assertTrue(np.isfinite(model.train_on_batch(x,data['y'])))
        self.assertTrue(any(not np.array_equal(a,b.numpy()) for a,b in zip(initial,model.trainable_weights)))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'rms.keras';model.save(path)
            loaded=keras.models.load_model(path)
            self.assertIsInstance(loaded.optimizer,keras.optimizers.RMSprop)
            np.testing.assert_allclose(model(x).numpy(),loaded(x).numpy(),rtol=1e-5,atol=1e-4)


if __name__=='__main__':unittest.main()

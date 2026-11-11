"""Loss scale, MAE monitoring, and serialized loss regression checks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.models import build_neural
from src.experiment import train_neural, ROOT
from src.dataset import read_json

class LossTests(unittest.TestCase):
    def test_losses_monitor_mae_and_restore(self):
        from tensorflow import keras
        data=dict(history=np.zeros((4,18,2),'float32'),context=np.zeros((4,3),'float32'),y=np.full((4,36),10,'float32'))
        for loss in ['mean_absolute_error','huber','mean_squared_error']:
            cfg=read_json(ROOT/'configs/initial.json');cfg.update(loss=loss,huber_delta=800)
            cfg['training'].update(epochs=3,batch_size=2,patience=1,min_delta=0)
            model=build_neural('lstm',1,3,cfg)
            expected=10 if loss=='mean_absolute_error' else (50 if loss=='huber' else 100)
            scored=model.evaluate({'history':data['history'],'context':data['context']},data['y'],verbose=0,return_dict=True)
            self.assertAlmostEqual(scored['loss'],expected,places=4)
            with tempfile.TemporaryDirectory() as temp,patch('src.experiment.metrics',return_value={'mae':10.}):
                _,details=train_neural(model,data,data,'lstm',cfg,Path(temp))
                self.assertEqual(details['stopped_epoch'],2)
                curve=read_json(Path(temp)/'learning_curve.json')
                self.assertLess(curve[0]['train_mae'],11)
                if loss!='mean_absolute_error':self.assertGreater(curve[0]['train_loss'],40)
                loaded=keras.models.load_model(Path(temp)/'model.keras')
                if loss=='huber':
                    self.assertEqual(loaded.loss.get_config()['delta'],800)
                    self.assertAlmostEqual(float(loaded.loss(np.array([[0.]],'float32'),np.array([[1000.]],'float32'))),480000.)

if __name__=='__main__':unittest.main()

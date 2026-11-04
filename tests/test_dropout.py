import unittest
import numpy as np
from src.dataset import read_json
from src.models import build_neural
from src.experiment import ROOT

class DropoutTests(unittest.TestCase):
    def test_same_initial_weights_and_inference_but_stochastic_training(self):
        from tensorflow import keras
        for name,window,file in [('mlp',3,'epoch1000_mlp3.json'),('cnn',7,'cnn_daily.json')]:
            cfg=read_json(ROOT/'configs'/file)
            base=build_neural(name,window,51,cfg)
            weights=[v.numpy().copy() for v in base.trainable_weights]
            rng=np.random.default_rng(5)
            x=rng.normal(size=(4,window*36+51)).astype('float32') if name=='mlp' else {'history':rng.normal(size=(4,window*18,2)).astype('float32'),'context':rng.normal(size=(4,51)).astype('float32')}
            expected=base(x,training=False).numpy()
            cfg['head_dropout']=0.2
            changed=build_neural(name,window,51,cfg)
            self.assertEqual(len(weights),len(changed.trainable_weights))
            for a,b in zip(weights,changed.trainable_weights): np.testing.assert_array_equal(a,b.numpy())
            np.testing.assert_allclose(expected,changed(x,training=False).numpy(),rtol=1e-5,atol=1e-5)
            self.assertIsInstance(changed.layers[-2],keras.layers.Dropout)
            self.assertFalse(np.allclose(changed(x,training=True).numpy(),changed(x,training=True).numpy()))
            layer=changed.get_layer('head_dropout')
            values=layer(np.ones((20000,),dtype='float32'),training=True).numpy()
            self.assertTrue(np.isin(values,[0.0,1.25]).all())
            self.assertAlmostEqual(float(values.mean()),1.0,delta=0.03)

if __name__=='__main__': unittest.main()

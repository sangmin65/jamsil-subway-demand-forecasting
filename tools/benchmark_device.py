"""Short CPU/GPU training benchmark; never a model-selection experiment."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',choices=['cpu','gpu'],required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--epochs',type=int,default=5)
    args=p.parse_args()
    if args.device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    import numpy as np
    from src.dataset import load_frame,fit_preprocessor,build_window,inputs,read_json,sha256
    from src.models import build_neural
    from src.experiment import predict_neural
    if args.warmup<1 or args.epochs<1: raise ValueError('Positive warmup and measured epochs required')
    gpus=tf.config.list_physical_devices('GPU')
    if args.device=='gpu' and not gpus: raise RuntimeError('No GPU detected; refusing CPU fallback benchmark')
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu,True)
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=read_json(ROOT/'configs/epoch1000_lstm.json')
    cfg['adam']['learning_rate']=0.003
    frame=load_frame(args.source)
    splits,names=build_window(frame,cfg,fit_preprocessor(frame),read_json(ROOT/'data/reference/holidays_2024_2025.json'),14)
    model=build_neural('lstm',14,len(names),cfg)
    x=inputs(splits['train'],'lstm'); val=inputs(splits['validation'],'lstm')
    devices=sorted({str(v.value.device) for v in model.weights})
    if args.device=='gpu' and not any('GPU' in d for d in devices): raise RuntimeError('Model weights are not on GPU')
    rows=[]
    class Timing(tf.keras.callbacks.Callback):
        def on_epoch_begin(self,epoch,logs=None): self.started=time.perf_counter()
        def on_epoch_end(self,epoch,logs=None):
            train_seconds=time.perf_counter()-self.started
            predicted=predict_neural(self.model,val,16)
            if not np.isfinite(predicted).all(): raise ValueError('Non-finite validation predictions')
            rows.append({'epoch':epoch+1,'train_seconds':train_seconds,'train_and_validation_seconds':time.perf_counter()-self.started,'warmup':epoch<args.warmup})
    model.fit(x,splits['train']['y'],epochs=args.warmup+args.epochs,batch_size=16,shuffle=True,callbacks=[Timing()],verbose=2)
    measured=rows[args.warmup:]
    result={'purpose':'Device timing only; not a selected model or predictive score', 'device_requested':args.device,
            'tensorflow':tf.__version__,'keras':tf.keras.__version__,'build_info':tf.sysconfig.get_build_info(),
            'gpu_details':[tf.config.experimental.get_device_details(g) for g in gpus], 'weight_devices':devices,
            'model':'lstm','window':14,'batch_size':16,'learning_rate':0.003,'seed':42,
            'joined_sha256':sha256(args.source),'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'src').glob('*.py')},
            'rows':rows,'median_train_seconds':float(np.median([r['train_seconds'] for r in measured])),
            'median_train_and_validation_seconds':float(np.median([r['train_and_validation_seconds'] for r in measured])),
            'final_test_evaluated':False,'checkpoint_io_included':False}
    (args.output/'benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,default=str))

if __name__=='__main__': main()

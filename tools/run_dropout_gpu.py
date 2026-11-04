"""Approved six GPU dropout trials; stop on failed training or verification."""
import argparse, copy, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from run_tuning_batch import command, verify, save, ROOT
sys.path.insert(0,str(ROOT))
from src.dataset import read_json,sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True); p.add_argument('--kbo-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    manifest=read_json(ROOT/'configs/dropout_gpu.json')
    state={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'manifest':manifest,
           'gpu_lock_sha256':sha256(ROOT/'requirements-gpu-wsl-lock.txt'),
           'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'src').glob('*.py')},'final_test_evaluated':False,'results':[]}
    baselines={}
    try:
        for t in manifest['trials']:
            state['current']=t['id']; save(out/'status.json',state)
            cfg=read_json(ROOT/t['base_config']); cfg['models']=[t['model']]; cfg['windows']=[t['window']]
            cfg['adam']['learning_rate']=0.003; cfg['head_dropout']=t['dropout']; cfg['require_gpu']=True
            cfg['followup_approval']=manifest['approval']
            assert cfg['training']['epochs']==1000 and cfg['training']['patience']==50 and cfg['seed']==42
            config_path=out/(t['id']+'.json'); save(config_path,cfg); run=out/t['id']
            command([sys.executable,'-u','-m','src.experiment','train','--config',str(config_path),'--source',str(args.source),'--kbo-source',str(args.kbo_source),'--output',str(run)],out/(t['id']+'.log'))
            report=read_json(run/'report.json'); assert report['final_test_evaluated'] is False and len(report['conditions'])==1
            assert report['code_sha256']==state['code_sha256']
            c=report['conditions'][0]; folder=f"{t['model']}_{t['window']}d"
            assert c['weight_devices'] and all('GPU:' in d for d in c['weight_devices'])
            if t['dropout']==0: baselines[t['model']]=(report,run,copy.deepcopy(cfg))
            br,bpath,bc=baselines[t['model']]
            check=copy.deepcopy(cfg); check['head_dropout']=0.0; assert check==bc
            for key in ['joined_sha256','kbo_sha256','holidays_sha256']: assert report['sources'][key]==br['sources'][key]
            assert read_json(run/folder/'target_dates.json')==read_json(bpath/folder/'target_dates.json')
            verify(run,args.source,out/(t['id']+'_verify.log'))
            metrics=read_json(run/folder/'validation_metrics.json'); baseline=read_json(bpath/folder/'validation_metrics.json')
            curve=read_json(run/folder/'learning_curve.json')
            state['results'].append({**t,'condition':c,'metrics':metrics,'baseline_metrics':baseline,
                'mae_improvement':baseline['overall']['mae']-metrics['overall']['mae'],
                'improvement_pct':100*(baseline['overall']['mae']-metrics['overall']['mae'])/baseline['overall']['mae'],
                'last_train_mae':curve[-1]['train_mae'],'last_validation_mae':curve[-1]['mae'],
                'run':str(run),'model_reload_verified':True,'environment':report['environment'],'sources':report['sources']})
            save(out/'status.json',state)
        state['status']='completed'; state.pop('current',None); state['finished_utc']=datetime.now(timezone.utc).isoformat()
        save(out/'status.json',state)
        save(ROOT/'reports/metrics/dropout_gpu_2026-11-04.json',state)
    except Exception as e:
        state['status']='failed'; state['error']=repr(e); save(out/'status.json',state); raise

if __name__=='__main__': main()

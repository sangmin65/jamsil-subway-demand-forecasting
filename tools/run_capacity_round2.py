"""EXP-010: larger width or two extra layers, paired with original capacity at 3 seeds."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
from run_tuning_batch import ROOT, command, read_json, save, sha256, verify


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--kbo-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest_path=ROOT/'configs/capacity_round2.json'
    manifest=read_json(manifest_path)
    sources=dict(joined_sha256=sha256(args.source),kbo_sha256=sha256(args.kbo_source),
                 holidays_sha256=sha256(ROOT/'data/reference/holidays_2024_2025.json'))
    state=dict(status='preflight',started_utc=datetime.now(timezone.utc).isoformat(),
               manifest=manifest,manifest_sha256=sha256(manifest_path),sources=sources,
               code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'src').glob('*.py')},
               runner_sha256=sha256(Path(__file__)),results=[],previous_seed42=[],final_test_evaluated=False)
    save(out/'status.json',state)
    argv=['--source',str(args.source),'--kbo-source',str(args.kbo_source)]
    try:
        # Check every larger architecture on synthetic inputs before starting any fit.
        for model,spec in manifest['models'].items():
            for variant in ['width','depth']:
                cfg=read_json(ROOT/spec['baseline']/'config.json')
                cfg[model].update(spec[variant]);cfg['followup_approval']=manifest['approval']
                path=out/f'preflight_{model}_{variant}.json';save(path,cfg)
                command([sys.executable,'-m','src.experiment','preflight','--config',str(path),*argv,
                         '--output',str(out/f'preflight_{model}_{variant}')],out/f'preflight_{model}_{variant}.log')
        previous=read_json(ROOT/'reports/metrics/capacity_neural_2026-11-06.json')
        for row in previous['results']:
            run=ROOT/'outputs/capacity-neural-2026-11-06'/row['trial']
            report=read_json(run/'report.json')
            assert not report['final_test_evaluated']
            for key,value in sources.items():assert report['sources'][key]==value
            verify(run,args.source,out/f"previous_{row['trial']}_verify.log")
            state['previous_seed42'].append(dict(trial=row['trial'],run=str(run.relative_to(ROOT)),
                condition=row['condition'],metrics=row['metrics'],reload_verified=True))
        seed_rows=read_json(ROOT/'reports/metrics/seeds_lstm_2026-11-07.json')['results']
        state['status']='running'
        for model,spec in manifest['models'].items():
            base=ROOT/spec['baseline'];bc=read_json(base/'config.json');br=read_json(base/'report.json')
            folder=f"{model}_{spec['window']}d"
            for seed in manifest['seeds']:
                for variant in ['baseline','width','depth']:
                    trial=f'{model}_{variant}_seed{seed}'
                    state['current']=trial;save(out/'status.json',state)
                    reused=variant=='baseline' and (seed==42 or model=='lstm')
                    if reused:
                        run=base if seed==42 else ROOT/next(r['run'] for r in seed_rows if r['seed']==seed)
                    else:
                        cfg=copy.deepcopy(bc)
                        cfg.update(seed=seed,followup_approval=manifest['approval'])
                        if variant!='baseline':cfg[model].update(spec[variant])
                        assert cfg.get('context_mode','calendar_game')=='calendar_game' and cfg['require_gpu']
                        path=out/f'{trial}.json';save(path,cfg)
                        run=out/trial
                        command([sys.executable,'-u','-m','src.experiment','train','--config',str(path),*argv,
                                 '--output',str(run)],out/f'{trial}.log')
                    cfg=read_json(run/'config.json');report=read_json(run/'report.json')
                    c,=report['conditions']
                    assert cfg['seed']==seed and c['model']==model and c['window']==spec['window']
                    expected=copy.deepcopy(bc[model])
                    if variant!='baseline':expected.update(spec[variant])
                    assert cfg[model]==expected
                    for key in ['training','adam','head_dropout','excluded_test_dates']:
                        assert cfg.get(key)==bc.get(key)
                    assert cfg.get('context_mode','calendar_game')=='calendar_game'
                    assert not report['final_test_evaluated']
                    if not reused:assert report['code_sha256']==state['code_sha256']
                    for key,value in sources.items():assert report['sources'][key]==value
                    assert report['environment']['packages']==br['environment']['packages']
                    assert c['weight_devices'] and all('GPU:' in d for d in c['weight_devices'])
                    for filename in ['target_dates.json','context_features.json']:
                        assert read_json(run/folder/filename)==read_json(base/folder/filename)
                    with np.load(run/folder/'validation_predictions.npz') as a,np.load(base/folder/'validation_predictions.npz') as b:
                        np.testing.assert_array_equal(a['truth'],b['truth']);np.testing.assert_array_equal(a['dates'],b['dates'])
                    verify(run,args.source,out/f'{trial}_verify.log')
                    curve=read_json(run/folder/'learning_curve.json')
                    state['results'].append(dict(model=model,variant=variant,seed=seed,reused=reused,
                        run=str(run.relative_to(ROOT)),condition=c,metrics=read_json(run/folder/'validation_metrics.json'),
                        config_sha256=sha256(run/'config.json'),environment=report['environment'],reload_verified=True,
                        last_train_mae=curve[-1]['train_mae'],last_validation_mae=curve[-1]['mae'],
                        last50_best_improvement=min(r['mae'] for r in curve[:-50])-min(r['mae'] for r in curve) if len(curve)>50 else None))
                    save(out/'status.json',state)
        assert len(state['results'])==27 and sum(not r['reused'] for r in state['results'])==22
        state.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat());state.pop('current',None)
        save(out/'status.json',state)
        save(ROOT/'reports/metrics/capacity_round2.json',state)
    except Exception as error:
        state.update(status='failed',error=repr(error));save(out/'status.json',state);raise


if __name__=='__main__':main()

"""EXP-011: sequential LR-schedule, batch-size and optimizer comparisons."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
import sys

import numpy as np
from run_tuning_batch import ROOT, command, read_json, save, sha256, verify


def choose(state, stage, candidates):
    summaries={}
    for candidate in candidates:
        rows=[r for r in state['results'] if r['variant']==candidate]
        assert [r['seed'] for r in rows]==state['manifest']['seeds']
        values=[r['condition']['validation_mae'] for r in rows]
        summaries[candidate]=dict(mean_mae=mean(values),sample_std=stdev(values))
    winner=candidates[0]
    for candidate in candidates[1:]:
        if summaries[candidate]['mean_mae'] < summaries[winner]['mean_mae']-1e-6:
            winner=candidate
    state['stages'].append(dict(stage=stage,candidates=candidates,summary=summaries,winner=winner))
    return winner


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--kbo-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest_path=ROOT/'configs/learning_tuning.json';manifest=read_json(manifest_path)
    base=ROOT/manifest['baseline_run'];bc=read_json(base/'config.json');br=read_json(base/'report.json')
    sources=dict(joined_sha256=sha256(args.source),kbo_sha256=sha256(args.kbo_source),
                 holidays_sha256=sha256(ROOT/'data/reference/holidays_2024_2025.json'))
    state=dict(status='preflight',started_utc=datetime.now(timezone.utc).isoformat(),manifest=manifest,
        manifest_sha256=sha256(manifest_path),sources=sources,
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'src').glob('*.py')},
        runner_sha256=sha256(Path(__file__)),results=[],stages=[],final_test_evaluated=False)
    argv=['--source',str(args.source),'--kbo-source',str(args.kbo_source)]
    def checkpoint():save(out/'status.json',state)
    def fit(cfg,trial,mode='train'):
        path=out/f'{trial}.json';save(path,cfg)
        run=out/trial
        command([sys.executable,'-u','-m','src.experiment',mode,'--config',str(path),*argv,
                 '--output',str(run)],out/f'{trial}.log')
        return run
    def collect(variant,seed,run,reused,expected):
        cfg=read_json(run/'config.json');report=read_json(run/'report.json');c,=report['conditions']
        assert cfg==expected and cfg['seed']==seed
        assert cfg['lstm']==bc['lstm'] and cfg['models']==['lstm'] and cfg['windows']==[14]
        assert cfg.get('head_dropout',0)==0 and cfg.get('context_mode','calendar_game')=='calendar_game'
        assert cfg['training']['epochs']==2000 and cfg['training']['patience']==50
        assert cfg['training']['min_delta']==0 and cfg['training']['shuffle']==bc['training']['shuffle']
        assert cfg['excluded_test_dates']==bc['excluded_test_dates']
        assert c['model']=='lstm' and c['window']==14 and c['parameters']==8356
        assert not report['final_test_evaluated']
        if not reused:assert report['code_sha256']==state['code_sha256']
        for k,v in sources.items():assert report['sources'][k]==v
        assert report['environment']['packages']==br['environment']['packages']
        assert c['weight_devices'] and all('GPU:' in d for d in c['weight_devices'])
        folder='lstm_14d'
        for file in ['target_dates.json','context_features.json']:
            assert read_json(run/folder/file)==read_json(base/folder/file)
        with np.load(run/folder/'validation_predictions.npz') as a,np.load(base/folder/'validation_predictions.npz') as b:
            np.testing.assert_array_equal(a['truth'],b['truth']);np.testing.assert_array_equal(a['dates'],b['dates'])
        verify(run,args.source,out/f'{variant}_seed{seed}_verify.log')
        curve=read_json(run/folder/'learning_curve.json')
        events=[dict(epoch=r['epoch'],old=r['learning_rate'],new=r['next_learning_rate']) for r in curve if r.get('lr_reduced')]
        if not reused:
            assert len(events)==c['lr_reductions']
            steps_per_epoch=int(np.ceil(352/cfg['training']['batch_size']))
            assert c['total_optimizer_steps']==steps_per_epoch*c['stopped_epoch']
            initial=cfg[cfg.get('optimizer','adam')]['learning_rate']
            assert np.isclose(curve[0]['learning_rate'],initial)
            for previous,current in zip(curve,curve[1:]):
                assert current['learning_rate']==previous['next_learning_rate']
            if cfg.get('lr_plateau') is None:assert not events
        state['results'].append(dict(variant=variant,seed=seed,reused=reused,run=str(run.relative_to(ROOT)),
            config=cfg,config_sha256=sha256(run/'config.json'),condition=c,
            metrics=read_json(run/folder/'validation_metrics.json'),environment=report['environment'],
            reload_verified=True,lr_events=events,last_train_mae=curve[-1]['train_mae'],
            last_validation_mae=curve[-1]['mae'],
            last50_best_improvement=min(r['mae'] for r in curve[:-50])-min(r['mae'] for r in curve)))
        checkpoint()
    def train_group(variant,template):
        for seed in manifest['seeds']:
            cfg=copy.deepcopy(template);cfg.update(seed=seed,followup_approval=manifest['approval'])
            state['status']='running';state['current']=f'{variant}_seed{seed}';checkpoint()
            run=fit(cfg,state['current']);collect(variant,seed,run,False,cfg)
    def selected_config(variant):
        return copy.deepcopy(next(r['config'] for r in state['results'] if r['variant']==variant))
    checkpoint()
    try:
        # A short, non-selection audit checks backward compatibility with the old fit path.
        for k,v in sources.items():assert br['sources'][k]==v
        audit=copy.deepcopy(bc);audit['training']['epochs']=5;audit['followup_approval']=manifest['approval']
        state['current']='baseline_prefix_audit';checkpoint()
        audit_run=fit(audit,'baseline_prefix_audit')
        old=read_json(base/'lstm_14d/learning_curve.json')[:5]
        new=read_json(audit_run/'lstm_14d/learning_curve.json')
        for key in ['mae','train_mae']:
            np.testing.assert_allclose([r[key] for r in old],[r[key] for r in new],rtol=1e-6,atol=1e-5)
        state['baseline_prefix_audit']=dict(epochs=5,passed=True,run=str(audit_run.relative_to(ROOT)),not_used_for_selection=True)
        # Validate the alternative optimizer before long runs.
        rms=copy.deepcopy(bc);rms['optimizer']='rmsprop';rms['rmsprop']={**manifest['rmsprop'],'learning_rate':.001}
        fit(rms,'rmsprop_preflight','preflight')
        old_rows=read_json(ROOT/manifest['baseline_results'])['results']
        for seed in manifest['seeds']:
            run=ROOT/next(r['run'] for r in old_rows if r['model']=='lstm' and r['variant']=='baseline' and r['seed']==seed)
            cfg=read_json(run/'config.json')
            for key in ['adam','lstm','training','head_dropout','excluded_test_dates']:
                assert cfg.get(key)==bc.get(key)
            assert cfg.get('optimizer','adam')=='adam' and cfg.get('lr_plateau') is None
            state['current']=f'baseline_seed{seed}';checkpoint();collect('baseline',seed,run,True,cfg)
        scheduled=copy.deepcopy(bc);scheduled['lr_plateau']=manifest['lr_plateau']
        train_group('plateau',scheduled)
        selected=choose(state,'1_schedule',['baseline','plateau']);checkpoint()
        template=selected_config(selected)
        for batch in [8,32]:
            cfg=copy.deepcopy(template);cfg['training']['batch_size']=batch
            train_group(f'batch{batch}',cfg)
        selected=choose(state,'2_batch',[selected,'batch8','batch32']);checkpoint()
        template=selected_config(selected)
        cfg=copy.deepcopy(template);cfg['adam']['learning_rate']=.001
        train_group('adam_lr001',cfg)
        for rate,label in [(.003,'003'),(.001,'001')]:
            cfg=copy.deepcopy(template);cfg['optimizer']='rmsprop'
            cfg['rmsprop']={**manifest['rmsprop'],'learning_rate':rate}
            train_group(f'rmsprop_lr{label}',cfg)
        selected=choose(state,'3_optimizer',[selected,'adam_lr001','rmsprop_lr003','rmsprop_lr001'])
        assert len(state['results'])==21 and sum(not r['reused'] for r in state['results'])==18
        state.update(status='completed',winner=selected,finished_utc=datetime.now(timezone.utc).isoformat())
        state.pop('current',None);checkpoint()
        save(ROOT/'reports/metrics/learning_tuning.json',state)
    except Exception as error:
        state.update(status='failed',error=repr(error));checkpoint();raise


if __name__=='__main__':main()

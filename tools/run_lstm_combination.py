"""EXP-012: three fresh LSTM depth3/batch8 fits and paired 2x2 comparison."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
import sys
import numpy as np
from run_tuning_batch import ROOT, command, read_json, save, sha256, verify


def publish(state):
    groups={v:[r for r in state['results'] if r['variant']==v] for v in ['depth1_batch16','depth1_batch8','depth3_batch16','depth3_batch8']}
    stats={v:dict(mean_mae=mean(r['condition']['validation_mae'] for r in rows),sd=stdev(r['condition']['validation_mae'] for r in rows),mean_minutes=mean(r['condition']['seconds']/60 for r in rows)) for v,rows in groups.items()}
    state['summary']=stats
    state['winner']=min(stats,key=lambda v:stats[v]['mean_mae'])
    comparisons={}
    new=groups['depth3_batch8']
    for v in ['depth1_batch8','depth3_batch16']:
        gains=[next(r for r in groups[v] if r['seed']==n['seed'])['condition']['validation_mae']-n['condition']['validation_mae'] for n in new]
        comparisons[v]=dict(mean_gain=mean(gains),improvement_pct=100*mean(gains)/stats[v]['mean_mae'],winning_seeds=sum(g>0 for g in gains),paired_gains=gains)
    state['comparisons']=comparisons
    save(ROOT/'reports/metrics/lstm_combination.json',state)
    lines=['# EXP-012 LSTM 3층·배치8 조합 비교','',
      '- 승인: DEC-034. 새 학습3회, 기존9개 결과 재사용. 42·907394·892611 동일 시드. 매번 새 초기화.',
      '- 14일 입력, 각 LSTM32유닛·Dense32, Adam0.003 고정, MAE 손실, Dropout0. 최대2000회·조기종료50·min_delta0. 기존 데이터·분할·전처리 유지.',
      '- 2024년352일 학습 / 2025상반기181일 검증. 추가 행사 자료와 하반기 최종 평가 미사용. MSE·Huber 보류.',
      f"- 상태: 모든12개 비교 모델 복원 검증 완료. 시작 {state['started_utc']}, 종료 {state['finished_utc']} (UTC).",'',
      '| 조건 | 평균 MAE | 표본 SD | 평균 분 |','|---|---:|---:|---:|']
    for v,s in stats.items():lines.append(f"| {v} | {s['mean_mae']:.2f} | {s['sd']:.2f} | {s['mean_minutes']:.2f} |")
    lines+=['','## 기존 결과 대비','']
    for v,s in comparisons.items():lines.append(f"- {v} 대비 MAE 감소 {s['mean_gain']:+.2f}명 ({s['improvement_pct']:+.2f}%), 개선 {s['winning_seeds']}/3시드. 양수는 개선, 음수는 악화.")
    lines += [f"- 네 조건 중 평균 최저: {state['winner']}. 최종 모델 채택이나 독립 평가 결과가 아니다.",'',
      '| 조건 | 시드 | MAE | 최선/종료 에포크 | 종료 사유 | 분 | 재사용 |','|---|---:|---:|---|---|---:|---|']
    for v,rows in groups.items():
        for r in rows:
            c=r['condition'];lines.append(f"| {v} | {r['seed']} | {c['validation_mae']:.2f} | {c['best_epoch']}/{c['stopped_epoch']} | {c['stop_reason']} | {c['seconds']/60:.2f} | {r['reused']} |")
    lines += ['','## 해석과 한계','',
      '- 1층·배치8과 비교하면 깊이 추가의 효과, 3층·배치16과 비교하면 배치 감소의 효과를 확인한다. 두 변경의 효과가 단순히 더해진다고 가정하지 않는다.',
      '- 같은 검증기간을 반복 선택에 사용했으므로 평균 차이를 통계적 유의성이나 실전 일반화로 단정하지 않는다. 시드별 방향과 시간 대비 개선을 함께 본다.',
      '- 배치8/16은 에포크당 갱신44/22회로 동일 에포크가 동일 계산 예산은 아니다. 과거 재사용 시간은 통제된 벤치마크가 아니다.',
      '- 팀원 오차 분석은 고정된 기존 RF/LSTM으로 계속한다. 손실 변경은 분석 결과를 확인한 뒤 결정한다.',
      '- 추가 행사 자료의 불일치는 이 실험 입력에 포함되지 않았다. 야구 일정의 전날 가용성은 미검증이며 전체 이용객에는 다른 행사 영향도 포함될 수 있다.',
      '- 모든 조건의 승하차·월별·note별 지표와 학습곡선은 공유 JSON에 보존. 모델 복원 예측, 원본 해시·날짜·정답·환경·GPU를 확인했다.',
      '', '[전체 결과 JSON](../metrics/lstm_combination.json)','']
    (ROOT/'reports/experiments/EXP-012_lstm-combination.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--kbo-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=read_json(ROOT/'configs/lstm_combination.json');old=read_json(ROOT/manifest['baseline_results']);depth=read_json(ROOT/manifest['depth_results'])
    assert old['status']=='completed'
    sources=dict(joined_sha256=sha256(a.source),kbo_sha256=sha256(a.kbo_source),holidays_sha256=sha256(ROOT/'data/reference/holidays_2024_2025.json'))
    assert sources==old['sources']
    state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),manifest=manifest,sources=sources,
      code_sha256={str(f.relative_to(ROOT)):sha256(f) for f in (ROOT/'src').glob('*.py')},results=[],final_test_evaluated=False)
    base=ROOT/next(r['run'] for r in old['results'] if r['variant']=='batch8' and r['seed']==42)
    br=read_json(base/'report.json')
    def checkpoint():save(out/'status.json',state)
    def collect(variant,seed,run,reused,expected):
        cfg=read_json(run/'config.json');rep=read_json(run/'report.json');c,=rep['conditions']
        assert cfg==expected and cfg['seed']==seed and not rep['final_test_evaluated']
        assert cfg['models']==['lstm'] and cfg['windows']==[14] and cfg['adam']['learning_rate']==.003
        assert cfg.get('head_dropout',0)==0 and cfg.get('lr_plateau') is None
        assert cfg['training']['epochs']==2000 and cfg['training']['patience']==50
        assert all(rep['sources'][k]==v for k,v in sources.items())
        assert rep['environment']['packages']==br['environment']['packages']
        assert all('GPU:' in d for d in c['weight_devices']) and c['weight_devices']
        if not reused:assert rep['code_sha256']==state['code_sha256']
        for f in ['target_dates.json','context_features.json']:
            assert read_json(run/'lstm_14d'/f)==read_json(base/'lstm_14d'/f)
        with np.load(run/'lstm_14d/validation_predictions.npz') as x,np.load(base/'lstm_14d/validation_predictions.npz') as y:
            np.testing.assert_array_equal(x['truth'],y['truth']);np.testing.assert_array_equal(x['dates'],y['dates'])
        verify(run,a.source,out/f'{variant}_{seed}_verify.log')
        state['results'].append(dict(variant=variant,seed=seed,run=str(run.relative_to(ROOT)),reused=reused,config=cfg,condition=c,
           metrics=read_json(run/'lstm_14d/validation_metrics.json'),curve_path=str((run/'lstm_14d/learning_curve.json').relative_to(ROOT)),reload_verified=True))
        checkpoint()
    checkpoint()
    try:
        for variant,rows in [('depth1_batch16',[r for r in old['results'] if r['variant']=='baseline']),('depth1_batch8',[r for r in old['results'] if r['variant']=='batch8']),('depth3_batch16',[r for r in depth['results'] if r['model']=='lstm' and r['variant']=='depth'])]:
            assert [r['seed'] for r in rows]==manifest['seeds']
            for r in rows:
                run=ROOT/r['run'];state['current']=f'verify_{variant}_{r["seed"]}';checkpoint();collect(variant,r['seed'],run,True,read_json(run/'config.json'))
        for seed in manifest['seeds']:
            cfg=copy.deepcopy(next(r['config'] for r in state['results'] if r['variant']=='depth3_batch16' and r['seed']==seed))
            cfg['training']['batch_size']=8;cfg['followup_approval']=manifest['approval']
            assert cfg['lstm']['layers']==3 and cfg['lstm']['units']==32
            name=f'depth3_batch8_seed{seed}';state['current']=name;checkpoint();save(out/f'{name}.json',cfg)
            run=out/name
            command([sys.executable,'-u','-m','src.experiment','train','--config',str(out/f'{name}.json'),'--source',str(a.source),'--kbo-source',str(a.kbo_source),'--output',str(run)],out/f'{name}.log')
            collect('depth3_batch8',seed,run,False,cfg)
        assert len(state['results'])==12
        state.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat());state.pop('current',None)
        publish(state);checkpoint()
    except Exception as e:
        state.update(status='failed',error=repr(e));checkpoint();raise


if __name__=='__main__':main()

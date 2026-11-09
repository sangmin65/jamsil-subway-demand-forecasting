"""Publish completed EXP-010 paired capacity comparisons without training."""
from pathlib import Path
from statistics import mean,stdev
from datetime import datetime
import json

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    state=read(ROOT/'reports/metrics/capacity_round2.json')
    assert state['status']=='completed' and not state['final_test_evaluated']
    rows=state['results'];assert len(rows)==27 and all(r['reload_verified'] for r in rows)
    summary=dict(models={},new_training_count=sum(not r['reused'] for r in rows),final_test_evaluated=False,
        new_training_minutes=sum(r['condition']['seconds'] for r in rows if not r['reused'])/60,
        wall_minutes=(datetime.fromisoformat(state['finished_utc'])-datetime.fromisoformat(state['started_utc'])).total_seconds()/60)
    labels={'baseline':'기준','width':'너비 추가 확대','depth':'기준에서 두 층 추가'}
    for model in state['manifest']['models']:
        stats={}
        base={r['seed']:r for r in rows if r['model']==model and r['variant']=='baseline'}
        for variant in labels:
            group=[r for r in rows if r['model']==model and r['variant']==variant]
            assert [r['seed'] for r in group]==state['manifest']['seeds']
            vals=[r['condition']['validation_mae'] for r in group]
            delta=[base[r['seed']]['condition']['validation_mae']-r['condition']['validation_mae'] for r in group]
            stats[variant]=dict(mean_mae=mean(vals),sample_std=stdev(vals),
                other_two_seed_mean=mean(r['condition']['validation_mae'] for r in group if r['seed']!=42),
                mean_gain=mean(delta),gain_std=stdev(delta),winning_seeds=sum(d>0 for d in delta),
                pairs=[dict(seed=r['seed'],gain=d) for r,d in zip(group,delta)],
                parameters=group[0]['condition']['parameters'],
                new_training_minutes=sum(r['condition']['seconds'] for r in group if not r['reused'])/60,
                overall={k:mean(r['metrics']['overall'][k] for r in group) for k in group[0]['metrics']['overall']},
                groups={g:{label:{k:mean(r['metrics'][g][label][k] for r in group) for k in group[0]['metrics'][g][label]}
                           for label in group[0]['metrics'][g]} for g in ['month','note_status']})
        summary['models'][model]=stats
    save(ROOT/'reports/metrics/capacity_round2_summary.json',summary)
    lines=['# EXP-010 더 큰 너비·두 층 추가 비교','',
        '- 승인: [DEC-032](../decisions/training.md#dec-032), 명세 configs/capacity_round2.json.',
        '- 학습2024년352일 / 검증2025상반기181일. 경기 정보 포함, 기존 원본·전처리·시계열 분할 유지. 하반기 최종 평가 및 보충 행사 자료 미사용.',
        '- 동일 시드42·907394·892611에서 기준/너비/깊이를 비교했다. 기준5개 재사용·신규22개 학습, 비교27개. 재사용 모델도 복원 검증했다.',
        '- MLP3일: 기준64→32, 너비256→32, 깊이64→32→32→32. CNN7일: 기준필터8 한 층, 너비필터32 한 층, 깊이필터8 세 층. LSTM14일: 기준32유닛 한 층, 너비128유닛 한 층, 깊이32유닛 세 층.',
        '- Adam lr0.003, batch16, patience50. MLP/CNN 최대1000·LSTM 최대2000. Dropout MLP/LSTM0·CNN0.1 유지. 너비와 깊이를 동시에 변경하지 않았다. RF는 이번 신경망 구조 비교에서 재학습하지 않았다.','',
        '| 모델 | 조건 | MAE 평균 | 표본 SD | 기준 대비 감소 | 개선 시드 | 승차/하차 MAE | 파라미터 수 |','|---|---|---:|---:|---:|---|---|---:|']
    for model,stats in summary['models'].items():
        for variant,s in stats.items():
            win='—' if variant=='baseline' else f"{s['winning_seeds']}/3"
            lines.append(f"| {model.upper()} | {labels[variant]} | {s['mean_mae']:.2f} | {s['sample_std']:.2f} | {s['mean_gain']:+.2f} | {win} | {s['overall']['board_mae']:.2f}/{s['overall']['alight_mae']:.2f} | {s['parameters']:,} |")
    lines+=['','## 상태별 MAE — 세 시드 평균','','| 모델 | 조건 | 명시 취소 6일 | 취소+일반 1일 | 기록 없음·미확인 95일 | 일반 기록 78일 | 기타 1일 |','|---|---|---:|---:|---:|---:|---:|']
    for model,stats in summary['models'].items():
        for variant,s in stats.items():
            groups=s['groups']['note_status']
            lines.append('| '+model.upper()+' | '+labels[variant]+' | '+' | '.join(f"{groups[k]['mae']:.2f}" for k in ['explicit_cancellation','explicit_cancellation + no_special_note','no_record_unknown','no_special_note','other_note_unknown'])+' |')
    lines+=['','일반 기록은 note에 특별 표기가 없다는 뜻이다. 경기 실제 개최와 예측 전날의 일정 가용성을 확정한 분류가 아니다. 1일·6일 그룹은 표본이 작다.']
    lines+=['','감소가 양수이면 개선이다. 평균/SD는 시드별 MAE에서 계산했으며 앙상블 예측 지표가 아니다.','',
            '## 시드42를 제외한 두 시드 평균','','| 모델 | 기준 | 너비 확대 | 두 층 추가 |','|---|---:|---:|---:|']
    for model,s in summary['models'].items():
        lines.append('| '+model.upper()+' | '+' | '.join(f"{s[v]['other_two_seed_mean']:.2f}" for v in labels)+' |')
    lines+=['','## 시드별 결과와 학습 비용','','| 모델 | 조건 | 시드 | MAE | 기준−이번 | 최선/종료 | 종료 사유 | 분 | 재사용 |','|---|---|---:|---:|---:|---|---|---:|---|']
    for r in rows:
        c=r['condition'];b=next(x for x in rows if x['model']==r['model'] and x['variant']=='baseline' and x['seed']==r['seed'])
        reason={'early_stopping':'조기 종료','maximum_epochs':'상한 도달'}[c['stop_reason']]
        lines.append(f"| {r['model'].upper()} | {labels[r['variant']]} | {r['seed']} | {c['validation_mae']:.2f} | {b['condition']['validation_mae']-c['validation_mae']:+.2f} | {c['best_epoch']}/{c['stopped_epoch']} | {reason} | {c['seconds']/60:.2f} | {'예' if r['reused'] else '아니오'} |")
    lines+=['','재사용 행의 시간은 과거 실행 기록이다. 신규 학습·검증 시간 합계: '+f"{summary['new_training_minutes']:.2f}분. 준비·모델 복원 검증 시간은 제외한다. 전체 실행 경과 시간은 {summary['wall_minutes']:.2f}분이다.",'',
            '## 에포크 상한에 도달한 조건','','| 모델 | 조건 | 시드 | 종료 시 학습 MAE | 종료 시 검증 MAE | 마지막 50회 최선 MAE 개선 |','|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['condition']['stop_reason']=='maximum_epochs':
            lines.append(f"| {r['model'].upper()} | {labels[r['variant']]} | {r['seed']} | {r['last_train_mae']:.2f} | {r['last_validation_mae']:.2f} | {r['last50_best_improvement']:.2f} |")
    lines+=['','후반 개선은 마지막 50회 이전의 최저 검증 MAE와 전체 최저의 차이다. 학습 MAE는 에포크 내 배치 학습 지표이고 검증 MAE는 음수 예측을 0으로 보정한 지표이므로 두 값의 차이를 과적합의 단독 증거로 사용하지 않는다. 조기 종료된 조건은 마지막 50회에 기존 최저값을 갱신하지 못했다.','',
            '## 이전 확장 단계와 비교 — 시드42만','','| 모델 | 기존 기준 | 이전 너비 | 이번 너비 | 이전 한 층 추가 | 이번 두 층 추가 |','|---|---:|---:|---:|---:|---:|']
    for model in summary['models']:
        score=lambda v:next(r['condition']['validation_mae'] for r in rows if r['model']==model and r['variant']==v and r['seed']==42)
        old=lambda v:next(r['condition']['validation_mae'] for r in state['previous_seed42'] if r['trial']==f'{model}_{v}')
        lines.append(f"| {model.upper()} | {score('baseline'):.2f} | {old('width'):.2f} | {score('width'):.2f} | {old('depth'):.2f} | {score('depth'):.2f} |")
    lines+=['','이전 EXP-007은 단일 시드다. 위 표를 3시드 평균 표와 섞어 순위를 매기지 않는다.','','## 모델별 판단','']
    for model,stats in summary['models'].items():
        winner=min(stats,key=lambda v:stats[v]['mean_mae'])
        if winner=='baseline':
            lines.append(f"- {model.upper()}: 이번 3시드 평균에서는 기준 구조가 가장 낮다. 고정한 학습 조건에서 추가 확대의 이득을 확인하지 못했으므로 구조 확대의 우선순위를 낮춘다.")
        elif stats[winner]['winning_seeds']<3:
            s=stats[winner]
            lines.append(f"- {model.upper()}: 수치상 평균 최저는 {labels[winner]}이며 기준 대비 {s['mean_gain']:.2f}명 감소했다. 그러나 개선은 {s['winning_seeds']}/3시드로 방향이 일관되지 않으므로 이 평균만으로 구조 변경을 채택하기는 어렵다. 시드별 차이와 계산 비용을 함께 판단해야 한다.")
        else:
            s=stats[winner]
            lines.append(f"- {model.upper()}: {labels[winner]}가 3시드 평균 최저이며 기준 대비 {s['mean_gain']:.2f}명 감소, {s['winning_seeds']}/3시드에서 개선했다. 시드별 차이와 나머지 두 시드 평균, 계산 비용을 함께 보고 후속 후보로 판단한다.")
    lstm=summary['models']['lstm']
    costs={v:mean(r['condition']['seconds']/60 for r in rows if r['model']=='lstm' and r['variant']==v) for v in labels}
    lines += [f"- LSTM 깊이 확대의 평균 개선율은 {100*lstm['depth']['mean_gain']/lstm['baseline']['mean_mae']:.2f}%다. 기록된 실행당 평균 시간은 기준 {costs['baseline']:.2f}분 → 세 층 {costs['depth']:.2f}분({costs['depth']/costs['baseline']:.2f}배)이다. 기준 시간은 과거 실행이므로 통제된 속도 벤치마크는 아니다. 작은 개선은 인정하되 비용 대비 이득은 제한적이다.",
        '- 곡선에서 MLP/CNN은 학습 오차가 더 내려가는 동안 검증 오차가 정체되는 경향이 보인다. 추가 용량만으로 검증 성능이 개선되지 않았다. LSTM 세 층은 두 시드에서 2000회까지 완만한 개선이 이어져 완전 수렴했다고 단정할 수 없다.',
        '- 다음 실험 제안: 더 큰 구조 탐색의 우선순위를 낮추고 기존 한 층 LSTM에서 고정 학습률과 학습률 감소 스케줄을 비교한다. 후반에 작은 갱신을 허용하는 효과를 구조 변경과 분리해 확인할 수 있다. 세 층은 소폭 개선 후보로 보존하고, 채택하려면 남은 두 시드에서도 먼저 확인한다. 옵티마이저 종류·배치 크기까지 동시에 바꾸지 않는다. 이 제안은 아직 실행하지 않았다.','',
            '## 검증과 해석','','- 기준과 새 조건의 원본·KBO·공휴일 해시, 대상 날짜·정답 배열·입력 특성, 패키지 버전 및 저장 모델 복원 예측을 확인했다. GPU 가중치 배치를 확인했다.',
            '- 기존 하이퍼파라미터를 고정한 비교다. 구조별 학습률·규제 강도를 따로 최적화한 결과는 아니다. 파라미터 수가 다르므로 크기를 맞춘 깊이/너비의 순수 효과라고 해석하지 않는다.',
            '- 세 시드는 결과 확인 전 기존 목록의 첫 세 개로 고정했다. 시드42는 과거 구조 선택에 사용돼 나머지 두 시드 평균도 병기했다. 같은 검증 구간을 반복 사용했으므로 최종 일반화 성능이나 통계적 유의성을 보장하지 않는다.',
            '- 상한 도달은 수렴의 보장이 아니다. 학습/검증 곡선과 후반 개선을 확인한다. 일부 조건의 악화를 모든 큰 모델의 무용성으로 일반화하지 않는다.',
            '- note 그룹은 사후 분석용이며 기록 없음은 미확인이다. 전날 수요 및 일정의 실제 가용성 미검증은 유지한다.',
            '- 상세 월별·note·승하차 지표와 에포크·파라미터·환경은 공유 JSON, 전체 예측·모델·학습곡선·복원 로그는 outputs/capacity-round2-2026-11-09/에 보존한다.','',
            '[전체 결과 JSON](../metrics/capacity_round2.json) · [요약 JSON](../metrics/capacity_round2_summary.json) · [비교 그림](../figures/EXP-010_comparison.png) · [학습곡선](../figures/EXP-010_learning_curves.png)','']
    (ROOT/'reports/experiments/EXP-010_capacity-round2.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(summary,state)
    print(json.dumps({m:{v:dict(mae=s['mean_mae'],gain=s['mean_gain'],wins=s['winning_seeds']) for v,s in stats.items()} for m,stats in summary['models'].items()}))


def plot(summary,state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,(model,stats) in zip(axes,summary['models'].items()):
        bars=ax.bar(['Original','Wider','Deeper'],[s['mean_mae'] for s in stats.values()],
            yerr=[s['sample_std'] for s in stats.values()],capsize=4,color=['#237c93','#bd8433','#8b689e'])
        ax.bar_label(bars,labels=[f"{s['mean_mae']:.2f}" for s in stats.values()],padding=6)
        ax.set(title=model.upper(),ylabel='Validation MAE',ylim=(0,max(s['mean_mae']+s['sample_std'] for s in stats.values())*1.16))
    fig.suptitle('EXP-010 | mean +/- sample SD across 3 paired seeds')
    fig.savefig(ROOT/'reports/figures/EXP-010_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(14,12),layout='constrained')
    for i,model in enumerate(summary['models']):
        for j,seed in enumerate(state['manifest']['seeds']):
            ax=axes[i,j]
            for v,color in [('baseline','#237c93'),('width','#bd8433'),('depth','#8b689e')]:
                r=next(r for r in state['results'] if (r['model'],r['seed'],r['variant'])==(model,seed,v))
                curve=read(ROOT/r['run']/f"{model}_{r['condition']['window']}d"/'learning_curve.json')
                ax.plot([e['epoch'] for e in curve],[e['mae'] for e in curve],color=color,label=v)
                ax.plot([e['epoch'] for e in curve],[e['train_mae'] for e in curve],color=color,linestyle='--',alpha=.5)
            ax.set(title=f'{model.upper()} seed {seed}',xlabel='Epoch',ylabel='MAE',ylim=(0,650));ax.legend(fontsize=8)
    fig.suptitle('Validation (solid) / training (dashed); display clipped to 0-650, full curves saved')
    fig.savefig(ROOT/'reports/figures/EXP-010_learning_curves.png',dpi=120);plt.close(fig)


if __name__=='__main__':main()

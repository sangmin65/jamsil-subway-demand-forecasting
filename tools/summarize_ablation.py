"""Publish EXP-009 only after every paired run and restore verification completes."""
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
import json

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    base = read_json(ROOT/'reports/metrics/ablation_baselines_2026-11-09.json')
    states = {name: read_json(ROOT/f'reports/metrics/ablation_{name}_2026-11-09.json') for name in ['rf','lstm']}
    assert base['status'] == 'completed' and len(base['results']) == 2
    summary = dict(baselines={r['name']:r['metrics'] for r in base['results']}, models={},
                   final_test_evaluated=False, generated_utc=datetime.now(timezone.utc).isoformat())
    for name, state in states.items():
        assert state['status'] == 'completed' and not state['final_test_evaluated']
        assert state['sources'] == base['sources']
        assert len(state['results']) == 5 and all(r['both_model_reload_verified'] for r in state['results'])
        stats = {}
        for mode in ['included','excluded']:
            rows = [r[f'{mode}_metrics'] for r in state['results']]
            overall = {k: mean(r['overall'][k] for r in rows) for k in rows[0]['overall']}
            grouped = {g: {label: {k: mean(r[g][label][k] for r in rows) for k in rows[0][g][label]}
                           for label in rows[0][g]} for g in ['month','note_status']}
            stats[mode] = dict(overall=overall, groups=grouped,
                               sample_std=stdev(r['overall']['mae'] for r in rows),
                               other_four_seed_mean=mean(r[f'{mode}_metrics']['overall']['mae'] for r in state['results'] if r['seed'] !=42))
        delta = [r['excluded_minus_included_mae'] for r in state['results']]
        stats['paired_difference'] = dict(mean=mean(delta), sample_std=stdev(delta),
                                         seeds_helped_by_game=sum(d>0 for d in delta),
                                         pairs=[dict(seed=r['seed'], excluded_minus_included=r['excluded_minus_included_mae']) for r in state['results']])
        stats['included_reduction_vs_last_week_pct'] = 100*(1-stats['included']['overall']['mae']/summary['baselines']['last_week']['overall']['mae'])
        stats['game_reduction_relative_to_excluded_pct'] = 100*mean(delta)/stats['excluded']['overall']['mae']
        stats['new_training_seconds'] = sum(r['excluded_condition']['seconds'] for r in state['results'])
        # Average per-seed metrics, not metrics of an ensemble prediction.
        summary['models'][name] = stats
    save(ROOT/'reports/metrics/ablation_summary_2026-11-09.json', summary)
    lines = ['# EXP-009 단순 기준과 경기 정보 유무 비교', '',
        '- 승인: [DEC-031](../decisions/training.md#dec-031), 명세 configs/ablation_2026-11-09.json.',
        '- 학습 2024-01-15~12-31의 352일, 검증 2025-01-01~06-30의 181일. 기존 원본만 사용, 보충 행사/시범경기 자료 미사용. 하반기 최종 평가 미실행.',
        '- 지난주 기준은 목표일 7일 전의 같은 시간 실제 인원이다. 검증 기간의 이전 관측도 사용한다. 요일·시간대 평균은 학습 352일에서만 계산하고 검증 중 갱신하지 않는다.',
        '- RF 3일 나무200·깊이10·최소잎3, LSTM 14일 32유닛 한 층·Dense32·학습률0.003·batch16·최대2000·patience50·Dropout0. 모델별 다른 조건은 EXP-008 그대로다.',
        '- 경기 포함은 EXP-008 저장 모델 복원 검증 후 재사용. 제외는 동일 5시드로 처음부터 학습. 과거 수요와 달력20개만 유지하며 경기 유무·건수·시각·존재표시·팀 입력을 모두 제거했다.',
        '- 시드: 42·907394·892611·659988·622763. 평균은 각 시드 MAE의 평균이며 예측 앙상블의 성능이 아니다.', '',
        '| 조건 | 검증 MAE | 시드 표준편차 | 승차 MAE | 하차 MAE |', '|---|---:|---:|---:|---:|']
    labels={'last_week':'지난주 같은 시간','weekday_hour_mean':'학습 요일·시간대 평균'}
    for name, scores in summary['baselines'].items():
        s=scores['overall'];lines.append(f"| {labels[name]} | {s['mae']:.2f} | 해당 없음 | {s['board_mae']:.2f} | {s['alight_mae']:.2f} |")
    for name, stats in summary['models'].items():
        for mode, label in [('included','경기 포함'),('excluded','경기 제외')]:
            s=stats[mode]['overall'];lines.append(f"| {name.upper()} {label} | {s['mae']:.2f} | {stats[mode]['sample_std']:.2f} | {s['board_mae']:.2f} | {s['alight_mae']:.2f} |")
    lines += ['', '## 같은 시드에서 경기 정보의 추가 가치', '',
              '차이 = 경기 제외 MAE − 경기 포함 MAE. 양수이면 경기 포함이 좋다. 표준편차는 시드 간 표본 표준편차다.', '',
              '| 모델 | 시드 | 포함 MAE | 제외 MAE | 제외−포함 | 제외 최선/종료 에포크 | 종료 사유 | 신규 학습·검증 분 |',
              '|---|---:|---:|---:|---:|---|---|---:|']
    for name,state in states.items():
        for r in state['results']:
            c=r['excluded_condition'];ep=f"{c['best_epoch']}/{c['stopped_epoch']}" if name=='lstm' else '해당 없음'
            reason={'early_stopping':'조기 종료','maximum_epochs':'상한 도달'}.get(c.get('stop_reason'),'나무 학습 완료')
            lines.append(f"| {name.upper()} | {r['seed']} | {r['included_metrics']['overall']['mae']:.2f} | {r['excluded_metrics']['overall']['mae']:.2f} | {r['excluded_minus_included_mae']:+.2f} | {ep} | {reason} | {c['seconds']/60:.2f} |")
    for name,s in summary['models'].items():
        p=s['paired_difference'];lines += ['',f"- {name.upper()}: 제외−포함 평균 {p['mean']:+.2f}명, 차이의 표본 표준편차 {p['sample_std']:.2f}명. 경기 포함이 낮은 시드 {p['seeds_helped_by_game']}/5.",
            f"- {name.upper()} 시드42 제외한 네 시드 평균: 포함 {s['included']['other_four_seed_mean']:.2f}, 제외 {s['excluded']['other_four_seed_mean']:.2f}명."]
    lines += ['', '## 월별 확인', '', '| 월 | 지난주 | 평균표 | RF 포함 | RF 제외 | LSTM 포함 | LSTM 제외 |','|---|---:|---:|---:|---:|---:|---:|']
    for month in summary['baselines']['last_week']['month']:
        vals=[summary['baselines'][b]['month'][month]['mae'] for b in labels]
        vals += [summary['models'][m][v]['groups']['month'][month]['mae'] for m in ['rf','lstm'] for v in ['included','excluded']]
        lines.append('| '+month+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines += ['', '## 관찰과 다음 단계', '']
    for name,s in summary['models'].items():
        lines.append(f"- {name.upper()} 경기 포함은 지난주 기준보다 MAE {s['included_reduction_vs_last_week_pct']:.2f}% 감소. 경기 제외 대비 감소율 {s['game_reduction_relative_to_excluded_pct']:.2f}%. 신규 학습·검증 합계 {s['new_training_seconds']/60:.2f}분(복원 검사·준비 시간 제외).")
    lines += ['', '| 사후 note 그룹 (날짜 수) | RF 포함 | RF 제외 | LSTM 포함 | LSTM 제외 |', '|---|---:|---:|---:|---:|']
    for group, metric in summary['models']['rf']['included']['groups']['note_status'].items():
        vals=[summary['models'][m][v]['groups']['note_status'][group]['mae'] for m in ['rf','lstm'] for v in ['included','excluded']]
        lines.append('| '+f"{group} ({int(metric['days'])})"+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines += ['', '- 다음으로 검증 구간에서 오차가 큰 날짜·시간대를 확인하고 별도 수집한 행사 기록과 대조하는 것을 권장한다. 전체 평균 개선을 개별 경기/행사의 인과효과로 해석하지 않는다.',
              '- 이번에는 기존 하이퍼파라미터를 고정했다. 경기 제외 조건을 따로 최적화한 최선 모델과의 비교는 아니며, 추가 튜닝이나 신규 자료 결합은 실행하지 않았다.',
              '', '![기준 및 경기 정보 비교](../figures/EXP-009_comparison.png)', '']
    lines += ['[LSTM 5시드 학습·검증 곡선](../figures/EXP-009_lstm_curves.png)', '',
              '학습 곡선은 배치 학습 중 손실이며 검증 MAE와 계산 시점/음수 보정 여부가 다르다. 작은 취소 그룹에서는 전체 평균과 다른 경향이 나올 수 있으므로 날짜 수와 함께 해석한다.']
    lines += ['', '## 해석 범위와 검증', '',
        '- 원본/KBO/공휴일 해시, 학습·검증·보류 날짜, 정답 배열, 모델별 패키지 버전의 일치를 확인했다. 포함·제외 모델 모두 복원 예측 확인(rtol=1e-5, atol=1e-4). LSTM GPU 가중치 배치를 확인했다.',
        '- 입력 제거로 LSTM Dense 파라미터 수와 초기화 대응도 달라진다. 동일한 파라미터 수의 비교나 야구 개최의 인과효과가 아니다. 기존 최선 설정을 고정한 추가 예측 가치 비교다.',
        '- 과거 승하차에는 이미 야구/행사의 영향이 포함될 수 있다. 경기 입력을 빼는 것이 야구 영향 자체를 제거한다는 뜻은 아니다.',
        '- 모델별 과거 창 길이와 학습 조건이 다르다. 단순 기준보다 개선됐다는 사실만으로 경기 정보의 효과를 입증하지 않으며 포함/제외의 짝 비교를 별도로 본다.',
        '- 기존 검증을 반복 사용했으므로 최종 일반화 성능이나 통계적 유의성의 증명이 아니다. 시드42는 과거 선택에 사용돼 다른 네 시드 평균도 함께 보고했다.',
        '- 에포크 상한에서 종료한 조건은 수렴을 보장하지 않는다. 추가 학습으로 더 개선될 가능성과 현재 정해진 학습 예산에서의 비교를 구분한다.',
        '- 예측 전날 실제 수요 확보 및 사후 일정의 당시 가용성 미검증, 기록 없음은 미확인이라는 한계는 유지된다. note는 사후 그룹 평가용이며 입력이 아니다.',
        '- 전체/월별/note별 상세 지표는 공유 JSON, 모델·검증 예측·학습곡선·복원 로그는 outputs/ablation-2026-11-09/에 보관한다.',
        '- [요약 JSON](../metrics/ablation_summary_2026-11-09.json), [기준](../metrics/ablation_baselines_2026-11-09.json), [RF](../metrics/ablation_rf_2026-11-09.json), [LSTM](../metrics/ablation_lstm_2026-11-09.json).', '']
    (ROOT/'reports/experiments/EXP-009_ablation.md').write_text('\n'.join(lines), encoding='utf-8')
    plot(summary, states)
    print({k:v['paired_difference'] for k,v in summary['models'].items()})


def plot(summary, states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Last week', 'Weekday/hour mean', 'RF +game', 'RF -game', 'LSTM +game', 'LSTM -game']
    values = [v['overall']['mae'] for v in summary['baselines'].values()]
    deviations = [0,0]
    for model in ['rf','lstm']:
        for mode in ['included','excluded']:
            values.append(summary['models'][model][mode]['overall']['mae'])
            deviations.append(summary['models'][model][mode]['sample_std'])
    fig, ax = plt.subplots(figsize=(10,5), layout='constrained')
    bars=ax.bar(labels,values,yerr=deviations,capsize=4,color=['#8b9199','#8b9199','#237c93','#a9cbd3','#9b5c9c','#d9bcda'])
    ax.bar_label(bars, labels=[f'{v:.2f}' for v in values],padding=6)
    ax.set(ylabel='Validation MAE (passengers)',title='EXP-009 | 181 validation days | mean +/- sample SD across 5 seeds',ylim=(0,max(values)*1.17))
    dest=ROOT/'reports/figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'EXP-009_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(5,1,figsize=(10,15),layout='constrained')
    for ax,row in zip(axes,states['lstm']['results']):
        for mode,color in [('included','#237c93'),('excluded','#9b5c9c')]:
            curve=read_json(ROOT/row[f'{mode}_run']/'lstm_14d/learning_curve.json')
            ax.plot([r['epoch'] for r in curve],[r['mae'] for r in curve],color=color,label=f'{mode}: validation')
            ax.plot([r['epoch'] for r in curve],[r['train_mae'] for r in curve],color=color,alpha=.5,linestyle='--',label=f'{mode}: training')
        ax.set(title=f"LSTM seed {row['seed']}",xlabel='Epoch',ylabel='MAE',ylim=(0,650));ax.legend(fontsize=8,ncol=2)
    fig.suptitle('EXP-009 | full curves saved in outputs; display clipped to 0-650 MAE')
    fig.savefig(dest/'EXP-009_lstm_curves.png',dpi=120);plt.close(fig)


if __name__ == '__main__':
    main()

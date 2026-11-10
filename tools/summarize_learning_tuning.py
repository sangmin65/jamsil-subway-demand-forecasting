"""Publish completed EXP-011 results; never trains or evaluates the held-out test."""
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, stdev

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    state=read(ROOT/'reports/metrics/learning_tuning.json')
    rows=state['results']
    assert state['status']=='completed' and len(rows)==21 and all(r['reload_verified'] for r in rows)
    assert not state['final_test_evaluated'] and state['baseline_prefix_audit']['passed']
    assert sum(not r['reused'] for r in rows)==18
    variants=list(dict.fromkeys(r['variant'] for r in rows))
    baseline={r['seed']:r for r in rows if r['variant']=='baseline'}
    groups={v:[r for r in rows if r['variant']==v] for v in variants}
    summary=dict(winner=state['winner'],variants={},stages=state['stages'],final_test_evaluated=False,
        new_training_minutes=sum(r['condition']['seconds'] for r in rows if not r['reused'])/60,
        wall_minutes=(datetime.fromisoformat(state['finished_utc'])-datetime.fromisoformat(state['started_utc'])).total_seconds()/60)
    for variant,group in groups.items():
        assert [r['seed'] for r in group]==state['manifest']['seeds']
        values=[r['condition']['validation_mae'] for r in group]
        gains=[baseline[r['seed']]['condition']['validation_mae']-r['condition']['validation_mae'] for r in group]
        summary['variants'][variant]=dict(mean_mae=mean(values),sample_std=stdev(values),mean_gain=mean(gains),
            winning_seeds=sum(d>0 for d in gains),paired_gains=dict(zip(state['manifest']['seeds'],gains)),
            other_two_mean=mean(values[1:]),mean_minutes=mean(r['condition']['seconds']/60 for r in group),
            overall={k:mean(r['metrics']['overall'][k] for r in group) for k in group[0]['metrics']['overall']},
            groups={g:{label:{k:mean(r['metrics'][g][label][k] for r in group) for k in group[0]['metrics'][g][label]}
                       for label in group[0]['metrics'][g]} for g in ['month','note_status']})
    (ROOT/'reports/metrics/learning_tuning_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# EXP-011 LSTM 학습 방법 순차 비교','',
        '- 승인: [DEC-033](../decisions/training.md#dec-033), 명세 `configs/learning_tuning.json`. 2026-11-10 시작, 실제 시작/종료 UTC는 공유 JSON에 보존.',
        '- LSTM14일·32유닛 한 층·Dense32·Dropout0, 최대2000·조기종료50·min_delta0. 2024년352일 학습/2025상반기181일 검증. 기존 원본·경기 포함51특성·전처리 유지, 보충 자료와 하반기 최종 평가 미사용.',
        '- 같은 세 시드42·907394·892611. 신규18회+기준3개 재사용=21개 고유 결과. 매번 새 초기화이며 가중치를 이어 학습하지 않았다. 단계 간 기준 행은 재사용 참조다.',
        '- 스케줄: 검증MAE 0.1명 이상 개선이20회 없으면 lr×0.5, 하한0.00001, cooldown0. 조기종료의 min_delta0과 독립. 기록된 사용학습률과 다음학습률을 구분한다.',
        '- 1단계 정책 선택→2단계 배치8/16/32 선택→3단계 Adam/RMSprop×초기lr0.001/0.003 비교. RMSprop rho0.9·momentum0·epsilon1e-7·centeredFalse·weight_decayNone. Adam은 기존beta/epsilon/amsgrad 유지.',
        '- 각 단계 세 시드 평균MAE 최저를 다음 단계로 전달하고 1e-6 이내 동률은 이전 기준을 유지한다. 최종 모델 채택이나 독립 평가가 아닌 순차 탐색이다.','',
        '| 조건 ID | 옵티마이저 | 초기 LR | 스케줄 | 배치 | 평균 MAE | 표본 SD | 원래 기준 대비 감소 | 개선 시드 | 평균 분 |',
        '|---|---|---:|---|---:|---:|---:|---:|---|---:|']
    for v,group in groups.items():
        cfg=group[0]['config'];opt=cfg.get('optimizer','adam');s=summary['variants'][v]
        lines.append(f"| {v} | {opt} | {cfg[opt]['learning_rate']} | {'정체 시 감소' if cfg.get('lr_plateau') else '고정'} | {cfg['training']['batch_size']} | {s['mean_mae']:.2f} | {s['sample_std']:.2f} | {s['mean_gain']:+.2f} | {s['winning_seeds']}/3 | {s['mean_minutes']:.2f} |")
    lines+=['','감소가 양수면 개선이다. 평균은 시드별 점수 평균이며 앙상블 성능이 아니다. 기존 기준 시간은 과거 기록으로, 통제된 속도 벤치마크는 아니다.','',
            '## 단계별 비교와 선택','']
    for stage in state['stages']:
        reference=stage['candidates'][0]
        lines += [f"### {stage['stage']}",'',f"비교 기준: `{reference}`. 다음 단계 전달 조건: **`{stage['winner']}`**.",'',
                  '| 조건 | 평균 MAE | 단계 기준 대비 감소 | 개선 시드 | 시드42 제외 평균 |','|---|---:|---:|---|---:|']
        reference_rows={r['seed']:r for r in groups[reference]}
        for v in stage['candidates']:
            s=summary['variants'][v]
            delta=[reference_rows[r['seed']]['condition']['validation_mae']-r['condition']['validation_mae'] for r in groups[v]]
            lines.append(f"| {v} | {s['mean_mae']:.2f} | {mean(delta):+.2f} | {sum(d>0 for d in delta)}/3 | {s['other_two_mean']:.2f} |")
    lines+=['','## 시드별 에포크·시간·학습률 변화','','| 조건 | 시드 | MAE | 최선/종료 | 종료 사유 | 분 | LR 감소 횟수 | 갱신 횟수 |','|---|---:|---:|---|---|---:|---:|---:|']
    for r in rows:
        c=r['condition'];steps=c.get('total_optimizer_steps',c['stopped_epoch']*((352+r['config']['training']['batch_size']-1)//r['config']['training']['batch_size']))
        lines.append(f"| {r['variant']} | {r['seed']} | {c['validation_mae']:.2f} | {c['best_epoch']}/{c['stopped_epoch']} | {'조기 종료' if c['stop_reason']=='early_stopping' else '상한 도달'} | {c['seconds']/60:.2f} | {len(r['lr_events'])} | {steps} |")
    lines+=['','기존 기준의 갱신수는 종료 에포크×에포크당 배치 수로 계산했다. 신규 조건은 옵티마이저의 실제 iterations를 기록·확인했다.',
            f"신규18개 학습·검증 시간 합계 {summary['new_training_minutes']:.2f}분. 감사·준비·복원 등을 포함한 전체 경과 시간 {summary['wall_minutes']:.2f}분.",'',
            '## 학습률 감소 시점','']
    for r in rows:
        if r['config'].get('lr_plateau'):
            events=', '.join(f"{e['epoch']}회 뒤 {e['old']:.6g}→{e['new']:.6g}" for e in r['lr_events']) or '감소 발동 없음'
            lines.append(f"- {r['variant']} / 시드{r['seed']}: {events}.")
    lines+=['','## 최종 단계 선택 후보와 원래 기준의 상태별 오차','','| 조건 | 승차 | 하차 | 명시취소6일 | 기록없음95일 | 일반기록78일 |','|---|---:|---:|---:|---:|---:|']
    for v in dict.fromkeys(['baseline',state['winner']]):
        s=summary['variants'][v];g=s['groups']['note_status']
        lines.append(f"| {v} | {s['overall']['board_mae']:.2f} | {s['overall']['alight_mae']:.2f} | {g['explicit_cancellation']['mae']:.2f} | {g['no_record_unknown']['mae']:.2f} | {g['no_special_note']['mae']:.2f} |")
    lines+=['','note는 사후 그룹 분석용이다. 기록 없음은 미확인이고 일반 기록도 실제 개최나 전날 정보 가용성을 확정하지 않는다. 취소+일반1일/기타1일과 월별 지표는 JSON에 보존한다.','',
            '## 검증과 한계','',
            '- 기본 Adam 경로의 첫5에포크 학습/검증MAE가 이전 실행과 허용오차rtol1e-6/atol1e-5 내 일치했다. 이 감사 결과는 조건 선택에 사용하지 않았다. 신규 기능을 포함한 테스트12개 통과 후 실행했다.',
            '- 모든21개 저장 모델 복원, 원본/KBO/공휴일 해시, 대상날짜·정답배열·입력특성, GPU 가중치·패키지 버전을 확인했다. 새 실행의 src 해시도 일치한다.',
            '- 같은 검증 기간과 시드를 반복 사용하여 선택 편향이 있다. 세 시드 평균 차이를 통계적 유의성이나 미사용 기간의 일반화 성능으로 단정하지 않는다. 과거 선택에 사용된 시드42를 제외한 두 시드 평균도 병기했다.',
            '- 앞 단계 선택에 조건부인 탐색으로 모든 정책×배치×옵티마이저 조합의 전수 비교는 아니다. 평균 최저가 시드별로 일관되거나 실용적으로 큰 개선인지 별도로 판단해야 한다.',
            '- batch8/16/32의 에포크당 갱신은44/22/11회다. 동일2000에포크는 동일 계산 예산이 아니다. 배치 변경은 검증 추론 배치 크기도 함께 바뀌며 각 저장 모델의 같은 배치로 복원 검증했다.',
            '- 스케줄의 감소 시점/횟수는 조건별 학습 경로에 따라 다르다. 동일 정책에서 옵티마이저를 바꾸는 비교이지 학습률 경로까지 고정한 비교가 아니다. 상한 도달은 수렴을 보장하지 않는다.',
            f"- 선택된 최선 가중치로 예측을 복원했다. 저장 모델의 옵티마이저 상태를 사용한 이어 학습은 검증/실행하지 않았다. 원본 로그·예측·모델은 {state['output']}/에 보존한다.",
            '- 최초 실행은 Windows의 WSL 업데이트 설치 중 첫 조건804회에서 중단됐다. 부분 결과를 선택에 사용하지 않고 새 폴더에서 초기화해 재실행했다. 위 시간은 재실행 배치 기준이다. [중단 기록](../metrics/learning_tuning_interruption.json)에 원래 출력과 시스템 설치 이벤트를 기록했다.','',
            '[전체 JSON](../metrics/learning_tuning.json) · [요약 JSON](../metrics/learning_tuning_summary.json) · [비교 그림](../figures/EXP-011_comparison.png) · [학습곡선](../figures/EXP-011_learning_curves.png) · [학습률 경로](../figures/EXP-011_learning_rates.png)','']
    (ROOT/'reports/experiments/EXP-011_learning-tuning.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(state,summary,groups)
    print(json.dumps({v:{k:s[k] for k in ['mean_mae','sample_std','mean_gain','winning_seeds']} for v,s in summary['variants'].items()}))


def plot(state,summary,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    stats=summary['variants'];bars=ax.bar(list(stats),[s['mean_mae'] for s in stats.values()],yerr=[s['sample_std'] for s in stats.values()],capsize=4)
    ax.bar_label(bars,labels=[f"{s['mean_mae']:.2f}" for s in stats.values()],padding=6)
    ax.set(title='EXP-011: three paired seeds, mean +/- sample SD',ylabel='Validation MAE',ylim=(0,max(s['mean_mae']+s['sample_std'] for s in stats.values())*1.17))
    fig.savefig(ROOT/'reports/figures/EXP-011_comparison.png',dpi=150);plt.close(fig)
    for rates in [False,True]:
        fig,axes=plt.subplots(3,3,figsize=(15,12),layout='constrained')
        for i,stage in enumerate(state['stages']):
            for j,seed in enumerate(state['manifest']['seeds']):
                ax=axes[i,j]
                for v in stage['candidates']:
                    r=next(r for r in groups[v] if r['seed']==seed)
                    curve=read(ROOT/r['run']/'lstm_14d/learning_curve.json');epochs=[e['epoch'] for e in curve]
                    initial=r['config'][r['config'].get('optimizer','adam')]['learning_rate']
                    if rates:ax.step(epochs,[e.get('learning_rate',initial) for e in curve],where='post',label=v)
                    else:
                        line,=ax.plot(epochs,[e['mae'] for e in curve],label=v)
                        ax.plot(epochs,[e['train_mae'] for e in curve],color=line.get_color(),linestyle='--',alpha=.4)
                ax.set(title=f"{stage['stage']} / seed {seed}",xlabel='Epoch',ylabel='Learning rate' if rates else 'MAE')
                if rates:ax.set_yscale('log');ax.set_ylim(7e-6,.005)
                else:ax.set_ylim(0,650)
                ax.legend(fontsize=8)
        fig.suptitle('Learning rate used in each epoch' if rates else 'Validation solid / training dashed; display clipped to 0-650')
        suffix='learning_rates' if rates else 'learning_curves'
        fig.savefig(ROOT/f'reports/figures/EXP-011_{suffix}.png',dpi=120);plt.close(fig)


if __name__=='__main__':main()

"""Publish completed post-selection audit; no training or final-period access."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    a = json.loads((ROOT/'reports/metrics/robustness_audit.json').read_text(encoding='utf8'))
    lines = ['# ANALYSIS-007 후속 신뢰성·조건별 검증', '',
             '2026-11-11 · DEC-039 · 2025 상반기 181일 기존 시드 42 예측만 분석. 하반기 재평가와 최종 구성 변경 없음.', '',
             '## 데이터·평가 점검', '',
             '- 2호선 종합운동장역의 상반기까지 날짜×18시간 격자, 중복·누락, 유한한 비음수 정답 확인.',
             '- 전처리를 2024 자료로 재계산하여 저장된 평균·표준편차·팀 범주·행 수와 일치 확인.',
             '- RF 3일과 LSTM 14일 입력의 마지막 날짜는 전날. 같은 목표 날짜의 실제 승하차는 과거 입력에 포함하지 않음.',
             '- 두 모델의 날짜·정답·승하차 출력 순서 및 저장 예측 대조, 각 예측 음수 보정 후 50:50 결합 일치 확인.',
             '- context는 달력·수집 기록 수·예정 시각·홈/원정팀. note·gamecenter_status·verified·같은 행의 인원·제공 lag 컬럼은 입력에 포함하지 않음. note는 사후 그룹 분석에만 사용.',
             '- 추가 회귀 검사 3개: 미래 인원을 크게 바꿔도 전처리와 당일까지 입력 불변, 미래 전용 팀은 학습 어휘에 들어가지 않고 unseen 처리, 승하차 순서·음수 보정 MAE 손계산 일치. 기존 test_setup.py의 6개 검사도 통과(총9개).',
             '- **미해결:** 일정 수집 완전성·예측 전날 공개 여부와 승하차 데이터 공개 지연. 코드상 누수 검사가 통과해도 실제 운영 시점 가용성이 검증된 것은 아니다.', '',
             '## 결합 개선 폭의 불확실성', '',
             'RF의 일별 MAE − 결합의 일별 MAE. 양수는 결합 개선. 날짜별 36개 값을 함께 유지하고 두 모델에 같은 날짜를 적용하는 비순환 이동 블록 부트스트랩, 10,000회, 시드 42. 7일 주 분석, 3·14일 민감도. 마지막 블록은 181일에서 자른다.', '',
             '| 블록 길이 | 관측 개선(명) | percentile 95% 구간(명) | 재표집 개선 비율 |',
             '|---|---:|---:|---:|']
    for b in a['bootstrap']:
        lo, hi = b['percentile_95']
        lines.append(f"| {b['block_days']}일 | {b['observed_gain']:.2f} | {lo:.2f} ~ {hi:.2f} | {b['positive_fraction']:.2%} |")
    lines += ['', '**세 길이 모두 0을 포함한다.** 평균 개선은 관측했으나 안정적 우위를 강하게 주장할 근거는 부족하다. 재표집 개선 비율은 실제 우위의 확률이나 p값이 아니다. 반복적인 모델 선택의 편향을 보정하지 않으며, 계절 변화가 있는 단일 반년에서 구한 기술적 구간이다.', '',
              '## 조건별 비교', '', '각 조건은 서로 중첩된다. 기록 없음은 미확인이며 확정 비경기일이 아니다. 표본 수가 적은 집단과 다수 조건 비교는 탐색적으로 해석한다.', '',
              '| 조건 | 값 | 날짜 수 | 인원 관측 수 | RF MAE | LSTM MAE | 결합 MAE | 결합 개선 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in a['groups']:
        lines.append(f"| {r['group']} | {r['value']} | {r['days']} | {r['observations']} | {r['RF']['mae']:.2f} | {r['LSTM']['mae']:.2f} | {r['Average50']['mae']:.2f} | {r['gain']:+.2f} |")
    lines += ['', '- 상반기 평일 129일에서 결합 MAE가 17.77명 감소했으나 주말 52일에서는 25.56명 증가했다. 경기 기록이 있는 86일에서도 8.27명 증가했다. 전체 평균의 개선이 모든 조건의 개선을 의미하지 않는다.',
              '- 1~3월에는 개선, 4~6월에는 악화했다. 월·경기·주말 조건이 서로 겹치므로 이 표만으로 경기나 계절을 원인으로 확정하지 않는다. 사후 그룹에 따라 결합 비율을 바꾸는 새 모델은 만들지 않았다.',
              '', '## 시간대·방향', '', '| 시작 시각 | 방향 | 관측 수 | RF MAE | LSTM MAE | 결합 MAE | 결합 개선 |', '|---|---|---:|---:|---:|---:|---:|']
    for r in a['hourly']:
        lines.append(f"| {r['hour']} | {r['direction']} | {r['observations']} | {r['RF']:.2f} | {r['LSTM']:.2f} | {r['Average50']:.2f} | {r['gain']:+.2f} |")
    lines += ['', '- [재현 수치·출처 해시](../metrics/robustness_audit.json)', '- 재현: `tools/verify_robustness.py` 및 `tools/report_robustness.py`. 실행 폴더가 이미 있으면 덮어쓰지 않고 종료한다.', '']
    (ROOT/'reports/experiments/ANALYSIS-007_robustness.md').write_text('\n'.join(lines),encoding='utf8')
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for i,b in enumerate(a['bootstrap']):
        lo,hi=b['percentile_95'];v=b['observed_gain']
        axes[0].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=5)
    axes[0].axvline(0,color='black',linewidth=1);axes[0].set(yticks=range(3),yticklabels=['3 days','7 days','14 days'],xlabel='RF MAE - combined MAE (people)',title='Moving-block bootstrap: 95% intervals')
    for d in ('board','alight'):
        rows=[r for r in a['hourly'] if r['direction']==d]
        axes[1].plot([r['hour'] for r in rows],[r['gain'] for r in rows],label=d)
    axes[1].axhline(0,color='black',linewidth=1);axes[1].set(xlabel='Hour start',ylabel='RF MAE - combined MAE',title='Hourly improvement / deterioration');axes[1].legend()
    fig.suptitle('2025 H1 only | seed 42 | post-selection descriptive analysis');fig.tight_layout()
    fig.savefig(ROOT/'reports/figures/robustness_audit.png',dpi=150);plt.close(fig)
    path=ROOT/'reports/metrics/robustness_backtests.json'
    if not path.exists():
        print('Audit report ready; temporal backtests still pending.');return
    s=json.loads(path.read_text(encoding='utf8'));assert s['status']=='completed' and s['final_artifacts_unchanged']
    lines=['# EXP-016 2024년 시간순 후속 검증','', '2026-11-11 · DEC-039 · 하반기 최종 결과 보존 · 시드 42 고정.', '',
           '## 절차', '',
           '- RF 3일: 200나무·깊이10·최소잎3. LSTM 14일: 32유닛1층→달력/경기정보 결합→Dense32→36출력, Dropout0, MAE, Adam0.003, 배치8. 두 모델 각각 음수 보정 후 50:50.',
           '- 학습 기간 마지막 한 달을 내부 검증으로 사용하여 최대2000·patience50으로 최선 횟수를 선택한다. 내부 학습만으로 전처리 적합. 이후 전체 학습 기간으로 전처리를 다시 맞추고 초기화부터 선택 횟수만큼 재학습. 외부 두 달은 학습·조기 종료에 사용하지 않음.',
           '- 모든 목표는 2024-01-15부터 공통. 과거 실제 수요를 매일 갱신하는 하루 앞 예측. 기준 모델: 7일 전 동일 시간, 해당 학습 목표 날짜의 요일·시간 평균.',
           '- 구간별 전처리·설정·곡선·모델·예측을 새 outputs 폴더에 저장. RF와 LSTM 각각 저장·복원 예측 일치 확인.',
           '- 기존 설정은 전체2024 및 2025상반기를 사용해 개발한 설정이다. 외부 구간을 조기 종료에서 분리했어도, 이 분석을 완전히 새로운 독립 평가로 부르지 않는다. 짧은 학습 기간과 계절 변화가 동시에 달라져 원인을 하나로 단정하지 않는다.',
           '- 각 외부 구간의 월은 학습에서 아직 경험하지 못한 달이다. 월 원핫 칸은 고정되어 있어도 해당 월의 효과를 학습한 것은 아니다. 내부 검증의 마지막 한 달 역시 앞선 내부 학습에 없는 달이다. 이는 실제 시간순 초기 운영의 제약이며, 1년 전체를 학습한 기존2025예측과 같은 조건이 아니다.', '',
           '| 학습 종료 | 외부 검증 | 학습/검증 일수 | 내부 최선/종료 에포크 | 종료 사유 | 총 소요(초) |',
           '|---|---|---:|---:|---|---:|']
    pooled={}; all_truth=[]; all_dates=[]
    for f in s['folds']:
        t=f['inner_training'];lines.append(f"| {f['train_end']} | {f['start']}~{f['end']} | {f['train_days']}/{f['validation_days']} | {t['best_epoch']}/{t['stopped_epoch']} | {t['stop_reason']} | {f['seconds']:.1f} |")
        with np.load(ROOT/'outputs/robustness-2026-11-11'/f['start']/'predictions.npz') as z:
            all_truth.append(z['truth']);all_dates.extend(z['dates'].tolist())
            for name in f['metrics']:
                np.testing.assert_allclose(np.abs(z[name]-z['truth']).mean(),f['metrics'][name]['mae'])
                pooled.setdefault(name,[]).append(z[name])
    assert len(all_dates)==len(set(all_dates))
    lines += ['', '| 검증 기간 | 모델 | 전체 MAE | 승차 MAE | 하차 MAE | 승차 RMSE | 하차 RMSE |', '|---|---|---:|---:|---:|---:|---:|']
    for f in s['folds']:
        for n,m in f['metrics'].items():
            lines.append(f"| {f['start']}~{f['end']} | {n} | {m['mae']:.2f} | {m['board_mae']:.2f} | {m['alight_mae']:.2f} | {m['board_rmse']:.2f} | {m['alight_rmse']:.2f} |")
    truth=np.concatenate(all_truth); scores={n:float(abs(np.concatenate(p)-truth).mean()) for n,p in pooled.items()}
    lines+=['','184일 공통 외부 예측을 이어 붙여 계산한 MAE(구간 단순평균 아님):','']
    for n,v in scores.items():lines.append(f'- {n}: {v:.2f}명')
    lines+=['','## 해석','']
    for f in s['folds']:
        m=f['metrics'];gain=m['RF']['mae']-m['Average50']['mae']
        lines.append(f"- {f['start']}~{f['end']}: RF 대비 결합 MAE 감소 {gain:+.2f}명. 이 구간 최저 MAE는 {min(m,key=lambda n:m[n]['mae'])}.")
    lines+=['', '- 184일 합계에서는 결합이 RF보다 약2.20% 개선했지만 3구간 중 2구간은 RF가 더 좋았다. 전체 개선은 7~8월 개선의 영향이 크다. 모든 기간의 안정적 우위라고 주장하지 않는다.',
            '- 마지막 구간은 내부 검증(10월) MAE가 9회333.99명에서 최선, 59회334.39명으로 patience50 종료. 그동안 학습 손실은 456.33→418.25로 감소했다. 종료 규칙의 오류가 아니라 내부 기간에 민감한 에포크 선택이며, 충분히 학습된 모든 LSTM이 비시즌에 불리하다는 결론으로 확장하지 않는다. 외부 결과를 보고 추가 횟수로 재실행하지 않았다.',
            '- 마지막 구간의 결합124.96명은 전주 기준123.92명보다도 약간 높았다. 단순 기준과 비교하는 이유를 보여주는 사례다.',
            '- 각 구간의 학습/평가까지 총 소요 합계는 약34분38초이며, 앞선 감사·보고서 작성 시간은 제외한다.',
            '- 이번 결과는 안정성 진단이며 최종 구성/하반기 결과를 바꾸지 않는다. 경기 일정의 사전 가용성·실제 자료 공개 지연 한계는 유지된다.',
            '- [환경·설정·구간별 지표·보존 해시](../metrics/robustness_backtests.json)',
            '', '![기간별 비교](../figures/temporal_validation.png)', '', '![내부 학습·검증 곡선](../figures/temporal_learning_curves.png)',
            '- 실행: `./tools/run_wsl_python.ps1 tools/verify_robustness.py`, 보고서: `.venv/Scripts/python.exe tools/report_robustness.py`.', '']
    (ROOT/'reports/experiments/EXP-016_temporal-validation.md').write_text('\n'.join(lines),encoding='utf8')
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(3);width=.16
    for i,name in enumerate(pooled):ax.bar(x+(i-2)*width,[f['metrics'][name]['mae'] for f in s['folds']],width,label=name)
    ax.set(xticks=x,xticklabels=['Jul-Aug 2024','Sep-Oct 2024','Nov-Dec 2024'],ylabel='MAE (people)',title='Fixed configuration, inner epoch selection, outer temporal validation');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/temporal_validation.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,f in zip(axes,s['folds']):
        curve=json.loads((ROOT/'outputs/robustness-2026-11-11'/f['start']/'inner/learning_curve.json').read_text(encoding='utf8'))
        ax.plot([r['epoch'] for r in curve],[r['train_mae'] for r in curve],label='Training MAE')
        ax.plot([r['epoch'] for r in curve],[r['mae'] for r in curve],label='Inner validation MAE')
        ax.axvline(f['refit_epochs'],color='black',linestyle='--',label='Selected epoch')
        ax.set(title=f"Inner validation: {f['train_end'][:7]}",xlabel='Epoch',ylabel='MAE (people)');ax.legend(fontsize=7)
    fig.suptitle('Epoch selection only: outer two-month scores are not used');fig.tight_layout()
    fig.savefig(ROOT/'reports/figures/temporal_learning_curves.png',dpi=150);plt.close(fig)
    print(scores)


if __name__=='__main__':
    main()

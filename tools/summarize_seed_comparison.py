"""Summarize completed EXP-008 runs without additional model training."""
from pathlib import Path
from statistics import mean, stdev
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for group, count in [('rf', 10), ('lstm', 5)]:
        state = json.loads((ROOT / f'reports/metrics/seeds_{group}_2026-11-07.json').read_text(encoding='utf-8'))
        assert state['status'] == 'completed' and len(state['results']) == count
        assert not state['final_test_evaluated']
        rows.extend(state['results'])
    assert all(row['reload_verified'] for row in rows)
    labels = dict(rf_leaf2='RF 3일 최소 잎 2', rf_leaf3='RF 3일 최소 잎 3', lstm='LSTM 14일 32유닛 한 층')
    summary = {}
    for name in labels:
        group = [r for r in rows if r['name'] == name]
        scores = [r['condition']['validation_mae'] for r in group]
        fresh = [r['condition']['validation_mae'] for r in group if not r['reused']]
        summary[name] = dict(mean=mean(scores), sample_std=stdev(scores), minimum=min(scores),
                             maximum=max(scores), new_seed_mean=mean(fresh), new_seed_std=stdev(fresh),
                             board_mae_mean=mean(r['metrics']['overall']['board_mae'] for r in group),
                             alight_mae_mean=mean(r['metrics']['overall']['alight_mae'] for r in group))
    seeds = [r['seed'] for r in rows if r['name'] == 'rf_leaf2']
    paired = []
    for seed in seeds:
        a = next(r for r in rows if r['name'] == 'rf_leaf2' and r['seed'] == seed)
        b = next(r for r in rows if r['name'] == 'rf_leaf3' and r['seed'] == seed)
        paired.append(dict(seed=seed, leaf3_minus_leaf2=b['condition']['validation_mae'] - a['condition']['validation_mae']))
    output = dict(summary=summary, paired_rf=paired, final_test_evaluated=False)
    (ROOT / 'reports/metrics/seeds_summary_2026-11-07.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# EXP-008 RF·LSTM 시드 반복', '',
             '사용자 요청에 따라 세 설정에서 새 시드 네 개를 각각 학습하고 시드 42의 기존 결과를 재검증해 포함했다. 승인과 추출 방법은 [DEC-030](../decisions/training.md#dec-030)에 있다.',
             '', '- 새 시드 907394·892611·659988·622763은 결과 확인 전에 무작위 추출했다.',
             '- RF 3일은 나무 200·깊이 10에서 최소 잎 2와 3을 비교했다.',
             '- LSTM 14일은 32유닛 한 층·Dense 32, 학습률 0.003, batch 16, 최대 2000회, patience 50, Dropout 0이다.',
             '- 학습은 2024년 352일, 검증은 2025년 상반기 181일이다. 하반기 최종 평가 미사용.',
             '', '| 조건 | 5시드 평균 MAE | 표본 표준편차 | 최저~최고 | 새 4시드 평균 | 승차/하차 평균 MAE |',
             '|---|---:|---:|---|---:|---|']
    for name, values in summary.items():
        lines.append(f"| {labels[name]} | {values['mean']:.2f} | {values['sample_std']:.2f} | {values['minimum']:.2f}~{values['maximum']:.2f} | {values['new_seed_mean']:.2f} | {values['board_mae_mean']:.2f}/{values['alight_mae_mean']:.2f} |")
    lines += ['', '표준편차는 시드 간 변동을 나타내는 표본 표준편차(ddof=1)다. 시드 42는 기존 설정 탐색에 사용돼 새 네 시드 평균도 별도 보고한다.', '',
              '| 조건 | 시드 | 검증 MAE | 최선/종료 에포크 | 종료 사유 | 학습·검증 분 |',
              '|---|---:|---:|---|---|---:|']
    for r in rows:
        c = r['condition']
        epochs = f"{c['best_epoch']}/{c['stopped_epoch']}" if 'best_epoch' in c else '해당 없음'
        reason = {'early_stopping':'조기 종료','maximum_epochs':'상한 도달'}.get(c.get('stop_reason'), '나무 학습 완료')
        reused = ' (기존)' if r['reused'] else ''
        lines.append(f"| {labels[r['name']]} | {r['seed']}{reused} | {c['validation_mae']:.2f} | {epochs} | {reason} | {c['seconds']/60:.2f} |")
    lines += ['', '시드 42의 소요 시간은 당시 실행 기록이며 이번 신규 실행 시간에 포함하지 않는다.',
              '', '| 시드 | 최소 잎 3 MAE − 최소 잎 2 MAE |', '|---:|---:|']
    for pair in paired:
        lines.append(f"| {pair['seed']} | {pair['leaf3_minus_leaf2']:+.4f} |")
    lines += ['', '차이가 양수이면 최소 잎 2가 더 좋다.', '',
              '## 검증과 해석 범위', '',
              '- 각 실행에서 저장 모델 복원, 원본·KBO·공휴일 해시, 대상 날짜·입력 특성·패키지 버전을 확인했다. LSTM 학습 가중치의 GPU 배치 및 파라미터 수를 확인했다.',
              '- 동일 날짜 분할에서 학습 난수에 따른 변동을 비교했다. 다른 계절·기간에 대한 성능은 별도 시간순 평가로 확인해야 한다.',
              '- 전체·월별·note 상태별 상세 지표는 공유 JSON에, 예측·모델·학습 곡선과 복원 로그는 outputs/seeds-rf-2026-11-07 및 outputs/seeds-lstm-2026-11-07에 보존했다.',
              '- [요약 JSON](../metrics/seeds_summary_2026-11-07.json), [RF 상세 JSON](../metrics/seeds_rf_2026-11-07.json), [LSTM 상세 JSON](../metrics/seeds_lstm_2026-11-07.json).', '']
    (ROOT / 'reports/experiments/EXP-008_seeds.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    main()

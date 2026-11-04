"""Build EXP-006 tables from verified results; never train or evaluate test data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    state = json.loads((ROOT / 'reports/metrics/lstm_gpu_epochs_2026-11-04.json').read_text(encoding='utf-8'))
    assert state['status'] == 'completed' and len(state['results']) == 2
    assert not state['final_test_evaluated']
    first, second = state['results']
    lines = [
        '# EXP-006: LSTM 14일 GPU 최대 1000·2000회 비교', '',
        '- 승인: [DEC-028](../decisions/training.md#dec-028). 두 조건 학습과 모델 복원 검증 완료.',
        '- 환경: RTX 5060 / WSL2 Ubuntu 24.04 / TensorFlow 2.21.0 / Keras 3.15.1 / float32.',
        '- LSTM 32 units → 달력·경기 입력 결합 → Dense 32 ReLU → Dense 36 Linear, Dropout 없음.',
        '- 학습률 0.003, batch 16, seed 42, patience 50, min_delta 0. 최대 에포크만 1000·2000.',
        '- 각 조건은 같은 시드로 처음부터 학습. 1000회 체크포인트를 이어 학습하지 않았습니다.',
        '- 2024년 352일 학습 / 2025년 상반기 181일 검증. 하반기 최종 평가 미사용.',
        f"- UTC 시작 {state['started_utc']} / 종료 {state['finished_utc']}", '',
        '| 최대 에포크 | 검증 MAE | 승차 MAE | 하차 MAE | 최선/종료 에포크 | 종료 사유 | 학습·검증 분 |',
        '|---:|---:|---:|---:|---|---|---:|',
    ]
    for r in state['results']:
        assert r['model_reload_verified']
        c, m = r['condition'], r['metrics']['overall']
        reason = '조기 종료' if c['stop_reason'] == 'early_stopping' else '최대 횟수'
        lines.append(f"| {r['epochs']} | {m['mae']:.2f} | {m['board_mae']:.2f} | {m['alight_mae']:.2f} | {c['best_epoch']}/{c['stopped_epoch']} | {reason} | {c['seconds']/60:.2f} |")
    lines += ['', f"1000회 기준 대비 2000회 조건의 오차 감소: {second['mae_improvement']:+.4f}명 ({second['improvement_pct']:+.4f}%). 양수는 개선, 음수는 악화입니다.",
              f"공통 {second['common_prefix_epochs']}개 에포크의 검증 MAE 최대 절대 차이: {second['common_prefix_max_abs_mae_difference']:.10g}명.",
              '소요 시간은 조건별 모델 구성·학습·검증 기록이며 배치 전체의 프로세스 준비·별도 복원 검증 시간과 구분합니다.', '',
              '## 학습 곡선', '', '| 조건 | 마지막 학습 MAE | 마지막 검증 MAE | 마지막 50회 내 최선 MAE 개선 |', '|---|---:|---:|---:|']
    for r in state['results']:
        last = r['learning_curve'][-1]
        lines.append(f"| 최대 {r['epochs']}회 | {last['train_mae']:.2f} | {last['mae']:.2f} | {r['last_50_best_mae_gain']:.4f} |")
    lines += ['', '학습 MAE는 배치 손실, 검증 MAE는 음수 예측을 0으로 보정한 값입니다. 보고한 최종 모델 성능은 종료 에포크가 아니라 복원된 최선 에포크의 점수입니다.', '',
              '| 도달 에포크 | 1000회 조건 누적 최선 MAE | 2000회 조건 누적 최선 MAE |', '|---:|---:|---:|']
    for epoch in [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]:
        cells = []
        for r in state['results']:
            curve = r['learning_curve']
            cells.append(f"{min(x['mae'] for x in curve[:epoch]):.2f}" if len(curve) >= epoch else '미도달')
        lines.append(f"| {epoch} | {cells[0]} | {cells[1]} |")
    cpu = state['cpu_reference']
    lines += ['', '## 기존 CPU 결과와 비교', '',
              f"기존 CPU / TensorFlow 2.20의 LSTM 14일·학습률 0.003·최대1000회 결과는 MAE {cpu['validation_mae']:.2f}명, 최선/종료 {cpu['best_epoch']}/{cpu['stopped_epoch']}회, {cpu['seconds']/60:.2f}분입니다.",
              'GPU 조건은 OS·Python·TensorFlow·연산 경로가 달라 정확도 차이를 GPU 자체의 효과로 해석하지 않습니다. 종료 횟수까지 다르면 총 소요 시간의 비율을 순수 장치 가속 배율로 해석할 수 없습니다.', '',
              '## 검증과 한계', '',
              '- 두 모델의 실제 학습 가중치 GPU 배치와 저장 후 재로딩 예측·지표를 확인했습니다(rtol=1e-5, atol=1e-4).',
              '- 두 GPU 조건은 최대 에포크 외 설정이 같습니다. 기존 CPU와 원본·KBO·공휴일 해시 및 대상 날짜가 일치합니다.',
              '- 코드 해시를 실행 전후 확인했습니다. 모델 구조·학습 절차를 변경하지 않고 기존 실행기를 호출했습니다.',
              '- 단일 시드 결과이며 검증 반복 사용의 선택 편향이 있습니다. 경기 기록 없음·일정 공개 시점·수집 완전성의 한계는 유지됩니다.',
              '- [공유 JSON](../metrics/lstm_gpu_epochs_2026-11-04.json)에 조건별 설정·환경·해시·전체 학습 곡선·월별·note 상태별 지표를 저장했습니다.',
              '- 로컬 outputs/lstm-gpu-epochs-2026-11-04에 모델·예측·원본 로그를 보존합니다.', '',
              '## 결과 해석과 다음 검토', '',
              '- 이번 단일 시드에서는 학습 상한을 늘려 MAE가 277.92명에서 275.65명으로 2.27명(0.82%) 줄었습니다. 승차·하차 오차가 모두 감소했습니다.',
              '- 공통 1000개 에포크의 검증 MAE가 정확히 일치하므로, 이번 두 실행의 차이는 1000회 이후 학습에서 얻은 개선으로 해석할 수 있습니다.',
              '- 최선은 1283회, 종료는 1333회였습니다. 2000회까지 모두 실행한 결과가 아니며, 최선 1283회의 가중치를 복원했습니다.',
              '- 최종 학습 MAE는 더 낮아졌지만 검증 MAE는 1283회 이후 50회 동안 최선값을 갱신하지 못했습니다. 같은 patience 50에서 상한만 더 높여도 이번 시드의 종료는 달라지지 않으므로 상한 추가 확대의 우선순위는 낮습니다.',
              '- LSTM의 검증 최선은 이번 GPU 결과 275.65명으로 갱신됐습니다. 전체 모델에서는 기존 RF 3일 255.87명이 여전히 가장 좋습니다.',
              '- 다음에는 동일한 여러 시드로 개선의 재현성을 확인하거나, LSTM 유닛 수 등 다른 한 가지 조건을 비교하는 것이 낫습니다. 구체적인 후속 조건은 이번에 실행하지 않았습니다.',
              '- 소요 시간은 측정된 실행 시간입니다. 두 실행의 에포크당 시간 차이를 모델의 정확도 차이와 혼동하지 않으며 다른 작업 부하·캐시 등의 영향을 통제했다고 주장하지 않습니다.', '',
    ]
    (ROOT / 'reports/experiments/EXP-006_lstm-gpu-epochs.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

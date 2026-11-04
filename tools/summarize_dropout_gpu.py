"""Summarize verified EXP-005 results without training or evaluating test data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    state = json.loads((ROOT / 'reports/metrics/dropout_gpu_2026-11-04.json').read_text(encoding='utf-8'))
    assert state['status'] == 'completed' and len(state['results']) == 6
    assert not state['final_test_evaluated']
    lines = [
        '# EXP-005: GPU 기준과 Dropout 비교', '',
        '- 승인: [DEC-027](../decisions/training.md#dec-027). 6개 조건 학습 및 모델 복원 검증 완료.',
        '- 환경: WSL2 Ubuntu 24.04 / Python 3.12.3 / TensorFlow 2.21.0 / Keras 3.15.1 / RTX 5060.',
        '- MLP 3일·CNN 7일, 마지막 Dense 32 ReLU 뒤 Dropout 0.0·0.1·0.2, 출력 Dense 36 Linear.',
        '- Adam 학습률 0.003, 최대 1000회, patience 50, batch 16, seed 42. 매번 처음부터 학습.',
        '- 학습 2024년 352일 / 검증 2025년 상반기 181일. 하반기 최종 평가 미사용.',
        '- 입력·전처리·분할·음수 예측 0 보정 및 최선 가중치 복원 규칙 유지.',
        f"- UTC 시작: {state['started_utc']} / 종료: {state['finished_utc']}", '',
        '| 모델 | Dropout | MAE | 승차 MAE | 하차 MAE | GPU 기준 대비 오차 감소율 | 최선/종료 에포크 | 종료 사유 | 학습·검증 초 |',
        '|---|---:|---:|---:|---:|---:|---|---|---:|',
    ]
    for r in state['results']:
        assert r['model_reload_verified']
        c, m = r['condition'], r['metrics']['overall']
        reason = '조기 종료' if c['stop_reason'] == 'early_stopping' else '최대 횟수'
        lines.append(f"| {r['model'].upper()} {r['window']}일 | {r['dropout']:.1f} | {m['mae']:.2f} | {m['board_mae']:.2f} | {m['alight_mae']:.2f} | {r['improvement_pct']:+.2f}% | {c['best_epoch']}/{c['stopped_epoch']} | {reason} | {c['seconds']:.1f} |")
    lines += ['', '오차 감소율은 (동일 GPU의 Dropout 0 MAE−이번 MAE)/기준 MAE입니다. 음수는 악화입니다. 시간은 조건별 학습·검증 기록이며 별도 프로세스 준비·모델 복원 검증 시간을 포함한 배치 총시간과 다릅니다.', '', '## 모델별 최선', '']
    for model in ['mlp', 'cnn']:
        r = min((r for r in state['results'] if r['model'] == model), key=lambda r: r['condition']['validation_mae'])
        lines.append(f"- {model.upper()}: Dropout {r['dropout']:.1f}, MAE {r['condition']['validation_mae']:.2f}명.")
    lines += ['', '## 기존 CPU 결과와 환경 비교', '', '| 모델 | 기존 CPU / TF 2.20 MAE | GPU / TF 2.21 Dropout 0 MAE |', '|---|---:|---:|']
    for model, old in [('mlp', 298.2721), ('cnn', 328.2671)]:
        r = next(r for r in state['results'] if r['model'] == model and r['dropout'] == 0)
        lines.append(f"| {model.upper()} | {old:.2f} | {r['condition']['validation_mae']:.2f} |")
    lines += ['', 'OS·Python·TensorFlow·연산 경로가 함께 달라졌으므로 이 차이를 GPU 자체의 정확도 개선이나 Dropout 효과로 해석하지 않습니다.', '', '## 학습 곡선 확인', '', '| 조건 | 마지막 학습 MAE | 마지막 검증 MAE | 마지막 50회 내 최선 MAE 개선량 |', '|---|---:|---:|---:|']
    for r in state['results']:
        curve_path = ROOT / 'outputs/dropout-gpu-2026-11-04' / r['id'] / f"{r['model']}_{r['window']}d" / 'learning_curve.json'
        curve = json.loads(curve_path.read_text(encoding='utf-8'))
        gain = min(x['mae'] for x in curve[:-50]) - min(x['mae'] for x in curve) if len(curve) > 50 else None
        lines.append(f"| {r['id']} | {r['last_train_mae']:.2f} | {r['last_validation_mae']:.2f} | {gain:.3f} |")
    lines += ['', '학습 MAE는 학습 중 Dropout이 켜진 상태의 배치 손실이며 검증은 Dropout을 끄고 음수 예측을 보정합니다. 따라서 두 값의 차이만으로 과적합 정도를 정량 비교하지 않습니다. 조기 종료 조건은 마지막 50회에 최선값 갱신이 없어 개선량이 0입니다. 선택된 모델의 점수는 마지막 에포크가 아니라 최선 에포크입니다.', '', '## 검증과 한계', '',
        '- 실제 학습 가중치가 GPU:0에 배치되었음을 6개 조건 모두 확인했습니다.',
        '- 같은 모델 내 기준과 변경 조건의 원본·KBO·공휴일 해시, 대상 날짜, Dropout 외 설정이 일치합니다.',
        '- 6개 저장 모델을 다시 불러와 검증 예측·지표·최선 에포크를 확인했습니다(rtol=1e-5, atol=1e-4).',
        '- 구현 테스트 8개 통과. Dropout 유무에 따른 초기 가중치 일치, 검증 시 비활성화 및 학습 시 마스킹·보정 포함.',
        '- 단일 시드 탐색이며 검증 데이터 반복 사용에 따른 선택 편향이 있습니다. 작은 차이는 여러 시드로 재현성을 확인해야 합니다.',
        '- 경기 기록 없음은 미확인으로 유지합니다. 일정 공개 시점과 수집 완전성에 대한 기존 한계도 유지됩니다.',
        '- [공유 JSON](../metrics/dropout_gpu_2026-11-04.json)에 전체·승하차·월별·note별 지표와 코드·환경·원본 해시를 보존했습니다.',
        '- 조건별 곡선·모델·예측·실행 및 복원 로그는 로컬 outputs/dropout-gpu-2026-11-04에 보존합니다.', '',
        '## 해석과 다음 실험 제안', '',
        '- MLP는 Dropout 없음이 가장 좋았습니다. 0.1은 승차 오차가 소폭 줄었지만 하차 오차가 커져 전체 성능이 나빠졌고, 0.2도 기준보다 나빴습니다. 이번 결과에서 Dropout을 기본값으로 채택할 근거는 없습니다.',
        '- CNN은 0.1이 가장 좋았고 전체 오차가 약 0.76% 줄었습니다. 하차 오차 개선이 주된 이유이며 승차 오차는 소폭 증가했습니다. 0.2는 개선 폭이 더 작고 더 오래 학습했습니다.',
        '- CNN 기준은 최선 MAE 328.40에서 종료 시 343.81로 악화됐지만 0.1은 최선 325.91에서 종료 시 326.62였습니다. 이 실행에서는 후반 검증 오차가 덜 악화됐으나, Dropout이 모든 시드에서 과적합을 해결한다고 단정하지 않습니다.',
        '- 여섯 조건 모두 상한 전에 조기 종료했고 마지막 50회에 최선값이 갱신되지 않았습니다. MLP·CNN의 최대 횟수만 늘리는 실험은 우선순위가 낮습니다.',
        '- 기존 전체 검증 최선은 여전히 RF 3일 MAE 255.87명, 신경망 중에서는 기존 CPU LSTM 14일 MAE 278.57명입니다. 이번 Dropout 실험으로 순위는 바뀌지 않았습니다.',
        '- 다음 우선 후보는 LSTM 14일·학습률 0.003의 GPU 기준 1000회와 최대 2000회 비교(patience 50 유지)입니다. CPU 실험의 마지막 50회 개선량은 약 0.17명으로 작았으므로 큰 개선을 보장하지 않습니다. GPU 1000회 기준을 함께 두어 환경 변화와 학습 연장 효과를 구분해야 합니다.',
        '- CNN Dropout 0과 0.1은 동일한 여러 시드에서 짝지어 비교해 평균·변동성을 확인하는 것이 좋습니다. 실험 횟수는 늘어나지만 0.76% 차이가 반복되는지 판단할 수 있습니다.',
        '- 위 후속 실험은 제안이며 이번 실행에 포함하지 않았습니다.', '',
    ]
    (ROOT / 'reports/experiments/EXP-005_dropout-gpu.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

"""Verify saved validation predictions and model reloads; publish aggregate results only."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import read_json, load_frame, build_window, inputs
from src.evaluation import metrics
from src.experiment import predict_neural


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--verify-partial", action="store_true", help="Verify completed conditions only; do not publish final reports")
    args = p.parse_args()
    report = read_json(args.run / "report.json")
    if not args.verify_partial:
        assert len(report["conditions"]) == 9 and report["final_test_evaluated"] is False
    cfg = read_json(args.run / "config.json")
    pre = read_json(args.run / "preprocessor.json")
    frame = load_frame(args.source)
    holidays = read_json(ROOT / "data/reference/holidays_2024_2025.json")
    windows = {}
    rows = []
    for condition in report["conditions"]:
        model, window = condition["model"], condition["window"]
        out = args.run / f"{model}_{window}d"
        with np.load(out / "validation_predictions.npz") as saved:
            truth, pred, raw, dates = [saved[k] for k in ("truth", "prediction", "raw", "dates")]
        assert truth.shape == pred.shape == raw.shape == (181, 36)
        assert np.array_equal(pred, np.maximum(raw, 0))
        assert dates[0] == "2025-01-01" and dates[-1] == "2025-06-30"
        scored = metrics(truth, raw)
        stored = read_json(out / "validation_metrics.json")
        for k, v in scored.items():
            assert np.isclose(v, stored["overall"][k])
        if window not in windows:
            windows[window], _ = build_window(frame, cfg, pre, holidays, window)
        validation = windows[window]["validation"]
        assert np.array_equal(truth, validation["y"])
        x = inputs(validation, model)
        if model == "rf":
            import joblib
            restored = joblib.load(out / "model.joblib")
            reproduced = restored.predict(x)
        else:
            from tensorflow import keras
            restored = keras.models.load_model(out / "model.keras", compile=False)
            reproduced = predict_neural(restored, x, cfg["training"]["batch_size"])
        assert np.allclose(reproduced, raw, rtol=1e-5, atol=1e-4), "Reload changed predictions"
        row = {**condition, **scored, "reload_max_absolute_difference": float(np.max(np.abs(reproduced-raw))),
               "monthly": stored["month"], "note_status": stored["note_status"]}
        if model != "rf":
            curve = read_json(out / "learning_curve.json")
            best = min(curve, key=lambda e: e["mae"])
            assert best["epoch"] == condition["best_epoch"]
            assert np.isclose(best["mae"], scored["mae"])
            row["curve_summary"] = {"first_validation_mae": curve[0]["mae"], "last_validation_mae": curve[-1]["mae"],
                                    "last_train_mae": curve[-1]["train_mae"],
                                    "last_20_validation_change": curve[-1]["mae"]-curve[max(0,len(curve)-21)]["mae"]}
        rows.append(row)
        print(f"Verified {model} {window}d MAE={scored['mae']:.3f}", flush=True)
    if args.verify_partial:
        print(f"Verified {len(rows)} completed conditions; final report not generated")
        return
    rows.sort(key=lambda r: r["mae"])
    output = {"date": "2026-11-03", "run": args.run.name, "training_commit": report["environment"]["git_commit"],
              "source_hashes": report["sources"], "code_hashes": report["code_sha256"],
              "environment": report["environment"], "validation_days": 181, "seed": cfg["seed"],
              "final_test_evaluated": False, "all_saved_models_reloaded_and_verified": True, "ranking": rows}
    dest = ROOT / "reports/metrics/first_run_2026-11-03.json"
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# EXP-001: 1차 9개 조합 실행", "", "- 날짜: 2026-11-03 (Asia/Seoul)",
             "- 승인: DEC-023, 사용자 전달에 따른 팀 검토 완료.",
             f"- 학습 코드 기준: {output['training_commit']}",
             "- 기존 설정 그대로 실행. 2024년 352일 학습, 2025년 상반기 181일 검증. 하반기 최종 평가는 미실행.",
             "- 점수: 음수 0 보정 후 인원 단위 MAE. 승차·하차·시간대 동일 비중. 단일 시드 42.", "",
             "| 순위 | 모델 | 과거 일수 | 평균 MAE | 승차 MAE | 하차 MAE | 최선/종료 에포크 | 소요 초 |",
             "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        ep = f"{r['best_epoch']}/{r['stopped_epoch']}" if "best_epoch" in r else "해당 없음"
        lines.append(f"| {i} | {r['model']} | {r['window']} | {r['mae']:.2f} | {r['board_mae']:.2f} | {r['alight_mae']:.2f} | {ep} | {r['seconds']:.1f} |")
    lines += ["", "## 관찰과 다음 검토", "",
              f"현재 1차 검증 최저는 {rows[0]['model'].upper()} {rows[0]['window']}일 조건의 MAE {rows[0]['mae']:.2f}명입니다. 이는 시간대별 승차·하차 절대오차의 평균이며 하루 총인원 오차가 아닙니다.",
              "", "신경망 6개 조건은 모두 200에포크 상한에서 종료했습니다. MLP 3일과 모든 LSTM은 최선 시점이 200회로, 충분히 수렴했다고 단정할 수 없습니다. MLP 7·14일은 최선 시점이 184·188회이며 후반 학습 오차 감소가 검증 개선으로 이어지지 않았습니다. 따라서 모든 신경망에 일괄적으로 에포크를 늘리는 것보다 모델별 곡선에 근거해 후속 조건을 정해야 합니다.",
              "", "LSTM 세 입력 기간의 결과가 비슷하므로 과거 정보 활용과 학습 속도를 후속 진단할 가치가 있습니다. 현재 결과만으로 LSTM이 이 문제에 부적합하다고 결론내리지 않습니다. RF는 승차 RMSE가 큰 시간대 오차를 시사하므로 혼잡 시간대 사례 분석도 필요합니다.",
              "", "검증 비교 그래프: reports/figures/EXP-001_validation_mae.png. 학습 곡선: reports/figures/EXP-001_learning_curves.png.",
              "", "## 검증과 해석 범위", "", "모든 저장 모델을 다시 불러와 181일 검증 예측을 재계산하고 저장 예측과 일치함을 확인했습니다. 저장 예측에서 MAE·RMSE를 재계산했으며, 신경망 체크포인트가 학습 곡선의 최저 검증 MAE와 일치하는지도 확인했습니다.",
              "", "소요 시간은 각 조건의 모델 생성·학습·검증·저장을 포함하며 공통 데이터 준비 시간은 제외합니다. 세부 월별·note 상태별 지표, 음수 비율, 학습 곡선 요약은 metrics/first_run_2026-11-03.json에 있습니다.",
              "", "1차 검증 순위는 최종 평가 성능이나 통계적 우월성의 증명이 아닙니다. 최대 에포크에 도달한 신경망은 학습 곡선으로 미수렴 여부를 점검해야 합니다. 일정의 당시 가용성·수집 완전성과 실제 지하철 공개 시점은 미검증입니다.",
              "", "모델·개별 예측·전체 학습 곡선은 Git 제외 outputs/first-run-2026-11-03/에 보관합니다. 자동 튜닝·최종 평가·전체 자료 재학습은 수행하지 않았습니다."]
    (ROOT / "reports/experiments/EXP-001_first-run.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Plot aggregate validation results; no test predictions and no model training."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
font_manager.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
report = json.loads((ROOT / "reports/metrics/first_run_2026-11-03.json").read_text(encoding="utf-8"))
rows = report["ranking"]
out = ROOT / "reports/figures"
out.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5.8), layout="constrained")
palette = {"rf": "#157F8C", "mlp": "#4878A8", "lstm": "#B77A38"}
labels = [f"{r['model'].upper()} · {r['window']}일" for r in rows]
bars = ax.barh(labels, [r["mae"] for r in rows], color=[palette[r["model"]] for r in rows])
ax.invert_yaxis()
ax.bar_label(bars, fmt="%.2f", padding=5)
ax.set_xlim(0, max(r["mae"] for r in rows)*1.18)
ax.set_xlabel("검증 MAE (명) · 낮을수록 좋음")
ax.set_title("1차 검증 결과: 3개 모델 × 3개 입력 기간", loc="left", fontweight="bold", pad=20)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, alpha=.15)
ax.set_axisbelow(True)
fig.text(.01, -.025, "2025년 상반기 181일 · 음수 보정 후 승차/하차 평균 · 시드 42 · 하반기 최종 평가 미실행", fontsize=9)
fig.savefig(out / "EXP-001_validation_mae.png", dpi=180, bbox_inches="tight")
plt.close(fig)
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, layout="constrained")
for i, name in enumerate(("mlp", "lstm")):
    for j, window in enumerate((3, 7, 14)):
        curve = json.loads((ROOT / f"outputs/first-run-2026-11-03/{name}_{window}d/learning_curve.json").read_text(encoding="utf-8"))
        ax = axes[i, j]
        ax.plot([r["epoch"] for r in curve], [r["train_mae"] for r in curve], label="학습 MAE (보정 전)", color="#4878A8")
        ax.plot([r["epoch"] for r in curve], [r["mae"] for r in curve], label="검증 MAE (보정 후)", color="#B77A38")
        best = min(curve, key=lambda r:r["mae"])
        ax.scatter(best["epoch"], best["mae"], color="#B77A38", zorder=3)
        ax.set_title(f"{name.upper()} · {window}일 | 최선 {best['epoch']}회")
        ax.set_xlabel("에포크")
        ax.set_ylabel("MAE (명)")
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].legend(fontsize=8)
fig.suptitle("1차 신경망 학습 곡선 · 최선 검증 시점 표시", fontweight="bold")
fig.savefig(out / "EXP-001_learning_curves.png", dpi=180)
plt.close(fig)

"""Post-selection audit, paired block bootstrap and nested 2024 time backtests.

Never constructs a 2025 H2 split. Final artifacts are hash-guarded, not scored.
"""
import copy
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import TARGETS, TEAM_COLS, fit_preprocessor, make_context, inputs, read_json, sha256
from src.models import build_rf, build_neural
from src.experiment import save_json, train_neural, predict_neural, environment, note_statuses
from src.evaluation import metrics
from src.baselines import baseline_predictions

OUT = ROOT / 'outputs/robustness-2026-11-11'
HOLIDAYS = read_json(ROOT / 'data/reference/holidays_2024_2025.json')


def arrays(frame, cutoff, window):
    """Fit preprocessing on prefix only; target and history positions explicit."""
    fit = frame[frame.date <= pd.Timestamp(cutoff)]
    pre = fit_preprocessor(fit)
    pre['fit_period'] = ['2024-01-01', cutoff]
    assert pre['fit_rows'] == len(fit)
    raw, context, names = make_context(frame, pre, HOLIDAYS)
    days = pd.DatetimeIndex(sorted(frame.date.unique()))
    y = frame[TARGETS].to_numpy(dtype='float32').reshape(len(days), 18, 2)
    positions = np.arange(14, len(days))
    hist = np.stack([y[i-window:i].reshape(window*18, 2) for i in positions])
    assert all(days[i-1] < days[i] for i in positions)
    np.testing.assert_array_equal(hist[:, -18:], y[positions-1])
    dates = days[positions].strftime('%Y-%m-%d').to_numpy(dtype=str)
    d = dict(history=((hist-pre['mean'])/pre['scale']).astype('float32'), history_raw=hist,
             context=context[positions], context_raw=raw[positions], y=y[positions].reshape(-1, 36), dates=dates)
    return d, pre, names


def select(d, start, end):
    mask = (d['dates'] >= start) & (d['dates'] <= end)
    return {k: v[mask] for k, v in d.items()}


def bootstrap(gain, length, repeats=10000):
    """Paired non-circular moving blocks, full-day values, truncate to n days."""
    rng = np.random.default_rng(42)
    n = len(gain)
    starts = rng.integers(0, n-length+1, size=(repeats, int(np.ceil(n/length))))
    ix = (starts[:, :, None] + np.arange(length)).reshape(repeats, -1)[:, :n]
    draws = gain[ix].mean(axis=1)
    return dict(block_days=length, repeats=repeats, seed=42, observed_gain=float(gain.mean()),
                percentile_95=np.quantile(draws, [.025, .975]).tolist(),
                positive_fraction=float((draws > 0).mean()))


def audit(frame, source, kbo):
    saved = ROOT / 'outputs/hourly-rf-2026-11-11/baseline_predictions.npz'
    with np.load(saved) as z:
        dates = z['dates']; y = z['truth']; rp = z['rf']; lp = z['lstm']; ap = z['combined']
    assert len(dates) == 181 and dates[0] == '2025-01-01' and dates[-1] == '2025-06-30'
    np.testing.assert_allclose(ap, (rp+lp)/2, rtol=0, atol=0)
    checks = {}
    for label, window, folder, key in [('RF', 3, 'first-run-2026-11-03', 'rf_3d'),
                                       ('LSTM', 14, 'loss-comparison-2026-11-11/mae', 'lstm_14d')]:
        run = ROOT/'outputs'/folder
        d, pre, names = arrays(frame, '2024-12-31', window)
        old = read_json(run/'preprocessor.json')
        for field in ('mean', 'scale', 'teams', 'fit_rows', 'target_order'):
            assert pre[field] == old[field], field
        assert names == read_json(run/key/'context_features.json')
        assert not any(any(w in n for w in ('note', 'gamecenter', 'verified', '승차', '하차', 'lag_')) for n in names)
        val = select(d, '2025-01-01', '2025-06-30')
        np.testing.assert_array_equal(y, val['y']); np.testing.assert_array_equal(dates, val['dates'])
        with np.load(run/key/'validation_predictions.npz') as z:
            np.testing.assert_array_equal(z['truth'], y)
            np.testing.assert_array_equal(z['dates'], dates)
            np.testing.assert_allclose(np.maximum(z['raw'], 0), rp if label == 'RF' else lp, rtol=1e-5, atol=1e-4)
        checks[label] = dict(preprocessor_matches=True, dates_truth_order_matches=True,
                             stored_predictions_match=True, context_allowlist_checked=True,
                             history_ends_previous_day=True, fit_rows=pre['fit_rows'])
    gain = abs(rp-y).mean(axis=1)-abs(ap-y).mean(axis=1)
    boot = [bootstrap(gain, length) for length in (3, 7, 14)]
    rows = []; daily = frame.groupby('date').first().loc[pd.to_datetime(dates)]
    statuses = note_statuses(kbo)
    labels = {
        'month': np.array([d[:7] for d in dates]),
        'weekend': np.where(pd.to_datetime(dates).weekday >= 5, 'weekend', 'weekday'),
        'holiday': np.array(['holiday' if d in {x['date'] for x in HOLIDAYS['dates'] if x['is_holiday']} else 'not_holiday' for d in dates]),
        'note_status': np.array([statuses.get(d, 'no_record_unknown') for d in dates]),
        'collected_record': np.where(daily.has_kbo_record.to_numpy() == 1, 'record_present', 'no_record_unknown')}
    for group, values in labels.items():
        for value in sorted(set(values)):
            m = values == value
            r = dict(group=group, value=value, days=int(m.sum()), observations=int(m.sum()*36))
            for name, p in [('RF', rp), ('LSTM', lp), ('Average50', ap)]:
                r[name] = metrics(y[m], p[m])
            r['gain'] = r['RF']['mae']-r['Average50']['mae']; rows.append(r)
    hour_rows = []
    for h in range(18):
        for j, direction in enumerate(['board', 'alight']):
            k = h*2+j
            r = dict(hour=h+6, direction=direction, observations=181)
            for name, p in [('RF', rp), ('LSTM', lp), ('Average50', ap)]:
                r[name] = float(abs(p[:, k]-y[:, k]).mean())
            r['gain'] = r['RF']-r['Average50']; hour_rows.append(r)
    return dict(checks=checks, bootstrap=boot, groups=rows, hourly=hour_rows,
                source_sha256=sha256(source), kbo_sha256=sha256(kbo), prediction_sha256=sha256(saved),
                unresolved=['Retrospective schedule completeness and pre-forecast availability unverified',
                            'Previous-day passenger publication delay not verified',
                            'Repeated validation/model selection; confidence intervals do not adjust for selection'],
                context_features=names)


def main():
    OUT.mkdir(exist_ok=False)
    source = ROOT.parent/'인공지능공학 과제용/종합운동장역_2024-2025_모델입력_경기결합.csv'
    kbo = ROOT.parent/'인공지능공학 과제용/KBO원천데이터_정리.csv'
    guards = [ROOT/'configs/final_evaluation.json', ROOT/'reports/metrics/final_evaluation.json']
    guards += list((ROOT/'outputs').glob('final-evaluation*/**/*'))
    guards = [p for p in guards if p.is_file()]
    frozen = {str(p.relative_to(ROOT)): sha256(p) for p in guards}
    s = dict(status='running', phase='audit', final_test_evaluated=False, frozen_sha256=frozen, folds=[])
    def checkpoint(): save_json(OUT/'status.json', s)
    checkpoint()
    try:
        frame = pd.read_csv(source, encoding='utf-8-sig')
        frame = frame[(frame['호선']=='2호선') & (frame['역명']=='종합운동장역') & (frame['날짜']<='2025-06-30')].copy()
        frame['date'] = pd.to_datetime(frame['날짜']); frame['hour_actual'] = frame['시간대'].str.extract(r'^(\d+)시').astype(int)
        frame = frame.sort_values(['date', 'hour_actual']).reset_index(drop=True)
        assert pd.MultiIndex.from_frame(frame[['date','hour_actual']]).equals(pd.MultiIndex.from_product([pd.date_range('2024-01-01','2025-06-30'), range(6,24)]))
        assert np.isfinite(frame[TARGETS]).all().all() and (frame[TARGETS] >= 0).all().all()
        s['audit'] = audit(frame, source, kbo)
        save_json(ROOT/'reports/metrics/robustness_audit.json', s['audit']); checkpoint()
        cfg = read_json(ROOT/'outputs/loss-comparison-2026-11-11/mae/config.json')
        cfg['followup_approval'] = dict(date='2026-11-11', evidence='추천 순서대로 검증 진행, 하반기 보존, 기록 및 커밋 사용자 승인')
        s['config'] = cfg; s['environment'] = environment()
        for train_end, end in [('2024-06-30','2024-08-31'), ('2024-08-31','2024-10-31'), ('2024-10-31','2024-12-31')]:
            start = (pd.Timestamp(train_end)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            inner_start = pd.Timestamp(train_end).replace(day=1).strftime('%Y-%m-%d')
            inner_end = (pd.Timestamp(inner_start)-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            fold = OUT/start; fold.mkdir(); inner = fold/'inner'; inner.mkdir()
            s.update(phase='inner_epoch_selection', current_fold=start); checkpoint()
            begun = time.perf_counter(); prefix = frame[frame.date <= pd.Timestamp(end)]
            d, pre, names = arrays(prefix, inner_end, 14)
            train = select(d, '2024-01-15', inner_end); val = select(d, inner_start, train_end)
            save_json(inner/'preprocessor.json', pre)
            model = build_neural('lstm', 14, len(names), cfg)
            assert all('GPU:' in str(v.value.device) for v in model.trainable_weights)
            _, details = train_neural(model, train, val, 'lstm', cfg, inner)
            epochs = details['best_epoch']
            s.update(phase='full_prefix_refit', selected_epochs=epochs); checkpoint()
            d, pre, names = arrays(prefix, train_end, 14)
            train = select(d, '2024-01-15', train_end); test = select(d, start, end)
            save_json(fold/'preprocessor.json', pre); save_json(fold/'config.json', cfg)
            model = build_neural('lstm', 14, len(names), cfg)
            history = model.fit(inputs(train,'lstm'), train['y'], epochs=epochs, batch_size=8, shuffle=True, verbose=2)
            save_json(fold/'refit_learning_curve.json', history.history)
            lp = np.maximum(predict_neural(model, inputs(test,'lstm'), 8), 0)
            model.save(fold/'model.keras')
            from tensorflow import keras
            restored = keras.models.load_model(fold/'model.keras', compile=False)
            np.testing.assert_allclose(lp, np.maximum(predict_neural(restored, inputs(test,'lstm'),8),0), rtol=1e-5, atol=1e-4)
            rd, _, _ = arrays(prefix, train_end, 3)
            rt = select(rd, '2024-01-15', train_end); rv = select(rd, start, end)
            np.testing.assert_array_equal(rv['y'],test['y']); np.testing.assert_array_equal(rv['dates'],test['dates'])
            rf = build_rf(cfg); rf.fit(inputs(rt,'rf'),rt['y']); rp = np.maximum(rf.predict(inputs(rv,'rf')),0)
            joblib.dump(rf, fold/'model.joblib')
            np.testing.assert_array_equal(rp,np.maximum(joblib.load(fold/'model.joblib').predict(inputs(rv,'rf')),0))
            preds = dict(RF=rp, LSTM=lp, Average50=(rp+lp)/2, **baseline_predictions(prefix,train['dates'],test['dates']))
            np.savez_compressed(fold/'predictions.npz', dates=test['dates'], truth=test['y'], **preds)
            row = dict(train_start='2024-01-15',train_end=train_end,start=start,end=end,
                       inner_fit_end=inner_end,inner_validation_start=inner_start,
                       train_days=len(train['y']),validation_days=len(test['y']),inner_training=details,
                       refit_epochs=epochs,seconds=time.perf_counter()-begun,reload_verified=True,
                       metrics={n:metrics(test['y'],p) for n,p in preds.items()})
            s['folds'].append(row); save_json(fold/'report.json',row); checkpoint()
        assert all(sha256(ROOT/p)==h for p,h in frozen.items())
        s.update(status='completed',phase='complete',final_artifacts_unchanged=True)
        save_json(ROOT/'reports/metrics/robustness_backtests.json',s); checkpoint()
    except Exception as e:
        s.update(status='failed',error=repr(e)); checkpoint(); raise


if __name__ == '__main__':
    main()

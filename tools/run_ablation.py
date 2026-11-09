"""EXP-009: simple validation baselines, then paired RF/LSTM game ablations."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
from run_tuning_batch import ROOT, command, read_json, save, sha256, verify
from src.dataset import load_frame, fit_preprocessor, build_window
from src.baselines import baseline_predictions
from src.evaluation import grouped_metrics
from src.experiment import note_statuses


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['baselines', 'rf', 'lstm'], required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--kbo-source', type=Path, required=True)
    args = p.parse_args()
    out = ROOT / f'outputs/ablation-2026-11-09/{args.phase}'
    out.mkdir(parents=True, exist_ok=False)
    manifest = read_json(ROOT / 'configs/ablation_2026-11-09.json')
    hashes = dict(joined_sha256=sha256(args.source), kbo_sha256=sha256(args.kbo_source),
                  holidays_sha256=sha256(ROOT / 'data/reference/holidays_2024_2025.json'))
    state = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 manifest=manifest, sources=hashes, results=[], final_test_evaluated=False,
                 runner_sha256=sha256(Path(__file__)),
                 code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (ROOT/'src').glob('*.py')})
    save(out / 'status.json', state)
    try:
        frame = load_frame(args.source)
        cfg = read_json(ROOT / 'configs/initial.json')
        splits, _ = build_window(frame, cfg, fit_preprocessor(frame),
                                read_json(ROOT / 'data/reference/holidays_2024_2025.json'), 14)
        validation = splits['validation']
        statuses = note_statuses(args.kbo_source)
        if args.phase == 'baselines':
            predictions = baseline_predictions(frame, splits['train']['dates'], validation['dates'])
            for name, pred in predictions.items():
                np.savez_compressed(out / f'{name}.npz', dates=validation['dates'], truth=validation['y'], prediction=pred)
                state['results'].append(dict(name=name, metrics=grouped_metrics(validation['y'], pred, validation['dates'], statuses)))
            state['training_dates'] = splits['train']['dates'].tolist()
            state['validation_dates'] = validation['dates'].tolist()
        else:
            assert read_json(ROOT/'outputs/ablation-2026-11-09/baselines/status.json')['status'] == 'completed'
            if args.phase == 'lstm':
                assert read_json(ROOT/'outputs/ablation-2026-11-09/rf/status.json')['status'] == 'completed'
            old = read_json(ROOT / f'reports/metrics/seeds_{args.phase}_2026-11-07.json')
            references = [r for r in old['results'] if r['name'] == manifest['groups'][args.phase]]
            assert [r['seed'] for r in references] == manifest['seeds']
            for row in references:
                seed = row['seed']; baseline = ROOT / row['run']; model = args.phase
                ref = read_json(baseline / 'report.json')
                window = row['condition']['window']; folder = f'{model}_{window}d'
                for k,v in hashes.items():
                    assert ref['sources'][k] == v
                assert not ref['final_test_evaluated']
                state['current'] = f'{model}_seed{seed}'
                save(out / 'status.json', state)
                verify(baseline, args.source, out / f'seed{seed}_included_verify.log')
                config = copy.deepcopy(read_json(baseline / 'config.json'))
                assert config.get('context_mode', 'calendar_game') == 'calendar_game'
                config.update(models=[model], windows=[window], context_mode='calendar_only', followup_approval=manifest['approval'])
                assert config['seed'] == seed
                path = out / f'seed{seed}.json'; save(path, config)
                run = out / f'seed{seed}'
                command([sys.executable, '-u', '-m', 'src.experiment', 'train', '--config', str(path),
                         '--source', str(args.source), '--kbo-source', str(args.kbo_source), '--output', str(run)],
                        out / f'seed{seed}.log')
                report = read_json(run / 'report.json'); condition, = report['conditions']
                assert not report['final_test_evaluated']
                assert report['code_sha256'] == state['code_sha256']
                assert report['environment']['packages'] == ref['environment']['packages']
                for k,v in hashes.items(): assert report['sources'][k] == v
                assert read_json(run/folder/'target_dates.json') == read_json(baseline/folder/'target_dates.json')
                assert condition['context_features'] == 20
                if model == 'lstm': assert all('GPU:' in d for d in condition['weight_devices'])
                with np.load(run/folder/'validation_predictions.npz') as new, np.load(baseline/folder/'validation_predictions.npz') as included:
                    np.testing.assert_array_equal(new['truth'], included['truth'])
                    np.testing.assert_array_equal(new['dates'], included['dates'])
                verify(run, args.source, out / f'seed{seed}_excluded_verify.log')
                scores = read_json(run / folder / 'validation_metrics.json')
                included_scores = read_json(baseline / folder / 'validation_metrics.json')
                state['results'].append(dict(seed=seed, included_run=row['run'], excluded_run=str(run.relative_to(ROOT)),
                    included_condition=row['condition'], excluded_condition=condition,
                    included_metrics=included_scores, excluded_metrics=scores,
                    excluded_minus_included_mae=scores['overall']['mae']-included_scores['overall']['mae'],
                    both_model_reload_verified=True, environment=report['environment'],
                    excluded_config_sha256=sha256(path)))
                save(out / 'status.json', state)
        state.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
        state.pop('current', None)
        save(out / 'status.json', state)
        save(ROOT / f'reports/metrics/ablation_{args.phase}_2026-11-09.json', state)
    except Exception as error:
        state.update(status='failed', error=repr(error)); save(out / 'status.json', state)
        raise


if __name__ == '__main__':
    main()

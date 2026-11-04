"""EXP-006: independently train two GPU LSTMs and verify saved predictions."""
import argparse
import copy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_tuning_batch import ROOT, command, save, verify

sys.path.insert(0, str(ROOT))
from src.dataset import read_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--kbo-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / 'configs/lstm_gpu_epochs.json'
    manifest = read_json(manifest_path)
    cfg = read_json(ROOT / manifest['base_config'])
    cfg.update(models=['lstm'], windows=[manifest['window']], require_gpu=True,
               head_dropout=0.0, followup_approval=manifest['approval'])
    cfg['adam']['learning_rate'] = manifest['learning_rate']
    assert cfg['training']['patience'] == manifest['patience'] == 50
    assert cfg['training']['batch_size'] == manifest['batch_size'] == 16
    assert cfg['seed'] == manifest['seed'] == 42
    assert manifest['epochs'] == [1000, 2000] and manifest['window'] == 14
    state = {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'manifest': manifest, 'manifest_sha256': sha256(manifest_path),
        'base_config_sha256': sha256(ROOT / manifest['base_config']),
        'runner_sha256': sha256(Path(__file__)),
        'gpu_lock_sha256': sha256(ROOT / 'requirements-gpu-wsl-lock.txt'),
        'training_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'src').glob('*.py')},
        'final_test_evaluated': False, 'results': [],
    }
    cpu = ROOT / 'outputs/tuning-round1-2026-11-04/lstm_14d_lr003'
    cpu_report = read_json(cpu / 'report.json')
    state['cpu_reference'] = cpu_report['conditions'][0]
    state['cpu_environment'] = cpu_report['environment']
    try:
        for epochs in manifest['epochs']:
            trial = f'lstm_14d_epochs{epochs}'
            state['current'] = trial
            save(out / 'status.json', state)
            config = copy.deepcopy(cfg)
            config['training']['epochs'] = epochs
            path = out / f'{trial}.json'
            save(path, config)
            run = out / trial
            command([sys.executable, '-u', '-m', 'src.experiment', 'train', '--config', str(path),
                     '--source', str(args.source), '--kbo-source', str(args.kbo_source),
                     '--output', str(run)], out / f'{trial}.log')
            report = read_json(run / 'report.json')
            assert not report['final_test_evaluated'] and len(report['conditions']) == 1
            assert report['code_sha256'] == state['code_sha256']
            c = report['conditions'][0]
            assert c['weight_devices'] and all('GPU:' in d for d in c['weight_devices'])
            for key in ['joined_sha256', 'kbo_sha256', 'holidays_sha256']:
                assert report['sources'][key] == cpu_report['sources'][key]
            folder = run / 'lstm_14d'
            assert read_json(folder / 'target_dates.json') == read_json(cpu / 'lstm_14d/target_dates.json')
            verify(run, args.source, out / f'{trial}_verify.log')
            curve = read_json(folder / 'learning_curve.json')
            metrics = read_json(folder / 'validation_metrics.json')
            result = {
                'id': trial, 'epochs': epochs, 'config': config, 'condition': c, 'metrics': metrics,
                'environment': report['environment'], 'sources': report['sources'],
                'model_reload_verified': True, 'learning_curve': curve,
                'last_50_best_mae_gain': min(r['mae'] for r in curve[:-50]) - min(r['mae'] for r in curve)
                if len(curve) > 50 else None,
            }
            if state['results']:
                baseline = state['results'][0]
                check = copy.deepcopy(config)
                check['training']['epochs'] = baseline['epochs']
                assert check == baseline['config']
                n = min(len(curve), len(baseline['learning_curve']))
                result['common_prefix_epochs'] = n
                result['common_prefix_max_abs_mae_difference'] = max(
                    abs(curve[i]['mae'] - baseline['learning_curve'][i]['mae']) for i in range(n))
                b = baseline['metrics']['overall']['mae']
                result['mae_improvement'] = b - metrics['overall']['mae']
                result['improvement_pct'] = 100 * result['mae_improvement'] / b
            state['results'].append(result)
            save(out / 'status.json', state)
        state['status'] = 'completed'
        state.pop('current', None)
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(out / 'status.json', state)
        save(ROOT / 'reports/metrics/lstm_gpu_epochs_2026-11-04.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        save(out / 'status.json', state)
        raise


if __name__ == '__main__':
    main()

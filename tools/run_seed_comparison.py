"""Run EXP-008 repeated seeds, preserving each baseline's model and data settings."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys

from run_tuning_batch import ROOT, command, read_json, save, sha256, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--kbo-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=('rf', 'lstm'), required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / 'configs/seeds_2026-11-07.json'
    manifest = read_json(manifest_path)
    assert len(set(manifest['new_seeds'])) == 4 and 42 not in manifest['new_seeds']
    state = dict(status='running', started=datetime.now(timezone.utc).isoformat(),
                 manifest=manifest, manifest_sha256=sha256(manifest_path),
                 runner_sha256=sha256(Path(__file__)), final_test_evaluated=False, results=[],
                 code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'src').glob('*.py')})
    save(output / 'status.json', state)
    try:
        for name, spec in manifest['conditions'].items():
            if spec['model'] != args.group:
                continue
            baseline = ROOT / spec['baseline']
            reference = read_json(baseline / 'report.json')
            folder = f"{spec['model']}_{spec['window']}d"
            baseline_condition, = [c for c in reference['conditions']
                                   if c['model'] == spec['model'] and c['window'] == spec['window']]
            config = read_json(baseline / 'config.json')
            assert config['seed'] == 42 and not reference['final_test_evaluated']
            verify(baseline, args.source, output / f'{name}_baseline_verify.log')
            state['results'].append(dict(
                name=name, seed=42, reused=True, condition=baseline_condition,
                metrics=read_json(baseline / folder / 'validation_metrics.json'),
                environment=reference['environment'], sources=reference['sources'],
                run=spec['baseline'], reload_verified=True))
            save(output / 'status.json', state)
            for seed in manifest['new_seeds']:
                trial = f'{name}_seed{seed}'
                state['current'] = trial
                save(output / 'status.json', state)
                trial_config = copy.deepcopy(config)
                trial_config.update(seed=seed, models=[spec['model']], windows=[spec['window']],
                                    followup_approval=manifest['approval'])
                config_path = output / f'{trial}.json'
                save(config_path, trial_config)
                run = output / trial
                command([sys.executable, '-u', '-m', 'src.experiment', 'train', '--config', str(config_path),
                         '--source', str(args.source), '--kbo-source', str(args.kbo_source),
                         '--output', str(run)], output / f'{trial}.log')
                report = read_json(run / 'report.json')
                condition, = report['conditions']
                assert report['code_sha256'] == state['code_sha256']
                assert not report['final_test_evaluated']
                for key in ('joined_sha256', 'kbo_sha256', 'holidays_sha256'):
                    assert report['sources'][key] == reference['sources'][key]
                assert report['environment']['packages'] == reference['environment']['packages']
                for file in ('target_dates.json', 'context_features.json'):
                    assert read_json(run / folder / file) == read_json(baseline / folder / file)
                if spec['model'] == 'lstm':
                    assert condition['parameters'] == baseline_condition['parameters']
                    assert condition['weight_devices'] and all('GPU:' in d for d in condition['weight_devices'])
                verify(run, args.source, output / f'{trial}_verify.log')
                result = dict(name=name, seed=seed, reused=False, condition=condition,
                              metrics=read_json(run / folder / 'validation_metrics.json'),
                              environment=report['environment'], sources=report['sources'],
                              run=str(run.relative_to(ROOT)), reload_verified=True)
                if spec['model'] == 'lstm':
                    curve = read_json(run / folder / 'learning_curve.json')
                    result['last_train_mae'] = curve[-1]['train_mae']
                    result['last_validation_mae'] = curve[-1]['mae']
                state['results'].append(result)
                save(output / 'status.json', state)
        state.update(status='completed', finished=datetime.now(timezone.utc).isoformat())
        state.pop('current', None)
        save(output / 'status.json', state)
        save(ROOT / f'reports/metrics/seeds_{args.group}_2026-11-07.json', state)
    except Exception as error:
        state.update(status='failed', error=repr(error))
        save(output / 'status.json', state)
        raise


if __name__ == '__main__':
    main()

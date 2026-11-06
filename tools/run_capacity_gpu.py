"""Run EXP-007 neural GPU or random forest CPU conditions against local baselines."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from run_tuning_batch import ROOT, command, read_json, save, sha256, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--kbo-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=('neural', 'rf'), required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = read_json(ROOT / 'configs/capacity_gpu.json')
    state = dict(status='running', manifest=manifest, final_test_evaluated=False, results=[],
                 started=datetime.now(timezone.utc).isoformat(),
                 code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'src').glob('*.py')})
    try:
        for model, variants in manifest['models'].items():
            if (model == 'rf') != (args.group == 'rf'):
                continue
            baseline = ROOT / variants['baseline']
            reference = read_json(baseline / 'report.json')
            reference_condition, = [c for c in reference['conditions']
                                    if c['model'] == model and (model != 'rf' or c['window'] == 3)]
            folder = f"{model}_{reference_condition['window']}d"
            verify(baseline, args.source, out / f'{model}_reference_verify.log')
            for variant in (key for key in variants if key != 'baseline'):
                trial = f'{model}_{variant}'
                state['current'] = trial
                save(out / 'status.json', state)
                config = read_json(baseline / 'config.json')
                config['models'] = [model]
                config['windows'] = [reference_condition['window']]
                assert model == 'rf' or config['require_gpu']
                config[model].update(variants[variant])
                config['followup_approval'] = dict(manifest['approval'])
                if model == 'rf':
                    config['followup_approval']['evidence'] = manifest['rf_approval']
                path = out / f'{trial}.json'
                save(path, config)
                run = out / trial
                command([sys.executable, '-u', '-m', 'src.experiment', 'train', '--config', str(path),
                         '--source', str(args.source), '--kbo-source', str(args.kbo_source),
                         '--output', str(run)], out / f'{trial}.log')
                report = read_json(run / 'report.json')
                c, = report['conditions']
                assert not report['final_test_evaluated'] and report['code_sha256'] == state['code_sha256']
                if model != 'rf':
                    assert c['weight_devices'] and all('GPU:' in d for d in c['weight_devices'])
                    assert c['parameters'] > reference_condition['parameters']
                for key in ('joined_sha256', 'kbo_sha256', 'holidays_sha256'):
                    assert report['sources'][key] == reference['sources'][key]
                for file in ('target_dates.json', 'context_features.json'):
                    assert read_json(run / folder / file) == read_json(baseline / folder / file)
                assert report['environment']['packages'] == reference['environment']['packages']
                verify(run, args.source, out / f'{trial}_verify.log')
                metrics = read_json(run / folder / 'validation_metrics.json')
                state['results'].append(dict(
                    trial=trial, baseline=reference_condition, condition=c, metrics=metrics,
                    gain=reference_condition['validation_mae'] - c['validation_mae'],
                    environment=report['environment'], sources=report['sources'],
                    reload_verified=True))
                save(out / 'status.json', state)
        state.update(status='completed', finished=datetime.now(timezone.utc).isoformat())
        del state['current']
        save(out / 'status.json', state)
        save(ROOT / f'reports/metrics/capacity_{args.group}_2026-11-06.json', state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        save(out / 'status.json', state)
        raise


if __name__ == '__main__':
    main()

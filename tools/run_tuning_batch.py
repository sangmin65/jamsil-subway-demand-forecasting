"""Run the approved nine trials, verify restored models, and publish comparisons."""
import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import read_json, sha256

REPORT = ROOT / 'reports/experiments/EXP-004_tuning-round1.md'
METRICS = ROOT / 'reports/metrics/tuning_round1_2026-11-04.json'

def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)

def command(cmd, log):
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

def verify(run, source, log):
    command([sys.executable, 'tools/summarize_first_run.py', '--run', str(run),
             '--source', str(source), '--verify-partial'], log)

def publish(state):
    save(METRICS, state)
    lines = ['# EXP-004: 모델별 첫 파라미터 튜닝', '',
             '- 날짜: 2026-11-04에 승인·시작 (Asia/Seoul). 실제 UTC 시작·종료는 공유 JSON에 기록.',
             '- 상태: 승인한 9개 조건 학습 및 저장 모델 복원 검증 완료.',
             '- 승인: [DEC-026](../decisions/training.md#dec-026)',
             '- 명세: `configs/tuning_round1.json`. 실행 코드·기준 설정 SHA256은 공유 JSON에 보존.',
             '- 조건: RF 3일 최소 잎 크기 1·5·10 / MLP 3일·CNN 7일·LSTM 14일 학습률 0.0003·0.003.',
             '- 기준은 RF leaf 3, 신경망 학습률 0.001. 다른 설정은 각각의 기존 실행과 동일. 시드 42로 처음부터 학습.',
             '- 학습 2024년 352일 / 검증 2025년 상반기 181일. 하반기 최종 평가 미사용.', '',
             '| 모델·입력 | 변경 값 | 기준 MAE | 이번 MAE | 오차 감소율 | 최선/종료 에포크 | 종료 사유 | 소요 분 |',
             '|---|---:|---:|---:|---:|---|---|---:|']
    for row in state['results']:
        c=row['condition']; ep=f"{c['best_epoch']}/{c['stopped_epoch']}" if 'best_epoch' in c else '해당 없음'
        reason={'early_stopping':'조기 종료','maximum_epochs':'최대 횟수'}.get(c.get('stop_reason'),'나무 학습 완료')
        lines.append(f"| {c['model'].upper()} {c['window']}일 | {row['value']} | {row['baseline']['mae']:.2f} | {row['metrics']['overall']['mae']:.2f} | {row['improvement_pct']:+.2f}% | {ep} | {reason} | {c['seconds']/60:.2f} |")
    lines += ['', '오차 감소율은 (기준 MAE−이번 MAE)/기준 MAE입니다. 음수면 악화입니다.', '', '## 모델별 기준 포함 최선', '']
    for model in ['rf','mlp','cnn','lstm']:
        rows=[r for r in state['results'] if r['condition']['model']==model]
        first=rows[0]; options=[('기존 기준',first['baseline']['mae'])]+[(str(r['value']),r['metrics']['overall']['mae']) for r in rows]
        winner=min(options,key=lambda x:x[1])
        lines.append(f"- {model.upper()} {first['condition']['window']}일: {winner[0]}, MAE {winner[1]:.2f}명.")
    lines += ['', '## 검증 및 해석 범위', '',
              '- 기준 실행과 신규 실행의 원본·KBO·공휴일 해시, 학습·검증·평가 날짜가 일치합니다.',
              '- 기준 4개 실행 그룹과 새 9개 조건의 저장 모델을 재로딩하여 검증 예측 및 MAE·RMSE·최선 에포크를 확인했습니다. 재현 허용 오차 rtol=1e-5, atol=1e-4.',
              '- 2025년 하반기 정답은 평가 점수 계산이나 모델 선택에 사용하지 않았습니다.',
              '- 단일 시드와 기존 검증 최선 입력 기간에서의 탐색입니다. 검증 데이터 반복 사용에 따른 선택 편향 가능성이 있으며 작은 차이의 재현성과 다른 입력 기간은 후속 검토 대상입니다.',
              '- 최대 에포크에 도달한 조건은 수렴하지 않았을 수 있어 낮은 학습률 자체가 부적합하다고 단정하지 않습니다.',
              '- 전체 MAE뿐 아니라 승하차·월별·note 상태별 지표와 기준 대비 차이를 [공유 JSON](../metrics/tuning_round1_2026-11-04.json)에 저장했습니다. 기록 없음은 실제 비경기일로 확정하지 않습니다.',
              '- 일정의 예측 전 가용성·수집 완전성 등 기존 데이터 한계는 유지됩니다.']
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--kbo-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--commit-results',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    manifest=read_json(ROOT/'configs/tuning_round1.json')
    state={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
           'training_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           'manifest':manifest,'manifest_sha256':sha256(ROOT/'configs/tuning_round1.json'),
           'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'src').glob('*.py'))},
           'runner_sha256':sha256(Path(__file__)), 'final_test_evaluated':False,'results':[]}
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    hashes={'joined_sha256':sha256(args.source),'kbo_sha256':sha256(args.kbo_source),
            'holidays_sha256':sha256(ROOT/'data/reference/holidays_2024_2025.json')}
    state['sources']=hashes; save(out/'status.json',state)
    try:
        prepared=[]
        for trial in manifest['trials']:
            baseline=ROOT/trial['baseline_run']; br=read_json(baseline/'report.json')
            for k,v in hashes.items(): assert br['sources'][k]==v, 'Baseline input changed'
            cfg=read_json(ROOT/trial['base_config']); bc=read_json(baseline/'config.json')
            assert cfg==bc, 'Baseline config file differs from recorded run'
            cfg['models']=[trial['model']]; cfg['windows']=[trial['window']]
            section,key=trial['parameter'].split('.'); cfg[section][key]=trial['value']
            check=copy.deepcopy(cfg); check['models']=bc['models']; check['windows']=bc['windows']; check[section][key]=bc[section][key]
            assert check==bc, 'Unapproved config changes'
            cfg['followup_approval']=manifest['approval']
            path=out/(trial['id']+'.json'); save(path,cfg)
            prepared.append((trial,path,baseline,br))
        assert len(prepared)==9 and len({x[0]['id'] for x in prepared})==9
        verified=set()
        for trial,path,baseline,br in prepared:
            state['current']=trial['id']; save(out/'status.json',state)
            if baseline not in verified:
                verify(baseline,args.source,out/(trial['id']+'_baseline_verify.log')); verified.add(baseline)
            run=out/trial['id']
            command([sys.executable,'-u','-m','src.experiment','train','--config',str(path),
                     '--source',str(args.source),'--kbo-source',str(args.kbo_source),'--output',str(run)],out/(trial['id']+'.log'))
            report=read_json(run/'report.json'); assert len(report['conditions'])==1 and report['final_test_evaluated'] is False
            for k,v in hashes.items(): assert report['sources'][k]==v
            assert report['code_sha256']==state['code_sha256'], 'Training code changed during batch'
            c=report['conditions'][0]; folder=f"{c['model']}_{c['window']}d"
            assert read_json(run/folder/'target_dates.json')==read_json(baseline/folder/'target_dates.json')
            verify(run,args.source,out/(trial['id']+'_verify.log'))
            scores=read_json(run/folder/'validation_metrics.json'); bs=read_json(baseline/folder/'validation_metrics.json')
            improvement=bs['overall']['mae']-scores['overall']['mae']
            changes={group:{label:scores[group][label]['mae']-bs[group][label]['mae'] for label in scores[group]} for group in ['month','note_status']}
            row={**trial,'condition':c,'baseline':bs['overall'],'baseline_group_metrics':bs,'metrics':scores,
                 'mae_improvement':improvement,'improvement_pct':100*improvement/bs['overall']['mae'],
                 'group_mae_change_new_minus_baseline':changes,'run':str(run),'config_sha256':sha256(path),'model_reload_verified':True}
            state['results'].append(row); save(out/'status.json',state)
        state['status']='completed'; state['finished_utc']=datetime.now(timezone.utc).isoformat(); state.pop('current',None)
        publish(state); save(out/'status.json',state)
        if args.commit_results:
            assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==branch, 'Branch changed; results not committed'
            paths=[str(REPORT.relative_to(ROOT)),str(METRICS.relative_to(ROOT))]
            subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
            subprocess.run(['git','commit','--only','-m','experiment: 9개 튜닝 조건 검증과 기존 결과 비교 기록','--',*paths],cwd=ROOT,check=True)
            state['result_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            subprocess.run(['git','push','origin',branch],cwd=ROOT,check=True)
            state['results_pushed']=True; save(out/'status.json',state)
    except Exception as error:
        state['status']='failed' if len(state['results'])<9 else 'completed_publication_failed'
        state['error']=repr(error); save(out/'status.json',state); raise

if __name__=='__main__':
    main()

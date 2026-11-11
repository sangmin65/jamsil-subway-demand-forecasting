"""EXP-013 fresh seed42 MAE/Huber/MSE fits, restore and validation diagnostics."""
import argparse
import copy
import sys
from datetime import datetime, timezone
import numpy as np
from run_tuning_batch import ROOT,read_json,save,sha256,command,verify
from src.dataset import load_frame

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--kbo-source',required=True);a=p.parse_args()
    out=ROOT/'outputs/loss-comparison-2026-11-11';out.mkdir(exist_ok=False)
    previous=read_json(ROOT/'reports/metrics/learning_tuning.json')
    old=next(r for r in previous['results'] if r['variant']=='batch8' and r['seed']==42)
    base=ROOT/old['run'];base_report=read_json(base/'report.json')
    y=load_frame(a.source)[['승차인원','하차인원']].to_numpy().reshape(731,18,2)
    residual=abs(y[14:366]-y[7:359]).ravel()
    delta=float(round(float(np.quantile(residual,.9))/100)*100)
    assert delta==800
    sources=dict(joined_sha256=sha256(a.source),kbo_sha256=sha256(a.kbo_source),holidays_sha256=sha256(ROOT/'data/reference/holidays_2024_2025.json'))
    assert sources==previous['sources']
    state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),sources=sources,
        code_sha256={str(f.relative_to(ROOT)):sha256(f) for f in (ROOT/'src').glob('*.py')},
        huber_delta=delta,delta_reference=dict(period='2024-01-15/2024-12-31',days=352,method='7-day naive absolute residual q90 rounded to nearest 100 people',quantile90=float(np.quantile(residual,.9))),
        previous_mae=old['condition']['validation_mae'],results=[],final_test_evaluated=False)
    def checkpoint():save(out/'status.json',state)
    checkpoint()
    try:
        for variant,loss in [('mae','mean_absolute_error'),('huber','huber'),('mse','mean_squared_error')]:
            cfg=copy.deepcopy(read_json(base/'config.json'))
            cfg['loss']=loss
            if variant=='huber':cfg['huber_delta']=delta
            cfg['followup_approval']=dict(date='2026-11-11',reviewer='사용자',evidence='LSTM 1층·배치8 MAE/Huber/MSE 시드42 비교, delta 추천값 위임, 결과 비교·커밋')
            assert cfg['seed']==42 and cfg['training']['batch_size']==8 and cfg['lstm'].get('layers',1)==1
            state['current']=variant;checkpoint();config=out/f'{variant}.json';save(config,cfg)
            run=out/variant
            command([sys.executable,'-u','-m','src.experiment','train','--config',str(config),'--source',a.source,'--kbo-source',a.kbo_source,'--output',str(run)],out/f'{variant}.log')
            report=read_json(run/'report.json');c,=report['conditions']
            assert report['code_sha256']==state['code_sha256'] and not report['final_test_evaluated']
            assert all(report['sources'][k]==v for k,v in sources.items())
            assert report['environment']['packages']==base_report['environment']['packages']
            assert c['weight_devices'] and all('GPU:' in d for d in c['weight_devices'])
            verify(run,a.source,out/f'{variant}_verify.log')
            with np.load(run/'lstm_14d/validation_predictions.npz') as z,np.load(base/'lstm_14d/validation_predictions.npz') as b:
                np.testing.assert_array_equal(z['truth'],b['truth']);np.testing.assert_array_equal(z['dates'],b['dates'])
                if variant=='mae':np.testing.assert_allclose(z['prediction'],b['prediction'],rtol=1e-5,atol=1e-4)
                truth=z['truth'].reshape(-1,18,2);pred=z['prediction'].reshape(truth.shape)
                diagnostics={}
                for j,d in enumerate(['board','alight']):
                    t,v=truth[:,:,j],pred[:,:,j];e=v-t;idx=t.argmax(axis=1);ix=np.arange(len(t))
                    diag=dict(peak_mae=float(abs(e[ix,idx]).mean()),peak_bias=float(e[ix,idx].mean()),peak_within_hour=float((abs(v.argmax(axis=1)-idx)<=1).mean()),high={})
                    for q in [.8,.9,.95]:
                        threshold=float(np.quantile(t,q));mask=t>=threshold
                        diag['high'][str(q)]=dict(threshold=threshold,n=int(mask.sum()),mae=float(abs(e[mask]).mean()),normal_mae=float(abs(e[~mask]).mean()),shortage=float(np.maximum(-e[mask],0).mean()),miss_rate=float((v[mask]<threshold).mean()))
                    diagnostics[d]=diag
            state['results'].append(dict(variant=variant,config=cfg,condition=c,run=str(run.relative_to(ROOT)),metrics=read_json(run/'lstm_14d/validation_metrics.json'),diagnostics=diagnostics,reload_verified=True))
            checkpoint()
        state.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat());state.pop('current',None)
        state['winner']=min(state['results'],key=lambda r:r['condition']['validation_mae'])['variant']
        save(ROOT/'reports/metrics/loss_comparison.json',state)
        lines=['# EXP-013 LSTM 손실 함수 시드42 비교','',
            '- DEC-035 승인. 14일, LSTM32유닛1층·Dense32, 배치8, Adam0.003, Dropout0, 최대2000·patience50. 세 조건 모두 새 초기화. 변경은 손실 함수뿐.',
            '- 조기 종료·최선 복원은 음수0 보정 검증 MAE. train_loss와 train_mae를 분리. 손실별 수치 크기를 서로 성능 지표처럼 비교하지 않는다.',
            '- Huber delta800명: 2024-01-15~12-31 학습352일의 지난주 동일시간 기준 절대오차90백분위784명을 100명 단위 반올림. 검증·최종 평가 자료로 delta를 맞추지 않았다. 최적값을 주장하지 않는다.',
            '- Huber는 |오차|≤800에서 0.5×오차², 초과하면 800×(|오차|−400). 목표값은 원래 인원 단위. 동일 학습률에서도 손실별 기울기 크기는 다르므로 손실별 최적화까지 완료한 비교가 아니다.',
            '- 2024년352일 학습 / 2025상반기181일 검증. 추가 행사 정보·하반기 최종 평가 미사용. 단일 시드 탐색이며 통계적 유의성·일반화 주장 없음.','',
            '| 손실 | MAE | 기존 대비 감소 | 승차 MAE | 하차 MAE | 최선/종료 | 종료 사유 | 분 |','|---|---:|---:|---:|---:|---|---|---:|']
        for r in state['results']:
            c=r['condition'];m=r['metrics']['overall']
            lines.append(f"| {r['variant']} | {c['validation_mae']:.2f} | {state['previous_mae']-c['validation_mae']:+.2f} | {m['board_mae']:.2f} | {m['alight_mae']:.2f} | {c['best_epoch']}/{c['stopped_epoch']} | {c['stop_reason']} | {c['seconds']/60:.2f} |")
        lines+=['','## 피크·고수요 비교','', '상위10%는 승차/하차 각각 실제값 기준326개. 기본 ANALYSIS-001과 같은 사후 진단이다. 상위5%·20% 민감도는 JSON에 포함.','',
            '| 손실 | 방향 | 고수요 MAE | 일반 MAE | 고수요 평균 부족 인원 | 실제 피크 시점 MAE | 시각 ±1시간 |','|---|---|---:|---:|---:|---:|---:|']
        for r in state['results']:
            for d,z in r['diagnostics'].items():
                h=z['high']['0.9'];lines.append(f"| {r['variant']} | {d} | {h['mae']:.2f} | {h['normal_mae']:.2f} | {h['shortage']:.2f} | {z['peak_mae']:.2f} | {100*z['peak_within_hour']:.1f}% |")
        lines+=['',f"- 이번 세 조건의 전체 검증 MAE 최저: **{state['winner']}**. 최종 모델 선정이 아니다.",
            '- 다음 판단: 전체 MAE와 고수요 개선/일반 구간 악화를 함께 검토한 뒤 유망 조건을 동일3시드로 확인한다. 후속 학습 자동 실행 없음.',
            '- 세 모델 복원 예측 검증 완료, 원본 해시·정답·날짜·패키지 동일성 확인. 새 MAE 실행은 기존 배치8 예측과 rtol1e-5/atol1e-4로 일치.',
            '- 재현: tools/run_loss_comparison.py --source <기존 결합CSV> --kbo-source <기존 KBO CSV>. 출력 경로가 이미 있으면 덮어쓰지 않고 중단한다.',
            '- 전체 결과: [JSON](../metrics/loss_comparison.json). 조건별 학습곡선·예측·모델은 JSON의 run 경로 아래 lstm_14d에 보존.','']
        (ROOT/'reports/experiments/EXP-013_loss-comparison.md').write_text('\n'.join(lines),encoding='utf8')
        checkpoint()
    except Exception as e:
        state.update(status='failed',error=repr(e));checkpoint();raise

if __name__=='__main__':main()

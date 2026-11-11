"""EXP-014: head Dropout .1/.2, seed42, plus fixed RF 50:50 diagnostics."""
import argparse
import copy
import sys
from datetime import datetime, timezone
import numpy as np
from run_tuning_batch import ROOT, read_json, save, sha256, command, verify

def diagnose(y,p):
    e=p-y
    out=dict(mae=float(abs(e).mean()),directions={},monthly={})
    for j,d in enumerate(['board','alight']):
        a,b=y[:,:,j],p[:,:,j];err=b-a;ix=np.arange(len(a));peak=a.argmax(axis=1)
        z=dict(mae=float(abs(err).mean()),bias=float(err.mean()),peak_mae=float(abs(err[ix,peak]).mean()),
               peak_within_hour=float((abs(b.argmax(axis=1)-peak)<=1).mean()),high={})
        for q in [.8,.9,.95]:
            t=float(np.quantile(a,q));mask=a>=t
            z['high'][str(q)]=dict(threshold=t,n=int(mask.sum()),mae=float(abs(err[mask]).mean()),
                normal_mae=float(abs(err[~mask]).mean()),shortage=float(np.maximum(-err[mask],0).mean()),miss_rate=float((b[mask]<t).mean()))
        out['directions'][d]=z
    return out

def publish(s):
    s['best_lstm']=min(s['results'],key=lambda r:r['diagnostics']['lstm']['mae'])['dropout']
    s['best_average']=min(s['results'],key=lambda r:r['diagnostics']['average50']['mae'])['dropout']
    save(ROOT/'reports/metrics/lstm_dropout.json',s)
    lines=['# EXP-014 LSTM 마지막 Dropout 및 고정50:50 비교','',
      '- 2026-11-11 사용자 승인: LSTM Dropout 실험 진행. DEC-036. 시드42, 14일,32유닛1층→달력/경기 결합→Dense32(ReLU)→Dropout→출력36. LSTM 내부 dropout/recurrent_dropout은0 유지.',
      '- 비율0 기준은 EXP-013 MAE 재현 실행 재사용,0.1/0.2는 새 초기화. 배치8·Adam0.003·MAE·최대2000·patience50·min_delta0 유지. 조기 종료는 LSTM 검증MAE 기준이며 결합 점수로 에포크를 선택하지 않았다.',
      '- 각 LSTM의 음수0 보정 예측과 같은 시드RF3일·잎3 예측을50:50 평균. 가중치 탐색·추가 시드 학습·하반기 최종 평가·추가 행사 입력 없음.',
      '- 동일2025상반기181일,6516개 값. 고수요는 방향별 실제값 상위10%,각326개.5/20%민감도 JSON 포함. 사후 진단 기준이며 운영 혼잡 기준이 아니다.','',
      '| Dropout | LSTM MAE | 기준 대비 감소 | 50:50 MAE | 기존50:50 대비 감소 | 최선/종료 | 분 | 재사용 |','|---|---:|---:|---:|---:|---|---:|---|']
    base=s['results'][0]
    for r in s['results']:
        c=r['condition'];d=r['diagnostics']
        lines.append(f"| {r['dropout']} | {d['lstm']['mae']:.2f} | {base['diagnostics']['lstm']['mae']-d['lstm']['mae']:+.2f} | {d['average50']['mae']:.2f} | {base['diagnostics']['average50']['mae']-d['average50']['mae']:+.2f} | {c['best_epoch']}/{c['stopped_epoch']} | {c['seconds']/60:.2f} | {r['reused']} |")
    lines+=['','## 고수요·피크','', '| Dropout | 예측 | 방향 | 고수요 MAE | 일반 MAE | 고수요 부족인원 | 피크 MAE |','|---|---|---|---:|---:|---:|---:|']
    for r in s['results']:
        for kind,z in r['diagnostics'].items():
            for direction,d in z['directions'].items():
                h=d['high']['0.9'];lines.append(f"| {r['dropout']} | {kind} | {direction} | {h['mae']:.2f} | {h['normal_mae']:.2f} | {h['shortage']:.2f} | {d['peak_mae']:.2f} |")
    lines+=['',f"- 단독 최저 Dropout {s['best_lstm']}, 고정50:50 최저 Dropout {s['best_average']}. 단일시드 검증 결과이며 최종채택이나 통계적 유의성 주장이 아니다.",
      '- 모든 모델 복원 예측 검증 완료. 원본해시·날짜·정답·환경 동일 확인. 고수요·피크 악화와 전체MAE 개선을 구분해 후속3시드 검증 여부를 결정한다. 자동 확장 없음.',
      '- [전체JSON](../metrics/lstm_dropout.json)에 월별·승하차·고수요 지표와 종료사유·학습곡선 경로 보존. 소요시간은 재사용 기준의 과거 기록과 비교하므로 통제된 속도 벤치마크가 아니다.',
      '- 재현: tools/run_lstm_dropout.py --source <기존 결합CSV> --kbo-source <기존 KBO CSV>. 기존 출력 디렉터리는 덮어쓰지 않음.','']
    (ROOT/'reports/experiments/EXP-014_lstm-dropout.md').write_text('\n'.join(lines),encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--kbo-source',required=True);a=p.parse_args()
    out=ROOT/'outputs/lstm-dropout-2026-11-11';out.mkdir(exist_ok=False)
    old=read_json(ROOT/'reports/metrics/loss_comparison.json');base=ROOT/next(r['run'] for r in old['results'] if r['variant']=='mae')
    bc=read_json(base/'config.json');br=read_json(base/'report.json')
    assert bc['seed']==42 and bc['training']['batch_size']==8 and bc['loss']=='mean_absolute_error' and bc['lstm'].get('layers',1)==1
    sources=dict(joined_sha256=sha256(a.source),kbo_sha256=sha256(a.kbo_source),holidays_sha256=sha256(ROOT/'data/reference/holidays_2024_2025.json'))
    assert sources==old['sources']
    rf_run=ROOT/'outputs/first-run-2026-11-03'
    rf_report=read_json(rf_run/'report.json')
    assert all(rf_report['sources'][k]==v for k,v in sources.items())
    state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),sources=sources,results=[],final_test_evaluated=False,
        code_sha256={str(f.relative_to(ROOT)):sha256(f) for f in (ROOT/'src').glob('*.py')})
    def checkpoint():save(out/'status.json',state)
    checkpoint()
    try:
        command([sys.executable,'tools/summarize_first_run.py','--run',str(rf_run),'--source',a.source,'--verify-partial','--model','rf','--window','3'],out/'rf_verify.log')
        for rate in [0.,.1,.2]:
            state['current']=f'dropout{rate}';checkpoint()
            cfg=copy.deepcopy(bc);run=base
            if rate:
                cfg['head_dropout']=rate;cfg['followup_approval']=dict(date='2026-11-11',reviewer='사용자',evidence='LSTM1층·배치8 마지막 Dropout0.1/0.2 시드42 및 RF50:50 비교 제안 후 그거 실험하자 승인')
                name=f'dropout{int(rate*100)}';config=out/f'{name}.json';save(config,cfg);run=out/name
                command([sys.executable,'-u','-m','src.experiment','train','--config',str(config),'--source',a.source,'--kbo-source',a.kbo_source,'--output',str(run)],out/f'{name}.log')
            rep=read_json(run/'report.json');c,=rep['conditions']
            assert read_json(run/'config.json')==cfg and not rep['final_test_evaluated']
            assert all(rep['sources'][k]==v for k,v in sources.items()) and rep['environment']['packages']==br['environment']['packages']
            assert c['weight_devices'] and all('GPU:' in d for d in c['weight_devices'])
            if rate:assert rep['code_sha256']==state['code_sha256']
            verify(run,a.source,out/f'dropout{rate}_verify.log')
            with np.load(run/'lstm_14d/validation_predictions.npz') as z,np.load(base/'lstm_14d/validation_predictions.npz') as b,np.load(rf_run/'rf_3d/validation_predictions.npz') as rf:
                for other in [b,rf]:
                    np.testing.assert_array_equal(z['dates'],other['dates']);np.testing.assert_array_equal(z['truth'],other['truth'])
                y=z['truth'].reshape(181,18,2);pred=z['prediction'].reshape(y.shape).astype(float);rpred=rf['prediction'].reshape(y.shape).astype(float)
                assert np.isfinite(pred).all() and (pred>=0).all()
                avg=(pred+rpred)/2;diags={k:diagnose(y,v) for k,v in [('lstm',pred),('average50',avg)]}
                dates=z['dates'].astype(str)
                for k,v in [('lstm',pred),('average50',avg)]:
                    for month in range(1,7):
                        mask=np.array([int(d[5:7])==month for d in dates]);diags[k]['monthly'][str(month)]=float(abs(v[mask]-y[mask]).mean())
                np.testing.assert_allclose(diags['lstm']['mae'],c['validation_mae'])
                if rate==0:np.testing.assert_allclose(diags['average50']['mae'],read_json(ROOT/'reports/metrics/ensemble_analysis.json')['models']['Average50']['overall']['mae'])
                np.savez_compressed(out/f'average_dropout{rate}.npz',dates=dates,truth=y,prediction=avg)
            state['results'].append(dict(dropout=rate,reused=rate==0,run=str(run.relative_to(ROOT)),config=cfg,condition=c,diagnostics=diags,
                metrics=read_json(run/'lstm_14d/validation_metrics.json'),curve_path=str((run/'lstm_14d/learning_curve.json').relative_to(ROOT)),reload_verified=True))
            checkpoint()
        state.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat());state.pop('current',None);publish(state);checkpoint()
    except Exception as e:
        state.update(status='failed',error=repr(e));checkpoint();raise

if __name__=='__main__':main()

"""Frozen FINAL-001: existing seed42 RF versus RF/LSTM 50:50, no fitting."""
import argparse
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dataset import read_json,sha256,load_frame,build_window,inputs
from src.experiment import predict_neural,note_statuses,environment,save_json
from src.evaluation import grouped_metrics

def diagnostics(y,p,thresholds):
    a=y.reshape(-1,18,2);b=p.reshape(a.shape);out={}
    for j,d in enumerate(['board','alight']):
        t=thresholds[d];truth=a[:,:,j];pred=b[:,:,j];e=pred-truth;hi=truth>=t;ix=np.arange(len(a));peak=truth.argmax(axis=1)
        def group(mask):
            assert mask.sum()>0
            return dict(n=int(mask.sum()),mae=float(abs(e[mask]).mean()),bias=float(e[mask].mean()),shortage=float(np.maximum(-e[mask],0).mean()),under_fraction=float((e[mask]<0).mean()))
        out[d]=dict(threshold=t,high=group(hi),normal=group(~hi),miss_rate=float((pred[hi]<t).mean()),false_positive_count=int((pred[~hi]>=t).sum()),
            peak_mae=float(abs(e[ix,peak]).mean()),peak_bias=float(e[ix,peak].mean()),peak_within_hour=float((abs(pred.argmax(axis=1)-peak)<=1).mean()),daily_total_mae=float(abs(e.sum(axis=1)).mean()))
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--kbo-source',required=True);a=parser.parse_args()
    plan_path=ROOT/'configs/final_evaluation.json';plan=read_json(plan_path)
    out=ROOT/'outputs/final-evaluation-2026-11-11';out.mkdir(exist_ok=False)
    state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),plan=plan,plan_sha256=sha256(plan_path),new_training=False,final_test_evaluated=False)
    save_json(out/'status.json',state)
    try:
        state['environment']=environment();frame=load_frame(a.source);holidays=read_json(ROOT/'data/reference/holidays_2024_2025.json');statuses=note_statuses(a.kbo_source)
        state['sources']=dict(joined_sha256=sha256(a.source),kbo_sha256=sha256(a.kbo_source),holidays_sha256=sha256(ROOT/'data/reference/holidays_2024_2025.json'))
        predictions={};reference=None;state['verification']={};start=time.perf_counter()
        for name,spec in plan['models'].items():
            for path,digest in spec['sha256'].items():assert sha256(ROOT/path)==digest,'Frozen artifact changed'
            run=ROOT/spec['run'];cfg=read_json(run/'config.json');pre=read_json(run/'preprocessor.json');old=read_json(run/'report.json')
            assert all(old['sources'][k]==v for k,v in state['sources'].items())
            assert cfg['seed']==42 and cfg['excluded_test_dates']==plan['excluded_dates']
            if name=='RF':
                import joblib
                assert cfg['rf']['min_samples_leaf']==3 and cfg['rf']['n_estimators']==200 and cfg['rf']['max_depth']==10
                model=joblib.load(run/spec['subdirectory']/spec['artifact']);window=3;kind='rf'
                def predict(data):return model.predict(inputs(data,'rf'))
            else:
                from tensorflow import keras
                assert cfg['training']['batch_size']==8 and cfg.get('head_dropout',0)==0 and cfg['loss']=='mean_absolute_error' and cfg['lstm'].get('layers',1)==1
                model=keras.models.load_model(run/spec['subdirectory']/spec['artifact'],compile=False);window=14;kind='lstm'
                def predict(data):return predict_neural(model,inputs(data,'lstm'),8)
            splits,features=build_window(frame,cfg,pre,holidays,window)
            assert features==read_json(run/spec['subdirectory']/'context_features.json')
            saved_dates=read_json(run/spec['subdirectory']/'target_dates.json')
            for split in ['train','validation','test']:assert list(splits[split]['dates'])==saved_dates[split]
            val=predict(splits['validation'])
            with np.load(run/spec['subdirectory']/'validation_predictions.npz') as stored:
                np.testing.assert_array_equal(stored['truth'],splits['validation']['y']);np.testing.assert_allclose(val,stored['raw'],rtol=1e-5,atol=1e-4)
            test=splits['test'];dates=test['dates'];truth=test['y']
            assert len(dates)==178 and len(set(dates))==178 and dates[0]==plan['test_start'] and dates[-1]==plan['test_end']
            assert not set(dates)&set(plan['excluded_dates'])
            if reference is None:reference=(dates.copy(),truth.copy())
            np.testing.assert_array_equal(dates,reference[0]);np.testing.assert_array_equal(truth,reference[1])
            raw=predict(test);assert raw.shape==(178,36) and np.isfinite(raw).all()
            predictions[name]=np.maximum(raw,0)
            state['verification'][name]=dict(validation_reload_verified=True,preprocessor_reused=True,negative_fraction=float((raw<0).mean()))
        dates,truth=reference
        combined=(predictions['RF']+predictions['LSTM'])/2
        assert ((combined>=np.minimum(predictions['RF'],predictions['LSTM']))&(combined<=np.maximum(predictions['RF'],predictions['LSTM']))).all()
        state['results']={name:dict(metrics=grouped_metrics(truth,pred,dates,statuses),diagnostics=diagnostics(truth,pred,plan['high_demand_thresholds_from_validation'])) for name,pred in [('RF',predictions['RF']),('RF_LSTM_50',combined)]}
        np.savez_compressed(out/'test_predictions.npz',dates=dates,truth=truth,rf=predictions['RF'],lstm_component=predictions['LSTM'],combined=combined)
        save_json(out/'test_dates.json',dates.tolist())
        val=read_json(ROOT/'reports/metrics/ensemble_analysis.json')
        state['validation_reference']={n:val['models'][k] for n,k in [('RF','RF'),('RF_LSTM_50','Average50')]}
        r=state['results'];rf=r['RF']['metrics']['overall']['mae'];co=r['RF_LSTM_50']['metrics']['overall']['mae']
        state['comparison']=dict(mae_reduction=rf-co,improvement_pct=100*(1-co/rf))
        state.update(status='completed',final_test_evaluated=True,finished_utc=datetime.now(timezone.utc).isoformat(),evaluation_seconds=time.perf_counter()-start)
        save_json(ROOT/'reports/metrics/final_evaluation.json',state)
        lines=['# FINAL-001 고정 RF·LSTM50:50와 RF 단독 최종 평가','',
          '- 2026-11-11 사용자 승인 및 DEC-037. 최종 평가 전 configs/final_evaluation.json에 모델·전처리·설정SHA256과 결합비율을 고정했다. 시드42 RF3일·잎3 및 LSTM14일·1층32·배치8·MAE·Dropout0. 재학습 없음.',
          '- 학습2024년352일, 선택/조기종료2025상반기181일, 최종평가2025-07-01~12-31 중178일(6408개 인원값). 과거 탐색6일: '+', '.join(plan['excluded_dates'])+' 제외. 제외일 점수는 계산하지 않았다.',
          '- 매일 전날까지의 실제 수요를 입력하는 하루앞 예측이다. 하반기 전체를6개월 전에 예측한 것이 아니다. 전날23~24시 자료까지 확보 가정이며 실제 공개 지연·야구일정의 당시 가용성은 미검증.',
          '- 각 모델 음수예측을0으로 보정 후50:50 평균, 반올림 없음. 최종 결과에 따라 가중치·모델·날짜·입력을 조정하지 않았다. 학습 전처리와 최선가중치를 그대로 재사용.','',
          '| 구성 | 상반기 검증 MAE | 하반기 최종 MAE | 승차 MAE | 하차 MAE | 승차 RMSE | 하차 RMSE |','|---|---:|---:|---:|---:|---:|---:|']
        for name,z in r.items():
            m=z['metrics']['overall'];lines.append(f"| {name} | {state['validation_reference'][name]['overall']['mae']:.2f} | {m['mae']:.2f} | {m['board_mae']:.2f} | {m['alight_mae']:.2f} | {m['board_rmse']:.2f} | {m['alight_rmse']:.2f} |")
        lines+=['',f"- 고정50:50의 RF 대비 MAE 감소 {rf-co:+.2f}명({state['comparison']['improvement_pct']:+.2f}%). 양수 개선·음수 악화.",'','## 월별 최종 MAE','', '| 월 | 평가 일수 | RF | 고정50:50 |','|---|---:|---:|---:|']
        for month,m in r['RF']['metrics']['month'].items():lines.append(f"| {month} | {m['days']} | {m['mae']:.2f} | {r['RF_LSTM_50']['metrics']['month'][month]['mae']:.2f} |")
        lines+=['','## 고수요·피크','',
          '상반기에서 정한 고수요 임계값을 평가 전 고정: 승차1049.3명 이상, 하차1425.6명 이상. 하반기 상위10%를 다시 선택한 것이 아니므로 표본 비율은10%와 다를 수 있다. 실제 혼잡/안전 기준은 아니다. 피크는 일별 최대, 동률이면 가장 이른 시간.','',
          '| 구성 | 방향 | 고수요 표본수 | 고수요 MAE | 일반 MAE | 고수요 부족 인원 | 피크 MAE |','|---|---|---:|---:|---:|---:|---:|']
        for name,z in r.items():
            for d,v in z['diagnostics'].items():lines.append(f"| {name} | {d} | {v['high']['n']} | {v['high']['mae']:.2f} | {v['normal']['mae']:.2f} | {v['high']['shortage']:.2f} | {v['peak_mae']:.2f} |")
        lines+=['','## 재현·한계','',
          '- 두 모델의 저장된 검증예측 재현, 전체 대상 날짜·정답 완전일치, 원본/모델/전처리/설정SHA256 확인. 유한예측·평균범위 검사 통과. note상태별 점수는 JSON에 보존하며 기록없음을 비경기일로 확정하지 않음.',
          '- 상반기와 하반기는 수요 분포·계절·경기 구성이 달라 점수 차이를 모델 개선/악화의 단일 원인으로 설명할 수 없다. 단일시드·한 평가기간이므로 통계적 유의성이나 모든 기간 우위를 주장하지 않는다.',
          '- 하반기는 이제 공개된 평가 결과이므로 향후 이를 보고 수정한 모델에는 독립 최종평가로 다시 주장하지 않는다. 추가 실험은 후속 탐색으로 구분해야 한다.',
          '- 재현 명령: tools/evaluate_final.py --source <기존CSV> --kbo-source <기존KBO>. 기존출력 덮어쓰기 금지. 환경·원본·구성은 [전체JSON](../metrics/final_evaluation.json)에 기록. 출력예측은 outputs/final-evaluation-2026-11-11/test_predictions.npz에 보존.','']
        (ROOT/'reports/experiments/FINAL-001_frozen-evaluation.md').write_text('\n'.join(lines),encoding='utf8')
        save_json(out/'status.json',state)
        print(state['comparison'])
    except Exception as e:
        state.update(status='failed',error=repr(e));save_json(out/'status.json',state);raise

if __name__=='__main__':main()

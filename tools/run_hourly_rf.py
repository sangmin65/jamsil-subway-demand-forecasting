"""Validation-only case/cost audit followed by 18 two-output RF fits."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dataset import make_context,inputs,read_json,sha256,TARGETS
from src.models import build_rf
from src.experiment import predict_neural,environment,save_json,note_statuses
from src.evaluation import grouped_metrics

def prefix_data(source,pre,window):
    # Only construct train/validation arrays; never construct a test split.
    f=pd.read_csv(source,encoding='utf-8-sig')
    f=f[(f['호선']=='2호선')&(f['역명']=='종합운동장역')&(f['날짜']<='2025-06-30')].copy()
    f['date']=pd.to_datetime(f['날짜']);f['hour_actual']=f['시간대'].str.extract(r'^(\d+)시').astype(int)
    f=f.sort_values(['date','hour_actual']).reset_index(drop=True)
    days=pd.date_range('2024-01-01','2025-06-30')
    assert pd.MultiIndex.from_frame(f[['date','hour_actual']]).equals(pd.MultiIndex.from_product([days,range(6,24)]))
    y=f[TARGETS].to_numpy(dtype='float32').reshape(len(days),18,2)
    raw,context,names=make_context(f,pre,read_json(ROOT/'data/reference/holidays_2024_2025.json'))
    positions=np.arange(14,len(days));hist=np.stack([y[i-window:i].reshape(window*18,2) for i in positions])
    norm=((hist-pre['mean'])/pre['scale']).astype('float32');dates=days[positions].strftime('%Y-%m-%d').to_numpy(dtype=str)
    result={}
    for label,mask in [('train',dates<'2025-01-01'),('validation',dates>='2025-01-01')]:
        result[label]=dict(history=norm[mask],history_raw=hist[mask],context=context[positions][mask],context_raw=raw[positions][mask],y=y[positions][mask].reshape(-1,36),dates=dates[mask])
    assert len(result['train']['y'])==352 and len(result['validation']['y'])==181
    return result,names

def diagnostics(y,p):
    a=y.reshape(-1,18,2);b=p.reshape(a.shape);out={}
    for j,d in enumerate(['board','alight']):
        t,v=a[:,:,j],b[:,:,j];e=v-t;ix=np.arange(len(t));peak=t.argmax(axis=1)
        z=dict(peak_mae=float(abs(e[ix,peak]).mean()),total_mae=float(abs(e.sum(axis=1)).mean()),peak_within_hour=float((abs(v.argmax(axis=1)-peak)<=1).mean()),high={})
        for q in [.8,.9,.95]:
            threshold=float(np.quantile(t,q));h=t>=threshold
            z['high'][str(q)]=dict(threshold=threshold,n=int(h.sum()),mae=float(abs(e[h]).mean()),normal_mae=float(abs(e[~h]).mean()),shortage=float(np.maximum(-e[h],0).mean()),miss_rate=float((v[h]<threshold).mean()))
        out[d]=z
    return out

def case_audit(y,predictions,dates):
    y=y.reshape(-1,18,2);daily=[];summary={}
    for name,p in predictions.items():
        p=p.reshape(y.shape);rows=[]
        for i,date in enumerate(dates):
            for j,d in enumerate(['board','alight']):
                a,b=y[i,:,j],p[i,:,j];e=b-a;peak=int(a.argmax());pp=int(b.argmax());total=float(abs(a.sum()-b.sum()));absolute=float(abs(e).sum())
                row=dict(model=name,date=str(date),direction=d,total_signed_error=float(e.sum()),total_absolute_error=total,
                    sum_hourly_absolute_error=absolute,cancellation_component=absolute-total,
                    peak_time_error=pp-peak,peak_size_error=float(b.max()-a.max()),error_at_actual_peak=float(e[peak]),
                    actual_peak=int(peak+6),predicted_peak=int(pp+6),mae=float(abs(e).mean()))
                assert absolute+1e-6>=total
                rows.append(row);daily.append(row)
        summary[name]={d:dict(days=181,timing_over_one_hour=sum(abs(r['peak_time_error'])>1 for r in rows if r['direction']==d),
            actual_peak_under=sum(r['error_at_actual_peak']<0 for r in rows if r['direction']==d),
            hourly_absolute_error=sum(r['sum_hourly_absolute_error'] for r in rows if r['direction']==d),
            total_absolute_error=sum(r['total_absolute_error'] for r in rows if r['direction']==d),
            within_day_cancellation=sum(r['cancellation_component'] for r in rows if r['direction']==d)) for d in ['board','alight']}
    rf=abs(predictions['RF'].reshape(y.shape)-y).mean(axis=(1,2));avg=abs(predictions['Average50'].reshape(y.shape)-y).mean(axis=(1,2));gain=rf-avg
    best=np.argsort(-gain)[:5];worst=np.argsort(gain)[:5]
    def rows(indices):return [dict(date=str(dates[i]),rf_mae=float(rf[i]),average_mae=float(avg[i]),improvement=float(gain[i])) for i in indices]
    return dict(summary=summary,daily=daily,improved_days=int((gain>0).sum()),worsened_days=int((gain<0).sum()),tied_days=int((gain==0).sum()),
        best_cases=rows(best),worst_cases=rows(worst),selection='Top5 improvement and deterioration versus RF, daily mean over36 values, validation only; descriptive extreme cases')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--kbo-source',required=True);a=p.parse_args()
    out=ROOT/'outputs/hourly-rf-2026-11-11';out.mkdir(exist_ok=False)
    s=dict(status='running',phase='audit',new_training=True,final_test_evaluated=False,seed=42,results={},environment=environment())
    def checkpoint():save_json(out/'status.json',s)
    checkpoint()
    try:
        rf_run=ROOT/'outputs/first-run-2026-11-03';lstm_run=ROOT/'outputs/loss-comparison-2026-11-11/mae'
        cfg=read_json(rf_run/'config.json');s['rf_parameters']=cfg['rf'];s['sources']=dict(joined_sha256=sha256(a.source),kbo_sha256=sha256(a.kbo_source),holidays_sha256=sha256(ROOT/'data/reference/holidays_2024_2025.json'))
        s['model_hashes']={str(p.relative_to(ROOT)):sha256(p) for p in [rf_run/'rf_3d/model.joblib',lstm_run/'lstm_14d/model.keras']}
        for run in [rf_run,lstm_run]:assert all(read_json(run/'report.json')['sources'][k]==v for k,v in s['sources'].items())
        data,names=prefix_data(a.source,read_json(rf_run/'preprocessor.json'),3)
        ld,lnames=prefix_data(a.source,read_json(lstm_run/'preprocessor.json'),14)
        assert names==read_json(rf_run/'rf_3d/context_features.json') and lnames==read_json(lstm_run/'lstm_14d/context_features.json')
        for d in [data,ld]:
            np.testing.assert_array_equal(d['train']['y'],data['train']['y']);np.testing.assert_array_equal(d['validation']['y'],data['validation']['y'])
        y=data['validation']['y'];dates=data['validation']['dates'];x=inputs(data['validation'],'rf');xt=inputs(data['train'],'rf');lx=inputs(ld['validation'],'lstm')
        from tensorflow import keras
        rf=joblib.load(rf_run/'rf_3d/model.joblib');lstm=keras.models.load_model(lstm_run/'lstm_14d/model.keras',compile=False)
        rp=rf.predict(x);lp=predict_neural(lstm,lx,8)
        for run,sub,pred in [(rf_run,'rf_3d',rp),(lstm_run,'lstm_14d',lp)]:
            with np.load(run/sub/'validation_predictions.npz') as z:
                np.testing.assert_array_equal(y,z['truth']);np.testing.assert_array_equal(dates,z['dates']);np.testing.assert_allclose(pred,z['raw'],rtol=1e-5,atol=1e-4)
        rp=np.maximum(rp,0);lp=np.maximum(lp,0);ap=(rp+lp)/2
        preds={'RF':rp,'LSTM':lp,'Average50':ap}
        audit=case_audit(y,preds,dates);save_json(ROOT/'reports/metrics/error_case_audit.json',audit)
        np.savez_compressed(out/'baseline_predictions.npz',dates=dates,truth=y,rf=rp,lstm=lp,combined=ap)
        # Warmed single-day 36-output wall latency; .numpy synchronizes GPU completion.
        def rf_one(i):return np.maximum(rf.predict(x[i:i+1]),0)
        def lstm_one(i):return np.maximum(lstm({k:v[i:i+1] for k,v in lx.items()},training=False).numpy(),0)
        funcs={'RF':rf_one,'LSTM':lstm_one,'Average50':lambda i:(rf_one(i)+lstm_one(i))/2}
        for fn in funcs.values():
            for i in range(5):fn(i)
        timing={k:[] for k in funcs}
        for i in range(30):
            keys=list(funcs);keys=keys[i%3:]+keys[:i%3]
            for k in keys:
                start=time.perf_counter();v=funcs[k](i);timing[k].append((time.perf_counter()-start)*1000)
                assert v.shape==(1,36) and np.isfinite(v).all()
        costs={k:dict(median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),samples_ms=v) for k,v in timing.items()}
        costs['artifact_bytes']={'RF':(rf_run/'rf_3d/model.joblib').stat().st_size,'LSTM':(lstm_run/'lstm_14d/model.keras').stat().st_size}
        save_json(ROOT/'reports/metrics/prediction_cost.json',dict(environment=s['environment'],costs=costs,scope='Same WSL session; CPU RF n_jobs1, GPU LSTM, float32; prepared in-memory inputs; 5 warmups each,30 single-day trials, rotated order; excludes disk/model load and preprocessing; not controlled system load benchmark'))
        s['phase']='hourly_rf_training';checkpoint();models=[];seconds=[];columns=[]
        for h in range(18):
            s['current_hour']=h+6;checkpoint();m=build_rf(cfg);start=time.perf_counter();m.fit(xt,data['train']['y'][:,2*h:2*h+2]);seconds.append(time.perf_counter()-start)
            path=out/f'rf_hour{h+6:02}.joblib';joblib.dump(m,path);restored=joblib.load(path);v=restored.predict(x)
            np.testing.assert_allclose(v,m.predict(x),rtol=1e-12,atol=1e-12);columns.append(v);models.append(restored)
        hp=np.maximum(np.concatenate(columns,axis=1),0);assert hp.shape==y.shape
        preds['HourlyRF']=hp;statuses=note_statuses(a.kbo_source)
        for name,pred in preds.items():s['results'][name]=dict(metrics=grouped_metrics(y,pred,dates,statuses),diagnostics=diagnostics(y,pred))
        times=[]
        for i in range(35):
            start=time.perf_counter();v=np.concatenate([m.predict(x[i:i+1]) for m in models],axis=1);duration=(time.perf_counter()-start)*1000
            if i>=5:times.append(duration)
        s['hourly_rf_cost']=dict(fit_seconds=sum(seconds),per_hour_fit_seconds=seconds,model_bytes=sum(p.stat().st_size for p in out.glob('rf_hour*.joblib')),median_predict_ms=float(np.median(times)),p95_predict_ms=float(np.quantile(times,.95)),samples_ms=times)
        s['training_days']=352;s['validation_days']=181;s['input_features']=xt.shape[1];s['reload_verified']=True;s['total_trees']=18*cfg['rf']['n_estimators'];s['status']='completed';s.pop('current_hour',None)
        np.savez_compressed(out/'hourly_rf_validation.npz',dates=dates,truth=y,prediction=hp)
        save_json(ROOT/'reports/metrics/hourly_rf.json',s);checkpoint();print({n:z['metrics']['overall']['mae'] for n,z in s['results'].items()})
    except Exception as e:
        s.update(status='failed',error=repr(e));checkpoint();raise

if __name__=='__main__':main()

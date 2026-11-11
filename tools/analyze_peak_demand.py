"""Validation-only diagnostics; no fitting, ensemble, or held-out evaluation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def metrics(y, p):
    e = p-y
    return dict(n=int(y.size), mae=float(np.mean(abs(e))), bias=float(np.mean(e)),
                under_fraction=float(np.mean(e<0)), shortage=float(np.mean(np.maximum(-e,0))))

def main():
    source = ROOT/'outputs/team-handoff-2026-11-10/validation_predictions.json'
    records = read(source)['records']
    dates = sorted({r['date'] for r in records})
    assert len(dates)==181 and dates[0]=='2025-01-01' and dates[-1]=='2025-06-30'
    lookup={(r['date'],r['hour_start'],r['direction']):r for r in records}
    assert len(lookup)==len(records)==6516
    ordered=[lookup[d,h,k] for d in dates for h in range(6,24) for k in ['승차','하차']]
    y=np.array([r['actual'] for r in ordered]).reshape(181,18,2)
    models={k:np.array([r[f'{field}_prediction'] for r in ordered]).reshape(y.shape)
            for k,field in [('RF','rf'),('LSTM_b16','lstm')]}
    state=read(ROOT/'reports/metrics/learning_tuning.json')
    run=next(r for r in state['results'] if r['variant']=='batch8' and r['seed']==42)
    npz=ROOT/run['run']/'lstm_14d/validation_predictions.npz'
    with np.load(npz) as a:
        assert list(a['dates'].astype(str))==dates
        np.testing.assert_array_equal(a['truth'].reshape(y.shape),y)
        models['LSTM_b8']=a['prediction'].reshape(y.shape)
    for p in models.values():
        assert np.isfinite(p).all() and (p>=0).all()
    result=dict(seed=42,period=[dates[0],dates[-1]],days=181,observations=6516,
        final_test_evaluated=False,new_training=False,
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,npz]},
        rules=dict(high_demand='validation direction-specific linear quantile; >= threshold, include ties; descriptive only',
                   quantiles=[0.8,0.9,0.95],peak_tie='earliest hour',
                   near_peak='hours >=95% daily maximum; count contiguous peak regions',
                   weak_peak='daily maximum <=1.5*daily mean; heuristic sensitivity flag',
                   selection_metric='unchanged: overall validation MAE'),models={})
    daily=[]
    for name,p in models.items():
        out=dict(overall=metrics(y,p),monthly={},directions={})
        for month in range(1,7):
            mask=np.array([int(d[5:7])==month for d in dates])
            out['monthly'][str(month)]=metrics(y[mask],p[mask])
        for j,direction in enumerate(['board','alight']):
            a,b=y[:,:,j],p[:,:,j]
            actual_idx=a.argmax(axis=1); pred_idx=b.argmax(axis=1); ix=np.arange(181)
            shift=pred_idx-actual_idx; peak=b.max(axis=1)-a.max(axis=1)
            at_peak=b[ix,actual_idx]-a.max(axis=1)
            total=(b-a).sum(axis=1)
            near=a>=a.max(axis=1)[:,None]*.95
            regions=near[:,0].astype(int)+(near[:,1:] & ~near[:,:-1]).sum(axis=1)
            weak=a.max(axis=1)<=1.5*a.mean(axis=1)
            ties=(a==a.max(axis=1)[:,None]).sum(axis=1)
            z=dict(all=metrics(a,b),peak=dict(total_mae=float(abs(total).mean()),total_bias=float(total.mean()),
                size_mae=float(abs(peak).mean()),size_bias=float(peak.mean()),
                at_actual_peak_mae=float(abs(at_peak).mean()),at_actual_peak_bias=float(at_peak.mean()),
                timing_absolute_hours=float(abs(shift).mean()),within_one_hour=float((abs(shift)<=1).mean()),
                aligned_peak_days=int((abs(shift)<=1).sum()),
                aligned_at_peak_mae=float(abs(at_peak[abs(shift)<=1]).mean()),
                actual_tie_days=int((ties>1).sum()),predicted_tie_days=int(((b==b.max(axis=1)[:,None]).sum(axis=1)>1).sum()),
                multi_near_peak_days=int((regions>1).sum()),weak_peak_days=int(weak.sum())),high_demand={})
            for q in [.8,.9,.95]:
                threshold=float(np.quantile(a,q,method='linear')); high=a>=threshold; predicted=b>=threshold
                hi=metrics(a[high],b[high]); normal=metrics(a[~high],b[~high])
                hi.update(threshold=threshold,miss_rate=float((~predicted[high]).mean()),
                    false_positive_count=int((predicted & ~high).sum()),false_positive_rate=float(predicted[~high].mean()),
                    total_absolute_error_share=float(abs(b-a)[high].sum()/abs(b-a).sum()))
                assert hi['n']+normal['n']==a.size
                assert np.isclose((hi['mae']*hi['n']+normal['mae']*normal['n'])/a.size,z['all']['mae'])
                z['high_demand'][str(q)]=dict(high=hi,normal=normal)
            for i,date in enumerate(dates):
                daily.append(dict(model=name,direction=direction,date=date,total_error=float(total[i]),
                    actual_peak_hour=int(actual_idx[i]+6),predicted_peak_hour=int(pred_idx[i]+6),
                    peak_size_error=float(peak[i]),error_at_actual_peak=float(at_peak[i]),
                    timing_error_hours=int(shift[i]),actual_ties=int(ties[i]),near_peak_regions=int(regions[i]),weak_peak=bool(weak[i])))
            out['directions'][direction]=z
        result['models'][name]=out
    # Confirm historical score independently of the input file's metadata.
    assert np.isclose(result['models']['RF']['overall']['mae'],255.86966921276755)
    assert np.isclose(result['models']['LSTM_b8']['overall']['mae'],run['condition']['validation_mae'])
    folder=ROOT/'reports/metrics';folder.mkdir(exist_ok=True)
    (folder/'peak_demand_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (folder/'peak_demand_daily.json').write_text(json.dumps(daily,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    names=list(models);x=np.arange(3)
    for j,d in enumerate(['board','alight']):
        axes[0].bar(x+j*.35,[result['models'][n]['directions'][d]['high_demand']['0.9']['high']['mae'] for n in names],.35,label=d)
        axes[1].bar(x+j*.35,[result['models'][n]['directions'][d]['peak']['at_actual_peak_mae'] for n in names],.35,label=d)
        axes[2].bar(x+j*.35,[100*result['models'][n]['directions'][d]['peak']['within_one_hour'] for n in names],.35,label=d)
    for ax,title in zip(axes,['Top 10% demand MAE (people)','Error at actual daily peak (people)','Peak timing within 1 hour (%)']):
        ax.set_title(title);ax.set_xticks(x+.175,names);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('2025 H1 validation | fixed seed 42 | 181 days | demand thresholds shared across models')
    fig.tight_layout(); dest=ROOT/'reports/figures/peak_demand_analysis.png';dest.parent.mkdir(exist_ok=True);fig.savefig(dest,dpi=160);plt.close(fig)
    print('Saved diagnostics for 181 days, 6516 observations, three fixed-seed models.')

if __name__=='__main__':
    main()

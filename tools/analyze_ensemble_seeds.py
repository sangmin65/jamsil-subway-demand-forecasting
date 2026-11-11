"""Pair existing RF leaf3 and LSTM batch8 predictions within each approved seed."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_ensemble import complement, metrics, read, save, ROOT

def describe(y,p,dates):
    z=dict(overall=metrics(y,p),directions={},monthly={})
    for j,d in enumerate(['board','alight']):
        a,b=y[:,:,j],p[:,:,j];ix=np.arange(len(a));peak=a.argmax(axis=1)
        sub=dict(overall=metrics(a,b),peak_mae=float(abs(b[ix,peak]-a.max(axis=1)).mean()),
            peak_within_hour=float((abs(b.argmax(axis=1)-peak)<=1).mean()),high={})
        for q in [.8,.9,.95]:
            threshold=float(np.quantile(a,q));mask=a>=threshold
            sub['high'][str(q)]=dict(threshold=threshold,high=metrics(a[mask],b[mask]),normal=metrics(a[~mask],b[~mask]),
                miss_rate=float((b[mask]<threshold).mean()),false_positive_count=int((b[~mask]>=threshold).sum()))
            h=sub['high'][str(q)];assert np.isclose((h['high']['mae']*h['high']['n']+h['normal']['mae']*h['normal']['n'])/a.size,sub['overall']['mae'])
        z['directions'][d]=sub
    for month in range(1,7):
        mask=np.array([int(d[5:7])==month for d in dates]);z['monthly'][str(month)]=metrics(y[mask],p[mask])
    return z

def main():
    seeds=[42,907394,892611];rf_state=read(ROOT/'reports/metrics/seeds_rf_2026-11-07.json');lstm_state=read(ROOT/'reports/metrics/learning_tuning.json')
    assert rf_state['status']==lstm_state['status']=='completed'
    out=ROOT/'outputs/ensemble-seeds-2026-11-11';out.mkdir(exist_ok=True)
    state=dict(seeds=seeds,weights={'RF':.5,'LSTM_b8':.5},period=['2025-01-01','2025-06-30'],days=181,observations_per_seed=6516,
        new_training=False,final_test_evaluated=False,results=[],sources={},aggregation='Mean and sample SD of three paired-seed scores, not a six-model ensemble')
    reference=None
    for seed in seeds:
        rows=[next(r for r in rf_state['results'] if r['name']=='rf_leaf3' and r['seed']==seed),
              next(r for r in lstm_state['results'] if r['variant']=='batch8' and r['seed']==seed)]
        predictions={};runs={}
        for name,sub,row in zip(['RF','LSTM_b8'],['rf_3d','lstm_14d'],rows):
            assert row['reload_verified']
            run=ROOT/row['run'].replace('\\','/');cfg=read(run/'config.json');rep=read(run/'report.json')
            assert cfg['seed']==seed and not rep['final_test_evaluated']
            if name=='RF':assert cfg['rf']['min_samples_leaf']==3 and cfg['rf']['n_estimators']==200 and cfg['rf']['max_depth']==10
            else:assert cfg['training']['batch_size']==8 and cfg['lstm'].get('layers',1)==1 and cfg.get('loss','mean_absolute_error')=='mean_absolute_error'
            for key,value in lstm_state['sources'].items():assert rep['sources'][key]==value
            path=run/sub/'validation_predictions.npz';state['sources'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as a:
                dates=a['dates'].astype(str);y=a['truth'].reshape(181,18,2);pred=a['prediction'].reshape(y.shape).astype(float)
                assert dates[0]=='2025-01-01' and dates[-1]=='2025-06-30' and len(set(dates))==181
                if reference is None:reference=(dates.copy(),y.copy())
                np.testing.assert_array_equal(dates,reference[0]);np.testing.assert_array_equal(y,reference[1])
                assert np.isfinite(pred).all() and (pred>=0).all()
                np.testing.assert_allclose(metrics(y,pred)['mae'],row['condition']['validation_mae'],rtol=1e-10)
                predictions[name]=pred;runs[name]=str(run.relative_to(ROOT))
        predictions['Average50']=(predictions['RF']+predictions['LSTM_b8'])/2
        avg=predictions['Average50']
        assert ((avg>=np.minimum(predictions['RF'],predictions['LSTM_b8']))&(avg<=np.maximum(predictions['RF'],predictions['LSTM_b8']))).all()
        m={name:describe(y,p,dates) for name,p in predictions.items()}
        row=dict(seed=seed,runs=runs,models=m,complementarity=complement(y,predictions['RF'],predictions['LSTM_b8']),
            gains={n:m[n]['overall']['mae']-m['Average50']['overall']['mae'] for n in ['RF','LSTM_b8']})
        state['results'].append(row)
        np.savez_compressed(out/f'seed{seed}.npz',dates=dates,truth=y,prediction=avg)
    prior=read(ROOT/'reports/metrics/ensemble_analysis.json')
    for n in ['RF','LSTM_b8','Average50']:np.testing.assert_allclose(state['results'][0]['models'][n]['overall']['mae'],prior['models'][n]['overall']['mae'],atol=1e-8)
    state['summary']={n:dict(mean_mae=float(np.mean([r['models'][n]['overall']['mae'] for r in state['results']])),
        sample_sd=float(np.std([r['models'][n]['overall']['mae'] for r in state['results']],ddof=1))) for n in ['RF','LSTM_b8','Average50']}
    state['comparisons']={n:dict(mean_gain=float(np.mean([r['gains'][n] for r in state['results']])),
        improved_seeds=sum(r['gains'][n]>0 for r in state['results']),
        improvement_pct=100*(1-state['summary']['Average50']['mean_mae']/state['summary'][n]['mean_mae'])) for n in ['RF','LSTM_b8']}
    save(ROOT/'reports/metrics/ensemble_seeds.json',state)
    lines=['# ANALYSIS-003 RF·LSTM 50:50 동일3시드 비교','',
        '- 2026-11-11 사용자 승인: 50:50 유지, 시드42·907394·892611 비교·기존 결과 대조·기록·커밋.',
        '- RF3일·나무200·깊이10·잎3, LSTM14일·32유닛1층·배치8·MAE·Adam0.003. 저장된 여섯 모델 예측 재사용. 재학습·가중치 탐색·최종 평가 미실행.',
        '- 같은 시드 안에서 음수0 보정된 두 예측을 0.5씩 평균하고 점수를 계산한다. 시드별 점수의 평균±표본SD를 보고하며 여섯 모델을 한꺼번에 합친 앙상블 점수가 아니다.',
        '- 동일2025상반기181일×18시간×승하차2=6516개. 세 시드에서 같은 날짜를 사용하므로 독립 표본19548개라고 주장하지 않는다. 최종2025하반기는 미평가.',
        '', '| 시드 | RF MAE | LSTM MAE | 평균 MAE | RF 대비 감소 | LSTM 대비 감소 |','|---|---:|---:|---:|---:|---:|']
    for r in state['results']:
        lines.append(f"| {r['seed']} | {r['models']['RF']['overall']['mae']:.2f} | {r['models']['LSTM_b8']['overall']['mae']:.2f} | {r['models']['Average50']['overall']['mae']:.2f} | {r['gains']['RF']:+.2f} | {r['gains']['LSTM_b8']:+.2f} |")
    lines+=['','| 모델 | 동일3시드 평균 MAE | 표본 SD |','|---|---:|---:|']
    for n,z in state['summary'].items():lines.append(f"| {n} | {z['mean_mae']:.2f} | {z['sample_sd']:.2f} |")
    for n,z in state['comparisons'].items():lines.append(f"\n- {n} 대비 평균 {z['mean_gain']:.2f}명({z['improvement_pct']:.2f}%) 개선, 개선 시드 {z['improved_seeds']}/3.")
    lines+=['','## 방향·고수요·피크 (시드별 지표의 평균)','',
        '고수요: 실제값 상위10%, 승차1050명 이상/하차1426명 이상 각각326개. 사후 진단용. 상위5%·20% 민감도·누락률은 JSON 보존. 피크 동률이면 가장 이른 시각.','',
        '| 모델 | 방향 | 전체 MAE | 고수요 MAE | 일반 MAE | 고수요 부족 인원 | 실제 피크 MAE |','|---|---|---:|---:|---:|---:|---:|']
    for n in state['summary']:
        for d in ['board','alight']:
            z=[r['models'][n]['directions'][d] for r in state['results']]
            vals=[np.mean([v['overall']['mae'] for v in z]),np.mean([v['high']['0.9']['high']['mae'] for v in z]),np.mean([v['high']['0.9']['normal']['mae'] for v in z]),np.mean([v['high']['0.9']['high']['shortage'] for v in z]),np.mean([v['peak_mae'] for v in z])]
            lines.append(f'| {n} | {d} | '+' | '.join(f'{x:.2f}' for x in vals)+' |')
    lines+=['','## 월별 MAE (3시드 평균)','', '| 월 | RF | LSTM | 평균 |','|---|---:|---:|---:|']
    for month in range(1,7):lines.append('| '+str(month)+' | '+' | '.join(f"{np.mean([r['models'][n]['monthly'][str(month)]['mae'] for r in state['results']]):.2f}" for n in state['summary'])+' |')
    lines+=['','## 해석과 다음 단계','',
        '- 세 시드 모두 전체MAE가 두 단일 모델보다 낮아 시드42에서의 개선 방향이 유지됐다. 표본SD도 작았으나 세 번의 반복만으로 통계적 유의성을 주장하지 않는다.',
        '- RF 대비 승차 전체MAE는 개선하고 하차 전체MAE는 악화했다. 고수요 승차·하차 및 실제 피크 시점 MAE는 RF보다 악화한다. 3시드 평균에서도 월별1~3월 개선·4~6월 악화 패턴이 유지된다.',
        '- 50:50을 최종 평가 후보로 유지할 근거가 생겼다. 다음에는 RF 단독과 고정50:50 중 최종 비교 대상 및 단일시드/시드평균 예측 등 배포 방식, 최종평가 절차를 먼저 확정하는 것을 추천한다. 이번 점수250.60은 세 시드 점수 평균이며 단일 결합 모델의 점수가 아니다. 새 기간 평가나 가중치 탐색은 추가 실행하지 않았다.',
        '', '## 검증과 해석 범위','',
        '- 기존 시드42 ANALYSIS-002 결과와 세 모델 점수가 일치한다. 원본해시·정답·날짜 일치, 개별 기존MAE 재현, 평균 범위·구간 가중합을 검증했다. 여섯 저장 모델은 이전 실험에서 복원 검증을 완료한 예측을 재사용했다.',
        '- RF 과거5시드 평균과 빼서 개선율을 계산하지 않고 같은3시드 RF를 기준으로 계산했다. 표본SD는 학습 난수의 변동이며 신뢰구간이 아니다.',
        '- 새 기간 일반화나 통계적 유의성을 보장하지 않는다. 전체MAE 개선과 고수요·피크 및 월별 악화는 함께 판단한다. 추가 가중치 탐색은 실행하지 않았다.',
        '- 재현: python tools/analyze_ensemble_seeds.py. [전체 JSON](../metrics/ensemble_seeds.json), outputs/ensemble-seeds-2026-11-11/seed*.npz에 시드별 평균 예측 보존.','',
        '![시드별 비교](../figures/ensemble_seeds.png)','']
    (ROOT/'reports/experiments/ANALYSIS-003_ensemble-seeds.md').write_text('\n'.join(lines),encoding='utf8')
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3)
    for j,n in enumerate(state['summary']):ax.bar(x+(j-1)*.25,[r['models'][n]['overall']['mae'] for r in state['results']],.25,label=n)
    ax.set_xticks(x,[str(s) for s in seeds]);ax.set(xlabel='Seed (paired RF / LSTM)',ylabel='Validation MAE (people)',title='Fixed 50:50 prediction average | 2025 H1, 181 days');ax.legend();fig.tight_layout();fig.savefig(ROOT/'reports/figures/ensemble_seeds.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(summary=state['summary'],comparisons=state['comparisons'])))

if __name__=='__main__':main()

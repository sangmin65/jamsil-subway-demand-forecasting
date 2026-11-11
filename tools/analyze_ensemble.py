"""Fixed seed42 RF/LSTM prediction pairing and authorized 50:50 average."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_peak_demand import metrics

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def complement(y,a,b):
    x=(a-y).ravel();z=(b-y).ravel();ax=abs(x);az=abs(z)
    topx=ax>=np.quantile(ax,.9);topz=az>=np.quantile(az,.9)
    return dict(n=len(x),rf_better=float((ax<az).mean()),lstm_better=float((az<ax).mean()),tie=float((ax==az).mean()),
        signed_error_correlation=float(np.corrcoef(x,z)[0,1]),absolute_error_correlation=float(np.corrcoef(ax,az)[0,1]),
        both_under=float(((x<0)&(z<0)).mean()),both_over=float(((x>0)&(z>0)).mean()),opposite_sign=float((x*z<0).mean()),
        zero_error_present=float(((x==0)|(z==0)).mean()),
        top_error_rf_count=int(topx.sum()),top_error_lstm_count=int(topz.sum()),top_error_intersection=int((topx&topz).sum()),
        top_error_jaccard=float((topx&topz).sum()/(topx|topz).sum()),
        cancellation_gain=float(((ax+az)/2-abs((x+z)/2)).mean()))

def main():
    source=ROOT/'outputs/team-handoff-2026-11-10/validation_predictions.json'
    records=read(source)['records'];dates=sorted({r['date'] for r in records})
    lookup={(r['date'],r['hour_start'],r['direction']):r for r in records}
    assert len(lookup)==len(records)==6516 and len(dates)==181 and dates[0]=='2025-01-01' and dates[-1]=='2025-06-30'
    ordered=[lookup[d,h,k] for d in dates for h in range(6,24) for k in ['승차','하차']]
    y=np.array([r['actual'] for r in ordered]).reshape(181,18,2)
    rf=np.array([r['rf_prediction'] for r in ordered]).reshape(y.shape)
    loss_state=read(ROOT/'reports/metrics/loss_comparison.json')
    run=next(r for r in loss_state['results'] if r['variant']=='mae')
    file=ROOT/run['run']/'lstm_14d/validation_predictions.npz'
    with np.load(file) as a:
        np.testing.assert_array_equal(a['dates'].astype(str),dates);np.testing.assert_array_equal(a['truth'].reshape(y.shape),y)
        lstm=a['prediction'].reshape(y.shape).astype(float)
    assert np.isfinite(rf).all() and np.isfinite(lstm).all() and (rf>=0).all() and (lstm>=0).all()
    avg=(rf+lstm)/2
    assert ((avg>=np.minimum(rf,lstm))&(avg<=np.maximum(rf,lstm))).all()
    models={'RF':rf,'LSTM_b8':lstm,'Average50':avg}
    result=dict(seed=42,period=[dates[0],dates[-1]],days=181,observations=6516,weights={'RF':.5,'LSTM_b8':.5},
        prediction_rule='Average already clipped nonnegative predictions, no rounding; no weight search',
        final_test_evaluated=False,new_training=False,
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,file]},
        complementarity={'overall':complement(y,rf,lstm)},models={})
    for j,d in enumerate(['board','alight']):result['complementarity'][d]=complement(y[:,:,j],rf[:,:,j],lstm[:,:,j])
    for n,p in models.items():
        m=dict(overall=metrics(y,p),monthly={},hourly={},directions={})
        for month in range(1,7):
            mask=np.array([int(d[5:7])==month for d in dates]);m['monthly'][str(month)]=metrics(y[mask],p[mask])
        for h in range(18):m['hourly'][str(h+6)]=metrics(y[:,h,:],p[:,h,:])
        for j,d in enumerate(['board','alight']):
            a,b=y[:,:,j],p[:,:,j];ix=np.arange(181);peak=a.argmax(axis=1);err=b-a
            z=dict(overall=metrics(a,b),peak=dict(at_actual_peak_mae=float(abs(err[ix,peak]).mean()),at_actual_peak_bias=float(err[ix,peak].mean()),
                within_one_hour=float((abs(b.argmax(axis=1)-peak)<=1).mean()),total_mae=float(abs(err.sum(axis=1)).mean())),high={})
            for q in [.8,.9,.95]:
                threshold=float(np.quantile(a,q));high=a>=threshold
                z['high'][str(q)]=dict(threshold=threshold,high=metrics(a[high],b[high]),normal=metrics(a[~high],b[~high]),
                    miss_rate=float((b[high]<threshold).mean()),false_positive_count=int((b[~high]>=threshold).sum()),
                    false_positive_rate=float((b[~high]>=threshold).mean()))
                hi=z['high'][str(q)]['high'];lo=z['high'][str(q)]['normal']
                assert np.isclose((hi['mae']*hi['n']+lo['mae']*lo['n'])/a.size,z['overall']['mae'])
            m['directions'][d]=z
        result['models'][n]=m
    assert np.isclose(result['models']['RF']['overall']['mae'],255.86966921276755)
    assert np.isclose(result['models']['LSTM_b8']['overall']['mae'],run['condition']['validation_mae'])
    assert result['models']['Average50']['overall']['mae']<=sum(result['models'][n]['overall']['mae'] for n in ['RF','LSTM_b8'])/2+1e-8
    result['comparisons']={n:dict(mae_reduction=result['models'][n]['overall']['mae']-result['models']['Average50']['overall']['mae'],
        improvement_pct=100*(1-result['models']['Average50']['overall']['mae']/result['models'][n]['overall']['mae'])) for n in ['RF','LSTM_b8']}
    save(ROOT/'reports/metrics/ensemble_analysis.json',result)
    out=ROOT/'outputs/ensemble-analysis-2026-11-11';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'validation_predictions.npz',dates=dates,truth=y,rf=rf,lstm=lstm,prediction=avg)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for j,d in enumerate(['board','alight']):
        axes[0].scatter((rf-y)[:,:,j].ravel(),(lstm-y)[:,:,j].ravel(),s=3,alpha=.2,label=d)
    axes[0].axhline(0,color='gray',lw=.7);axes[0].axvline(0,color='gray',lw=.7);axes[0].set(xlabel='RF signed error',ylabel='LSTM signed error',title='Paired errors (people)');axes[0].legend()
    names=list(models);x=np.arange(3)
    axes[1].bar(names,[result['models'][n]['overall']['mae'] for n in names]);axes[1].set_title('Overall MAE (people)')
    for j,d in enumerate(['board','alight']):axes[2].bar(x+j*.35,[result['models'][n]['directions'][d]['high']['0.9']['high']['mae'] for n in names],.35,label=d)
    axes[2].set_xticks(x+.175,names);axes[2].set_title('Top 10% demand MAE (people)');axes[2].legend()
    fig.suptitle('2025 H1 validation | seed 42 | 181 days / 6516 values | fixed 50:50 weights')
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/ensemble_analysis.png',dpi=160);plt.close(fig)
    lines=['# ANALYSIS-002 RF·LSTM 보완성 및 50:50 예측 평균','',
        '- 사용자 승인(2026-11-11): 오차 보완성 분석과50:50 평균 비교·결과 기록·커밋. 새 학습 없음.',
        '- RF3일(나무200·깊이10·잎3) 및 LSTM14일(1층32유닛·배치8·MAE·Adam0.003), 시드42. LSTM은 EXP-013 MAE 재현 실행. 기존 EXP-011 예측과 복원 일치 확인된 모델이다.',
        '- 동일2025상반기181일×18시간×승하차2=6516개. 음수0 보정한 두 예측을 각각0.5배하여 합산. 반올림·가중치 탐색·실제값에 따른 모델 선택 없음. 하반기 최종 평가 미사용.',
        '- 고수요는 승차·하차 각각 검증 실제값 상위10%(326개), 민감도5/20%, 경계동률 포함. 사후 진단이며 운영 혼잡 기준이 아니다. 피크 동률은 가장 이른 시간.','',
        '| 모델 | 전체 MAE | 승차 MAE | 하차 MAE | 평균 부호오차 |','|---|---:|---:|---:|---:|']
    for n,m in result['models'].items():lines.append(f"| {n} | {m['overall']['mae']:.2f} | {m['directions']['board']['overall']['mae']:.2f} | {m['directions']['alight']['overall']['mae']:.2f} | {m['overall']['bias']:.2f} |")
    lines+=['','## 고수요·일반·피크','', '| 모델 | 구분 | 고수요 MAE | 일반 MAE | 고수요 부족 인원 | 고수요 누락률 | 실제 피크 MAE |','|---|---|---:|---:|---:|---:|---:|']
    for n,m in result['models'].items():
        for d,z in m['directions'].items():
            h=z['high']['0.9'];lines.append(f"| {n} | {d} | {h['high']['mae']:.2f} | {h['normal']['mae']:.2f} | {h['high']['shortage']:.2f} | {h['miss_rate']:.1%} | {z['peak']['at_actual_peak_mae']:.2f} |")
    lines+=['','## 월별 MAE','', '| 월 | RF | LSTM | 평균 |','|---|---:|---:|---:|']
    for month in range(1,7):lines.append('| '+str(month)+' | '+' | '.join(f"{result['models'][n]['monthly'][str(month)]['mae']:.2f}" for n in names)+' |')
    c=result['complementarity']['overall']
    lines+=['','## 보완성','',f"- 관측값별 RF 우세 {c['rf_better']:.1%}, LSTM 우세 {c['lstm_better']:.1%}, 동률 {c['tie']:.1%}. 우세 횟수는 오차 크기와 달라 평균 성능의 순위를 대신하지 않는다.",
        f"- 부호오차 상관 {c['signed_error_correlation']:.3f}, 절대오차 상관 {c['absolute_error_correlation']:.3f}. 함께 과소 {c['both_under']:.1%}, 함께 과대 {c['both_over']:.1%}, 반대 방향 {c['opposite_sign']:.1%}, 적어도 한쪽 정확히0 {c['zero_error_present']:.1%}.",
        f"- 각 모델 절대오차 상위10% 집단: RF {c['top_error_rf_count']}개, LSTM {c['top_error_lstm_count']}개, 교집합 {c['top_error_intersection']}개, Jaccard {c['top_error_jaccard']:.3f}. 이는 실제 수요 상위10%와 다른 집단이다.",
        f"- 두 단일 MAE의 평균 대비 오차 상쇄 이득 {c['cancellation_gain']:.2f}명. 평균 예측 MAE는 각 MAE의 산술평균과 다르며, 더 좋은 단일 모델보다 낮다고 보장되지 않는다.",'']
    for n,c in result['comparisons'].items():lines.append(f"- {n} 대비 평균 예측 MAE 감소 {c['mae_reduction']:+.2f}명 ({c['improvement_pct']:+.2f}%). 양수 개선·음수 악화.")
    lines+=['','## 해석과 다음 단계','',
        '- 전체 MAE는 두 단일 모델보다 낮지만, RF 대비 하차 전체·고수요 승하차·실제 피크 시점 오차는 악화했다. 일반 하차 구간은 두 모델보다 개선했다. 평균 성능 개선과 피크 대응 개선을 구분한다.',
        '- 평균의 월별 MAE는 RF 대비1~3월 개선,4~6월 악화. LSTM 대비1~2월 악화,3~6월 개선이다. 모든 월에서 더 좋은 모델이 된 것은 아니다.',
        '- 약25%의 반대 방향 오차가 일부 상쇄되지만 오차 상관은 높고, 큰 오차652개 중487개가 겹친다. 두 모델 모두 놓치는 급증을 평균만으로 해결했다고 해석하지 않는다.',
        '- 다음 추천: 50:50을 그대로 고정하고 동일3시드42·907394·892611에서 각 RF/LSTM 예측쌍을 비교하여 개선 방향과 평균·표준편차를 확인한다. 각 시드별 두 모델 예측을 먼저 평균한 뒤 점수를 산출한다. 시드 점수 평균을 예측 평균으로 대신하지 않는다. 가중치 탐색·새 학습·3시드 확장은 이번에 실행하지 않았다.',
        '', '## 검증·한계','',
        '- 날짜·키 유일성, 정답 배열 완전 일치, 원본 SHA256, 기존 두 모델MAE 재현, 평균이 두 예측 사이에 위치함, 구간 가중 평균MAE 재현, 절대오차 삼각부등식 확인.',
        '- 단일 시드의 반복 사용된 검증기간 분석. 3/5시드 평균이나 독립 최종 평가가 아니다. 조건별 개선은 다음 시드 검증의 후보일 뿐이다. 새 학습 및 결합 비율 튜닝 미실행.',
        '- 재현: python tools/analyze_ensemble.py (NumPy·Matplotlib). 상세 시간대·월별·승하차·고수요5/10/20%·피크 결과와 입력 SHA256은 [JSON](../metrics/ensemble_analysis.json). 개별 평균 예측은 outputs/ensemble-analysis-2026-11-11/validation_predictions.npz.',
        '', '![비교 그림](../figures/ensemble_analysis.png)','']
    (ROOT/'reports/experiments/ANALYSIS-002_ensemble.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(dict(comparisons=result['comparisons'],complementarity=result['complementarity']),ensure_ascii=False))

if __name__=='__main__':main()

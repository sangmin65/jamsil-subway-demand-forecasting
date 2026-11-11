"""Average saved predictions across seeds, keeping model-family weights fixed."""
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from analyze_ensemble_seeds import describe, read, save, ROOT

def main():
    prior=read(ROOT/'reports/metrics/ensemble_seeds.json')
    assert prior['seeds']==[42,907394,892611]
    stacks={'RF':[],'LSTM':[]};hashes={};reference=None
    for row in prior['results']:
        for label,key,sub in [('RF','RF','rf_3d'),('LSTM','LSTM_b8','lstm_14d')]:
            path=ROOT/row['runs'][key]/sub/'validation_predictions.npz'
            relative=str(path.relative_to(ROOT));digest=hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest==prior['sources'][relative]
            hashes[relative]=digest
            with np.load(path) as z:
                y=z['truth'].reshape(181,18,2);dates=z['dates'].astype(str);p=z['prediction'].reshape(y.shape).astype(float)
                if reference is None:reference=(y.copy(),dates.copy())
                np.testing.assert_array_equal(y,reference[0]);np.testing.assert_array_equal(dates,reference[1])
                assert np.isfinite(p).all() and (p>=0).all()
                np.testing.assert_allclose(abs(p-y).mean(),row['models'][key]['overall']['mae'])
                stacks[label].append(p)
    y,dates=reference
    assert len(set(dates))==181 and dates[0]=='2025-01-01' and dates[-1]=='2025-06-30'
    rf=np.mean(stacks['RF'],axis=0);lstm=np.mean(stacks['LSTM'],axis=0);combined=(rf+lstm)/2
    np.testing.assert_allclose(combined,np.mean([(a+b)/2 for a,b in zip(stacks['RF'],stacks['LSTM'])],axis=0),rtol=1e-12,atol=1e-12)
    predictions={'RF_seed42':stacks['RF'][0],'LSTM_seed42':stacks['LSTM'][0],
        'Combined_seed42':(stacks['RF'][0]+stacks['LSTM'][0])/2,'RF_seedmean':rf,'LSTM_seedmean':lstm,'Combined_seedmean':combined}
    result=dict(seeds=prior['seeds'],period=[dates[0],dates[-1]],observations=6516,new_training=False,final_test_evaluated=False,
        rule='Clip each constituent prediction at zero, equal average within three seeds, then RF/LSTM family weights 0.5/0.5. No rounding or weight search.',sources=hashes,
        models={n:describe(y,p,dates) for n,p in predictions.items()},prior_mean_of_scores=prior['summary'])
    pairs=[('RF_seedmean','RF_seed42'),('LSTM_seedmean','LSTM_seed42'),('Combined_seedmean','Combined_seed42'),('Combined_seedmean','RF_seedmean')]
    result['comparisons']={f'{a}_vs_{b}':dict(mae_reduction=result['models'][b]['overall']['mae']-result['models'][a]['overall']['mae'],
        improvement_pct=100*(1-result['models'][a]['overall']['mae']/result['models'][b]['overall']['mae'])) for a,b in pairs}
    for new,old in [('RF_seedmean','RF'),('LSTM_seedmean','LSTM_b8'),('Combined_seedmean','Average50')]:
        assert result['models'][new]['overall']['mae']<=prior['summary'][old]['mean_mae']+1e-8
    save(ROOT/'reports/metrics/seed_prediction_average.json',result)
    out=ROOT/'outputs/seed-prediction-average-2026-11-11';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'validation_predictions.npz',dates=dates,truth=y,rf=rf,lstm=lstm,combined=combined)
    lines=['# ANALYSIS-005 세 시드 예측 평균 및 RF·LSTM 결합','',
        '- 2026-11-11 사용자 승인: RF3시드 예측 평균 및 RF·LSTM50:50 평균을 기존 결과와 비교·기록·커밋. 최종 평가는 결과 보고 결정. 새 학습·최종평가 미실행.',
        '- 시드42·907394·892611. RF3일·나무200·깊이10·잎3, LSTM14일·1층32유닛·배치8·MAE·Adam0.003·Dropout0. 기존 여섯 모델 예측 재사용.',
        '- 각 모델의 음수0 보정 예측을 동일 날짜·시간·승하차에서 평균. RF평균=(RF42+RF907394+RF892611)/3, LSTM평균도 동일. 결합=0.5×RF평균+0.5×LSTM평균으로 각 모델 가중치1/6. 시드별50:50 예측을 평균한 값과 수치적으로 같음. 가중치 탐색·반올림 없음.',
        '- 동일2025상반기181일·6516개 값. 현재 계산은 예측을 합친 하나의 구성에 대한 점수이며 과거 세 시드 점수의 평균±SD와 구분. 이번 구성의 반복SD를 산출하지 않음.','',
        '| 구성 | 전체 MAE | 승차 MAE | 하차 MAE | 모델 수 |','|---|---:|---:|---:|---:|']
    for n,z in result['models'].items():
        count=6 if n=='Combined_seedmean' else (3 if n.endswith('seedmean') else (2 if n.startswith('Combined') else 1))
        lines.append(f"| {n} | {z['overall']['mae']:.2f} | {z['directions']['board']['overall']['mae']:.2f} | {z['directions']['alight']['overall']['mae']:.2f} | {count} |")
    lines+=['','## 과거 점수 평균과의 구분','', '| 모델 | 기존3시드 점수 평균 | 이번3시드 예측 평균의 점수 |','|---|---:|---:|']
    for new,old in [('RF_seedmean','RF'),('LSTM_seedmean','LSTM_b8'),('Combined_seedmean','Average50')]:lines.append(f"| {new} | {prior['summary'][old]['mean_mae']:.2f} | {result['models'][new]['overall']['mae']:.2f} |")
    lines+=['','## 고수요·피크','', '실제 인원 상위10%, 승차/하차 각각326개, 사후 진단. 상위5/20% 민감도는 JSON. 피크 동률은 가장 이른 시간.','',
        '| 구성 | 방향 | 고수요 MAE | 일반 MAE | 고수요 부족인원 | 피크시점 MAE |','|---|---|---:|---:|---:|---:|']
    for n,z in result['models'].items():
        for d,v in z['directions'].items():
            h=v['high']['0.9'];lines.append(f"| {n} | {d} | {h['high']['mae']:.2f} | {h['normal']['mae']:.2f} | {h['high']['shortage']:.2f} | {v['peak_mae']:.2f} |")
    lines+=['','## 월별 MAE','', '| 월 | RF42 | 기존50:50(42) | RF3평균 | 6모델결합 |','|---|---:|---:|---:|---:|']
    for month in range(1,7):lines.append('| '+str(month)+' | '+' | '.join(f"{result['models'][n]['monthly'][str(month)]['mae']:.2f}" for n in ['RF_seed42','Combined_seed42','RF_seedmean','Combined_seedmean'])+' |')
    for k,c in result['comparisons'].items():lines.append(f"\n- {k}: MAE 감소 {c['mae_reduction']:+.2f}명({c['improvement_pct']:+.2f}%). 양수 개선.")
    lines+=['','## 해석 및 최종 평가 전 판단','',
        '- 이번 구성 중 전체MAE 최저는6모델결합250.16명. 기존 시드42 한 쌍250.55 대비0.39명(0.16%) 개선에 모델 수2→6개가 필요하다. 실행 비용을 직접 벤치마크하지 않았으므로 시간3배라고 단정하지 않는다.',
        '- RF3시드 평균은 RF42 대비0.40명 개선. 결합은 RF3시드 평균 대비5.32명(2.08%) 개선하므로 시드 평균 효과와 모델 종류 결합 효과를 구분할 수 있다.',
        '- 고수요·피크는 RF3평균이6모델결합보다 낮은 오차를 유지한다.6모델결합은 기존 시드42 결합보다 고수요 승차/하차와 피크 MAE가 소폭 악화했다. 전체MAE 개선이 피크 대응 개선을 의미하지 않는다.',
        '-6모델결합은 RF3평균 대비1~3월 개선,4~6월 악화. 기존 시드42 결합 대비1·2·4월 개선,3·5·6월 악화. 평가기간 구성에 따른 순위 변화 가능성을 남긴다.',
        '- 권고: 검증 전체MAE 최저 후보는6모델결합, 단순성 후보는기존 시드42 한 쌍, 고수요 비교 기준은RF3평균으로 정리한다.0.39명의 추가 이득만으로6모델을 필수로 권하지 않는다. 최종 구성을 사용자와 확정한 뒤 독립 평가로 넘어가며 이번에는 최종평가를 실행하지 않았다.',
        '', '## 검증·한계','',
        '- 앞선 분석에 기록된6개 예측 파일 SHA256 동일, 날짜·정답 완전 일치, 기존 각 시드MAE 재현 확인. 평균 순서 등가성과 MAE 삼각부등식, 구간 가중합 확인. 저장 모델의 기존 복원검증 결과를 재사용했다.',
        '- RF 과거5시드 평균과 직접 차이를 계산하지 않음. 반복 활용한 검증기간에서의 탐색이며 독립 최종 성능·통계적 유의성을 주장하지 않는다. 전체MAE와 고수요·피크 성능의 장단점을 분리한다.',
        '- 재현: python tools/analyze_seed_average.py. [상세JSON](../metrics/seed_prediction_average.json). 개별 예측은 outputs/seed-prediction-average-2026-11-11/validation_predictions.npz.','',
        '![구성 비교](../figures/seed_prediction_average.png)','']
    (ROOT/'reports/experiments/ANALYSIS-005_seed-prediction-average.md').write_text('\n'.join(lines),encoding='utf8')
    names=list(predictions);fig,ax=plt.subplots(figsize=(10,4));ax.bar(names,[result['models'][n]['overall']['mae'] for n in names]);ax.set(ylabel='MAE (people)',title='Prediction averaging | 2025 H1 validation, 181 days');ax.tick_params(axis='x',labelrotation=20);fig.tight_layout();fig.savefig(ROOT/'reports/figures/seed_prediction_average.png',dpi=160);plt.close(fig)
    print({n:z['overall']['mae'] for n,z in result['models'].items()});print(result['comparisons'])

if __name__=='__main__':main()

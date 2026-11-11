"""Publish validation-only diagnostics and hourly RF results; no model fitting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))

def main():
    s=read(ROOT/'reports/metrics/hourly_rf.json');audit=read(ROOT/'reports/metrics/error_case_audit.json');cost=read(ROOT/'reports/metrics/prediction_cost.json')['costs']
    assert s['status']=='completed' and not s['final_test_evaluated'] and s['reload_verified']
    out=ROOT/'outputs/hourly-rf-2026-11-11'
    with np.load(out/'baseline_predictions.npz') as z:
        dates=z['dates'].astype(str);y=z['truth'].reshape(181,18,2);preds={n:z[k].reshape(y.shape) for n,k in [('RF','rf'),('LSTM','lstm'),('Average50','combined')]}
    with np.load(out/'hourly_rf_validation.npz') as z:
        np.testing.assert_array_equal(z['truth'].reshape(y.shape),y);np.testing.assert_array_equal(z['dates'],dates);preds['HourlyRF']=z['prediction'].reshape(y.shape)
    for name,p in preds.items():np.testing.assert_allclose(abs(p-y).mean(),s['results'][name]['metrics']['overall']['mae'])
    chosen=[audit['best_cases'][0]['date'],audit['worst_cases'][0]['date']]
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    case_details=[]
    for row,date in enumerate(chosen):
        i=int(np.where(dates==date)[0][0]);gain=abs(preds['RF'][i]-y[i])-abs(preds['Average50'][i]-y[i])
        h,j=np.unravel_index(np.argmax(gain) if row==0 else np.argmin(gain),gain.shape)
        case_details.append(dict(date=date,hour=int(h+6),direction=['board','alight'][j],actual=float(y[i,h,j]),rf=float(preds['RF'][i,h,j]),lstm=float(preds['LSTM'][i,h,j]),average=float(preds['Average50'][i,h,j]),error_reduction=float(gain[h,j])))
        for j,d in enumerate(['board','alight']):
            ax=axes[row,j];ax.plot(range(6,24),y[i,:,j],color='black',label='Actual')
            for name in ['RF','LSTM','Average50']:ax.plot(range(6,24),preds[name][i,:,j],label=name)
            ax.set(title=f'{date} / {d}',xlabel='Hour start',ylabel='People');ax.set_ylim(bottom=0);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Validation only | largest daily improvement / deterioration vs RF | seed42')
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/combination_cases.png',dpi=160);plt.close(fig)
    lines=['# ANALYSIS-006 상반기 오차유형·결합사례·예측비용','',
      '- 2026-11-11 사용자 요청. 시드42 RF3일·기존 LSTM14일배치8 및 고정50:50. 2025상반기181일만 사용. 하반기 예측/점수/최종평가 파일은 읽거나 갱신하지 않았다.',
      '- 날짜별 총량오차, 시간별 절대오차 합, 피크 시각·최대크기·실제피크시점 오차를 계산. 합계절대오차≤시간별절대오차합을 검증. 그 차이는 하루 안에서 부호오차가 상쇄된 양이며 원인을 뜻하지 않는다.',
      '- 총량·시각·규모 실패는 겹칠 수 있으므로 임의로 하나의 원인 분류에 배정하지 않는다. 하루 피크는 승하차별최대, 동률이면 가장 이른 시간.','',
      '| 모델 | 방향 | 피크시각 ±1시간 초과 일수 | 실제피크 과소예측 일수 | 일총량절대오차 합/시간절대오차 합 |','|---|---|---:|---:|---:|']
    for n,dirs in audit['summary'].items():
        for d,v in dirs.items():lines.append(f"| {n} | {d} | {v['timing_over_one_hour']}/181 | {v['actual_peak_under']}/181 | {v['total_absolute_error']/v['hourly_absolute_error']:.1%} |")
    lines+=['','| 모델 | 방향 | 일총량 MAE | 피크크기 MAE | 피크시각 평균절대차(시간) |','|---|---|---:|---:|---:|']
    for n in audit['summary']:
        for d in ['board','alight']:
            rows=[r for r in audit['daily'] if r['model']==n and r['direction']==d]
            lines.append(f"| {n} | {d} | {np.mean([r['total_absolute_error'] for r in rows]):.2f} | {np.mean([abs(r['peak_size_error']) for r in rows]):.2f} | {np.mean([abs(r['peak_time_error']) for r in rows]):.2f} |")
    lines+=['','- 총량만으로 평가하면 하루 안에서 상쇄된 시간대 오차를 놓친다. 특히 RF승차의 총량절대오차합은 시간별절대오차합의39.5%에 불과하다. 그렇다고 나머지60.5%를 모두 피크 시각 문제로 단정하지는 않는다.',
      '- 결합은 하차피크 시각 ±1시간초과 일수를39→34일로 줄였지만 실제피크 과소예측 일수는122→131일로 늘렸다. 시각과 규모를 함께 봐야 하는 사례다.',
      '',f"- 일별MAE 개선 {audit['improved_days']}일, 악화 {audit['worsened_days']}일, 동률 {audit['tied_days']}일. 횟수는 개선 크기를 대신하지 않는다.",'','## 성공·실패 사례','',
      '일별36개 평균MAE의 RF대비 감소량 상위/하위5일. 극단 사례이며 무작위 대표 표본이 아니다.','', '| 구분 | 날짜 | RF MAE | 결합 MAE | 감소량 |','|---|---|---:|---:|---:|']
    for label,key in [('개선','best_cases'),('악화','worst_cases')]:
        for r in audit[key]:lines.append(f"| {label} | {r['date']} | {r['rf_mae']:.2f} | {r['average_mae']:.2f} | {r['improvement']:+.2f} |")
    lines+=['','위 각1위 날짜에서 개선/악화에 가장 크게 기여한 단일 관측값:','']
    for r in case_details:lines.append(f"- {r['date']} {r['hour']}시 {r['direction']}: 실제{r['actual']:.0f}, RF{r['rf']:.2f}, LSTM{r['lstm']:.2f}, 결합{r['average']:.2f}; 절대오차 감소{r['error_reduction']:+.2f}명.")
    lines+=['','- 개선사례6/27에서는 RF가 크게 놓친21시승차피크를 LSTM이 더 가깝게 예측해 평균이 RF를 보완했다. 악화사례4/19에서는 RF가 상대적으로 잘 맞힌17시승차피크를 LSTM이 낮게 예측해 평균이 RF의 예측을 끌어내렸다. 구체 행사 원인을 확인한 분석은 아니다.',
      '', '![사례 곡선](../figures/combination_cases.png)','','## 예측 비용','',
      '- 같은 WSL세션, RF CPU(n_jobs1)·LSTM GPU. 입력은 메모리에 준비한 한 날짜, 출력36개. 모델마다5회워밍업 후30회 측정, 순서 회전. GPU는 .numpy()로 완료 동기화. 입력준비/디스크로딩/초기 GPU기동은 제외하므로 서비스 전체지연이 아니다. 다른프로세스 부하를 통제한 벤치마크도 아니다.',
      '', '| 구성 | 중앙값 ms | p95 ms | 저장파일 MiB |','|---|---:|---:|---:|']
    for n in ['RF','LSTM','Average50']:
        size=sum(cost['artifact_bytes'].values()) if n=='Average50' else cost['artifact_bytes'][n]
        lines.append(f"| {n} | {cost[n]['median_ms']:.2f} | {cost[n]['p95_ms']:.2f} | {size/1024**2:.3f} |")
    lines+=['','- 파일 크기는 저장형식이 달라 직접 파라미터 효율로 해석하지 않는다. 신경망파일의 옵티마이저 상태 등도 포함될 수 있으며 RAM/VRAM실사용량과 다르다.',
      '- [오차상세 JSON](../metrics/error_case_audit.json), [비용/환경 JSON](../metrics/prediction_cost.json). 날짜별 진단은 오차상세daily에 보존. 기존 학습시간은 각 실험 기록을 참조하며 이번 비용은 추론비용이다.','']
    (ROOT/'reports/experiments/ANALYSIS-006_cases-cost.md').write_text('\n'.join(lines),encoding='utf8')
    lines=['# EXP-015 시간대별 RF 시드42 후속 탐색','',
      '- 2026-11-11 사용자 승인, DEC-038. 하반기 최종평가 이후 시행한 후속 탐색이며 본 실험은 상반기 검증만 사용. 기존최종모델·결과를 변경하지 않는다.',
      '- 기존 36출력 RF1개를 시간대별2출력(승차·하차) RF18개로 변경. 각모델 나무200·깊이10·잎3·seed42·n_jobs1 및 나머지RF파라미터 동일. 입력은 모든 모델에서 같은 과거3일 전체36값/일 및 예측일 달력·경기특성. 입력특성/목표정규화 추가 없음.',
      '- 각시간대별 모델이므로 총나무 수200→3600. 출력 분리와 전체 모델용량 증가가 동반되므로 순수한 출력구조 효과만 분리한 동일계산예산 실험은 아니다.',
      '- 저장 전처리 재사용,2024학습352일/2025상반기181일·6516개. 데이터는2025-06-30이하만 배열로 구성하며 test split을 만들지 않았다. 과거RF/LSTM검증정답·예측 재현 확인.18개모델 각각저장·복원예측 일치.','',
      '| 구성 | 전체 MAE | 승차 MAE | 하차 MAE | 승차 RMSE | 하차 RMSE |','|---|---:|---:|---:|---:|---:|']
    for n,r in s['results'].items():
        m=r['metrics']['overall'];lines.append(f"| {n} | {m['mae']:.2f} | {m['board_mae']:.2f} | {m['alight_mae']:.2f} | {m['board_rmse']:.2f} | {m['alight_rmse']:.2f} |")
    base=s['results']['RF']['metrics']['overall']['mae'];new=s['results']['HourlyRF']['metrics']['overall']['mae']
    lines+=['',f'- 기존RF 대비MAE 감소 {base-new:+.2f}명({100*(1-new/base):+.2f}%). 양수개선·음수악화.','',
      '| 구성 | 방향 | 고수요 MAE | 일반 MAE | 부족 인원 | 실제피크 MAE |','|---|---|---:|---:|---:|---:|']
    for n in ['RF','Average50','HourlyRF']:
        for d,v in s['results'][n]['diagnostics'].items():
            h=v['high']['0.9'];lines.append(f"| {n} | {d} | {h['mae']:.2f} | {h['normal_mae']:.2f} | {h['shortage']:.2f} | {v['peak_mae']:.2f} |")
    c=s['hourly_rf_cost'];lines+=['',f"- RF18개 총학습 {c['fit_seconds']:.2f}초, 파일합계 {c['model_bytes']/1024**2:.2f}MiB. 단일날짜순차예측 중앙값 {c['median_predict_ms']:.2f}ms, p95 {c['p95_predict_ms']:.2f}ms(5워밍업/30측정). 비용은 앞선 기준모델 측정 후 별도 측정이며 완벽한 동시부하 통제가 아니다.",
      '- 고수요는 상반기 실제값 상위10% 방향별326개,5/20%민감도는JSON. 임계값은 모든모델 동일. 단일시드 및 반복검증기간의 탐색으로 일반화/통계적유의성 주장 없음.',
      '- 전체MAE304.78로 기존RF255.87보다48.91명(19.11%) 악화. 승하차MAE·RMSE·고수요·일반·피크오차 모두 기존RF보다 높았다. 이번설정은 채택하거나3시드로 확대할 근거가 약해 보류를 권한다. 시간대분리가 항상 나쁘다는 일반화는 하지 않는다. 공동출력의 규제/정보공유가 도움이 됐을 가능성은 가설이며 이번 비교만으로 원인을 확정하지 않는다.',
      '- 같은 환경에서 기존RF 추론중앙값6.59ms 대비 시간대별RF112.16ms로 계산부담도 늘었다. 과거 최초RF학습시간5.12초는 환경·시점이 달라 이번57.52초와 통제된 속도비교로 사용하지 않는다.',
      '- 이번 새RF와 LSTM의 추가결합이나 다른시드는 실행하지 않았다. 결과 검토 후 결정. [전체결과/월별/note별 JSON](../metrics/hourly_rf.json).',
      '- 재현: WSL tools/run_hourly_rf.py --source <기존CSV> --kbo-source <기존KBO>, Windows tools/report_hourly_rf.py. outputs/hourly-rf-2026-11-11에 모델18개·상반기예측 보존. 기존출력 덮어쓰기 금지.','']
    (ROOT/'reports/experiments/EXP-015_hourly-rf.md').write_text('\n'.join(lines),encoding='utf8')
    fig,ax=plt.subplots(figsize=(8,4));names=list(s['results']);ax.bar(names,[s['results'][n]['metrics']['overall']['mae'] for n in names]);ax.set(title='Validation only | seed42 | 181 days | follow-up exploration',ylabel='MAE (people)');fig.tight_layout();fig.savefig(ROOT/'reports/figures/hourly_rf.png',dpi=160);plt.close(fig)
    print(case_details)

if __name__=='__main__':main()

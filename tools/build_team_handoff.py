"""Freeze a team handoff snapshot and validation-only review inputs; no training."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from statistics import mean
import json
import zipfile
from xml.sax.saxutils import escape

import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf';BUNDLE=ROOT/'outputs/team-handoff-2026-11-10'
OUT.mkdir(parents=True,exist_ok=True);BUNDLE.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

now=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
state=read(ROOT/'outputs/learning-tuning-2026-11-10-retry1/status.json')
snap=dict(captured_at=now,status=state['status'],current=state.get('current'),completed=len(state['results']),
    total_unique=21,new_total=18,stages=state['stages'],completed_conditions={},final_test_evaluated=False)
for variant in dict.fromkeys(r['variant'] for r in state['results']):
    rows=[r for r in state['results'] if r['variant']==variant]
    snap['completed_conditions'][variant]=dict(completed_seeds=len(rows),mean_mae=mean(r['condition']['validation_mae'] for r in rows),
        seeds=[r['seed'] for r in rows],complete=len(rows)==3)
save(ROOT/'reports/metrics/team_handoff_snapshot_2026-11-10.json',snap)

# Fixed historical representatives, never a rolling best seed or the ongoing trial.
rf_run=ROOT/'outputs/first-run-2026-11-03'
lstm_run=ROOT/'outputs/lstm-gpu-epochs-2026-11-04/lstm_14d_epochs2000'
with np.load(rf_run/'rf_3d/validation_predictions.npz') as f:
    dates=f['dates'].copy();truth=f['truth'].copy();rf=f['prediction'].copy()
with np.load(lstm_run/'lstm_14d/validation_predictions.npz') as f:
    np.testing.assert_array_equal(f['dates'],dates);np.testing.assert_array_equal(f['truth'],truth)
    lstm=f['prediction'].copy()
assert len(dates)==181 and str(dates[0])=='2025-01-01' and str(dates[-1])=='2025-06-30'
assert truth.shape==rf.shape==lstm.shape==(181,36)
daily=[];records=[]
for i,date in enumerate(dates):
    day=str(date);r=float(np.abs(rf[i]-truth[i]).mean());l=float(np.abs(lstm[i]-truth[i]).mean())
    daily.append(dict(date=day,rf_mae=r,lstm_mae=l,review_priority_score=(r+l)/2))
    for hour in range(6,24):
        for channel,direction in enumerate(['승차','하차']):
            j=(hour-6)*2+channel
            records.append(dict(date=day,hour_start=hour,direction=direction,actual=float(truth[i,j]),
                rf_prediction=float(rf[i,j]),lstm_prediction=float(lstm[i,j])))
save(BUNDLE/'validation_predictions.json',dict(period='2025-01-01/2025-06-30',seed=42,
    models={'rf':'3일, 나무200, 깊이10, 최소잎3','lstm':'14일, 32유닛1층, Adam0.003, batch16, 최대2000'},
    explanation='시드42의 고정된 대표 모델로 사례 분석. 5시드 평균이나 EXP-011 최선 모델의 예측이 아님. 예측은 음수0 보정, 반올림 안 함.',
    sources=[str(rf_run.relative_to(ROOT)),str(lstm_run.relative_to(ROOT))],records=records))
priority=sorted(daily,key=lambda r:r['review_priority_score'],reverse=True)[:10]
save(BUNDLE/'priority_dates.json',dict(rule='RF/LSTM 각 일별36출력 MAE의 평균 상위10일. 공동 검토 순서이며 모델 선택 지표가 아님.',dates=priority))
events=read(ROOT/'data/external/supplement_2026-11-09/seoul_events.json')
candidates=[r for r in events if r['venue']!='잠실야구장' and r.get('start_date') and r.get('end_date') and r['start_date']<='2025-06-30' and r['end_date']>='2025-01-01']
priority_days=[r['date'] for r in priority]
for r in candidates:
    r['overlapping_priority_dates']=[d for d in priority_days if r['start_date']<=d<=r['end_date']]
candidates.sort(key=lambda r:(not bool(r['overlapping_priority_dates']),r['start_date'],r['id']))
save(BUNDLE/'event_candidates.json',dict(note='기간 겹침은 후보 추출일 뿐 실제 공연일·개최 여부 확인이 아님. 게시시각과 실제 회차 확인 필요.',records=candidates))
review=read(ROOT/'reports/metrics/supplement_review_2026-11-09.json')
save(BUNDLE/'baseball_discrepancies.json',{k:review[k] for k in ['additions','original_only_preserve','changed_notes_review','limitations']})
save(BUNDLE/'event_verification_template.json',dict(template=[dict(date='',event_name='',venue='',session_start='',status='unknown',
    official_url='',supporting_url='',published_at=None,known_by_previous_day='unknown',is_setup_or_teardown='unknown',
    evidence='',checked_by='',checked_at='')],rule='확인된 실제 회차만 한 행. 빈칸/unknown은 근거 없음을 뜻하며 0이나 비행사로 바꾸지 않음.'))
(BUNDLE/'README.txt').write_text(f'''인공지능공학 3팀 작업 자료 / {now}

1. validation_predictions.json: 2025년 상반기181일 x 18시간 x 2방향 = 6516행. RF/LSTM 각각 시드42 고정 대표 모델.
   Python: rows=json.load(open(path,encoding="utf-8"))["records"]; df=pd.DataFrame(rows)
   signed_error = prediction - actual (양수=과대예측), absolute_error=abs(signed_error)
   일별 MAE는 같은 날짜36개 출력의 절대오차 평균. 모델별로 따로 계산한다.
2. priority_dates.json: 두 모델 일별MAE 평균 상위10일. 두 사람의 공동 점검 시작 목록.
3. event_candidates.json: 2025년 상반기와 기간이 겹치는 잠실 비야구장 행사 후보. 겹치는 기간의 매일이 실제 행사일인 것은 아니다.
4. event_verification_template.json: 에이전트 검증 기록의 필드 예시. 팀원 검증 과제가 아니다.
5. baseball_discrepancies.json: 신규 시범8건, 원본만 있는 취소1건, note 변경9건. 기존 원본을 덮어쓰지 않는다.

정정: 행사 자료의 날짜/시각/장소/출처 검증은 수집한 에이전트가 담당한다. 아래 후보와 확인 양식은 참고용이며 팀원에게 검증을 맡기는 과제가 아니다.
준석: 실험기록에서 실험 이유-변경 조건-결과-해석을 정리. 1/3/5시드와 CPU/GPU 비교 범위 구분. 발표용 핵심 시행착오5개와 비교표, 근거 문서/그림을 제출.
재헌: 모델별 오차큰 날짜10개/시간대·방향20개, 대표 과대·과소 사례 그래프3개, 관찰과 원인 가설을 구분한 해석3~5줄. 행사 근거는 에이전트가 검증해 별도 전달한다.
공통: 원본/학습코드 변경·새 학습·하반기 최종 평가 없이 검증/분석 자료만 전달. 날씨 실측/최종 관중/사후 취소를 전날 입력으로 넣지 않는다.
제출: 준석 실험 비교표+해석 정리본, 재헌 오차표+그래프3개+해석+분석코드. 전체 검증자료를 기반으로 한 사후 분석이며 인과관계를 증명하지 않는다.
''',encoding='utf-8')

history=[
 ('001','기본 9조합','RF/MLP/LSTM x 3·7·14일','RF3일 255.87. 신경망은 200회 상한에서 종료.'),
 ('002','학습 횟수 확대','최대1000·조기종료50, MLP/LSTM','MLP3일 300.03, LSTM14일 302.71. MLP7·14일은 개선 없음.'),
 ('003','하루별 1D CNN','입력3·7·14일','CNN7일 328.95로 세 조건 중 최저. RF보다 높음.'),
 ('004','첫 파라미터 튜닝','RF 잎1·5·10, 신경망lr0.0003·0.003','기준 잎3 유지. lr0.003에서 MLP298.27/CNN328.27/LSTM278.57.'),
 ('005','GPU·Dropout','MLP/CNN Dropout0·0.1·0.2','GPU 기준: MLP0에서290.31, CNN0.1에서325.91. CPU와 환경 차이 있음.'),
 ('006','LSTM GPU 횟수','최대1000 vs2000, patience50','277.92→275.65. 최선1283/종료1333. 공통1000회 곡선 일치.'),
 ('007','첫 용량 확대','RF3조건·신경망 너비/깊이6조건','RF 잎2 단일시드255.80, 잎3 대비0.07 차이. 신경망 확대6조건은 모두 악화.'),
 ('008','시드 반복','RF 잎2·3, LSTM 1층 / 5시드','RF 잎3 평균256.31±1.01, 잎2 256.82±1.10. LSTM273.90±1.28.'),
 ('009','단순 기준·경기 유무','RF/LSTM 포함·제외 5시드','지난주379.43, 요일·시간대평균450.74. 경기 포함이 두 모델 모두5/5 개선.'),
 ('010','더 큰 너비·두 층 추가','MLP256/CNN32/LSTM128, 각 두 층 추가 /3시드','MLP/CNN 실용 이득 없음. LSTM3층273.85→273.37(0.18%), 시간 약2.81배.'),
 ('011','학습 방법 비교','스케줄→배치→옵티마이저 /3시드','완료. Adam0.003·배치8 평균270.46, 기준273.85 대비1.24% 개선(3/3).'),
]

pdfmetrics.registerFont(TTFont('Korean','C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('KoreanBold','C:/Windows/Fonts/malgunbd.ttf'))
pdfmetrics.registerFontFamily('Korean',normal='Korean',bold='KoreanBold')
navy=colors.HexColor('#15334B');teal=colors.HexColor('#087F8C');gray=colors.HexColor('#52616B')
styles={
 'title':ParagraphStyle('title',fontName='KoreanBold',fontSize=22,leading=30,textColor=navy,spaceAfter=10),
 'sub':ParagraphStyle('sub',fontName='Korean',fontSize=9,leading=14,textColor=gray,spaceAfter=10),
 'h':ParagraphStyle('h',fontName='KoreanBold',fontSize=12,leading=18,textColor=teal,spaceBefore=10,spaceAfter=6),
 'body':ParagraphStyle('body',fontName='Korean',fontSize=10,leading=16,wordWrap='CJK',spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Korean',fontSize=9,leading=14,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='Korean',fontSize=8.5,leading=13,textColor=gray,wordWrap='CJK',spaceAfter=6)}
story=[]
def p(text,kind='body'):story.append(Paragraph(text,styles[kind]))
def title(n,text):p(f'인공지능공학 3팀 / 팀 공유 / {n}','sub');p(text,'title')
def table(rows,widths):
 t=Table([[Paragraph(str(c),styles['cell']) for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DFEDF1')),
     ('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#D4DEE5')),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
     ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7)]));story.append(t)

title('01','현재까지의 결론')
p(f'기준 시각: {now} / 완료 EXP-001~011','sub')
p('<b>EXP-011의 21개 결과와 모델 복원 검증이 완료됐다. 팀원들은 오늘 실험 결과의 해석과 큰 오차의 사례 분석에 집중한다.</b> 행사 자료 검증은 수집한 에이전트가 담당한다. 이전 행사검증 분담을 정정한 공유본이며 검증 평가 결과이며 하반기 최종 성능표가 아니다.')
p('공통 실험 기준','h')
table([['항목','내용'],['대상·목표','2024~2025년 종합운동장역 2호선. 전날 기록 확보를 가정해 다음 날06~24시의 시간대별 승차·하차36개 값 예측.'],
 ['시간순 분할','2024년352일 학습 / 2025년상반기181일 검증 / 하반기178일 최종 평가 보류.'],
 ['입력·평가','과거 시간대별 수요+요일·월·공휴일+경기 기록·예정 시각·팀. 원래 인원 단위 MAE, 음수 예측0 보정.'],
 ['주의','MAE는 시간대별 인원 오차이며 하루 총오차나 정확도(%)가 아니다. 기록 없음은 미확인. note는 사후 분석용, gamecenter_status는 판정에 쓰지 않는다.']],[91,416])
p('현재의 주요 근거','h')
table([['비교','결과·해석'],['기존 5시드 비교','RF3일·최소잎3: 256.31±1.01 / LSTM14일1층: 273.90±1.28. 동일 검증기간에서 RF가 낮음.'],
 ['경기 정보 포함/제외','RF 256.31/389.97, LSTM 273.90/300.31. 두 모델 모두5/5시드에서 포함이 낮음. 야구의 인과효과를 증명한 것은 아님.'],
 ['오늘 완료된 배치 비교','LSTM 3시드: batch8 270.46 / batch16 273.85 / batch32 275.97. 배치8은3/3 개선했지만 계산량 증가.'],
 ['평균 비교 원칙','1시드·3시드·5시드 수치를 섞어 개선율을 계산하지 않는다. CPU/GPU 환경 차이도 별도 표시한다.']],[113,394])
p('모델 역할: RF는 표 형태 비선형 비교 기준, MLP는 기본 신경망, LSTM은 시간 흐름, 1D CNN은 하루 안의 국소 패턴을 학습한다. 큰 모델이 항상 유리하지 않았다는 시행착오가 핵심이다.','small')
story.append(PageBreak())

title('02','실험 이력: 기본 비교와 개선')
p('EXP-001~007은 주로 시드42의 단일 실행. 각 값은 당시 조건의 결과이며 한 표의 공정한 최종 순위가 아니다.','sub')
table([['실험','변경·질문','결과와 배운 점']]+[[f'EXP-{n}<br/>{name}',setting,result] for n,name,setting,result in history[:7]],[102,157,248])
p('해석할 때 유지할 원칙','h')
p('1. 200회에서 끝난 LSTM을 보고 모델 자체가 나쁘다고 결론내리지 않고, 학습 횟수와 학습률을 확인했다.<br/>2. GPU 전환 때 운영체제·TensorFlow 버전도 달라져 같은 GPU의 기준을 다시 만들었다.<br/>3. RF 잎2의 단일시드0.07명 개선은 작아서 다음 단계에서 여러 시드로 검증했다.')
p('상세 설정·최선/종료에포크·시간·한계는 저장소 reports/experiments/EXP-001~007 문서에 보존.','small')
story.append(PageBreak())

title('03','반복 검증과 현재 진행 상황')
table([['실험','비교','핵심 결과']]+[[f'EXP-{n}<br/>{name}',setting,result] for n,name,setting,result in history[7:]],[102,157,248])
p('EXP-011 공유 시점의 완료 조건만','h')
rows=[['조건','완료 시드','평균 MAE']]
for v,s in snap['completed_conditions'].items():
 if s['complete']:rows.append([v,'3/3',f"{s['mean_mae']:.2f}"])
table(rows,[295,80,132])
p(f"진행 상태: 고유 결과 {snap['completed']}/21 완료(기존 기준3개 포함). 현재 EXP-011 학습은 종료됐다. 기존 기준3개를 포함한다.",'small')
p('학습률 감소 정책은 277.82명으로 고정273.85명보다 높았다. 최선은 Adam0.003·배치8이다. RMSprop0.003/0.001은280.11/297.78명으로 개선되지 않았다.')
p('소폭 개선과 한계','h')
p('LSTM3층은3시드 모두 소폭 개선됐지만 평균0.48명 감소에 시간은 기록상 약2.81배였다. 데이터는 일별352개이며 검증을 반복 사용했다. 구조 확대가 무조건 무의미하다고 단정하거나 최종 일반화 성능으로 발표하지 않는다.')
p('처음 실행은 Windows의 WSL 자동 업데이트 시점에 중단돼 부분 결과를 제외하고 새로 초기화했다. 하반기 최종 평가는 한 번도 실행하지 않았다.','small')
story.append(PageBreak())

title('04','오늘 밤 작업 제안: 실험 해석')
p('담당 제안: 준석 / 행사 자료 검증을 맡긴 이전 안내를 정정. 역할은 팀 사정에 따라 교환 가능.','sub')
p('<b>목표:</b> 팀이 왜 실험을 했고 무엇을 배웠는지 설명할 수 있도록 결과를 해석한다. 숫자만 나열하는 요약에서 한 단계 더 나아간다.')
table([['순서','해야 할 일·완료 기준'],['먼저 30분','PDF와 experiment_reports 안의 EXP-001~011을 읽고 입력기간·횟수·학습률·Dropout·용량·시드·경기유무 비교의 흐름을 파악.'],
 ['오늘 핵심','핵심 시행착오5개를 “가설/실험 이유-바꾼 조건-결과-배운 점”으로 작성. 개선하지 않은 실험도 포함.'],
 ['비교 점검','1시드/3시드/5시드 평균을 구분하고 같은 조건끼리 개선량 계산. CPU/GPU 환경 차이를 GPU 정확도 효과로 쓰지 않음.'],
 ['해석 질문','왜 RF가 현재 유리한지, 층 확대 이득이 작은 이유는 무엇인지 가능한 설명을 작성. 데이터352일·과적합·학습예산 등은 가설과 관찰을 구분.'],
 ['제출물','모델·설정 비교표+해석1~2쪽+발표할 핵심5개. 각 주장에 근거 실험ID/그림을 적고 EXP-011은 완료된 검증 평가라고 표시.']],[100,407])
p('작성 예시','h')
p('“200회에서 LSTM 성능이 낮아 학습 부족 가능성을 생각했다. 최대 횟수를 늘리자 개선됐으므로 첫 결과만으로 LSTM이 부적합하다고 판단하지 않았다.”처럼 관찰과 판단을 연결한다. 모든 데이터에서 성립하는 결론으로 확대하지 않는다.')
p('행사 자료 검증은 에이전트 담당','h')
p('수집한 서울시 행사231건·KBO 잠실333건의 원문 대응과 내용 검증을 에이전트가 수행한다. 실제 회차/설치·철거/장소/취소/전날 공개 여부를 구분한 확인·불일치·미확인 목록을 별도로 제공한다. 팀원에게 미완료 검증을 넘기지 않는다.')
p('검증 전 행사 후보는 모델 입력이나 오차 원인의 확정 근거로 쓰지 않는다. 확인할 수 없는 항목은 미확인으로 남기고 새 데이터 결합은 별도 결정한다.','small')
story.append(PageBreak())

title('05','오늘 밤 작업 제안: 오차 분석')
p('담당 제안: 재헌 / 새 학습 없이 제공된 검증 예측으로 바로 시작.','sub')
p('<b>목표:</b> 평균 MAE만으로 보이지 않는 실패 사례를 찾고, 어떤 데이터 보강이 필요한지 근거를 만든다.')
table([['순서','해야 할 일·제출물'],['1. 목록 공유','validation_predictions.json으로 모델별 오차 큰 날짜10개와 날짜·시간·방향20개를 정리. 상위 날짜를 먼저 준석에게 전달.'],
 ['2. 사례 분석','과대예측1일·과소예측1일·두 모델 차이가 큰1일 등 대표3사례의 실제/예측 시간대 그래프. 승차·하차를 구분하고 날짜·모델·시드 표시.'],
 ['3. 근거 대조','에이전트가 별도로 검증한 행사 자료가 오면 대조. 그전에는 과대/과소·시간대 패턴을 분석. 확인된 사실/가능한 설명/미확인을 구분하고 인과관계로 단정하지 않음.'],
 ['4. 제출','오차목록표+그래프3개+해석3~5줄+분석코드 또는 노트북. 모델별 일별MAE 계산이 제공된 예측의 전체MAE와 맞는지 확인.']],[100,407])
p('계산 기준과 분석 자료','h')
p('signed_error = prediction - actual: 양수는 과대, 음수는 과소예측. 절대오차를 사용해 날짜별36개 값을 평균하면 일별MAE다. 두 모델은 따로 집계한다. 제공자료는 <b>시드42의 RF잎3와 기존 LSTM배치16</b>의 고정 대표 예측이며, 오늘 배치8 후보나5시드 앙상블 예측이 아니다.')
p('오늘의 완료 기준','h')
p('준석: 실험 비교표·해석1~2쪽·핵심 시행착오5개. 재헌: 상위오차표·대표 그래프3개·해석·분석코드. 두 사람: 발표에서 설명할 관찰과 아직 모르는 점을 함께 정리한다. 행사 원문 검증은 에이전트가 별도로 담당한다.')
p('원본 데이터·학습 설정은 그대로 두고 분석 결과를 별도로 만든다. 하반기 최종 평가, 새 학습, 실측 날씨·최종 관중·사후 취소의 예측 입력 추가는 지금 하지 않는다. 검증 완료 자료를 모은 뒤 별도 데이터 버전과 비교 실험을 결정한다.','small')
p('상세 근거: 저장소 reports/experiments/EXP-001~011, reports/data_review.md, reports/metrics/team_handoff_snapshot_2026-11-10.json. 작업 브랜치: feat/initial-experiment-setup.','small')

def footer(canvas,doc):
 canvas.setFont('Korean',8);canvas.setFillColor(gray)
 canvas.drawString(44,27,'인공지능공학 3팀 | 검증 결과·업무 제안 | 최종 평가 미실행')
 canvas.drawRightString(A4[0]-44,27,f'{doc.page} / 5')
pdf=OUT/'인공지능공학3팀_실험요약_오늘작업_2026-11-10.pdf'
SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=38,bottomMargin=46,
    title='인공지능공학 3팀 실험 요약 및 오늘 작업',author='인공지능공학 3팀').build(story,onFirstPage=footer,onLaterPages=footer)
zip_path=OUT/'인공지능공학3팀_분석작업자료_2026-11-10.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for file in sorted(BUNDLE.iterdir()):
  if file.is_file():z.write(file,file.name)
 for file in sorted((ROOT/'reports/experiments').glob('EXP-0*.md')):z.write(file,'experiment_reports/'+file.name)
 for file in (ROOT/'reports/figures').glob('EXP-010*.png'):z.write(file,'figures/'+file.name)
print(json.dumps(dict(pdf=str(pdf),zip=str(zip_path),snapshot=now,prediction_rows=len(records),event_candidates=len(candidates),completed=snap['completed']),ensure_ascii=True))

"""Build the dated team/presentation handoff from reviewed reports; no training."""
from pathlib import Path
from xml.sax.saxutils import escape
import hashlib
import json
import subprocess
import zipfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
OUT.mkdir(parents=True,exist_ok=True)
BUNDLE=ROOT/'outputs/presentation-handoff-2026-11-12'
BUNDLE.mkdir(parents=True,exist_ok=True)
pages=[]
def page(title, subtitle, items): pages.append((title,subtitle,items))
def p(text): return ('p',text)
def h(text): return ('h',text)
def table(headers,rows,widths): return ('table',headers,rows,widths)
def pic(name,height): return ('image',name,height)

page('팀 공유 요약 · 최종 결론','2026.11.12 | 인공지능공학 3팀 | 결과 기준 커밋 94c0cf1',[
 p('프로젝트: 야구 일정과 과거 승하차 기록을 이용한 종합운동장역 2호선의 다음 날 시간대별 수요 예측. RF·MLP·LSTM·1D CNN을 비교하고, 최종적으로 시드 42 RF와 LSTM의 예측을 50:50으로 결합했다.'),
 h('발표에서 사용할 핵심 결론'),
 p('상반기 검증으로 선택한 RF·LSTM 고정 50:50 조합은 별도 하반기 178일 평가에서 MAE 241.04명으로 RF 단독 246.98명보다 2.41% 개선됐다. 다만 고수요·피크 오차와 RMSE는 RF보다 높아 급증 대응의 한계가 남았다.'),
 table(['평가 구분','RF','결합','해석'],[['2025 상반기 검증 181일','255.87','250.55','모델 선택에 사용'],['2025 하반기 최종 178일','246.98','241.04','선택 후 고정 모델 평가'],['2024 후속 검증 184일','270.69','264.75','3구간 중 결합 우위 1구간']], [145,65,65,220]),
 p('표의 단위는 MAE(명). 서로 다른 기간의 점수를 빼서 튜닝 효과라고 설명하지 않는다. MAE 241.04명은 시간대별 승차 또는 하차 예측의 절대오차 평균이며, 하루 총인원 오차나 정확도 백분율이 아니다.'),
 h('팀원이 기억할 네 가지'),
 p('1. 복잡한 구조보다 입력 정보와 학습 설정의 검증이 중요했다.\n2. 경기 정보 포함은 같은 조건의 제외 실험보다 좋았다.\n3. 결합은 평균 오차를 줄였지만 모든 기간·조건에서 우수하지 않았다.\n4. 콘서트·농구 등 추가 행사 자료는 학습에 사용하지 않았다.'),
 p('현재 상태: 승인된 실험·추가 검증 종료. 최종 모델과 하반기 결과 유지. 이 자료는 발표용 정리이며 새 학습이나 재평가를 실행하지 않았다. 상세 근거: FINAL-001, ANALYSIS-007, EXP-016.')])

page('데이터·문제 정의와 모델 선정','실험 결과를 이해하기 위한 공통 조건',[
 table(['항목','내용'],[['대상','종합운동장역 2호선, 2024-01-01~2025-12-31'],['예측 목표','다음 날 06~24시의 18개 1시간 구간 × 승차·하차 = 36개 출력'],['분할','2024 학습 352일 / 2025 상반기 검증 181일 / 하반기 최종 178일'],['입력','과거 3·7·14일 승하차, 요일·월 원핫, 공휴일, 수집 경기 기록 유무·건수, 예정 시각, 홈·원정팀'],['전처리','신경망 과거 인원은 학습 자료 기준 표준화, 예정 분은 1440으로 나눔. RF는 원래 수치. 목표 인원은 원 단위 유지.'],['정보의 경계','실제 관중·점수·경기 종료·note·gamecenter_status·verified는 입력 제외. 기록 없음은 미확인.']], [90,405]),
 p('전날 23~24시까지의 실제 인원 확보를 가정한다. 평가 기간에도 지나간 날짜의 실제 기록을 다음 날 입력으로 갱신한다. 6개월 전체를 한 번에 예측하는 장기 예측이 아니다.'),
 table(['모델','선정 이유','장점과 한계'],[['RF','비선형 표형 데이터의 비교 기준','학습·추론 부담이 작고 기준 성능이 좋음. 시간 순서를 별도 구조로 학습하지 않음.'],['MLP','과거 기록을 펼친 기본 신경망','구조가 단순하고 빠름. 순서 정보를 구조적으로 처리하지 않음.'],['LSTM','시간 순서의 의존성 학습','여러 날의 연속 패턴 표현. 학습 시간이 길고 데이터·학습 설정에 민감.'],['1D CNN','시간대 주변의 국소 패턴 학습','하루 안에서 필터 공유. 이번 구조·설정에서는 RF/LSTM보다 오차가 컸음.']], [55,150,290]),
 p('처음 14일은 모든 입력 기간의 공통 비교를 위해 제외했다. 하반기는 사전 탐색에 사용한 9/25·26·27·28·30, 10/1의 6일을 제외했다. 자료 전체를 야구 경기일로만 제한하지 않았다. 근거: EXP-001~003, FINAL-001.')])

page('실험 흐름 1 · 기본 모델에서 재현성까지','EXP-001~008 | 아래 수치는 2025 상반기 검증 MAE(명)',[
 table(['실험','질문·실행','핵심 결과와 판단'],[
 ['001 기본 9조합','RF·MLP·LSTM × 3·7·14일, seed42','RF 3일 255.87 최저. MLP 3일 312.64, LSTM 14일 410.42. 200회 상한만으로 신경망 부적합을 단정하지 않음.'],
 ['002 학습 횟수','최대1000·patience50, MLP/LSTM','MLP 3일 300.03, LSTM 14일 302.71. MLP 7·14일은 기존 최선 갱신 없음. 상한과 patience를 함께 변경.'],
 ['003 CNN 추가','하루별 공유 1D CNN, 3·7·14일','335.83 / 328.95 / 329.38. 7일 선택. 날짜 경계를 넘어 필터를 적용하지 않음.'],
 ['004 첫 튜닝','RF 잎 크기, 신경망 학습률 9조건','RF leaf3 유지. lr0.003에서 MLP298.27, CNN328.27, LSTM278.57. 단일 시드 탐색.'],
 ['005 GPU·Dropout','MLP3일/CNN7일의 0·0.1·0.2','GPU 기준 MLP0=290.31, CNN0.1=325.91. CNN은 같은 GPU의 0 대비0.76% 개선. 환경 변경을 정확도 개선 원인으로 단정하지 않음.'],
 ['006 GPU LSTM','최대1000 vs2000, patience50','277.92→275.65. 최선1283/종료1333. 상한 확대 후 실제 개선 확인.'],
 ['007 폭·깊이·RF','신경망 6변형, RF 3변형','신경망 변형 모두 기준보다 악화. RF leaf2=255.80으로 leaf3보다0.07명 개선에 그쳐 시드 확인.'],
 ['008 시드 반복','5시드로 RF leaf2/3, LSTM 비교','RF leaf3 평균256.31, leaf2 256.82. 단일 seed42의 작은 이득이 평균에서는 역전. LSTM(batch16)273.90.']], [80,145,270]),
 p('5시드: 42, 907394, 892611, 659988, 622763. 후속 3시드 비교는 42, 907394, 892611. 시드는 날짜 순서를 바꾸는 장치가 아니라 초기화·표본 추출 등의 난수 변동을 비교하기 위한 값이다.'),
 p('여기에는 CPU/GPU 환경과 최대 학습 횟수가 다른 단계가 섞여 있다. 각 변경의 효과는 같은 실험 안의 기준과 비교한다. 숫자만 연결해 하나의 통제된 개선 곡선으로 설명하지 않는다.')])

page('실험 흐름 2 · 입력·구조·학습 방법 검증','EXP-009~016 | 별도 표기 외 2025 상반기 검증 MAE(명)',[
 table(['실험','핵심 결과','얻은 교훈'],[
 ['009 기준·경기 제거','전주379.43, 요일·시간 평균450.74. 5시드 RF 경기 포함256.31 / 제외389.97, LSTM273.90 / 300.31.','같은 비교에서 두 모델 모두5/5시드 개선. 경기 입력의 예측상 유용성 확인; 인과 증명은 아님.'],
 ['010 추가 용량','3시드 LSTM 1층273.85→3층273.37. MLP 추가 깊이297.75→297.73, CNN 확대는 악화.','LSTM 깊이는0.48명 개선이나 시간 증가. 작은 평균 차이와 시드별 방향을 함께 판단.'],
 ['011 학습 방법','3시드 batch16=273.85, batch8=270.46, batch32=275.97. LR 감소277.82, RMSprop0.003=280.11.','Adam0.003 고정·batch8 선택. 배치8은3/3시드 개선. 동일 에포크가 동일 갱신 수는 아님.'],
 ['012 설정 조합','3시드 1층·batch8=270.46, 3층·batch8=271.78.','각각 좋았던 변경을 합쳐도 이득이 더해지지 않음. 1층 유지.'],
 ['013 손실 함수','seed42 MAE269.26 / Huber288.76 / MSE346.58. Huber delta800.','일부 피크는 개선해도 일반 구간 오차가 늘어 전체 MAE 악화. delta는 학습 자료 기준으로 정함.'],
 ['014 LSTM Dropout','seed42 0/0.1/0.2=269.26/280.09/283.34. 결합도250.55/256.78/258.41.','Dropout0 유지. 규제 추가가 항상 일반화 개선은 아님.'],
 ['015 시간대별 RF','seed42 RF18개 MAE304.78, 기존255.87보다19.11% 악화.','총 나무200→3600, 예측 비용도 증가. 하반기 평가 후 시행한 후속 탐색.'],
 ['016 시간순 검증','2024년3구간184일 RF270.69 / 결합264.75. 결합 우위1/3구간.','후속 안정성 검증. 과거 구간별 재학습이며 기존 최종 모델을 교체하지 않음.']], [85,220,190]),
 p('EXP-011은 정책→배치→옵티마이저 순차 탐색으로 모든 조합의 전수 검색이 아니다. EXP-013은 같은 학습률을 사용했으므로 손실별 최적화까지 마친 비교가 아니다. EXP-015·016은 최종 평가 후의 후속 작업으로 구분한다.')])

page('결합 선택의 근거와 최종 설정','ANALYSIS-001~005 · DEC-037 · 최종 평가 전 고정',[
 table(['구성','상반기 MAE','역할'],[['RF 단독 (seed42)','255.87','단일 모델 비교 기준'],['LSTM 단독 (seed42)','269.26','RF와 다른 오차 패턴'],['RF·LSTM 50:50 (seed42)','250.55','선택한 최종 구성'],['RF 3시드 예측 평균','255.47','추가 평균화 비교'],['RF3 + LSTM3, 6모델 평균','250.16','0.39명 추가 이득에 모델 수2→6']], [230,110,155]),
 p('같은 세 시드 안에서 결합을 비교하면 점수 평균은 RF255.73±0.89, LSTM270.46±1.21, 결합250.60±0.55였다(표본SD). 3/3시드에서 결합이 개선됐다. 이 값은 여섯 예측을 합친 한 모델의 점수250.16과 다르다.'),
 h('선택한 모델'),
 p('RF: 과거3일, 나무200, 최대깊이10, 최소잎표본3, seed42.\nLSTM: 과거14일×18시간×2채널 → LSTM32유닛1층 → 예측일 정보 결합 → Dense32(ReLU) → 출력36(선형). LSTM 내부 tanh/sigmoid, Dropout0.\n학습: MAE, Adam0.003, 배치8, 최대2000, patience50, min_delta0. 선택된 seed42 LSTM은 최선1710회/종료1760회.\n결합: 0.5×max(RF예측,0) + 0.5×max(LSTM예측,0), 반올림 없음.'),
 p('검증 최저 숫자만 보면6모델 결합이지만, 추가 이득0.39명과 복잡도를 함께 고려해 사용자 결정으로 기존 seed42 한 쌍을 최종 평가 전에 고정했다. 하반기를 보고 시드나 비율을 다시 고르지 않았다.'),
 p('피크·고수요 분석(ANALYSIS-001), 예측 평균(002), 동일3시드 비교(003), Dropout 곡선(004), 세 시드 예측 평균(005)을 순서대로 수행했다. Dropout은 학습·검증 격차를 줄여도 오차가 악화할 수 있었으며, 격차만으로 과적합 유무를 확정하지 않았다.')])

page('하반기 최종 평가 · 발표의 주 결과','FINAL-001 | 2025-07~12, 제외6일을 뺀178일·6,408개 값',[
 table(['지표','RF','고정 50:50'],[['전체 MAE','246.98','241.04'],['승차 MAE','278.41','269.00'],['하차 MAE','215.54','213.07'],['승차 RMSE','857.46','880.37'],['하차 RMSE','526.40','548.92']], [235,130,130]),
 p('전체 MAE는5.94명(2.41%) 감소했고 승하차 MAE와 7~12월 각 월의 전체 MAE도 개선됐다. 그러나 큰 오차에 민감한 RMSE는 악화했다. 2024년 학습 모델을 그대로 사용했으며 상반기를 포함해 재학습하지 않았다.'),
 table(['급증 관련 지표','RF 승차/하차','결합 승차/하차'],[['고수요 MAE','1663.42 / 1063.64','1757.37 / 1136.88'],['실제 피크 시점 MAE','1984.09 / 1076.86','2160.90 / 1262.68']], [195,150,150]),
 p('고수요는 상반기에서 정한 임계값(승차1049.3명·하차1425.6명)을 평가 전에 고정했다. 하반기 표본은 승차314개·하차328개이며 실제 안전·혼잡 기준이 아니다. 피크는 각 날짜·승하차별 최대 인원 시간대다.'),
 pic('final_evaluation.png',190),
 p('이 평가의 결론은 평균적인 시간대 오차의 소폭 개선이다. 모든 상황에서 우수하거나 급증·안전 대응까지 검증됐다고 주장하지 않는다. 출처: FINAL-001 및 reports/metrics/final_evaluation.json.')])

page('후속 검증 · 얼마나 안정적인가','ANALYSIS-007 / EXP-016 | 최종 모델·하반기 결과 유지',[
 p('상반기 일별 MAE 차이를 연속 날짜 블록으로 10,000회 재표집했다. 7일 블록의 개선 폭95% 구간은 -5.29~14.89명으로0을 포함한다. 3일·14일에서도 동일하다. 평균 개선 관측과 확실한 우위 주장은 구분해야 한다. 반복 선택 편향·계절 변화를 해소하는 분석은 아니다.'),
 table(['2024 외부 검증','RF','LSTM','결합','결합 판단'],[['7~8월 (62일)','404.31','392.48','372.07','개선'],['9~10월 (61일)','293.48','339.73','295.45','악화'],['11~12월 (61일)','112.09','159.25','124.96','악화'],['합계184일','270.69','297.67','264.75','약2.20% 개선']], [140,75,75,75,130]),
 p('각 학습 기간 마지막 한 달에서 에포크를 정한 뒤 전체 학습 기간으로 재학습했다. 최선/종료는1116/1166,602/652,9/59회. 마지막 구간의 이른 종료는 내부 기간에 대한 민감성을 보여준다. 기존에 사용한 자료·설정이므로 완전히 새로운 독립 외부 평가로 부르지 않는다.'),
 table(['상반기 조건','표본 날짜','결합의 RF 대비 변화'],[['평일','129일','MAE17.77명 감소'],['주말','52일','MAE25.56명 증가'],['수집 경기 기록 있음','86일','MAE8.27명 증가']], [195,110,190]),
 p('월별로는 상반기1~3월 개선,4~6월 악화. 월·경기·주말 집단이 겹치므로 원인을 단정하지 않는다. 기록 없음은 비경기일로 확정하지 않았다.'),
 p('검사9개 통과: 입력 시점, 미래값 변경 시 전처리 불변, 미관측 팀 처리, 날짜·정답·출력 순서, 지표·조기 종료 등. 저장 모델 복원과 예측 점수 재계산 완료. 구간 학습·평가 합계 약34분38초. 최종 평가 파일의 보존 해시 확인.')])

page('결합의 성공·실패와 예측 비용','ANALYSIS-006 | 2025 상반기 seed42 사례',[
 p('181일 중 일별 MAE는 결합이 RF보다125일 개선,56일 악화했다. 아래는 개선·악화가 가장 큰 날짜에서 기여가 큰 관측값으로, 무작위 대표 사례가 아니다.'),
 table(['사례: 승차','실제','RF','LSTM','결합'],[['6/27 21~22시: 보완','10512','3965','10029','6997'],['4/19 17~18시: 악화','9325','7455','1334','4394']], [175,80,80,80,80]),
 p('성공 사례에서는 LSTM이 RF가 놓친 급증을 보완했다. 실패 사례에서는 LSTM의 낮은 예측이 상대적으로 잘 맞춘 RF 예측을 끌어내렸다. 특정 행사가 원인이라는 인과 확인은 하지 않았다.'),
 pic('combination_cases.png',265),
 table(['구성','하루36개 출력의 추론 중앙값'],[['RF (CPU)','6.59ms'],['LSTM (GPU)','13.38ms'],['RF·LSTM 50:50','19.82ms'],['시간대별 RF18개','112.16ms']], [245,250]),
 p('입력이 메모리에 준비된 상태, 워밍업5회 후30회 측정. 파일 읽기·전처리·모델 로딩 제외. 실제 서비스 전체 응답 시간이나 인력 배치 효과를 측정한 결과는 아니다.')])

page('한계점과 향후 발전 계획','한계는 결과를 숨기기 위한 설명이 아니라 적용 범위의 명시',[
 table(['한계','발표에서 정확히 설명할 내용'],[['범위','2호선 종합운동장역, 2개 연도 자료. 다른 역·노선·연도에서도 같은 성능이라고 보장하지 못함.'],['사전 정보 가용성','사후 수집 경기 일정의 완전성·전날 공개 여부와 승하차 공개 지연 미검증. 전날 실제 기록 확보 가정이 필요.'],['누락 행사','다른 공연·농구·마라톤 등의 영향이 전체 승하차 정답에는 섞일 수 있으나 입력은 야구 중심. 추가 행사 자료는 학습하지 않음.'],['평가·선택 편향','상반기 반복 튜닝, 작은 시드 수, 단일 하반기 평가. 95% 구간도 모델 선택 편향을 보정하지 않음.'],['급증 대응','최종 전체 MAE는 개선했지만 고수요·피크 오차와 RMSE 악화. 안전·혼잡 대응으로 바로 연결하기 어려움.'],['통제 범위','CPU/GPU·라이브러리 차이, 모델용량 증가, 배치별 갱신 수 차이가 있음. 모든 실험이 같은 계산 예산은 아님.'],['목표의 한계','승하차 유동량은 승강장 동시 체류, 열차 혼잡, 대기열·안전 등급과 다름. MAE를 정확도%로 변환하지 않음.']], [100,395]),
 h('향후 과제 - 이번에 실행한 결과와 구분'),
 p('1. 예측 시점에 공개돼 있던 일정·수요 자료의 스냅샷과 공개 지연 검증.\n2. 확인된 실제 회차 기준의 콘서트·농구 등 행사 정보 추가. 설치·철거 기간을 행사일로 펼치지 않기.\n3. 기상·운영 정보 등 추가 입력을 한 종류씩 검증하고, 수집 시점과 결측 처리 기록.\n4. 고수요 과소예측을 줄이는 목표·평가 방법 및 시기별 안정성 검증.\n5. 이미 본 하반기를 새 독립 평가처럼 재사용하지 않고 새로운 미사용 기간 확보.'),
 p('추가 행사 자료 대조에서 출처 불일치·누락이 발견됐지만 해당 보충 자료는 기존 학습 입력에 들어가지 않았다. 따라서 그 오류가 기존 모델을 직접 오염시킨 것은 아니다. 근거: reports/data_review.md, FINAL-001, ANALYSIS-007.')])

page('발표 구성안 · 무엇을 어느 순서로 보여줄까','권장 본문12장 + 부록 | 실제 발표 시간에 맞춰 조정',[
 table(['장','슬라이드 제목','핵심 메시지·자료'],[['1','문제와 목적','종합운동장역 다음 날 시간대별 승하차 예측. 안전 효과를 이미 검증했다고 말하지 않기.'],['2','데이터와 예측 시점','대상·기간·36출력·하루 앞 입력. 데이터 공개 지연 가정 명시.'],['3','시간순 분할과 MAE','352/181/178일, 제외6일, 전처리 학습 범위. MAE 단위 설명.'],['4','네 모델의 선정 이유','RF/MLP/LSTM/CNN의 역할과 장단점. 신경망만 고집하지 않은 이유.'],['5','기본 비교와 시행착오','입력3/7/14일, 에포크·학습률, RF3일·LSTM14일을 남긴 과정.'],['6','효과가 있었던 변경','경기 정보 유무 EXP-009, 배치8 EXP-011. 같은 조건의 비교만 제시.'],['7','효과가 작거나 없던 변경','너비·깊이 확대, 3층+배치8, Dropout·손실 변경. 실패도 근거로 설명.'],['8','결합을 선택한 이유','개별 오차 보완, 세 시드 결과, 6모델 대비 단순성. 50:50 사전 고정.'],['9','하반기 최종 평가','241.04명,2.41% 개선. final_evaluation.png. 고수요·RMSE 악화도 함께.'],['10','어디서 성공·실패했나','combination_cases.png. 극단 사례임을 명시.'],['11','추가 검증·한계','robustness_audit.png, temporal_validation.png. 우위가 모든 기간에 유지되지는 않음.'],['12','최종 결론·향후 과제','평균 오차 소폭 개선, 급증 한계, 사전 가용성·다른 행사·미사용 기간 검증.']], [30,145,320]),
 h('팀원 분담 제안'),
 p('자료·방법 담당: 범위·컬럼·시간순 분할·누수 방지 설명.\n실험 담당: 변경 이유·기준·결과·채택 여부를 묶어 설명.\n오차 분석 담당: 성공/실패·피크·조건별 오차와 해석 한계 설명.\n발표 취합 담당: 표마다 기간·시드 수·단위 확인, 결론 문장 통일. 실제 담당자 배정은 팀에서 결정.'),
 p('본문에는 핵심 비교만 두고 16개 실험 전체표, 세부 설정, 학습 곡선, 출처는 부록으로 배치한다. 서로 다른 기간의 숫자를 한 순위표에 섞지 않는다.')])

page('예상 질문과 전달 전 점검','팀원 간 설명을 통일하기 위한 답변 원고',[
 h('Q. MAE241명이면 오차가 큰 것 아닌가?'),p('시간대별 승차/하차 예측의 절대오차 평균이다. 비교 기준과 수요 수준·피크 오차를 함께 봐야 한다. 하반기 RF보다5.94명 개선했지만 큰 급증의 오차는 여전히 크다. 정확도 몇%라고 환산하지 않는다.'),
 h('Q. LSTM 단독이 RF보다 나쁜데 왜 결합했나?'),p('두 모델이 틀리는 양상이 달라 평균 예측이 보완될 수 있다. 상반기 같은3시드 모두 결합이 개선했고, 고정 후 하반기도 평균MAE가 개선됐다. 모든 날에 보완되는 것은 아니다.'),
 h('Q. 층과 유닛·Dropout을 늘리면 더 좋아야 하지 않나?'),p('용량·규제를 늘리는 것과 검증 성능 개선은 다르다. 큰 구조의 개선은 작거나 일관되지 않았고, 최종 LSTM의 Dropout0.1/0.2는 오차가 증가했다. 이번 설정의 결과이지 모든 구조에 대한 결론은 아니다.'),
 h('Q. 3시드에서 좋아졌는데 왜 확실한 우위가 아니라고 하나?'),p('시드는 같은 날짜에서 학습 난수 변동을 보는 것이다. 다른 시기의 안정성과 다르다. 날짜 블록 분석의95%구간은0을 포함했고 후속3기간 중2기간에서 RF가 더 좋았다.'),
 h('Q. 경기 정보가 효과적이라는 것은 야구의 인과 효과인가?'),p('같은 학습 조건에서 경기 입력 포함 모델이 제외 모델보다 예측을 잘했다는 뜻이다. 다른 요인과 겹칠 수 있어 야구가 인원 증가의 원인임을 정량 증명한 것은 아니다.'),
 h('Q. 새 행사 데이터 오류 때문에 실험을 폐기해야 하나?'),p('해당 보충 자료는 학습에 사용하지 않아 직접 오염은 없다. 다만 기존 정답에 다른 행사 영향이 섞일 수 있고 사후 야구 일정의 사전 가용성 한계는 남는다.'),
 h('Q. 실제 현장에 바로 적용할 수 있나?'),p('전날 자료 확보와 당시 공개 일정 확인, 급증 대응 검증이 추가로 필요하다. 현재는 수업 프로젝트의 수요 예측 실험이며 역내 혼잡·안전 개선 효과를 실증하지 않았다.'),
 h('제출 전 체크'),p('표마다 기간·지표·단위·시드 수 기재 / 250.60(점수 평균)과250.16(예측 평균) 구분 / 하반기 제외6일 명시 / 실패 실험과 한계 포함 / 외부 자료·코드 출처 확인 / 실제 발표시간·담당자 확정.')])

sources=sorted((ROOT/'reports/experiments').glob('*.md'))
page('근거 문서와 재현 자료','공유 ZIP의 sources 폴더에 원문 포함 | 원본 데이터·대형 모델 제외',[
 p('원문 기록의 “미실행·다음 제안”은 각 실험 당시 상태다. 현재 최종 구성은 FINAL-001과 configs/final_evaluation.json, 후속 검증은 ANALYSIS-007·EXP-016을 기준으로 본다.'),
 *[p(f'{f.stem}: sources/{f.name}') for f in sources],
 p('데이터 검토: sources/data_review.md. 최종 고정 설정: sources/final_evaluation.json. 주요 지표 JSON: sources/metrics/. 그림 원본: figures/. 발표용 문장과 표: 팀공유_발표원고_2026-11-12.txt.'),
 p('자료 기준은 기존 결과 커밋94c0cf1. 이 정리는 새 학습·최종 재평가를 수행하지 않는다. 문서 생성 코드: tools/build_final_handoff.py. 출처 파일 SHA256 목록은 공유 ZIP의 source_manifest.json에 보존한다.')])

pdfmetrics.registerFont(TTFont('Malgun','C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunB','C:/Windows/Fonts/malgunbd.ttf'))
styles={
 'title':ParagraphStyle('title',fontName='MalgunB',fontSize=20,leading=27,textColor=colors.HexColor('#12344a'),spaceAfter=10),
 'sub':ParagraphStyle('sub',fontName='Malgun',fontSize=9,leading=13,textColor=colors.HexColor('#5a6a75'),spaceAfter=18),
 'p':ParagraphStyle('p',fontName='Malgun',fontSize=9,leading=14,wordWrap='CJK',spaceAfter=9),
 'h':ParagraphStyle('h',fontName='MalgunB',fontSize=11,leading=16,textColor=colors.HexColor('#156b7a'),spaceBefore=7,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Malgun',fontSize=8,leading=12,wordWrap='CJK'),
 'head':ParagraphStyle('head',fontName='MalgunB',fontSize=8,leading=12,textColor=colors.white,wordWrap='CJK')}
def para(t,s='p'): return Paragraph(escape(str(t)).replace('\n','<br/>'),styles[s])
story=[]; text=[]
for index,(title,subtitle,items) in enumerate(pages):
    if index: story.append(PageBreak())
    story.extend([para(title,'title'),para(subtitle,'sub')]);text.extend([title,subtitle,''])
    for item in items:
        kind=item[0]
        if kind in ('p','h'):
            story.append(para(item[1],kind));text.extend([item[1],''])
        elif kind=='table':
            _,head,rows,widths=item
            cells=[[para(c,'head') for c in head]]+[[para(c,'cell') for c in row] for row in rows]
            t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#12344a')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f1f5f7'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#cad7df'))]))
            story.extend([t,Spacer(1,10)]);text.extend([' | '.join(head)]+[' | '.join(row) for row in rows]+[''])
        elif kind=='image':
            path=ROOT/'reports/figures'/item[1]
            im=Image(str(path));ratio=min(495/im.imageWidth,item[2]/im.imageHeight)
            im.drawWidth=im.imageWidth*ratio;im.drawHeight=im.imageHeight*ratio
            story.extend([im,Spacer(1,9)]);text.extend([f'[그림: figures/{item[1]}]',''])
def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#cad7df'));canvas.line(50,42,545,42)
    canvas.setFont('Malgun',8);canvas.setFillColor(colors.HexColor('#5a6a75'))
    canvas.drawString(50,29,'인공지능공학 3팀 | 종합운동장역 수요 예측 | 2026.11.12')
    canvas.drawRightString(545,29,str(doc.page))
pdf=OUT/'팀공유_실험결과_최종결론_2026-11-12.pdf'
SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=42,bottomMargin=55,title='종합운동장역 수요 예측 - 실험 결과와 최종 결론',author='인공지능공학 3팀').build(story,onFirstPage=footer,onLaterPages=footer)
txt=ROOT/'reports/팀공유_발표원고_2026-11-12.txt';txt.write_text('\n'.join(text),encoding='utf8')
assert len(PdfReader(pdf).pages)==len(pages), 'Unexpected overflow: review pagination'
message='''팀원들 공유드립니다. 실험과 추가 검증까지 마무리했고, 발표용 PDF·원고·그림·근거 기록을 묶었습니다.

최종 구성은 시드42 RF(3일)+LSTM(14일,1층,batch8,Dropout0)의 50:50입니다.
하반기 최종178일 MAE는241.04명으로 RF246.98명보다2.41% 개선됐습니다.
다만 고수요·피크 오차와 RMSE는 악화됐고, 후속 기간 검증에서도 항상 결합이 좋지는 않았습니다.
추가 공연·농구 자료는 학습하지 않았으며, 일정의 전날 공개 여부·승하차 공개 지연은 한계로 남습니다.

발표는 데이터/평가 설계→모델 비교→튜닝 시행착오→결합 선정→하반기 결과→실패 사례/한계 순서로 구성하면 됩니다.
숫자는 PDF를 기준으로 통일하고, 모든 표에 평가기간·MAE단위(명)·시드 수를 적어주세요.
241명은 하루 총인원 오차가 아니라 시간대별 승차/하차 절대오차 평균입니다.
PDF10쪽에 발표 구성과 분담 제안,11쪽에 예상 질문 답변이 있습니다.
'''
msg=ROOT/'reports/팀전달_메시지_2026-11-12.txt';msg.write_text(message,encoding='utf8')
files=sources+[ROOT/'reports/data_review.md',ROOT/'configs/final_evaluation.json']
metric_names=['final_evaluation','ensemble_seeds','seed_prediction_average','robustness_audit','robustness_backtests','prediction_cost','hourly_rf','loss_comparison','lstm_dropout']
files += [ROOT/'reports/metrics'/f'{name}.json' for name in metric_names]
manifest={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(BUNDLE/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
bundle=BUNDLE/'팀공유_발표자료묶음_2026-11-12.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    for f in [pdf,txt,msg,BUNDLE/'source_manifest.json']:z.write(f,f.name)
    for f in files:z.write(f,('sources/metrics/' if f.parent.name=='metrics' else 'sources/')+f.name)
    for name in ['final_evaluation.png','combination_cases.png','robustness_audit.png','temporal_validation.png','temporal_learning_curves.png','ensemble_seeds.png','seed_prediction_average.png']:
        z.write(ROOT/'reports/figures'/name,'figures/'+name)
print(json.dumps(dict(pdf=str(pdf),pages=len(pages),text=str(txt),message=str(msg),zip=str(bundle)),ensure_ascii=False))

"""Build the Korean team-review handout with embedded fonts (no training)."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/인공지능공학3팀_1차실험_설정검토.pdf"
C = json.loads((ROOT / "configs/initial.json").read_text(encoding="utf-8"))
pdfmetrics.registerFont(TTFont("Korean", "C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("KoreanBold", "C:/Windows/Fonts/malgunbd.ttf"))
pdfmetrics.registerFontFamily("Korean", normal="Korean", bold="KoreanBold")
NAVY = colors.HexColor("#15334B")
TEAL = colors.HexColor("#087F8C")
GRAY = colors.HexColor("#52616B")
styles = {
    "title": ParagraphStyle("title", fontName="KoreanBold", fontSize=23, leading=32, textColor=NAVY, spaceAfter=8),
    "sub": ParagraphStyle("sub", fontName="Korean", fontSize=10, leading=16, textColor=GRAY, spaceAfter=14),
    "h": ParagraphStyle("h", fontName="KoreanBold", fontSize=13, leading=19, textColor=TEAL, spaceBefore=12, spaceAfter=7),
    "body": ParagraphStyle("body", fontName="Korean", fontSize=10.2, leading=16, wordWrap="CJK", spaceAfter=6),
    "small": ParagraphStyle("small", fontName="Korean", fontSize=9, leading=14, textColor=GRAY, wordWrap="CJK", spaceAfter=5),
    "cell": ParagraphStyle("cell", fontName="Korean", fontSize=9.5, leading=14.5, wordWrap="CJK"),
}
story = []


def p(text, kind="body"):
    story.append(Paragraph(text, styles[kind]))


def table(rows, widths=(108, 399)):
    cells = [[Paragraph(str(cell), styles["cell"]) for cell in row] for row in rows]
    t = Table(cells, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EDF4F7")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.45, colors.HexColor("#D9E3E8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(t)


def title(number, name, subtitle):
    p(f"{number} / 인공지능공학 3팀", "sub")
    p(name, "title")
    p(subtitle, "sub")


title("01", "1차 실험 설정 검토", "종합운동장역 수요 예측 | 2026.11.02 | 팀 최종 확인 요청")
p("<b>환경·코드 준비와 사전 점검 완료 / 실제 학습·성능 측정은 아직 미실행</b>")
p("아래 조건으로 1차 실험을 시작해도 되는지 확인 부탁드립니다. 초기값은 최적값이 아니며, 검증 결과를 보고 조정하고 이유와 결과를 기록합니다.")
p("실험 범위와 분할", "h")
table([
    ("대상", "2024~2025년 종합운동장역 <b>2호선</b>. 1차 행사 정보는 야구이며 다른 행사는 후속 확장."),
    ("예측 목표", "전날 23~24시까지의 기록 확보를 가정하고, 다음 날 06~24시 18개 시간대의 승차·하차 <b>36개 값</b> 예측."),
    ("비교 조건", "랜덤포레스트·MLP·LSTM × 과거 3·7·14일 = <b>총 9개 조합</b>. 같은 예측 대상 날짜와 정보로 비교."),
    ("학습 / 검증", "2024년 학습 <b>352일</b> / 2025년 1~6월 검증 <b>181일</b>. 학습 대상은 14일 공통 입력 확보 후 1월 15일부터."),
    ("최종 평가", "2025년 7~12월 <b>178일</b>. 기존 탐색일 9/25·26·27·28·30, 10/1 제외. 이 날짜의 관측은 이후 날짜의 과거 입력으로 사용 가능."),
])
p("사용하는 정보", "h")
table([
    ("과거 이용량", "각 입력 기간 안의 시간대별 승차·하차. 3일 조건에 7일 lag를 별도로 추가하지 않음."),
    ("예측일 달력", "요일 7개·월 12개 원핫 + 공휴일 여부. 주말과 공휴일은 구분."),
    ("예측일 경기", "수집 기록 유무·건수, 경기 1·2의 예정 시각·홈팀·원정팀. 팀은 원핫, 시각은 자정부터 분과 존재 표시."),
])
p("신경망의 과거 인원은 승차·하차 각각 표준화합니다. 기준과 팀 범주는 2024년 자료에서만 만들며, 예측 정답은 원래 인원 단위로 유지합니다.", "small")
story.append(PageBreak())

title("02", "모델 구조와 학습 설정", "핵심 구조·수치는 사용자와 합의 / 세부 초기값은 위임에 따라 선정")
table([
    ("랜덤포레스트", f"트리 <b>{C['rf']['n_estimators']}개</b>, 최대 깊이 <b>{C['rf']['max_depth']}</b>, 리프 최소 사례 <b>{C['rf']['min_samples_leaf']}개</b>.<br/>과거 기록을 펼친 값 + 예측일 정보 → 36개 값 동시 예측."),
    ("MLP", "과거 기록을 펼친 값 + 예측일 정보 → <b>64 → 32뉴런</b>(ReLU) → 36개 선형 출력."),
    ("LSTM", "시간대별 과거 입력 → <b>32유닛 1층</b> → 마지막 은닉 출력에 예측일 정보 결합(A안) → <b>32뉴런</b>(ReLU) → 36개 선형 출력."),
])
p("모델 선정 이유와 확인할 한계", "h")
p("<b>RF:</b> 표 형태 입력의 비선형 관계를 학습하는 비교 기준. 시간 흐름을 순차 처리하지 않고 드문 혼잡 구간을 평균화할 수 있습니다.<br/><b>MLP:</b> 같은 정보를 사용하는 기본 신경망. 입력 기간이 길면 가중치 수가 늘고, 시간 순서를 구조적으로 학습하지는 않습니다.<br/><b>LSTM:</b> 시간대 사이 변화 흐름을 순차 학습. 계산 부담과 긴 기간의 정보 손실 가능성을 확인합니다.")
p("MLP·LSTM 공통 학습 설정", "h")
table([
    ("학습 손실", "<b>MAE</b>. 승차·하차·18개 시간대 동일 비중. 학습 중에는 보정 전 원래 출력을 사용."),
    ("최적화 / 배치", f"<b>Adam / 학습률 {C['adam']['learning_rate']} 고정 / 배치 {C['training']['batch_size']}</b>. 학습률 자동 감소 없음."),
    ("횟수 / 종료", f"최대 <b>{C['training']['epochs']}에포크</b>, 매 에포크 검증. <b>{C['training']['patience']}회 연속 개선 없음</b>이면 종료. min_delta=0, 검증 MAE 최저 모델 복원."),
    ("재현성", "시드 42. 학습 사례는 섞되 사례 내부 시간순은 유지. 단일 시드 결과이므로 반복 실험의 안정성까지 검증한 것은 아님."),
])
p("위임받아 선정한 세부값", "h")
p("신경망 예정 시각은 분/1,440, RF는 원래 스케일을 사용합니다. RF 분할 기준은 squared_error이며 평가는 MAE입니다. 초기 Dropout·L1/L2·BatchNorm·기울기 절단은 사용하지 않습니다. 초기화와 Adam/LSTM 세부값은 DEC-022 및 실행 구성에 기록했습니다.", "small")
p("LSTM은 3·7·14일에 각각 54·126·252개 관측 시점을 읽습니다. 00~06시 관측 공백을 0명으로 채우지 않으며, 24시간 연속 관측으로 설명하지 않습니다.", "small")
story.append(PageBreak())

title("03", "평가 원칙과 팀 확인 요청", "실험 전 확인할 내용 / 후속 튜닝과 한계")
p("평가와 후속 튜닝", "h")
table([
    ("선택 지표", "<b>검증 승차 MAE와 하차 MAE의 평균</b>. 음수 예측은 0으로 보정하고 소수점 유지. 모델 선택·조기 종료 모두 같은 보정 후 지표 사용."),
    ("함께 볼 결과", "보정 전 MAE·음수 예측 비율, 승차·하차 RMSE, 월별·note 상태별 성능과 표본 수, 학습 곡선·시간."),
    ("튜닝 원칙", "2025년 상반기 검증 결과로 조정. 변경 이유·조건·결과를 기록하며 하반기 최종 평가로 반복 선택하지 않음."),
    ("후속 후보", "LSTM에 예측일 정보를 매 시점 함께 넣는 B안, 로그 입력 변환, 손실·모델 크기·학습률 등. 다른 행사 정보는 향후 확장."),
])
p("반드시 함께 이해해야 할 데이터 한계", "h")
p("<b>1.</b> 경기 기록 없음은 ‘실제 비경기일’이 아니라 <b>미확인</b>입니다. 수집 건수에는 취소 기록도 포함됩니다.")
p("<b>2.</b> <b>gamecenter_status는 사용하지 않습니다.</b> 경기 상태의 사후 분석은 note만 사용합니다. note와 점수·관중·실제 경기 시각 등 사후 결과는 모델 입력에서 제외합니다.")
p("<b>3.</b> 일정 수집본이 예측 전날 공개되어 있던 내용인지는 미검증입니다. 전체 일정의 완전성도 미검증이므로 <b>사후 수집 일정 기반 실험</b>이라는 한계를 명시합니다.")
p("<b>4.</b> 전날 지하철 기록을 다음 날 예측 전에 확보할 수 있다는 가정입니다. 실제 자료 공개 지연은 아직 확인하지 않았습니다.")
p("준비 상태와 검토 요청", "h")
p("전용 CPU 환경·버전 고정, 전처리·학습 코드 준비 완료. 단위 테스트 6개와 실제 자료 9개 조합 사전 점검을 통과했습니다. <b>실제 학습, 성능 측정, 학습부터 저장·복원까지 전체 실행 검증은 아직 하지 않았습니다.</b>")
table([
    ("팀 확인 요청", "① 입력 정보·분할·평가 기준에 수정할 부분이 있는지<br/>② 현재 설정으로 1차 실험을 시작해도 되는지<br/>③ 알고 있는 추가 데이터 문제나 누락이 있는지"),
    ("회신 예시", "‘현재 설정으로 1차 진행 동의’ 또는 ‘수정 요청: 항목 / 이유’. 실제 회신 후 검토 완료로 기록."),
    ("목표 일정", "11/3(화) 팀 확인 후 1차 실행·검증 / 11/4(수) 결과·문제점 정리 및 보완 논의. 완료를 보장하는 일정은 아님."),
])
p("기준: configs/initial.json, DEC-001~022, docs/RUNBOOK.md#setup-verification<br/>코드 기준: 1d9a6b5 / 브랜치: feat/initial-experiment-setup<br/><link href='https://github.com/sangmin65/jamsil-subway-demand-forecasting'>github.com/sangmin65/jamsil-subway-demand-forecasting</link>", "small")


def footer(c, doc):
    c.setStrokeColor(colors.HexColor("#D9E3E8"))
    c.line(44, 40, 551, 40)
    c.setFont("Korean", 8)
    c.setFillColor(GRAY)
    c.drawString(44, 26, "인공지능공학 3팀 | 팀 검토용 · 학습 미실행 | 2026.11.02")
    c.drawRightString(551, 26, f"{doc.page} / 3")


OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=(595.28, 841.89), leftMargin=44, rightMargin=44,
                        topMargin=38, bottomMargin=52, title="인공지능공학 3팀 1차 실험 설정 검토", author="인공지능공학 3팀")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)

# 종합운동장역 지하철 수요 예측

2024~2025년 종합운동장역 승하차 및 KBO 경기 데이터를 활용한 인공지능공학 팀 프로젝트입니다. 여러 모델을 비교해 다음 날 시간대별 승차·하차 수요를 예측합니다.

## 목표와 상태

- 2026-11-09 [EXP-009 단순 기준·경기 정보 비교](reports/experiments/EXP-009_ablation.md) 완료. 동일 검증 181일에서 지난주 MAE 379.43, 학습 요일·시간대 평균 450.74명. 5시드 평균은 RF 경기 포함/제외 256.31/389.97, LSTM 포함/제외 273.90/300.31명이며 두 모델 모두 5/5시드에서 경기 포함이 낮았습니다. 보충 행사 자료와 하반기 최종 평가는 사용하지 않았습니다.
- 야구 입력은 수집 기록 유무·건수, 경기 1·2 예정 시각과 홈팀·원정팀입니다. 별도 인기도 점수와 note·gamecenter_status는 입력에서 제외합니다. 팀은 원핫, 시각은 분과 존재 표시를 사용합니다.
- 1차 달력 입력은 요일·월·공휴일 여부입니다. 공휴일 참조 목록은 `data/reference/holidays_2024_2025.json`에 있습니다.
- 1차는 확보한 야구 정보를 중심으로 진행합니다. 콘서트·다른 행사 정보는 향후 발전 계획이며, 야구 경기일만으로 날짜를 제한하지 않습니다.
- 데이터 기간: **2024-01-01~2025-12-31** (합의 완료)
- **2026-11-04 수요일까지 1차 실행**, 부족한 자료는 이후 보충합니다.
- 여러 모델의 검증 성능을 비교해 최종 모델을 선택합니다.
- 1차 대상은 **2호선**, 학습 모델은 **랜덤포레스트·MLP·LSTM**으로 확정했습니다. 이후 사용자 승인으로 단독 1D CNN을 추가했습니다(EXP-003).
- 오늘 23~24시까지 기록 확보를 가정하고 다음 날 06~24시의 승차·하차 총 36개 값을 예측합니다. 실제 자료 공개 지연은 미검증입니다.
- 과거 **3일·7일·14일**을 세 모델에 각각 적용해 총 **9개 기본 조합**을 비교합니다.
- 분할: **2024년 학습 / 2025년 상반기 검증 / 하반기 최종 평가**. 공통 과거 입력과 탐색일 제외 적용 후 대상은 352/181/178일입니다.
- 경기 기록이 없는 날짜는 **미확인**으로 유지합니다.
- 모델 선택은 전체 검증 날짜의 승차·하차 평균 MAE입니다. RMSE·월별·note 상태별 성능도 보고합니다.
- 주요 설정은 사용자와 결정했으며 조정 가능한 초기 세부값은 DEC-022의 위임에 따라 선정했습니다. 전체 설정의 팀 검토 후 학습합니다.
- 2026-11-03 팀 확인 완료를 전달받아 9개 기본 조합의 1차 학습·검증을 완료했습니다. 현재 검증 최저는 **RF 3일, MAE 255.87명**입니다. 하반기 최종 평가는 미실행이며, 자세한 결과와 한계는 [EXP-001](reports/experiments/EXP-001_first-run.md)을 참고하세요. [실행 안내](docs/RUNBOOK.md)와 `configs/initial.json`으로 실행 조건을 확인할 수 있습니다.

## 구조

```text
.github/             PR·이슈 양식
configs/             승인된 실험 설정
data/raw/            원본 데이터 (Git 제외)
data/processed/      가공 데이터 (Git 제외)
data/external/       추가 자료 (Git 제외)
notebooks/           탐색·학습·평가 노트북
src/                 재사용할 처리·학습·평가 코드
outputs/             로컬 모델·예측·로그 (Git 제외)
reports/experiments/ 실험 기록
reports/decisions/   결정 기록
reports/metrics/     공유할 소규모 성능표
reports/figures/     공유할 핵심 그래프
docs/                운영 원칙·양식·점검 결과
```

## 시작 순서

1. [협업 규칙](CONTRIBUTING.md)과 [실험 원칙](docs/EXPERIMENT_GUIDE.md#policy)을 읽습니다.
2. [데이터 안내](docs/DATA_DICTIONARY.md#data-preparation)에 따라 원본을 준비합니다. 아직 이 저장소에 복사하지 않았습니다.
3. [실행 안내](docs/RUNBOOK.md)에 따라 Python 3.12 가상환경과 고정된 의존성을 설치합니다. 사전 점검은 학습 없이 수행할 수 있습니다.
4. 모델·분할·지표·설정을 협의하고 [결정 양식](docs/EXPERIMENT_GUIDE.md#decision-template)에 남깁니다.
5. 승인된 조건으로 구현·실행하고 [실험 양식](docs/EXPERIMENT_GUIDE.md#experiment-template)에 실제 결과를 기록합니다.

## 문서

- [실험 가이드: 원칙·초기 설정·기록 양식](docs/EXPERIMENT_GUIDE.md)
- [결정 기록: 범위·분할·평가](reports/decisions/scope.md)
- [결정 기록: 입력 변수·전처리](reports/decisions/features.md)
- [결정 기록: 학습 설정·실행 승인](reports/decisions/training.md)
- [팀원 제공 데이터 사전](docs/DATA_DICTIONARY.md)
- [데이터 수집·검증 보고서](reports/data_review.md)
- [1차 실험 결과](reports/experiments/EXP-001_first-run.md)
- [하루별 1D CNN 결과](reports/experiments/EXP-003_cnn-daily.md)
- [커밋·브랜치·PR 규칙](CONTRIBUTING.md)
- [AI 작업 지침](AGENTS.md)

검증 데이터로 모델을 선택하고 최종 평가 데이터는 선택 후 사용합니다. 성능 개선 여부와 관계없이 실제 시행착오와 한계를 보고합니다. 승하차 수요를 역내 동시 체류 인원이나 안전 등급으로 해석하지 않습니다.

## 현재 상태와 다음 작업 (2026-11-06)

- [EXP-002](reports/experiments/EXP-002_epoch1000.md): MLP·LSTM 3·7·14일 최대 1,000회·patience 50 비교 완료.
- [EXP-003](reports/experiments/EXP-003_cnn-daily.md): 하루별 1D CNN 3·7·14일 비교 완료.
- [EXP-004](reports/experiments/EXP-004_tuning-round1.md): RF 최소 잎 크기·신경망 학습률 9개 조건 완료. 전체 검증 최선은 RF 3일 MAE 255.87명, 신경망 최선은 LSTM 14일·학습률 0.003의 278.57명입니다.
- [EXP-005](reports/experiments/EXP-005_dropout-gpu.md): GPU MLP 3일·CNN 7일 각각 Dropout 0.0·0.1·0.2, 총 6개 조건 및 모델 복원 검증 완료. MLP는 Dropout 없음 290.31명, CNN은 0.1에서 325.91명으로 모델별 최선입니다. CNN은 동일 GPU 기준 대비 0.76% 개선됐습니다. 단일 시드 결과로 전체 모델 순위는 바뀌지 않았습니다.
- [EXP-006](reports/experiments/EXP-006_lstm-gpu-epochs.md): LSTM 14일 GPU 최대 1000·2000회 비교 및 모델 복원 검증 완료. MAE 277.92→275.65명으로 0.82% 개선됐습니다. 최대 2000회 조건은 최선 1283회·종료 1333회이며 patience 50으로 조기 종료됐습니다. 공통 1000회 검증 MAE는 정확히 일치했습니다. 현재 신경망 최선은 이번 LSTM 275.65명이며 전체 최선은 RF 255.87명입니다.
- [EXP-007 모델 용량 비교](reports/experiments/EXP-007_capacity.md)에서 RF 단일 설정 3개와 신경망 폭·깊이 6개를 완료했습니다. 검증 최저 MAE는 RF 3일·최소 잎 2의 255.80명으로 기존 최소 잎 3보다 0.07명 낮았습니다. 신경망 6개 변형은 모두 각 기준보다 오차가 컸습니다. 신경망 최선은 LSTM 14일·32유닛 한 층의 275.65명입니다.
- 최종 모델 선택 뒤 하반기 최종 평가 절차를 준비합니다. 현재까지 하반기 최종 평가는 미실행입니다.
- 팀원 초대·원격 협업 설정, 데이터 공유 방식·이용 조건은 별도 확인이 필요합니다.
- 최종 제출일과 발표 시간은 실제 공지를 확인해야 합니다. 계획서는 25분, OT는 30분 내외로 안내되어 있습니다.

현재 상태는 이 README, 실행 방법은 RUNBOOK, 실제 설정은 configs의 실험별 JSON, 선택 근거와 결과는 reports에서 관리합니다.

GPU 환경은 WSL2·Ubuntu 24.04·TensorFlow 2.21.0·RTX 5060입니다. 실제 LSTM 학습 및 테스트를 확인했고 Dropout 구현 후 테스트 8개가 통과했습니다. LSTM 단기 비교에서 같은 WSL CPU 대비 학습+검증 에포크 중앙값이 5.836초→0.618초(약 9.44배)로 줄었습니다. 모델별 속도 향상을 보장하는 수치는 아닙니다. 기존 Windows CPU 실험과 환경이 달라 정확도 비교 시 GPU 기준 조건도 함께 실행합니다. 실행은 tools/run_wsl_python.ps1, 재현 버전은 requirements-gpu-wsl-lock.txt, 상세 설치·측정 이력은 [RUNBOOK](docs/RUNBOOK.md)을 참고합니다.

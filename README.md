# 종합운동장역 지하철 수요 예측

2024~2025년 종합운동장역 승하차 및 KBO 경기 데이터를 활용한 인공지능공학 팀 프로젝트입니다. 여러 모델을 비교해 다음 날 시간대별 승차·하차 수요를 예측합니다.

## 목표와 상태

- 데이터 기간: **2024-01-01~2025-12-31** (합의 완료)
- **2026-11-04 수요일까지 1차 실행**, 부족한 자료는 이후 보충합니다.
- 여러 모델의 검증 성능을 비교해 최종 모델을 선택합니다.
- 1차 대상은 **2호선**, 학습 모델은 **랜덤포레스트·MLP·LSTM**으로 확정했습니다. 추가 학습 모델은 보류합니다.
- 입력 기간, 예측 시점·출력, 분할 날짜, 선택 지표, 기준 예측 구성, 파라미터·손실 가중치는 설명과 사용자 답변 후 확정합니다.
- 현재는 기본 구조·운영 문서 준비 단계입니다. 학습 코드·모델·측정된 성능은 없습니다.

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

1. [협업 규칙](CONTRIBUTING.md)과 [실험 원칙](docs/EXPERIMENT_POLICY.md)을 읽습니다.
2. [데이터 안내](data/README.md)에 따라 원본을 준비합니다. 아직 이 저장소에 복사하지 않았습니다.
3. Python·패키지·CPU/GPU 환경을 확인하고 [환경 양식](docs/templates/ENVIRONMENT_TEMPLATE.md)에 기록합니다. 패키지 버전과 설치 방법은 아직 확정하지 않았습니다.
4. 모델·분할·지표·설정을 협의하고 [결정 양식](docs/templates/DECISION_TEMPLATE.md)에 남깁니다.
5. 승인된 조건으로 구현·실행하고 [실험 양식](docs/templates/EXPERIMENT_TEMPLATE.md)에 실제 결과를 기록합니다.

## 문서

- [현재 상태와 할 일](docs/PROJECT_STATUS.md)
- [커밋·브랜치·PR 규칙](CONTRIBUTING.md)
- [AI 작업 지침](AGENTS.md)

검증 데이터로 모델을 선택하고 최종 평가 데이터는 선택 후 사용합니다. 성능 개선 여부와 관계없이 실제 시행착오와 한계를 보고합니다. 승하차 수요를 역내 동시 체류 인원이나 안전 등급으로 해석하지 않습니다.

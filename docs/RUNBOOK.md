# 1차 실행 준비와 명령

저장소 루트의 PowerShell에서 실행합니다. 원본은 저장소 밖 기존 위치에서 읽으며 수정·복사하지 않습니다. `configs/initial.json`은 실제 초기 수치, DEC-022는 코드에 명시된 전처리·초기화 등 고정 정책입니다. 이 정책을 바꾸면 코드·결정 기록도 함께 변경합니다.

초기 수치는 설정 JSON에서 변경하고 전처리·초기화·출력 정책 변경은 관련 코드와 결정 기록에 반영합니다. 결정은 `reports/decisions/DEC-XXX_주제.md`, 실험은 `reports/experiments/EXP-XXX_주제.md`에 해당 양식을 이용해 기록합니다. 승인 근거와 미확인 사항을 구분합니다.

환경 정보는 실행 보고서의 `environment`와 `requirements-lock.txt`에 기록하므로 별도 빈 환경 양식은 사용하지 않습니다. Python·패키지 버전, OS, 실제 인식 장치를 확인하며 설치 성공과 학습 성공을 구분합니다. 환경 문제가 생기면 CPU·RAM·GPU 드라이버·제약과 실제 실행 결과를 실험 기록에 추가합니다.

## 환경 설치

Python 3.12를 사용합니다. 이 컴퓨터에는 저장소의 `.venv`에 환경을 준비합니다. 새 컴퓨터에서는 다음과 같이 설치합니다.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
./.venv/Scripts/python.exe -m pip check
```

`requirements.txt`는 직접 의존성, `requirements-lock.txt`는 실제 확인한 환경의 전체 버전 목록입니다. Windows 기본 환경은 CPU 실행 기준이며, GPU 사용을 보장하지 않습니다.

## 학습 없는 사전 점검

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -v
./.venv/Scripts/python.exe -m src.experiment preflight --source '../인공지능공학 과제용/종합운동장역_2024-2025_모델입력_경기결합.csv' --kbo-source '../인공지능공학 과제용/KBO원천데이터_정리.csv' --output outputs/preflight-001
```

출력 경로는 매번 새 이름을 사용합니다. 기존 실행을 덮어쓰지 않습니다. 원본 무결성·완전한 시간대 격자·실제 입력의 유한값·학습/검증/평가 대상 수를 검사합니다. 신경망은 모델 생성과 합성 0 입력의 순전파만 확인합니다. 모델 학습, 실제 검증·평가 예측 및 성능 측정은 하지 않습니다.

저장물: 입력 원본/설정 해시, 환경 정보, 표준화 기준·팀 범주, 입력 특징 이름, 대상 날짜, 모델 구성·파라미터 수, 조건별 준비 결과. 원본의 사후 일정 사용과 미검증 한계를 보고서에 표시합니다.

## 팀 검토 후 1차 학습

실제 팀 검토가 완료되면 `configs/initial.json`의 `team_review`에 `approved: true`, 검토자, 확인 시점을 기록합니다. 단순 실행을 위해 허위로 승인 표시하지 않습니다. 2026-11-03 사용자 전달에 따른 팀 확인 완료를 기록하고 EXP-001을 실행했습니다.

```powershell
./.venv/Scripts/python.exe -m src.experiment train --source '../인공지능공학 과제용/종합운동장역_2024-2025_모델입력_경기결합.csv' --kbo-source '../인공지능공학 과제용/KBO원천데이터_정리.csv' --output outputs/first-run-001
```

9개 기본 조합을 학습하고 검증 결과만 비교합니다. RF는 squared_error로 학습하되 MAE로 평가합니다. 신경망은 원래 출력의 MAE로 학습하며 보정 후 검증 MAE로 체크포인트·조기 종료를 결정합니다. 마지막 배치를 버리지 않습니다. 실제 모델·검증 예측·학습 곡선·월별 및 note 상태별 지표·시간을 저장합니다. note는 사후 평가 그룹에만 사용하고 입력에는 넣지 않습니다. 기록 없는 날은 `no_record_unknown`입니다.

시드가 하나이므로 모델 간 차이가 난수 변동에 견고하다고 주장하지 않습니다. 200회 상한에 도달하면 수렴 여부를 검토합니다. 원래 인원 출력의 학습이 느릴 가능성도 확인합니다. 오류는 `failure.json`으로 남기고 실패 조건을 유효한 모델 성능으로 집계하지 않습니다. 부분 완료 결과는 유지합니다.

최종 평가 날짜는 분할 점검에만 사용하며 학습 명령은 최종 평가 예측·점수를 계산하지 않습니다. 최종 모델 선택과 평가 절차 확정 후 별도 실행 경로를 준비합니다. 자동으로 전체 기간에 재학습하지 않습니다.

## 구현 근거

- [Keras LSTM](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Keras Adam](https://keras.io/api/optimizers/adam/)
- [scikit-learn RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

학습 중 검증 지표는 명시적인 콜백에서 계산합니다. 최선 모델 복원·동점 유지·20회 기다림은 단위 테스트로 별도 확인합니다.

## EXP-001 결과 재검증과 그래프

```powershell
./.venv/Scripts/python.exe tools/summarize_first_run.py --run outputs/first-run-2026-11-03 --source '../인공지능공학 과제용/종합운동장역_2024-2025_모델입력_경기결합.csv'
```

모든 저장 모델을 재로딩하고 검증 예측·지표·최선 에포크를 확인한 뒤 공유용 집계를 생성합니다. 최종 평가는 하지 않습니다. `--verify-partial`은 완료된 조건만 확인하고 최종 보고서를 만들지 않습니다.

그래프는 학습 환경과 분리된 `outputs/plot-env`에 matplotlib 3.10.7을 설치해 `tools/plot_first_run.py`로 생성했습니다. Windows 맑은 고딕 글꼴을 사용합니다. 이 보조 환경은 Git에 포함하지 않으며 학습 의존성을 변경하지 않았습니다.

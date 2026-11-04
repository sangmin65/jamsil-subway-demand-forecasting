# 1차 실행 준비와 명령

저장소 루트의 PowerShell에서 실행합니다. 원본은 저장소 밖 기존 위치에서 읽으며 수정·복사하지 않습니다. `configs/initial.json`은 실제 초기 수치, DEC-022는 코드에 명시된 전처리·초기화 등 고정 정책입니다. 이 정책을 바꾸면 코드·결정 기록도 함께 변경합니다.

초기 수치는 설정 JSON에서 변경하고 전처리·초기화·출력 정책 변경은 관련 코드와 결정 기록에 반영합니다. 결정은 `reports/decisions/`의 주제별 문서에 DEC 번호로 추가하고, 실험은 `reports/experiments/EXP-XXX_주제.md`에 해당 양식을 이용해 기록합니다. 승인 근거와 미확인 사항을 구분합니다.

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

<a id="setup-verification"></a>

## 2026-11-02 사전 점검 이력

아래는 최초 학습 전의 확인 기록입니다. 이후 팀 승인과 실제 학습·저장·복원 검증은 [EXP-001](../reports/experiments/EXP-001_first-run.md)에서 완료했습니다. 아래 과거의 미실행 표현을 현재 상태로 해석하지 않습니다.

- 날짜: 2026-11-02
- 상태: 설정·코드·환경 준비 및 학습 없는 사전 점검 완료. 팀 검토·실제 학습 미완료.
- 실행 안내: `docs/RUNBOOK.md`

### 구현

`configs/initial.json`에 초기 수치를 기록했습니다. 전처리, RF·MLP·LSTM 생성, 원래 인원 MAE 학습, 음수 보정 후 검증 MAE에 따른 체크포인트·조기 종료, 학습곡선과 note/월별 검증 평가, 원본·코드·설정 해시와 환경 기록을 구현했습니다. 자동 최종 평가·전체 기간 재학습은 하지 않습니다.

표준화와 팀 범주는 2024년만 사용합니다. 과거 입력은 대상일 직전까지로 구성합니다. 원본 lag 열을 추가 입력하지 않으며 3일 조건에 7일 lag를 넣지 않습니다. 경기 일정은 사후 수집본이라는 한계를 유지합니다.

### 실제 확인 결과

- `pip check`: 의존성 충돌 없음.
- 단위 테스트 6개 통과: 미래 데이터 변조에도 학습 통계·범주 불변, 시간 경계·정답 정렬·공통 날짜, 미등록 팀과 기록 없음 구분, 보정 MAE와 동일 비중, 조기 종료 동점·기다림 초기화, 팀 검토 확인.
- 실제 자료 사전 점검: RF·MLP·LSTM × 3·7·14일, 9개 조건 통과.
- 모든 조건 학습/검증/평가 대상 수 352/181/178일 일치.
- 부가 입력 51개 값. 신경망은 합성 0 입력 2개에 대해 유한한 36개 출력을 확인했습니다.
- 팀 미검토 상태에서 실제 train CLI 호출이 모델 학습 전에 오류로 차단됨을 확인했습니다. 이는 의도한 동작이며 실제 검토 상태는 변경하지 않았습니다.
- 마지막 사전 점검에서 `/physical_device:CPU:0`만 확인되었습니다. 두 차례 점검 사이 원본·참조·설정 해시는 동일했습니다. 환경·코드 해시를 포함한 상세 결과는 로컬 `outputs/preflight-002/report.json`에 있습니다.

| 모델 | 3일 | 7일 | 14일 |
|---|---:|---:|---:|
| MLP 파라미터 수 | 13,508 | 22,724 | 38,852 |
| LSTM 파라미터 수 | 8,356 | 8,356 | 8,356 |

환경은 Python 3.12.14 / TensorFlow 2.20.0 / Keras 3.15.1 / scikit-learn 1.7.2 / pandas 2.3.3 / NumPy 2.5.4입니다. 전체 버전은 `requirements-lock.txt`에 기록했습니다. 가상환경과 모델·원본·로컬 상세 출력은 Git에서 제외합니다.

### 범위와 남은 확인

실제 모델의 fit은 한 번도 실행하지 않았습니다. 따라서 수렴·학습 시간·예측 성능 및 실제 학습부터 저장·복원까지 전체 실행이 검증되었다고 주장하지 않습니다. 현재는 전처리·모델 생성·순전파와 정책 로직을 확인한 상태입니다. 첫 승인 실행에서 학습 경로와 저장·복원도 확인합니다.

설치 완료 전에 수행한 TensorFlow import는 설치 중인 파일 때문에 실패했습니다. 설치 완료 후 사전 점검에서는 정상 로드되었습니다. 합성 테스트의 NumPy 문자열을 pandas Timestamp에 전달하던 타입 오류도 수정 후 전체 테스트 통과를 확인했습니다. 실제 성능 실패나 데이터 보정으로 기록하지 않습니다.

예측일 일정의 당시 가용성, 공식 목록 완전성, 실제 지하철 자료 공개 시점은 여전히 미검증입니다. 팀 확인과 모델 선택 후 최종 평가 절차도 남아 있습니다.

## EXP-002: 최대 1,000에포크·patience 50

DEC-024에 따라 MLP 3일과 LSTM 3·7·14일을 같은 시드로 처음부터 실행합니다. 두 명령을 순서대로 실행하며 각 출력 디렉터리는 새 경로여야 합니다. 기존 EXP-001은 보존합니다.

```powershell
./.venv/Scripts/python.exe -m src.experiment train --config configs/epoch1000_mlp3.json --source '../인공지능공학 과제용/종합운동장역_2024-2025_모델입력_경기결합.csv' --kbo-source '../인공지능공학 과제용/KBO원천데이터_정리.csv' --output outputs/epoch1000-2026-11-03/mlp3
./.venv/Scripts/python.exe -m src.experiment train --config configs/epoch1000_lstm.json --source '../인공지능공학 과제용/종합운동장역_2024-2025_모델입력_경기결합.csv' --kbo-source '../인공지능공학 과제용/KBO원천데이터_정리.csv' --output outputs/epoch1000-2026-11-03/lstm
```

로컬 순차 실행기의 `status.json`과 `mlp3.log`·`lstm.log`는 `outputs/epoch1000-2026-11-03/`에 저장합니다. 종료 상태와 개별 실행의 `report.json`·`failure.json`을 함께 확인합니다. 9개 조건을 전제로 한 EXP-001 요약 스크립트를 이 4개 조건에 그대로 적용하지 않습니다.

## EXP-003: 하루별 단독 1D CNN

```powershell
./.venv/Scripts/python.exe -m src.experiment preflight --config configs/cnn_daily.json --source '../인공지능공학 과제용/종합운동장역_2024-2025_모델입력_경기결합.csv' --kbo-source '../인공지능공학 과제용/KBO원천데이터_정리.csv' --output outputs/cnn-preflight-2026-11-04
./.venv/Scripts/python.exe -m src.experiment train --config configs/cnn_daily.json --source '../인공지능공학 과제용/종합운동장역_2024-2025_모델입력_경기결합.csv' --kbo-source '../인공지능공학 과제용/KBO원천데이터_정리.csv' --output outputs/cnn-daily-2026-11-04
./.venv/Scripts/python.exe tools/summarize_first_run.py --run outputs/cnn-daily-2026-11-04 --source '../인공지능공학 과제용/종합운동장역_2024-2025_모델입력_경기결합.csv' --verify-partial
```

재실행 시 출력은 새 경로를 사용합니다. 마지막 명령은 완료된 조건의 모델 복원·검증 예측을 확인하며 EXP-001 보고서를 덮어쓰지 않습니다. CNN 결과는 EXP-003에 기록합니다.

## EXP-004: 승인한 9개 튜닝 조건 순차 실행

```powershell
./.venv/Scripts/python.exe -u tools/run_tuning_batch.py --source '../인공지능공학 과제용/종합운동장역_2024-2025_모델입력_경기결합.csv' --kbo-source '../인공지능공학 과제용/KBO원천데이터_정리.csv' --output outputs/tuning-round1-2026-11-04 --commit-results
```

출력은 새 경로여야 합니다. 명세와 기준 전체 설정의 일치를 검사하고, 각 조건의 전체 설정을 로컬에 기록합니다. 기준 모델과 신규 모델을 재로딩해 검증하며 기준 대비 전체·월별·note 상태별 오차를 비교합니다. `status.json`은 완료된 검증 결과와 현재 조건을 담고, 조건별 로그도 출력 디렉터리에 저장합니다. 9개 완료 후 EXP-004와 공유 JSON을 생성합니다. `--commit-results`는 이 두 결과 파일만 커밋하고 시작한 브랜치에 push하며 브랜치가 바뀌면 자동 커밋을 중단합니다. 코드·원본 변경 또는 학습·검증 오류 시 중단합니다.

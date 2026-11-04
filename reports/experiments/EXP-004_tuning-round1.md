# EXP-004: 모델별 첫 파라미터 튜닝

- 날짜: 2026-11-04
- 상태: 사용자 승인 완료, 9개 조건 실행 준비. 완료 결과는 아직 없음.
- 승인·비교 원칙: [DEC-026](../decisions/training.md#dec-026)
- 설정: `configs/tuning_round1.json`
- RF 3일: 최소 잎 크기 1·5·10. MLP 3일·CNN 7일·LSTM 14일: 학습률 0.0003·0.003.
- 기준: RF leaf 3, 각 신경망 learning_rate 0.001. 나머지 설정은 각 기준 실행과 동일.
- 순차 실행·검증·비교 도구: `tools/run_tuning_batch.py`
- 로컬 진행 상태: `outputs/tuning-round1-2026-11-04/status.json`. 완료된 조건만 results에 추가됩니다.
- 9개 조건 완료 후 이 문서와 공유 지표 JSON을 생성하고 결과 파일만 커밋·push합니다. 오류 시 로컬 상태와 로그를 남기고 중단합니다.
- 하반기 최종 평가 미사용. 단일 시드, 동일 검증 데이터에서의 탐색입니다.

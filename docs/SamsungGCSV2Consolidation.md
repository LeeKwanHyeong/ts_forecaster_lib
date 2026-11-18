# 삼성 GCS 실험 코드의 v2 통합

## 현재 기준선

- 작업일: 2026-11-18
- 통합 대상 저장소: `ts_forecaster_lib`
- 통합 대상 브랜치: `ts_forecaster_lib_v2`
- v2 시작 커밋: `391fb996168aaa0aeca4bbc514b4d4adcb4042cc`
- 삼성 원본 커밋: `a09203ee83feac3785a65d97cd5069ce4610e750`
- 구현 위치: 원본과 분리한 `integration/samsung-gcs-into-v2` worktree
- 기존 `samsung_gcs`의 미커밋 파일과 index 보존
- 모델 본체, 공통 학습 엔진, 공통 API, 기존 체크포인트 규격, 버전 `0.2.0` 유지

## 통합 방식

v2를 공통 개발 기준으로 유지하고 삼성 전용 도구만 선별 이식하는 방식.
이식한 코드와 제외 사유를 확정한 뒤 `ours` 전략의 병합 커밋으로 삼성 원본의 이력도 연결하는 구성.
이 전략은 삼성 파일로 v2를 덮어쓰는 동작이 아니라 검증한 v2 트리를 보존하면서 원본 커밋을 병합 이력에 기록하는 동작.
단순 병합 모의 검사에서 확인한 8개 충돌 파일의 구형 구현을 복원하지 않는 결정.

| 삼성 변경 | 처리 |
|---|---|
| `52ce0f3`, `7cb4384`, `217d3b3`, `c68aba3` | 기존 v2 공개 AB 실행기와 sweep 유지, 시각화·결과 집계 및 기존 파일명 진입점 추가 |
| `190b01d`, `73e79ec` | 기존 v2 제외 정책 유지, 생성 artifact나 테스트 fixture를 일괄 삭제하지 않는 범위 |
| `cccd52e` | v2에 이미 반영된 모델 문서·보고서 유지 |
| `22a9fb5` | v2의 동일한 TinyTimeMixer 구현 유지 |
| `a09203e` | v2의 대응 커밋 `391fb99` 유지, 학습 메타데이터 처리 보존 |
| 미커밋 `patchtst_loss_comparison` | 진단용 도구와 과거 결과 문서만 복사·정리, 생성 데이터와 체크포인트 제외 |
| 미커밋 신규 모델 이식 코드 | v2의 기존 정식 모델/API와 중복되므로 재이식 제외 |
| 로컬 환경 파일, FFT/marimo 예제, SELLM 학습 노트북 | 삼성 실험 연결 범위 밖이며 원본 작업 공간에 보존 |

기존 `head_flatten` 모드, 직접 `MultiPartExoDataModule` 생성, 직접 `run_total_train_weekly` 호출, 하드코딩된 Windows 경로 복원 제외.
구형 노트북과 전체 삼성 소스는 삼성 브랜치의 기존 커밋으로 추적 가능한 상태.

## 실행 구조

```text
auto_run_architecture_hyper.py (기존 파일명의 CLI 진입점)
  -> run_samsung_gcs_patchtst_sweep.py
  -> run_exogenous_model_ab.py
  -> v2 공개 train / load_predictor
  -> 계획 주차별 long-format parquet

auto_run_architecture_hyper_validation_utils.py
  -> sweep manifest와 모든 계획 주차의 parquet 검증
  -> far_metrics.py의 8회 갱신 FAR
  -> 시드·배치·아키텍처별 요약
  -> auto_run_architecture_hyper_plot_utils.py의 그래프
```

### 계획 생성과 학습 실행

저장소 루트에서 실행하는 예제. `PYTHONPATH=src`와 필요한 개발·시각화 의존성을 갖춘 Python 사용.
`--execute`가 없으면 manifest와 실행 명령만 생성하고 학습은 수행하지 않는 기본 동작.

```bash
PYTHONPATH=src python -m model_test.exogenous_test.run_samsung_gcs_patchtst_sweep \
  --artifact-root artifacts/samsung_gcs_v2/run01 \
  --plan-weeks 202547 202548 202549 202550 202551 202552 202601 202602 \
  --architectures arch_small --batch-sizes 64 --seeds 42 142 \
  --lookback 52 --horizon 27 --num-workers 0 \
  --target-source /path/to/target.parquet \
  --exo-source /path/to/exogenous.parquet \
  --scope-label example_cohort
```

- 실제 로컬 학습은 위 명령에 `--execute`를 명시하는 별도 작업
- `--target-source`, `--exo-source`는 기존 AB 실행기의 입력 경로 선택 기능
- 명시한 외생변수 파일이 없으면 다른 데이터로 자동 대체하지 않는 정책
- `--scope-label`은 보고서 표시 이름이며 법인·사이트를 필터링하거나 권한을 부여하는 기능과 무관
- 기존 파일명 사용 시 `PYTHONPATH=src python src/model_test/exogenous_test/auto_run_architecture_hyper.py` 뒤에 동일 옵션 지정
- `202552 + 1` 같은 정수 연산 대신 실제 ISO 주차 목록 사용
- manifest 형식 버전은 2이며 동일 폴더에 다른 설정을 덮어쓰는 동작 차단
- 과거 버전 1 manifest는 별도 폴더에서 명령을 다시 생성하여 사용

### 저장 결과 평가

```bash
PYTHONPATH=src python -m model_test.exogenous_test.auto_run_architecture_hyper_validation_utils \
  --manifest artifacts/samsung_gcs_v2/run01/sweep_manifest.json \
  --output-dir artifacts/samsung_gcs_v2/run01/evaluation
```

- 평가 대상: 같은 아키텍처·배치·시드 안의 연속된 8회 예측 갱신
- 가중치: 오래된 순서부터 `0.03, 0.03, 0.07, 0.07, 0.15, 0.15, 0.25, 0.25`
- 각 target에 대한 최신 8개 계획 주차가 실제 달력에서도 연속된 경우만 사용
- 전체 계획 주차 목록에 포함된 target만 집계하며 예측 horizon 뒤쪽을 평가 기간에 임의 추가하지 않는 범위
- 부품별 8회 관측이 완성되고 모든 예측이 0이 아닌 경우만 수량 가중 집계
- 부품별 실적 불일치, 중복 행, 잘못된 ISO 주차, 미래 계획 주차, NaN/Inf, 음수 수량 차단
- 실적 null은 미관측 상태로 유지하고 0 수요로 바꾸지 않는 정책
- 음수 예측은 기본 오류 처리. 절단 평가가 필요하면 `--clip-negative` 명시
- 절단을 선택해도 `raw_forecasts.parquet`에는 원 예측을 보존하고 `evaluation_policy.json`에 처리 정책과 음수 건수 기록
- 누락 parquet는 실행 실패로 처리하고 일부 계획 주차만 성공한 결과를 완전한 평가처럼 표시하지 않는 정책
- 결과: `far_by_target_week.*`, `far_summary.*`, `evaluation_policy.json`, `plots/`

그래프의 X축은 YYYYWW 정수 크기가 아닌 관측 순서 기반 좌표 사용.
연말과 53주를 포함한 주차 라벨, 최대 tick 수, quantile 열이 없는 point 예측 그래프 지원.

## Loss 비교 도구

`src/model_test/patchtst_loss_comparison/README.md`의 독립 실행 절차 사용.
서로 다른 커밋의 native pretrain/fine-tuning을 별도 프로세스에서 실행하고 입력·설정·초기 가중치를 비교하는 진단 도구.
현재 결과를 과거 고정 문장으로 설명하지 않고 실제 summary와 지표에서 보고서를 생성하도록 정리한 구성.

`RESULTS.md`는 2026-11-15 삼성 브랜치 비교의 과거 기록이며 이번 v2 운영 성능 검증 결과와 구분.
이번 통합에서는 소규모 합성 데이터의 CPU smoke만 수행하며 고객 데이터 재학습과 대규모 성능 비교 제외.

## 검증

- 변경 전 v2: 865 passed, 6 skipped, 46 subtests passed
- 통합 집중 검증: 40 passed
- 최종 전체 검증: 903 passed, 6 skipped, 46 subtests passed
- 건너뛴 6개: NVIDIA CUDA 전용 smoke 테스트
- CLI dry-run: 8개 계획 주차의 manifest와 명령 생성 확인, 학습 미실행
- 테스트 환경: 임시 venv, Python 3.12.10, PyTorch 2.9.1, NumPy 2.1.3, Polars 1.31.0
- 기존 conda 및 프로젝트 `.venv` 변경 없음
- Hugging Face offline 설정으로 실행하며 실제 LLM 다운로드 없음
- CUDA 학습, 실제 Qwen 백본, 운영 데이터 성능, Runtime 배포 미검증
- 원본 추적·미추적 파일의 집계 SHA-256: `311b6c809a0e8f6132e2e9c0c762f6063fcd615bb1c186315bb8d03deeadad1f`
- 원본 Git status, unstaged diff, staged diff의 작업 전후 일치 확인
- 구조화된 검증 기록: `docs/evidence/SamsungGCSV2Consolidation20261118.json`

```bash
PYTHONPATH=src python -m pytest -q tests \
  src/model_test/patchtst_loss_comparison/test_common.py
```

## 남은 작업

- 다음 작업: 신규 모델은 통합된 `ts_forecaster_lib_v2`에서 별도 패키지로 개발
- 승인 필요: 운영 데이터 대규모 학습과 공용 Runtime 배포
- 이번 범위 밖: ReasonCast 구현과 예측 성능 개선 주장

# Chronos-2 27주 zero-shot 검증

## 최초 검증 기준선

- 작업 브랜치: `ts_forecaster_lib_v2`
- 시작 커밋: `599eb8b4c76db6b4dadc59a9ec9a602759cf1243`
- 범위: 독립 모델·backbone·입력 계약·한글 주석·제한된 실행 예제·테스트 추가
- 기존 모델 본체, 모델 등록부, 공통 학습 API, 공통 체크포인트 로더 변경 없음
- 공통 설정 변경은 `chronos2` 선택 의존성 추가만 포함
- 공식 `amazon/chronos-2` 가중치 사용, fine-tuning과 신규 pretrain 미수행

## 실행 환경

- 별도 임시 환경: `/private/tmp/ts-forecaster-chronos2-env-20261118`
- 기존 conda `ai_env`, `demand_engine` 및 프로젝트 `.venv` 변경 없음
- Python 3.12.10, PyTorch 2.9.1, NumPy 2.1.3, Polars 1.31.0
- Chronos forecasting 2.3.2, Transformers 4.57.6, CPU FP32, 추론 thread 2개
- 가중치 확보 후 검증은 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`로 수행
- 추론 요청의 외부 LLM/API 전송과 DB 쓰기 없음

## 가중치와 구조 검증

- 공식 revision: `29ec3766d36d6f73f0696f85560a422f50e8498c`
- 가중치와 설정 파일 모두 고정 SHA-256 검증
- 기본 점 예측 shape `(B,27,1)`, 분위수 shape `(B,27,3)` 확인
- 공식 Pipeline의 원 예측과 어댑터 결과 비교
- 부품별 독립 예측과 묶음 예측 일치 확인
- 미래 표현 `(B,2,768)`을 공식 head와 역정규화에 다시 넣어 원 예측 복원 확인
- 미래 외생변수, 결측 context, 0 수요, 상수 수요 추론 확인
- 학습 가능한 파라미터 0개, 공식 모델 가중치 갱신 미수행

모델 코드와 재현 명령: `src/modeling_module/models/Chronos2/README.md` 참고.

## 제한된 실행 결과

합성 계절·추세·간헐 시계열 3개와 기존 로컬 스냅샷의 8개 부품을 분리한 실행.
과거 52주만 예측에 전달하고 미래 27주 정답은 평가에서만 사용하는 구성.

| 입력 | 시계열 수 | 시계열당 예측 주수 | 예측 구간 | 실행 성격 |
|---|---:|---:|---|---|
| 합성 패턴 | 3 | 27 | 202601~202627 | 원 모델 작동·입력·출력 검증 |
| 로컬 parquet 표본 | 8 | 27 | 202518~202544 | 제한된 스냅샷의 원 단위 지표 점검 |

`artifacts/chronos2/`의 CSV와 summary를 증적으로 사용하며 생성 파일과 공식 가중치는 Git 제외.
| 입력 | 수요만 사용한 MAE | 수요만 사용한 WAPE | 달력 외생변수 사용 MAE | 달력 외생변수 사용 WAPE |
|---|---:|---:|---:|---:|
| 합성 패턴 3개 | 0.496371 | 2.893030% | 0.504070 | 2.937900% |
| 로컬 표본 8개 | 4.260620 | 17.680958% | 4.092381 | 16.982794% |

검증된 생성 위치:

- `artifacts/chronos2/synthetic_zero_shot_27_20261118_verified/`
- `artifacts/chronos2/local_snapshot_zero_shot_27_20261118_verified/`
- `artifacts/chronos2/official/29ec3766d36d6f73f0696f85560a422f50e8498c/`

실제 수요만 사용하는 CPU 예측 시간은 합성 3개 약 0.082초, 로컬 8개 약 0.170초로 기록.
가중치 로딩을 제외한 해당 실행의 측정값이며 일반적인 처리량이나 서비스 응답 시간 보장과 구분.
달력 변수는 합성 표본에서는 악화되고 로컬 표본에서는 개선된 결과이므로 일관된 성능 개선으로 해석하지 않는 기준.

비교용 마지막 관측값 반복 예측의 WAPE는 합성 표본 21.944277%, 로컬 표본 13.160422%로 기록.
로컬 8개 부품에서는 Chronos2가 이 단순 기준보다 낮은 성능을 보인 결과.
이번 구현·실행 완료와 운영 모델 채택 판단을 분리하며, 여러 시점·패턴의 비교 없이 우수성을 주장하지 않는 기준.

## 검증 상태

- 변경 전 기준 테스트: 903 passed, 6 skipped, 46 subtests passed
- Chronos2 집중 테스트: 47 passed
- 공식 가중치의 실제 모델 테스트: 5개 포함
- 실제 가중치를 포함한 전체 회귀: 950 passed, 6 skipped, 46 subtests passed
- 건너뛴 6개는 NVIDIA CUDA 전용 테스트
- 기본 BLAS/OpenMP 설정의 최초 전체 실행은 계산 지연으로 중단. 모델 코드는 유지하고 해당 테스트 프로세스에 `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`을 적용한 재실행에서 완료
- 위 스레드 설정은 테스트 프로세스에만 적용하며 라이브러리 import 시 전역 환경을 변경하지 않는 구성
- Chronos 선택 의존성이 없는 환경: 945 passed, 11 skipped, 46 subtests passed
- 위 11개는 CUDA 전용 6개와 명시적으로 비활성화한 공식 가중치 테스트 5개
- public wheel 로컬 빌드 및 별도 디렉토리 설치 후 선택 의존성 없이 import·모델 생성 확인
- 공식 가중치는 wheel에 포함하지 않고 코드와 사용 문서만 포함하는 구성
- 기존 추적 모델 파일의 작업 전후 집계 SHA-256 일치: `9116e0a0c4959320d95f60130ce99162a0ca3ef29473e83575ac82e77c12cce4`
- 구조화된 증적: `docs/evidence/Chronos2ZeroShot20261118.json`

## 해석과 남은 작업

- 이번 결과는 zero-shot 실행 검증이며 삼성 운영 성능 개선의 근거와 구분
- 과거부터 학습된 foundation model의 가중치를 사용했지만 이번 부품·시점의 추가 학습은 없는 의미의 zero-shot
- 로컬 데이터 출처가 운영/합성 중 무엇인지와 사전학습 데이터 중복 여부는 미확인
- 여러 시점·수요 유형의 로컬 비교는 후속 `Chronos2BalancedBacktest20261119.md`에서 완료
- 실제 운영 평가를 위한 데이터 출처 확인 및 법인별 동일 조건 비교는 별도 작업
- GPU/MPS 성능과 운영 Runtime 배포 미검증
- ReasonCast의 별도 의미 보정 모듈은 미구현. 현재 모델에 자동으로 적용된 상태가 아닌 구분
- 승인 필요: 대규모 운영 학습, 전체 데이터 실행, 공용 Runtime 배포
- 이 문서의 지표·환경은 2026-11-18 최초 실행 당시 기록이며 이후 저장소 반영 상태와 구분

## 후속 상태

- 2026-11-19 단일 작업 폴더 `ts_forecaster_lib`로 통합 완료, 별도 연결 worktree 제거
- 기존 코드·가중치·실험 결과의 해시 대조 및 v2↔삼성 브랜치 전환 확인
- 통합 폴더에서 공식 가중치 포함 전체 회귀 973 passed / 6 CUDA skipped / 46 subtests passed
- 최초 실행 증적의 경로와 수치는 당시 기록으로 보존, 현재 산출물은 기존 폴더의 동일 상대 경로에서 사용
- 커밋 대상은 구현·테스트·재현 문서·요약 증적이며 원천 데이터·공식 가중치·로컬 백업은 제외

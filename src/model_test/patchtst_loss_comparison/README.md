# PatchTST loss 수정 전후 비교 실행기

수정 전후의 **실제 저장소 학습 함수**를 각각 별도 프로세스에서 실행하는 재현용 코드.
운영 코드/DB/Runtime 변경이나 운영 체크포인트 교체를 수행하지 않는 로컬 실험.

## 파일 구성

- `common.py`: 고정 설정, 학습 이력만 이용한 층별 표본 추출, 시간 분할, 달력 외생변수, 평가 지표
- `worker.py`: 지정한 커밋의 native pretrain/fine-tuning 실행, 원래 학습 동작을 바꾸지 않는 관측 hook
- `run.py`: 데이터 스냅샷 생성, 전후 독립 프로세스 실행, 입력과 초기 가중치/설정 일치 확인
- `compare.py`: 저장된 실제 예측값 재평가, 시드별 CSV/그래프/결과 문서 생성
- `test_common.py`: 지표, ISO 53주/연말 경계, 달력 특성의 스냅샷 독립성 테스트
- `RESULTS.md`: 2026-11-15 실행 결과와 해석상의 제한
- `results/`: 원천 hash, 선택 표본, 로그, epoch별 loss, 예측값, 체크포인트 등 생성 증적. Git 제외

## 실행 환경

아래 환경과 `RESULTS.md`는 2026-11-15의 과거 비교 기록. v2에서 다시 실행할 때는 현재 의존성과 실행 환경을 별도 기록하는 기준.
이번 v2 통합 검증은 소규모 합성 데이터의 CPU smoke이며 과거 64개 부품의 전체 실험 재실행과 구분.

학습 실행 환경: 프로젝트 `.venv`의 Python 3.12.10, PyTorch 2.13.0, NumPy, Polars.
그래프 생성 환경: 로컬 conda `ai_env`의 Python 3.12.12, NumPy 2.4.0, Matplotlib 3.10.8.
학습은 전부 같은 프로젝트 `.venv`에서 CPU FP32로 실행. 그래프 환경으로 학습하거나 재평가 모델을 실행하지 않는 구분.
추가 패키지 설치, conda 환경 변경, 원격 GPU 사용 없이 수행한 실험.

## 재실행

저장소 루트에서 별도 worktree 생성. 기존 미커밋 파일을 사용하지 않는 방식.

```bash
git worktree add --detach /private/tmp/patchtst-before 22a9fb5f84c43f957d091c269999dd31f489bcf3
git worktree add --detach /private/tmp/patchtst-after a09203ee83feac3785a65d97cd5069ce4610e750
```

결과 경로는 기존 실험과 다른 새 디렉토리 지정. 실패한 실행 폴더/체크포인트를 자동 덮어쓰지 않는 정책.
v2를 비교하려면 `--after`에 해당 커밋을 checkout한 별도 worktree 경로 지정.
원천이 다른 위치에 있으면 `--target-source`와 선택적인 `--exo-source`로 명시.
외생 파일을 생략하면 파일 존재 시 교집합 정보만 기록하고 날짜에서 생성한 달력 외생변수를 사용.
기존 결과 재사용 시 코드 커밋·데이터 hash·시드·LR·설정·smoke 여부 불일치 차단.
비교 대상 `--before`, `--after`는 미커밋 변경이 없는 worktree만 허용. 원본의 작업 중인 삼성 디렉토리를 비교 대상으로 지정하지 않는 기준.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python src/model_test/patchtst_loss_comparison/run.py \
  --repo "$PWD" \
  --before /private/tmp/patchtst-before \
  --after /private/tmp/patchtst-after \
  --output "$PWD/src/model_test/patchtst_loss_comparison/results/replay"
```

`--smoke` 추가 시 검증 구간만 사용하는 짧은 실행 점검. 기본 비교와 구분하여 `smoke` 폴더에 저장.

```bash
MPLCONFIGDIR=/private/tmp/patchtst-mpl /opt/anaconda3/envs/ai_env/bin/python \
  src/model_test/patchtst_loss_comparison/compare.py \
  --results src/model_test/patchtst_loss_comparison/results/replay \
  --report src/model_test/patchtst_loss_comparison/results/replay/RESULTS.md

PYTHONPATH=src python -m pytest -q src/model_test/patchtst_loss_comparison/test_common.py
```

## 해석 기준

- Native pretrain loss의 좌표계는 affine 설정에 따라 달라지는 구조. 값의 크기만으로 전후 품질 판단 불가
- 고정 마스크·affine 제거 좌표계 재구성 오차와 최종 원 단위 예측 지표를 함께 확인
- 전후 학습 정답 구간, 난수 시드, 데이터 로더 generator seed, 설정 및 공통 초기 가중치 동일
- 정답 스케일 수정과 최소 마스크 보장을 함께 반영한 비교. affine 한 가지 요인만 분리한 실험은 아님
- 원천은 기존 로컬 스냅샷이며 운영/합성 생성 출처 미확인. 삼성 운영 실험의 재현으로 해석 불가
- 테스트 수요가 0인 비율 약 83%. 단일 시점 결과를 모든 수요 패턴과 운영 기간으로 일반화 불가
- 원 예측과 동일한 비음수 절단 후 예측은 별도 집계. 음수 예측을 숨기거나 유리한 결과만 선택하지 않는 방식

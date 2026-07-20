# Non-SELLM RTX 5090 bootstrap

이 절차는 core forecasting 모델의 private wheel을 RTX 5090에서 재현 가능하게 검증하기 위한
환경 계약입니다. SELLM 코드와 extra dependency를 wheel 또는 overlay 설치 집합에 포함하지 않습니다.

## 고정된 환경 경계

- base Python: `/opt/miniconda3/envs/ai_env/bin/python`
- Python: CPython 3.12.13
- PyTorch: `2.11.0+cu130`
- CUDA runtime: `13.0`
- GPU: `NVIDIA GeForce RTX 5090`, capability `(12, 0)`, `sm_120`
- venv overlay: `scikit-learn==1.7.2`의 CPython 3.12 manylinux wheel과 고정 SHA-256
- private wheel: `1private-cp312-none-any`, 승인된 clean Git commit provenance 필수

공유 `ai_env`는 CUDA/PyTorch와 core dependency를 읽기 전용 baseline으로만 제공합니다.
Bootstrap은 `--system-site-packages` venv를 새로 만들고, 충돌하는 scikit-learn과 private wheel을
그 venv 안에만 설치합니다. 공유 환경을 uninstall하거나 수정하지 않습니다.

`--system-site-packages` 특성상 공유 `ai_env`에 선재한 패키지는 venv에서도 import-visible합니다.
현재 base에는 `accelerate`, `safetensors`, `tokenizers`, `transformers`가 이미 있으므로 이 절차가
그 패키지들의 물리적 부재까지 보장하지는 않습니다. 대신 private wheel의 member/metadata/registry에
SELLM이 없고, bootstrap smoke 중 해당 dependency들이 import되지 않았음을 검증합니다. 완전한
dependency 물리 격리는 CUDA/PyTorch까지 독립 설치하는 별도 환경 사양이 필요합니다.

## 준비

dirty checkout은 private wheel source에 포함될 수 있으므로 clean detached checkout에서 빌드합니다.
`--skip-install-check`는 사용하지 않습니다.

```bash
git status --porcelain
EXPECTED_BUILDER_COMMIT="<independently-approved-full-40-character-sha>"
test "$(git rev-parse HEAD)" = "$EXPECTED_BUILDER_COMMIT"

BASE_PYTHON=/opt/miniconda3/envs/ai_env/bin/python
"$BASE_PYTHON" tools/build_private_wheel.py

WHEEL_PATH="$PWD/dist/private/modeling_module-0.1.1-1private-cp312-none-any.whl"
EXPECTED_WHEEL_SHA256="$(sha256sum "$WHEEL_PATH" | awk '{print $1}')"
```

첫 명령은 아무것도 출력하지 않아야 합니다. 현재 개발 checkout이 dirty하면 별도 clean clone 또는
detached checkout을 사용합니다. `EXPECTED_BUILDER_COMMIT`은 빌드 checkout이 임의로 정하는 값이
아니라, 배포 전에 독립적으로 승인한 commit과 일치해야 합니다.

## Bootstrap 실행

새로운 절대 경로를 `VENV_DIR`로 지정합니다. 이 경로는 repository 밖으로 resolve되어야 하며
`/`, `$HOME` 자체, repository root와 그 하위 경로는 거부됩니다. 기존 환경도 기본적으로 거부하며
자동 삭제하지 않습니다.

```bash
VENV_DIR="$HOME/.venvs/ts_forecaster_non_sellm_b015f9a"

EXPECTED_BUILDER_COMMIT="$EXPECTED_BUILDER_COMMIT" \
EXPECTED_WHEEL_SHA256="$EXPECTED_WHEEL_SHA256" \
tools/bootstrap_5090_non_sellm.sh "$WHEEL_PATH" "$VENV_DIR"
```

성공 조건:

- wheel archive·filename·SHA-256·manifest·bytecode·non-SELLM 경계가 venv 생성 전에 통과
- venv의 `sys.base_prefix`와 torch import 경로가 승인된 `ai_env`를 가리킴
- `pip check`가 `No broken requirements found.`로 종료
- wheel manifest의 builder commit이 기대 commit과 일치하고 dirty flag가 `false`
- package는 새 venv에서, internal registry는 `.pyc`에서 import
- `scikit-learn==1.7.2`가 venv 안에서 import되고 requirements에 기록된 wheel hash와 일치
- 설치 파일·public API·model registry 어디에도 SELLM이 포함되지 않음
- base에 선재한 SELLM 계열 dependency가 bootstrap smoke에서 import되지 않음
- 실제 CUDA tensor kernel과 `probe_device("cuda")` 통과

출력의 마지막 `PYTHON_BIN=...`은 설치된 private wheel을 직접 검증하거나 서비스 runtime에
지정할 수 있습니다.

```bash
"$VENV_DIR/bin/python" -I -c \
'import modeling_module, sklearn, torch; print(modeling_module.__file__, sklearn.__version__, torch.cuda.get_device_name(0))'
```

`src/model_test/total_train/dsio_total_running.py`는 현재 checkout의 `src`를 `sys.path` 앞에 추가합니다.
따라서 `PYTHON_BIN`만 이 venv로 바꾼 DSIO runner 실행은 설치된 private wheel 검증이 아니라 checkout
source 검증입니다. DSIO smoke는 승인된 clean non-SELLM checkout에서 별도 수행하며, private-wheel
install gate와 같은 증거로 취급하지 않습니다.

같은 venv를 의도적으로 다시 검증할 때만 `REUSE_VENV=1`을 사용합니다. 첫 성공 시 venv root에
`TS_FORECASTER_5090_BOOTSTRAP.json` receipt가 원자적으로 기록됩니다. 재사용은 wheel과 overlay를
다시 설치하지 않는 verification-only 동작이며, wheel hash·commit·constraints hash·base prefix와
모든 runtime 값이 receipt와 정확히 일치해야 합니다. 입력이 달라졌다면 기존 venv를 덮어쓰지 말고
새 `VENV_DIR`을 사용합니다.

```bash
REUSE_VENV=1 \
EXPECTED_BUILDER_COMMIT="$EXPECTED_BUILDER_COMMIT" \
EXPECTED_WHEEL_SHA256="$EXPECTED_WHEEL_SHA256" \
tools/bootstrap_5090_non_sellm.sh "$WHEEL_PATH" "$VENV_DIR"
```

## 사용하지 않는 파일

- `environment.yml`, `requirements.txt`: Mac/ARM export이므로 5090 bootstrap에 사용하지 않습니다.
- `environment.5090-sellm.yml`, `docs/sellm_5090_setup.md`: SELLM 전용이며 별도 트랙입니다.

# TimeMixer lineage integration plan

이 문서는 NHITS와 ExoTST 실행 기준선을 닫은 다음 진행할 TimeMixer 원본 계보 이식의
출처, 수식 경계, public 계약, 파일 책임과 검증 순서를 고정합니다. 현재 단계에서는
TimeMixer를 registry에 등록하거나 학습 가능 모델로 노출하지 않습니다.

## Decision

첫 artifact는 별도 suffix 없이 `timemixer`로 명명하고 다음 범위만 지원합니다.

- 논문의 long/short-term forecasting에 공통으로 쓰이는 point forecast 계산 그래프
- average-pooling 기반 multiscale 입력
- moving-average seasonal/trend decomposition
- Past-Decomposable-Mixing(PDM)
- Future-Multipredictor-Mixing(FMM)의 scale별 예측 합산
- channel-independent 단일 target 입력 `[B, L, 1]`
- library 표준 point 출력 `[B, H, 1]`
- endogenous-only public 학습, 저장, 로드, 예측

첫 이식에는 DFT decomposition, max/conv downsampling, future temporal feature,
multivariate channel-dependent 경로, exogenous 입력, distribution/quantile 출력,
imputation, anomaly detection, classification을 포함하지 않습니다. 이들은 원본 artifact의
수식을 조건문으로 오염시키지 않고, 실제 요구와 검증 계획이 생길 때 별도 확장으로 다룹니다.

## Provenance baseline

| Item | Frozen value |
|---|---|
| Paper | `TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting`, ICLR 2024 |
| arXiv | `2405.14616v1`, 2024-05-23 |
| Official repository | `https://github.com/kwuking/TimeMixer` |
| Repository commit | `e24610583b36fdd8c76cc17a8df4e65759a5f460` |
| Last commit touching `models/TimeMixer.py` | `38a3507595048d998d12f00d37b66987d03295fc` |
| Upstream license | Apache License 2.0 |
| Investigation date | 2026-07-23 |

Pinned source checksums at the repository commit are:

| Upstream file | SHA-256 |
|---|---|
| `models/TimeMixer.py` | `817d62f4aac54c8566e560f6d3785856e31c8ee51460279ee3d0a4823f11d4be` |
| `layers/Embed.py` | `ab492ea2f68459bbcf3cbffdd1beb75b24d0d70248d017a313a3b470316aaa2b` |
| `layers/Autoformer_EncDec.py` | `48745b4bb647355e9845792a855df9c59fd7df7fcc664c765351fec390c4073e` |
| `layers/StandardNorm.py` | `cc1c0bc65b7b094bbe83f988fb05b86272a59638c030c3781aa52ce8880379df` |
| `LICENSE` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

`main`을 다시 내려받아 암묵적으로 기준을 바꾸지 않습니다. 새 upstream 수정이 필요하면
commit, source hash, parity 결과를 함께 갱신하는 별도 변경으로 처리합니다.

Apache-2.0 소스를 이식하는 commit에는 upstream license 사본과 출처를 포함하고, 수정된
파일에 library wrapper 및 validation 변경 사실을 표시합니다. upstream에는 별도 `NOTICE`
파일이 없음을 확인했습니다.

## Paper and upstream boundary

논문 본체는 다음 계산을 정의합니다.

1. 과거 입력을 average pooling하여 fine-to-coarse multiscale series로 만듭니다.
2. 각 scale을 seasonal과 trend로 분해합니다.
3. seasonal은 fine-to-coarse bottom-up residual mixing을 수행합니다.
4. trend는 coarse-to-fine top-down residual mixing을 수행합니다.
5. 각 scale의 mixed representation을 독립 predictor로 horizon에 투영합니다.
6. scale별 예측을 합하여 최종 point forecast를 만듭니다.

현재 official `main`에는 논문 이후의 기능도 함께 들어 있습니다. `use_future_temporal_feature`,
DFT decomposition, max/conv downsampling과 forecasting 외 task는 source provenance에는
포함되지만 첫 public artifact의 capability에는 포함되지 않습니다.

Backbone parity를 위해 pinned upstream의 forecasting 모듈 구조와 parameter layout을 먼저
보존합니다. Public wrapper는 `x_mark_enc=None`, `use_future_temporal_feature=0`,
`task_name="long_term_forecast"`로 호출합니다. long/short 구분은 계산 그래프가 아니라
dataset 및 horizon 설정이므로 public artifact를 둘로 나누지 않습니다.

## Upstream characterization

Pinned source를 직접 실행한 tiny reference는 다음과 같습니다.

| Setting | Value |
|---|---|
| Input/output | `[2, 16, 1] -> [2, 4, 1]` |
| `d_model`, `d_ff`, `e_layers` | `4`, `8`, `1` |
| Scales | `down_sampling_layers=2`, `window=2`, `avg` |
| Decomposition | `moving_avg=3` |
| Embedding | `timeF`, no temporal marks |
| Parameter count | `1,039` |
| State-dict entries | `39` |
| Result | finite CPU forward |

동일 tiny 설정에서 `channel_independence=0`, `enc_in=c_out=3`인 upstream 경로도
`[2, 16, 3] -> [2, 4, 3]` finite forward를 확인했습니다. 다만 library의 현재 public target
schema는 단일 target이므로 이 경로는 첫 artifact에서 노출하지 않습니다.

확인된 upstream 경계는 구현 전에 명시적으로 막습니다.

- moving-average kernel은 출력 길이를 유지하도록 양의 홀수여야 합니다.
- `lookback // down_sampling_window**down_sampling_layers`는 1 이상이어야 합니다.
- 논문 기본인 average pooling만 허용해 각 scale의 길이와 temporal linear layer를 일치시킵니다.
- 현재 upstream의 conv downsampler는 forward 경로에서 생성되어 parameter/state-dict 및 반복
  호출 재현성 문제가 있으므로 첫 public artifact에서 제외합니다.
- DFT 경로는 `top_k`와 최소 scale의 FFT bin 관계, odd-length `irfft` 길이를 별도로 해결해야
  하므로 논문 본체의 parity 이후 별도 artifact 후보로 둡니다.
- 알 수 없는 downsampling method를 조용히 통과시키는 upstream 동작은 허용하지 않고 config
  생성 시 실패시킵니다.

## Target public contract

### Registry

| Field | Value |
|---|---|
| key | `timemixer` |
| family | `timemixer` |
| aliases | `timemixerbase`, `timemixercanonical` |
| class | `TimeMixerModel` |
| checkpoint aliases | `TimeMixer`, `TimeMixerCanonical` |
| exogenous policy | `none` |
| output capability | point only |

### Tensor contract

- Input: finite floating tensor `[batch, lookback, 1]`
- Output: finite floating tensor `[batch, horizon, 1]`
- `future_exo`, `past_exo_cont`, `past_exo_cat`: non-empty 입력이면 즉시 오류
- lookback/channel mismatch: model boundary에서 즉시 오류
- missing values: data policy에서 처리하며 backbone 내부에서 묵시적으로 대체하지 않음
- distribution 또는 quantile loss: model construction 전에 public validation에서 거부

### Configuration

`TimeMixerConfig(TrainingConfig)`는 다음 model field를 가집니다.

| Field | Initial default | Validation |
|---|---:|---|
| `y_dim` | `1` | 정확히 `1` |
| `d_model` | `16` | positive |
| `d_ff` | `32` | positive |
| `e_layers` | `2` | positive |
| `moving_avg` | `25` | positive odd integer |
| `down_sampling_layers` | `3` | non-negative integer |
| `down_sampling_window` | `2` | integer greater than `1` when layers are nonzero |
| `down_sampling_method` | `"avg"` | `Literal["avg"]` |
| `decomp_method` | `"moving_avg"` | `Literal["moving_avg"]` |
| `channel_independence` | `True` | must remain `True` in v1 |
| `use_norm` | `True` | boolean |
| `dropout` | `0.1` | `[0, 1)` |
| `embed` | `"timeF"` | upstream-compatible value retained for state parity |
| `freq` | `"h"` | upstream-compatible value retained for state parity |
| `use_future_temporal_feature` | `False` | must remain `False` in v1 |
| `use_exogenous_mode` | `False` | must remain `False` |

Scale lengths are derived only from `lookback`, window and layer count. Derived dimensions are not
serialized as independent config values.

## File ownership

| File | Responsibility |
|---|---|
| `TimeMixer/configs.py` | public config and strict scale validation |
| `TimeMixer/backbone.py` | Normalize, decomposition, scale mixing, PDM and FMM numerical core |
| `TimeMixer/TimeMixer.py` | library tensor wrapper and endogenous boundary |
| `TimeMixer/__init__.py` | stable public model/config exports |
| `TimeMixer/LICENSE.upstream` | Apache-2.0 source license copy |
| `models/model_builder.py` | config normalization and construction |
| `models/registry.py` | key, aliases and capability metadata |
| `api/train.py` | `TimeMixerArchitectureConfig` and preflight rejection |
| `training/model_trainers/timemixer_train.py` | point-only `CommonTrainer` integration |
| `training/model_trainers/total_train.py` | family runner only |
| `utils/checkpoint.py` | config reconstruction and predictor loading |

DB, Engine Run, tenant, writer 또는 service 책임은 이 모델 family에 추가하지 않습니다.

## Required validation gates

### Gate 1. Numerical identity

- pinned upstream과 같은 seed 및 tiny config로 exact output parity
- input gradient와 모든 사용 parameter gradient parity
- scale별 intermediate shape 및 FMM sum parity
- batch/channel independence 확인
- parameter count, state-dict key/shape와 public import 기준선 고정

### Gate 2. Library contract

- config validation과 unsupported capability rejection
- builder, registry, aliases, public architecture config 연결
- finite forward/backward와 optimizer step
- endogenous dataloader를 통한 1-epoch/100-step smoke
- checkpoint save/load 후 exact prediction parity
- `load_predictor(...).predict(...)` DataFrame schema 검증
- legacy checkpoint path에 영향이 없는지 전체 회귀 테스트

### Gate 3. Runtime

- Mac은 CPU unit/smoke만 수행
- RTX 5090에서 CUDA forward/backward, save/load/predict 수행
- seed 11/22/33 동일 데이터/loss/budget 정확도 비교
- training/inference latency와 peak VRAM 기록
- PatchMixer, PatchTST, NHITS와 같은 protocol에서 비교

공식 논문 benchmark 수치를 library 이식의 정확도 근거로 직접 사용하지 않습니다. 데이터,
split, target schema와 training protocol이 같은 실험만 promotion 근거로 사용합니다.

## Implementation order

1. upstream license와 source checksum fixture를 repository에 고정합니다.
2. upstream forecasting backbone을 wrapper 없이 이식하고 output/gradient parity를 닫습니다.
3. `TimeMixerConfig`와 invalid-scale characterization test를 추가합니다.
4. `[B,L,1] -> [B,H,1]` endogenous wrapper와 unsupported exogenous guard를 추가합니다.
5. builder, registry, public architecture config, checkpoint 계약을 연결합니다.
6. trainer, save/load, `load_predictor().predict()` 통합 smoke를 추가합니다.
7. 전체 CPU regression을 실행합니다.
8. RTX 5090 CUDA smoke와 100-step benchmark를 실행합니다.
9. 동일 데이터, loss, seed 11/22/33 비교 후 기본 family 전략을 결정합니다.

각 단계는 바로 앞 단계의 기준선이 통과한 뒤 진행합니다. parity가 깨진 상태에서 public API나
정확도 실험으로 넘어가지 않습니다.

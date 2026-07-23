# PatchMixer frozen baseline

이 문서는 PatchMixer 모델 축소 시점의 계보, 수치 계약, checkpoint 지원 범위와 검증 근거를
고정합니다. 활성 모델은 `patchmixer`와 `patchmixer_exo` 두 개입니다.

## Paper lineage

| Item | Frozen value |
|---|---|
| Repository | `https://github.com/Zeying-Gong/PatchMixer` |
| Commit | `cfc6c1386e7fe1633f92ef4b258ff1a4649008b4` |
| `models/PatchMixer.py` Git blob | `bf3867109192da6cd8816f4aec8ab0bf16ec80af` |
| License | MIT |

`PatchMixerModel`은 upstream tensor layout, separable convolution, dual forecasting heads,
channel-independent output과 RevIN을 보존합니다. Exogenous fusion은 이 parity 경계 밖입니다.

## Machine-enforced baseline

### Endogenous paper model

Fixture seed는 `20260724`, config는 lookback/horizon `16/4`, channels `2`, patch/stride `4/2`,
kernel `3`, `d_model=8`, `e_layers=1`, dropout `0`입니다.

| Contract | Value |
|---|---|
| Output shape | `[2, 4, 2]` |
| Parameters | `996` |
| State-dict entries | `24` |
| State schema SHA-256 | `d5013ef1b2f334455e719f0c163d141bb8c4d7542d895b22b5363a98ed65cf19` |
| Upstream equation parity | `rtol=2e-6`, `atol=5e-7` |
| Gradient contract | input 및 모든 trainable parameter finite/nonzero |
| Channel independence | 다른 channel perturbation의 교차 영향 `0` |

고정 output fixture는 `tests/test_patchmixer_model_baseline_contract.py`, upstream 식을 직접 계산하는
parity fixture는 `tests/test_patchmixer_lineage_contract.py`에 있습니다.

### Exogenous model

Fixture seed는 `20260727`, config는 lookback/horizon `8/2`, target channel `1`, past continuous
`2`, past categorical `2`, future continuous `2`입니다.

| Contract | Value |
|---|---|
| Output shape | `[2, 2]` |
| Parameters | `13,992` |
| State-dict entries | `50` |
| Exogenous state entries | `10` |
| State schema SHA-256 | `a65b168fabdbe45764e28d9d811b67f727eae4da4b4791379b1c8c86ea1f2090` |
| Gradient contract | target/past/future 및 모든 trainable parameter finite/nonzero |
| Inheritance contract | `_PatchMixerLegacyModel` 비상속 |

Output fixture는 `[[0.13809253, 0.12295340], [1.09814000, 1.11265635]]`이며 output,
normalized, bounded-normalized 경로의 gradient가 각각 고정되어 있습니다.
Point forward에서 사용하지 않던 distribution-only state 5개는 제거했습니다. 이전 exogenous
state에 이 key가 있어도 strict-load pre-hook이 소비하며, 사용되는 weight와 output은 동일합니다.

## Historical model decision

동일 Walmart weekly dataset과 seed 11/22/33 비교에서 논문 기반 모델은 과거 Enhanced보다 rolling
MAE 기준 2/3 seed에서 이겼고 seed-wise 평균 개선은 `+6.220%`였습니다. Last-origin MAE도 2/3
seed 승리, 평균 `+1.222%`였습니다.

BF16 batch-64, 20 warm-up 후 100 training-step 비교에서 논문 기반 모델은 과거 Enhanced 대비
throughput `1.574x`, mean step latency `36.46%` 감소, peak allocated VRAM `88.29%` 감소,
parameter `98.92%` 감소를 기록했습니다. 이 결과를 근거로 논문 모델을 `patchmixer` 기본으로
승격하고 Enhanced 신규 학습을 종료했습니다.

## Historical exogenous decision

Walmart weekly seed 11/22/33에서 full gated fusion은 endogenous 대비 rolling MAE 평균
`-1.814%`, last-origin MAE 평균 `-3.904%`로 일관된 개선에 실패했습니다. Past gate activation의
약 `78.62%`가 0.05 미만 또는 0.95 초과로 포화되었습니다.

Future residual coordinate 비교 결과는 다음과 같습니다. 양수는 endogenous 대비 MAE 개선입니다.

| Strategy | Rolling improvement | Last-origin improvement | Decision |
|---|---:|---:|---|
| output shift | `+0.520%` | `-2.040%` | default coordinate 유지 |
| normalized shift | `-0.196%` | `-5.820%` | opt-in only |

따라서 `patchmixer_exo`는 외생 입력 capability를 제공하지만 정확도 기본 모델은 아닙니다.

## Checkpoint inventory and policy

| Location | Artifact | Format | Policy |
|---|---|---|---|
| local `raw_data/fit/Xpatchtst/20260119` | `weekly_PatchMixerBase_L52_H27.pt` | pre-version, `BaseModel` | unsupported until migration |
| local `raw_data/fit/Xpatchtst/20260119` | `weekly_PatchMixerQuantile_L52_H27.pt` | pre-version, `QuantileModel` | unsupported until migration |
| RTX 5090 `tsf_full_runs` | `weekly_PatchMixer_L104_H27.pt` | v3, former Enhanced | load-only exact restore |
| RTX 5090 `tsf_regression_runs` | `weekly_PatchMixerQuantile_L104_H27.pt` | v3 quantile | load-only exact restore |
| repository fixtures | PatchMixer v1/v2 Normal/StudentT/point | v1/v2 | strict structural restore |

pre-version `BaseModel`과 `QuantileModel`은 현재 class와 state schema가 다르고, 해당 구조를 만든
source가 git history에 고정되어 있지 않습니다. Public loader는 non-strict 요청에서도 부분 로드를
허용하지 않으며 v3 migration을 요구합니다.

## Current validation gates

- public PatchMixer registry key는 `patchmixer`, `patchmixer_exo`만 노출
- paper/exogenous output, gradient, state schema와 parameter count 고정
- active model CPU train/checkpoint/load/predict smoke
- retired v3 Enhanced/Quantile exact load-only restore
- repository v1/v2 distribution fixture strict restore
- pre-version incompatible schema fail-closed
- full CPU regression
- RTX 5090 CUDA forward/backward, 100-step performance와 real checkpoint restore

## RTX 5090 consolidation validation

2026-07-23 격리 source snapshot을 RTX 5090에서 Python 3.12.13, PyTorch 2.11.0+cu130,
CUDA 13.0, driver 595.71.05로 검증했습니다.

- paper/exogenous baseline, normalized checkpoint, legacy load-only 및 NHITS/ExoTST CUDA 묶음:
  `23 passed`
- dead-state 제거 후 exogenous 전체 gradient와 active CUDA backward/save/load parity:
  `11 passed`
- v3 Enhanced 실물 checkpoint: strict restore, 85 state entries, finite `[2,27]`
- v3 Quantile 실물 checkpoint: strict restore, 197 state entries, finite `[2,3,27]`

BF16 batch 64, seed 20260721, AdamW/MSE, 20 warm-up 후 100 measured training-step 결과입니다.
DataLoader와 host-to-device transfer는 시간에 포함하지 않았습니다.

| Model | Parameters | Mean step | Throughput | Peak allocated VRAM |
|---|---:|---:|---:|---:|
| `patchmixer` | 76,564 | 2.682 ms | 23,863 samples/s | 22.96 MiB |
| `patchmixer_exo` | 7,892,613 | 4.644 ms | 13,781 samples/s | 206.45 MiB |

Exogenous capability는 paper model 대비 step latency가 73.16% 높고 throughput이 42.25% 낮으며,
peak allocated VRAM은 8.99배입니다. 서로 다른 입력 capability의 비용 비교이며 정확도 우위를
뜻하지 않습니다.

원본 결과는 `artifacts/benchmarks/patchmixer_active_5090.json`에 고정하며 SHA-256은
`8d107d2a1dbaf96ca5076f5ab6eea47776ff1f696003df372bfffe9917bf6ee6`입니다.

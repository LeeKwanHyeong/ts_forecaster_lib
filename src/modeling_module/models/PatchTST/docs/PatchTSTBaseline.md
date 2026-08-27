# PatchTST lineage and implementation baseline

이 문서는 PatchTST의 논문 계보와 현재 repository 구현 기준선을 분리해 고정합니다.

## Research lineage

- Paper: <https://arxiv.org/abs/2211.14730>
- Official repository: <https://github.com/yuqinie98/PatchTST>
- Paper: ICLR 2023, *A Time Series is Worth 64 Words*

공식 PatchTST의 핵심은 subseries patch를 Transformer token으로 사용하는 것과 각 channel을
독립적으로 처리하면서 embedding/Transformer weight를 공유하는 것입니다. Self-supervised
masked patch reconstruction도 논문 범위에 포함됩니다.

현재 구현이 계승하는 범위는 patching, positional encoding, Transformer encoder, RevIN과
target-only masked pretraining입니다. 다음 항목은 논문 원본 parity 범위 밖입니다.

- target과 past exogenous를 같은 patch projection에서 결합
- future exogenous token cross-attention
- mean-pooled point/distribution/quantile heads
- `Normal`/`StudentT` packed distribution output
- strict endogenous/exogenous artifact routing
- `c_in > 1`에서 channel-mixing patch projection

따라서 `patchtst_base`는 논문 기반 확장 모델이며 upstream 수식·state dict의 exact port가
아닙니다. 공식 benchmark 결과를 현재 모델의 결과로 인용하지 않습니다.

## Frozen source identity

계산 그래프 기준선은 repository commit
`43f5ec8c9cbc89eaed2a28d7fb011d86b5303428`의 다음 Git blob입니다.

| File | Git blob |
|---|---|
| `supervised/PatchTST.py` | `8fd033e32d2247f6af02442de5c1c4e68deefb8b` |
| `supervised/backbone.py` | `7104d734acd0f28d26cbbb09a9f129d908b51e44` |
| `common/backbone_base.py` | `5bb7fd4a42ecb707075cab5301e32e9a90f17a0a` |
| `common/configs.py` | `90c471a3760867377aa1fe1a4536f708310c8536` |
| `supervised/variants.py` | `6a580289c172d89957d93eae7371dcbbff869acc` |

동일 값은 [provenance.py](../provenance.py)에 machine-readable constant로 고정합니다.

## Fixed characterization fixture

계약 테스트는 lookback/horizon 8/2, patch length/stride 4/2, end padding, `d_model=8`,
`d_ff=16`, encoder 1 layer, 2 attention heads, dropout 0, `c_in=1`, RevIN off를 사용합니다.
Exogenous fixture는 past continuous 1개와 future continuous 1개를 추가합니다.

State schema hash는 정렬된 `name | shape | dtype` 목록의 SHA-256입니다.

| Variant | Parameters | State keys | State schema SHA-256 |
|---|---:|---:|---|
| Point endogenous | 706 | 29 | `5117c80bdc1fd89f4801bfa7fda7440ed81f7f4e71c31f4fc9a6042bf6caae8d` |
| Point exogenous | 1,676 | 50 | `278bb77462e07d0d45262403a9b444fd1fe3b2caef012aebab47d56f5902ca4e` |
| Quantile endogenous | 2,614 | 31 | `dcfb96f61a4ccaa62b71c813404f98e89d483f7bac9a801377242ded55ce8253` |
| Quantile exogenous | 3,584 | 52 | `ae0305caebc0bbbbe4919986548f75cde57d32a2952b8522e7e2d4de4cb8d93e` |
| Normal endogenous | 2,361 | 32 | `06bf9aa6e29a0b2cd575ca8a88ca1fc6fdb57a31cce0ed5ca4e96f4bb64e9f32` |
| Normal exogenous | 3,331 | 53 | `3acafa3c70d7cb1f833d95f9094defa29015132942115ef613e4834427e524be` |

## Artifact responsibility baseline

| Capability | Canonical artifact |
|---|---|
| Endogenous point | `patchtst_base` |
| Exogenous point | `patchtst_exogenous` |
| Endogenous distribution | `patchtst_base` |
| Exogenous distribution | `patchtst_exogenous` |
| Endogenous quantile | `patchtst_quantile` |
| Exogenous quantile | `patchtst_quantile_exogenous` |

`patchtst_base`와 `patchtst_quantile`의 exogenous config routing은 legacy compatibility이며 신규
학습의 권장 경로가 아닙니다. Explicit exogenous artifact는 family default에 포함하지 않습니다.

## Checkpoint boundary

- 기존 `PatchTSTModel`/`PatchTSTQuantileModel` checkpoint class name과 base aliases를 유지합니다.
- Strict variant는 공통 계산 모델과 동일 config에서 state dict schema가 일치해야 합니다.
- Base key의 legacy exogenous config는 strict exogenous subclass로 복원할 수 있어야 합니다.
- `architecture_variant`와 `exogenous_fusion_strategy` metadata는 explicit artifact에서 보존합니다.
- Public restore가 지원하는 distribution은 `Normal`과 `StudentT`입니다.

## Production sl_only baseline

202545 운영 비교 기준은 `SSL_MODE=sl_only`로 학습한 endogenous
`patchtst_base`입니다. 이 checkpoint는 full SSL qualification의 입력이나
출력 경로로 재사용하지 않으며 덮어쓰지 않습니다.

| Item | Frozen value |
|---|---|
| Data cutoff | `202544` |
| Seed / epochs | `42 / 8` |
| Capacity | `d_model=128`, `n_layers=2`, `d_ff=512` |
| Parameters | `403,099` |
| Checkpoint SHA-256 | `2674a5b01a882a7d3bf36af598d787136d2c15181879307989a8206a43fa2d78` |

전체 machine-readable 계약은
[`PatchTSTProductionSLOnlyBaseline.json`](PatchTSTProductionSLOnlyBaseline.json)에
고정합니다. Full SSL 실험은 별도 artifact root를 사용하고 승격 결정 전에는
이 baseline의 Demand Engine registry 항목을 변경하지 않습니다.
Pretrain 입력, backbone 이식, 카테고리 누수 방지 경계는
[`PatchTSTFullSSLContract.md`](PatchTSTFullSSLContract.md)에 정의합니다.

2026-07-24 RTX 5090 seed 11/22/33 qualification에서 현재 `full` 경로는
`sl_only` 대비 평균 MAE·WAPE가 `10.30%` 악화되고 학습 시간이 `24.73%`
증가했습니다. 따라서 운영 전략은 `sl_only`를 유지합니다. 세부 결과와
겹치는 Pretrain patch의 shortcut 분석은
[`PatchTSTSSL5090Qualification.md`](PatchTSTSSL5090Qualification.md)에
고정합니다.

## Accuracy status

이 기준선은 구조 및 호환성을 고정한 것이며 정확도 승격 근거가 아닙니다. PatchTST의 기본 모델
전략은 아직 PatchMixer와 동일한 Walmart seed 11/22/33 및 RTX 5090 프로토콜로 비교되지
않았습니다. Cross-family 비교 전에는 PatchTST가 PatchMixer 또는 통계 기준선보다 우수하다고
표현하지 않습니다.

## Baseline update rule

다음 중 하나를 변경할 때는 이 문서와 characterization fixture를 함께 갱신합니다.

1. Patching 또는 input projection geometry
2. Encoder attention 및 normalization
3. RevIN 위치와 distribution denormalization
4. Future cross-attention 또는 past feature fusion
5. Output head shape, parameter 이름 또는 checkpoint alias
6. Artifact capability routing 및 family expansion

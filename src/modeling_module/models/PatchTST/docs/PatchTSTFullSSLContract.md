# PatchTST Full SSL Contract

## Scope

`full`은 PatchTST의 target-history SSL pretrain과 supervised forecasting을
연속으로 실행하는 학습 전략입니다.

| `SSLConfig.mode` | 실행 범위 |
|---|---|
| `sl_only` | supervised 학습만 실행 |
| `ssl_only` | masked patch pretrain만 실행 |
| `full` | masked patch pretrain 후 supervised 학습 실행 |

202545 운영 checkpoint는
[`PatchTSTProductionSLOnlyBaseline.json`](PatchTSTProductionSLOnlyBaseline.json)에
고정된 `sl_only` 기준선입니다. Full SSL qualification은 반드시 다른
artifact root에서 실행하며 기존 운영 checkpoint를 덮어쓰지 않습니다.

## Pretrain Input Boundary

Pretrain은 학습 loader의 6-tuple 또는 7-tuple에서 첫 번째 target-history
tensor `x`만 사용합니다. 다음 값은 SSL reconstruction 입력으로 전달하지
않습니다.

- supervised target `y`
- 과거 연속형·카테고리 외생변수
- 미래 연속형·카테고리 외생변수
- series ID

Pretrain 모델 config에서도 past/future exogenous 차원과 미래 카테고리
cardinality를 모두 0으로 강제합니다.

## Separate Pretrain Stride

Supervised forecasting의 patch stride와 SSL reconstruction의 patch stride는
서로 다른 목적을 가지므로 별도 설정으로 관리합니다.

- `architecture.patchtst.stride`: supervised forecasting에만 적용
- `ssl.pretrain_stride`: SSL pretraining에만 적용
- `ssl.pretrain_stride=None`: 기존 checkpoint 재현을 위해 supervised
  stride를 그대로 재사용

이번 leakage-safe 후보는 다음과 같습니다.

| Item | Candidate |
|---|---:|
| `lookback` | `52` |
| `patch_len` | `13` |
| supervised `stride` | `6` |
| `pretrain_stride` | `13` |
| `mask_ratio` | `0.4` |

`pretrain_stride=patch_len`이면 Pretrain patch가 서로 겹치지 않으면서
52주 입력을 13주 단위 4개 patch로 빈틈없이 나눕니다. 따라서 가려진
patch의 값이 이웃한 unmasked patch에 중복 포함되는 shortcut을 제거합니다.
이 설정은 새 qualification 후보이며 기존 5090 결과나 production
`sl_only` 기준선을 소급 변경하지 않습니다.

## Transfer Boundary

SSL checkpoint에서 supervised 모델로 복원할 수 있는 state-dict prefix는
`backbone.*`뿐입니다.

`backbone.patch_embed.*`는 supervised `backbone.input_proj.*`로
변환합니다. 과거 외생변수로 input projection이 넓어진 경우에는 target
patch에 해당하는 기존 열만 복원하고 새 외생변수 열은 supervised 모델의
초기값을 유지합니다.

다음 supervised 전용 모듈은 SSL checkpoint에 같은 이름과 shape의 키가
있더라도 복원하지 않습니다.

- `head.*`
- `future_cat_embedding.*`
- `future_fuser.*`
- `revin_layer.*`

호환 가능한 backbone key가 하나도 없으면 random initialization으로
조용히 진행하지 않고 즉시 실패합니다. 실제 복원 key와 보호 prefix는
`pretrain_load_report`로 학습 결과에 남깁니다.

새 Pretrain checkpoint는 `patchtst.ssl.pretrain.v1` 계약을 저장합니다.
계약에는 target-history 입력 범위, 입력 채널 수, `lookback`, `patch_len`,
Pretrain stride, patch 수, coverage 유형, mask ratio, loss type, 예정된
supervised stride가 포함됩니다.

- Pretrain과 supervised의 stride 및 patch 수가 다른 것은 허용합니다.
- `patch_len` 또는 입력 채널 수가 다르면 같은 input projection 의미를
  보장할 수 없으므로 복원을 거부합니다.
- checkpoint에 예정된 supervised stride가 기록된 경우 실제 target
  stride와 다르면 복원을 거부합니다.
- 구형 checkpoint에 이 계약이 없으면 기존처럼 state-dict shape 기반
  복원을 유지하고 결과에 `legacy_shape_only`로 표시합니다.
- supervised 위치 인코딩은 target patch 수에 맞춰 초기화하며 SSL 전용
  reconstruction head와 mask token은 복원하지 않습니다.

## Categorical Leakage Boundary

카테고리 vocabulary는 DataModule이 window 또는 series split을 먼저
확정한 뒤, 학습 window가 참조하는 source row만 사용해 생성합니다.

- validation에만 있는 값은 vocabulary에 들어가지 않습니다.
- validation 및 운영의 신규 값은 UNK ID `0`으로 변환합니다.
- 동일 vocabulary와 fingerprint를 supervised train, validation, final
  checkpoint, strict restore에서 공유합니다.
- SSL pretrain은 category ID tensor 자체를 소비하지 않습니다.

## Artifact Layout

Public `train()`의 `full` 실행은 다음 artifact를 분리해 생성합니다.

```text
<save_dir>/
  pretrain/
    patchtst_pretrain_best.pt
    pretrain_cfg.json
  <frequency>_PatchTSTExogenous_L<lookback>_H<horizon>.pt
  training_manifest.json
```

Pretrain artifact에는 reconstruction 모델 state가 들어가며, 최종
checkpoint에는 supervised head, 카테고리 embedding, future fusion,
exogenous schema, vocabulary와 fingerprint가 들어갑니다. Pretrain
artifact에는 `pretrain_contract`가, 최종 supervised checkpoint metadata에는
같은 계약과 backbone transfer 요약이 함께 저장됩니다.

## Verification

계약 테스트는 다음 전체 경로를 실행합니다.

```text
training-only vocabulary fit
-> target-history SSL pretrain
-> backbone-only transfer
-> categorical supervised learning
-> final checkpoint save
-> strict load
-> known/UNK future category forecast
```

이 계약은 학습 경로의 정상 연결을 증명합니다. `full`이 `sl_only`보다
정확하다는 승격 근거는 아니며, 실제 데이터의 동일 split·seed
qualification을 별도로 수행해야 합니다.

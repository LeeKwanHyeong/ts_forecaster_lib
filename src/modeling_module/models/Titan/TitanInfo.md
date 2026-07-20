# Titan implementation notes

> **Deprecated:** Titan은 신규 운영 학습과 DSIO default에서 제외됩니다. Public `train(...)`에서
> Titan family 또는 artifact를 요청하면 `FutureWarning`이 발생합니다. Registry key와 기존 지원
> checkpoint load는 deprecation 기간 동안 유지하지만 신규 검증 범위는 추가하지 않습니다.

이 문서는 현재 repository의 Titan 구현과 public API 계약을 설명합니다. 최종 지원 범위는
repository root의 `README.md`와 `README.package.md` 표를 기준으로 합니다.

## Registered artifacts

| Artifact key | Memory path | Decoder | Output mode |
|---|---:|---:|---|
| `titan_base` | 없음 | `TitanCrossAttnDecoder` | point, `Normal`, `StudentT` |
| `titan_lmm` | LMM | `TitanCrossAttnDecoder` | point, `Normal`, `StudentT` |
| `titan_seq2seq` | LMM | `TitanCrossAttnDecoder` | point, `Normal`, `StudentT` |

세 artifact 모두 현재 `has_decoder=True`입니다. `titan_lmm`과 `titan_seq2seq`는 현재
`has_memory=True`, `has_decoder=True`로 구조가 같으며, 등록 이름만으로 서로 다른 decoder
topology를 가정하면 안 됩니다. Titan quantile artifact는 없습니다.

Distribution checkpoint의 public prediction 결과는 현재 location을 `{"point": ...}`로
노출합니다.

## Data contract

- target input: `(B, lookback, 1)`
- past continuous exogenous: 선택 사항; encoder input에 결합
- future continuous exogenous: 선택 사항; decoder의 horizon query에 주입
- categorical exogenous: public training과 prediction API에서 fail-fast

Future width가 설정된 checkpoint는 public prediction 시 batch 공용 `(horizon,
future_exo_dim)` 또는 `(B, horizon, future_exo_dim)`을 요구합니다. 누락과
batch/horizon/width 오류는 즉시 실패하고, width가 0인 모델은 non-empty future tensor를
거부합니다.

## Current data flow

1. target을 optional RevIN으로 정규화합니다.
2. target과 구성된 past continuous feature를 합쳐 `MemoryEncoder`에 전달합니다.
3. LMM variant는 encoder output을 `LMM`으로 보강합니다.
4. 모든 등록 variant가 `TitanCrossAttnDecoder`로 horizon token을 생성합니다. Future exogenous가
   구성된 경우 horizon별 query에 투영됩니다.
5. linear head가 point 또는 distribution parameter를 출력하고 location/scale을 target scale로
   복원합니다.

현재 모델에는 과거 문서가 언급하던 `TrendCorrector`, `FeatureModel`, `LMMSeq2Seq`,
`TestTimeMemoryManager` 기반 TTA가 public 등록 구조로 존재하지 않습니다. Masked self-attention
decoder나 별도 trend 보정도 현재 계약이 아닙니다.

## Code map

- [registered model variants](Titans.py)
- [configuration](common/configs.py)
- [encoder backbone](backbone.py)
- [cross-attention decoder](common/decoder.py)
- [trainer integration](../../training/model_trainers/titan_train.py)

기존 regression은 지원 checkpoint 호환을 보존하기 위한 안전망으로만 유지합니다. 신규 성능 검증,
distribution matrix 확장, 5090 promotion 대상에는 Titan을 포함하지 않습니다.

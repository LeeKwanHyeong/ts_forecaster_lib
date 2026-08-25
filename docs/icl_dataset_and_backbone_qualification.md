# ICL Dataset and Backbone Qualification

## 목적

AutoTimes와 SELLM은 동일한 봉인 Episode를 사용합니다. Endogenous Episode는
수요 이력만 포함하고, Exogenous Episode는 과거 관측 Feature와 미래에 미리 알 수
있는 Feature를 역할별로 분리합니다. 모델은 Artifact에 저장된 Feature 순서와
Source Revision hash가 checkpoint와 일치할 때만 실행됩니다.

## Exogenous Episode 계약

| 구분 | 내용 |
|---|---|
| 과거 Feature | Query와 Demonstration의 과거 구간에서 관측된 값 |
| 미래 Feature | Forecast 시점에 이미 승인되어 있는 달력·계획 값 |
| Source Revision | 외생변수 원천의 변경 불가능한 버전 |
| Schema hash | 과거·미래 Feature 순서와 Source Revision의 SHA256 |
| Artifact | `episodes.parquet`과 `manifest.json` |

AutoTimes는 과거와 미래 Feature를 서로 겹치지 않는 숫자 Token 채널에 넣습니다.
SELLM은 Demonstration 외생변수를 의미 Prompt로 요약하고, Query의 과거·미래
외생변수를 별도 Prompt Token으로 전달합니다. 어느 모델도 누락 Feature를 0으로
추정하거나 다른 Schema로 자동 대체하지 않습니다.

Feature 순서는 아래 23개로 고정합니다. 순서나 Source Revision이 달라지면 Schema
hash도 달라지므로 기존 checkpoint와 함께 사용할 수 없습니다.

1. `sin_annual`, `cos_annual`, `sin_semi`, `cos_semi`
2. `sin_quarter`, `cos_quarter`, `week_of_year_norm`, `peak_season_flag`
3. `is_year_start`, `is_year_end`, `is_q_start`, `is_q_end`
4. `lifecycle_pre_launch_flag`, `lifecycle_active_flag`, `lifecycle_service_ended_flag`
5. `lifecycle_age_years`, `lifecycle_remaining_years`, `post_lifecycle_years`
6. `warranty_years`, `warranty_active_flag`, `weeks_to_warranty_end_years`
7. `weeks_since_warranty_end_years`, `lifecycle_source_observed_flag`

## 5090 Qualification 기준

`tools/qualify_icl_backbones_5090.py`는 Mock 모델이 아닌 봉인된 로컬 Hugging Face
모델을 AutoTimes와 SELLM의 실제 backbone으로 사용합니다. 기존 Qwen2-0.5B
디렉터리는 계속 지원하고, seal manifest가 있는 모델은 revision, 라이선스, 구조,
파라미터 수와 파일 SHA256까지 검증합니다.

- 입력은 승인 Manifest와 SHA256이 일치하는 V100 수요 Parquet과 봉인된
  Operation Part/Warranty Snapshot만 사용합니다.
- 연속 이력이 충분한 자재를 이력 길이와 자재 코드 순서로 결정적으로 선택합니다.
- H26을 운영 기준 Episode로 생성하고 H27은 autoregressive horizon 경계 진단에만
  사용합니다.
- H26은 Train·Validation·Test를 모두 사용합니다. 동일 Source의 연속 이력으로
  H27까지 세 구간을 만들 수 없으므로 H27 진단은 서로 겹치지 않는 Train·Test만
  사용하고 Validation 기반 모델 선택은 수행하지 않습니다.
- Train, Validation, Test의 정답 구간이 서로 겹치지 않도록 경계 Episode를
  제외합니다. Receipt에는 split별 정답 시작·종료 주차를 함께 기록합니다.
- 외생변수는 임의 수요 기반 값이 아니라 승인된 ISO 달력 12개와
  Lifecycle/Warranty 11개 Feature를 사용합니다.
- Operation Part Manifest의 조직 범위, Source Revision, 행·자재 수와 논리 content
  hash를 검증하고 수요 Artifact와 자재 키 Coverage가 일치해야 합니다.
- Promotion, Weather, Macro, Outage는 승인 원천이 없으므로 Qualification 입력에
  포함하지 않습니다.
- 학습 시간, GPU peak memory, MAE, WAPE를 기록합니다.
- checkpoint를 다시 로드해 동일 Episode 예측의 최대 절대 차이를 검증합니다.
- Receipt에는 원본 수요 행이나 Secret을 기록하지 않습니다.

기본 실행은 4개 자재, 1 epoch의 개발 Qualification입니다. 이 결과는 실행 계약과
재현성을 확인하는 기준선이며, Production 정확도 승격 근거로 사용하지 않습니다.
승격 판단에는 전체 승인 표본과 다중 Seed 평가가 별도로 필요합니다.

```bash
python tools/qualify_icl_backbones_5090.py \
  --target-source /approved/input/tb_master_target.parquet \
  --input-manifest /approved/input/manifest.json \
  --operation-part-source /approved/exogenous/tb_mst_oper_part.parquet \
  --operation-part-manifest /approved/exogenous/operation_part_snapshot_manifest.json \
  --llm-local-path /approved/models/Qwen2-0.5B \
  --output-root /artifacts/icl-backbone-qualification/run-id
```

## 누수 방지 강화 후 Qualification 기준선

`9cdb709` clean checkout과 `ai_env`에서 동일 Source, Seed 42, 4개 자재,
1 epoch 조건으로 다시 실행했습니다.

| 구분 | 값 |
|---|---|
| Source Commit | `9cdb709b3ddeed3991e2f8b0ac5ef5b5ce213ef5` |
| Receipt SHA256 | `3031a027e739d52b5875f3ddc74b6448cceee8b6f2a1e435302df4f9bd63762b` |
| H26 Manifest | `41651cad878a0091f43e73865e0631bedebd487b353f2dde87113aec9711840a` |
| H27 Manifest | `62b6cde5fbd791ae6204e71d3dd64d0b057f72041d68d1a01c9cb697b1719e98` |

H26 Train 정답은 `202208~202333`, Validation은 `202334~202407`, Test는
`202408~202433`입니다. H27 진단 Train 정답은 `202308~202334`, Test는
`202408~202434`이며 Validation은 사용하지 않았습니다. 선택된 자재는 Episode
생성 전에 공통 주차 교집합으로 정렬했으므로 자재 간에도 split 정답 구간이
겹치지 않습니다.

| 모델 | Horizon | WAPE | MAE | Peak GPU | Reload 최대 오차 |
|---|---:|---:|---:|---:|---:|
| AutoTimes | H26 | 161.2% | 11.069 | 3,575.6 MiB | 0.0 |
| SELLM | H26 | 136.8% | 9.392 | 2,484.5 MiB | 0.0 |
| AutoTimes | H27 | 178.6% | 12.025 | 3,783.3 MiB | 0.0 |
| SELLM | H27 | 114.0% | 7.677 | 3,178.5 MiB | 0.0 |

네 checkpoint의 물리 SHA256, Episode Parquet SHA256, manifest hash와 receipt seal을
독립적으로 다시 계산해 모두 일치함을 확인했습니다. 이 결과는 누수 없는 실행
계약과 checkpoint 재현성의 현재 기준선입니다. 표본과 epoch가 작으므로 Production
모델 승격 근거로는 사용하지 않습니다.

## 선행 외생변수 Qualification 증적 (Superseded)

5090에서 `DSE/C100/V100/V100`의 다음 원천을 사용해 H26/H27를 검증했습니다.
이 실행은 split 정답 구간 비중첩 검사를 추가하기 전에 만들어진 선행 증적입니다.
따라서 실제 외생변수 전달, 모델 실행, checkpoint 재로딩 결과는 보존하지만 현재
코드가 생성하는 Episode manifest의 Qualification seal로 재사용하지 않습니다.
누수 방지 강화 이후의 정식 증적은 같은 Source로 도구를 다시 실행해 새 receipt를
발급해야 합니다.

| 구분 | 식별자 |
|---|---|
| 수요 Source Revision | `7242d4aa3d69bb7719eb21478f53be0637700c4128d272c261a8e403cfbc3cd1` |
| Operation Part Source Revision | `m3-canonical-v2-sh24-r1-20260723-db-full-load-18-v1` |
| Operation Part Manifest SHA256 | `869ec7b37614858567994dbb8ec0bb01eb3aaf2b28ef1bbdd9b47bf066247a7b` |
| Operation Part content SHA256 | `286b5c36034c150df6db92ac0a6f3ba302367064ea054b75a1dd5dc0f77e5300` |
| 결합 Exogenous Revision | `22bc67d775c45b55c878d41de6e627aa3526bad86f1aa1a7caa6933d10b67a35` |
| 자재 Coverage | `7,000 / 7,000` |
| Receipt SHA256 | `556a9f179d8b1d1717a80c84b0623402afc68cf22f7b66b44ab422e987d53234` |

4개 자재와 Seed 42, 1 epoch 기준 결과는 다음과 같습니다. WAPE는 비율을
백분율로 변환해 표시했습니다.

| 모델 | Horizon | WAPE | MAE | Peak GPU | Reload 최대 오차 |
|---|---:|---:|---:|---:|---:|
| AutoTimes | H26 | 426.6% | 14.315 | 7,142.9 MiB | 0.0 |
| SELLM | H26 | 246.0% | 8.255 | 4,014.8 MiB | 0.0 |
| AutoTimes | H27 | 278.8% | 9.266 | 7,649.2 MiB | 0.0 |
| SELLM | H27 | 264.3% | 8.786 | 4,018.0 MiB | 0.0 |

모든 checkpoint 물리 SHA256과 Episode Manifest seal이 일치했고 재로딩 예측
오차는 0이었습니다. 다만 표본과 epoch가 작고 WAPE가 높으므로 이 결과는 실제
외생변수 전달과 당시 코드의 재현성 검증에만 사용하며 모델 승격 근거나 현재
누수 방지 계약의 통과 증적으로 사용하지 않습니다.

## Qwen2-1.5B 확장 Qualification 기준선

### Backbone seal과 4-series smoke

RTX 5090의 `/home/leekwanhyeong/models/Qwen2-1.5B`에 다음 backbone을 고정했습니다.
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True` 조건에서
tokenizer, 가중치 load와 forward가 정상 동작했습니다.

| 구분 | 값 |
|---|---|
| Model ID | `Qwen/Qwen2-1.5B` |
| Revision | `8a16abf2848eda07cc5253dec660bf1ce007ad7a` |
| License | `apache-2.0` |
| Hidden size / layers | `1536 / 28` |
| Parameter count | `1,543,714,304` |
| Model safetensors SHA256 | `6f3a62caedc5c5278275bf5eed428806eeac8df927a3d333c3c850402c24cdeb` |
| Backbone manifest SHA256 | `e538d044d4a4ca50e17e353e368730d54827936c8a884645f1041cb383fc78a3` |

`c250893` clean checkout에서 기존 0.5B smoke와 동일한 4개 자재, seed 42,
1 epoch로 실행했습니다. H26/H27 Episode manifest는 각각
`41651cad878a0091f43e73865e0631bedebd487b353f2dde87113aec9711840a`,
`62b6cde5fbd791ae6204e71d3dd64d0b057f72041d68d1a01c9cb697b1719e98`
로 기존 기준선과 정확히 일치합니다. Receipt SHA256은
`01e3839297837bae91652f7e0386aaef3a1757b7eb766043355a25d23da40a78`입니다.

| 모델 | Horizon | WAPE | MAE | Peak GPU | Reload 최대 오차 |
|---|---:|---:|---:|---:|---:|
| AutoTimes | H26 | 277.2% | 19.028 | 8,484.3 MiB | 0.0 |
| SELLM | H26 | 108.6% | 7.453 | 6,494.0 MiB | 0.0 |
| AutoTimes | H27 | 211.2% | 14.217 | 9,026.3 MiB | 0.0 |
| SELLM | H27 | 98.7% | 6.642 | 7,053.6 MiB | 0.0 |

4-series 결과에서 1.5B가 모든 모델을 일관되게 개선한 것은 아닙니다. SELLM은
0.5B보다 좋아졌지만 AutoTimes는 나빠졌으므로 backbone 크기만으로 모델 우위를
판정하지 않습니다. H27은 계속 horizon 경계 진단으로만 사용합니다.

### Batch size 결정

H26 batch probe에서 batch 8은 AutoTimes가 24,296.2 MiB allocated,
29,786.0 MiB reserved를 사용해 32GB GPU의 실행 여유가 부족했습니다. Batch 4는
AutoTimes 14,165.1/17,088.0 MiB, SELLM 6,554.2/7,676.0 MiB로 안정적이어서
256-series 확장 기준을 batch 4로 고정했습니다. Best validation state를 CPU에
보관한 이후 AutoTimes 실행 중 GPU 사용량도 약 14.9 GiB로 유지됐습니다.

### H26 256-series 수렴 결과

`8d28d43` clean checkout, seed 42, batch 4에서 256개 자재를 사용했습니다. 두
모델의 Episode manifest는
`2be49f5c09ebc447e22363d93120f5461ebd0f09687ff80e9f81a1c834ca68bf`
로 동일합니다. AutoTimes는 LR `1e-3`, SELLM은 LR `1e-4`를 사용했습니다.

| 모델 | Epoch | Validation MAE | Validation WAPE |
|---|---:|---:|---:|
| AutoTimes | 1 | 5.008 | 50.3% |
| AutoTimes | 2 | 2.631 | 26.4% |
| AutoTimes | 3 | 2.355 | 23.7% |
| AutoTimes | 4 | **2.090** | **21.0%** |
| AutoTimes | 5 | 2.443 | 24.6% |
| SELLM | 1 | 2.606 | 26.2% |
| SELLM | 2 | 2.773 | 27.9% |
| SELLM | 3 | 2.452 | 24.6% |
| SELLM | 4 | 2.396 | 24.1% |
| SELLM | 5 | **2.178** | **21.9%** |

| 모델 | Test MAE | Test WAPE | 학습 시간 | Peak GPU | Checkpoint SHA256 |
|---|---:|---:|---:|---:|---|
| AutoTimes | 3.735 | 55.5% | 333.3s | 14,165.1 MiB | `fb13283e344537a0352619de36a01e452b5db674d9f5d04157f3236134512e89` |
| SELLM | 4.083 | 60.7% | 61.1s | 6,554.1 MiB | `4ebc20013dc72236d0eedead4c61b8d58ed23dd25720c64a4f426c1c7bbc641a` |

AutoTimes는 epoch 4가 Validation MAE·WAPE 최저점이고 epoch 5에서 악화됐으므로
현재 고정 epoch 후보는 4입니다. SELLM은 epoch 5까지 개선이 이어져 5가 현재
최선이지만 수렴 종료점은 아닙니다. 후속 실험에서는 SELLM만 6~10 epoch 범위를
확인해야 합니다.

SELLM은 LR `1e-3`에서 non-finite가 발생했고, `3e-4`도 1 epoch는 통과했지만
5-epoch 실행 중 다시 실패했습니다. `1e-4`에서만 5 epoch 전체가 안정적으로
완료됐으므로 Qwen2-1.5B SELLM 확장 기준 LR은 `1e-4`입니다. 두 최종 checkpoint는
strict reload 예측 최대 오차 0을 확인했습니다. AutoTimes와 SELLM의 aggregate
Receipt SHA256은 각각
`1bbde6639b8f09051cffed78cc1b313e767cce642cf6f75ba5b5f07af585c616`,
`db36824aa6ac783623f8e50fcaeaf7f1d3440d64afdc9c0202911397fd3aea61`
입니다.

이 단일 seed 256-series 결과는 1.5B 확장 실행 기준선이며 Production 승격
근거는 아닙니다. 모델 선택에는 동일 조건의 0.5B 비교와 다중 seed 검증이
추가로 필요합니다.

## Qwen2-0.5B 동일 조건 비교

Backbone 크기 효과만 분리하기 위해 Qwen2-1.5B 확장 실행과 같은 256-series,
Episode split, seed 42, batch 4, 5 epoch를 사용했습니다. AutoTimes LR은 `1e-3`,
SELLM LR은 `1e-4`로 동일하게 유지했습니다. 네 실행의 H26 Episode manifest는
모두 `2be49f5c09ebc447e22363d93120f5461ebd0f09687ff80e9f81a1c834ca68bf`
입니다.

기존 `/home/leekwanhyeong/models/Qwen2-0.5B` 파일을 새로 다운로드하지 않고
공식 revision `91d2aff3f957f99e4c74c962f2f408dcc88a18d8`과 Apache-2.0
라이선스로 봉인했습니다. Parameter count는 `494,032,768`, model safetensors
SHA256은 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`,
backbone manifest SHA256은
`d2541896d94b231d9ba121cd11a024286e2aaffbcaa0b55f83908717eabf6942`입니다.
Offline local-only load와 forward도 통과했습니다.

### Epoch별 Validation

| 모델 | Backbone | 최적 Epoch | Validation MAE | Validation WAPE |
|---|---|---:|---:|---:|
| AutoTimes | Qwen2-0.5B | 5 | **2.083** | **20.9%** |
| AutoTimes | Qwen2-1.5B | 4 | 2.090 | 21.0% |
| SELLM | Qwen2-0.5B | 4 | **2.176** | **21.9%** |
| SELLM | Qwen2-1.5B | 5 | 2.178 | 21.9% |

AutoTimes 0.5B는 epoch 5까지 개선됐습니다. SELLM 0.5B는 epoch 4가 최저점이고
epoch 5에서 소폭 악화됐습니다. 동일 조건에서 1.5B의 최적 Validation MAE는
AutoTimes가 0.31%, SELLM이 0.06% 나빠 사실상 동등한 범위입니다.

### Test·비용 비교

| 모델 | Backbone | Test MAE | Test WAPE | 학습 시간 | Peak GPU |
|---|---|---:|---:|---:|---:|
| AutoTimes | Qwen2-0.5B | 3.780 | 56.2% | 280.1s | 6,128.9 MiB |
| AutoTimes | Qwen2-1.5B | **3.735** | **55.5%** | 333.3s | 14,165.1 MiB |
| SELLM | Qwen2-0.5B | **3.977** | **59.1%** | 47.4s | 2,844.3 MiB |
| SELLM | Qwen2-1.5B | 4.083 | 60.7% | 61.1s | 6,554.1 MiB |

AutoTimes 1.5B는 Test MAE·WAPE를 1.17% 개선했지만 학습시간이 19.0%, Peak GPU가
131.1% 증가했습니다. SELLM 1.5B는 Test MAE·WAPE가 2.68% 악화됐고 학습시간은
28.8%, Peak GPU는 130.4% 증가했습니다. 따라서 현재 단일 seed 기준으로는
1.5B가 비용 증가를 정당화할 만큼 일관된 정확도 개선을 제공하지 않습니다.

Qwen2-0.5B checkpoint SHA256은 AutoTimes
`94fbcfbc100388e54a9668cc72341e8145134861f198d9902b39fa392c1866de`,
SELLM `e10dc0370c36857d601069c5c137fd4fe0b9e292e233f7cfb4fc7ebe86ec0576`이며
두 checkpoint의 strict reload 최대 오차는 0입니다. Aggregate receipt SHA256은
각각 `7f4d055e00a4600e74387ca29253f51cbfbda7e465a5a6d27315a2efb0a20c01`,
`acaf4af7358fac09e63ff367b40f01dff2dc9eacb7bd4ddf75302216c2f4cc6f`입니다.

기본 ICL backbone은 Qwen2-0.5B로 확정합니다. AutoTimes와 SELLM의 운영 후보
Qualification은 위에서 봉인한 model ID, revision과 manifest를 사용합니다.
Qwen2-1.5B는 기본 실행과 모델 선택에서 제외하고 연구용 비교 대상으로만
유지합니다. 이 backbone 결정은 SELLM checkpoint의 Production 승인을 의미하지
않습니다.

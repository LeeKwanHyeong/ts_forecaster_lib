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

`tools/qualify_icl_backbones_5090.py`는 Mock 모델이 아닌 로컬
`Qwen2-0.5B`를 AutoTimes와 SELLM의 실제 backbone으로 사용합니다.

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

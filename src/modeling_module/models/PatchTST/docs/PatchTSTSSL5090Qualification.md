# PatchTST Full SSL RTX 5090 Qualification

## Decision

2026-07-24 기준 현재 `full` 학습 경로는 운영 승격하지 않습니다.
202545 운영 전략은 기존 `sl_only`를 유지합니다.

세 seed 모두 `full`이 MAE와 WAPE를 개선하지 못했고, 학습 시간과 peak
VRAM도 증가했습니다. 기존 production checkpoint와 registry는 변경하지
않았습니다.

이 결과는 SSL 일반의 무효를 의미하지 않습니다. 현재 구현은 supervised
patch 설정을 Pretrain에도 그대로 사용해 masked reconstruction에 겹침
shortcut이 존재합니다. 따라서 결론의 범위는 현재 구현된 `full` 경로로
제한합니다.

## Fixed Conditions

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB |
| Driver / Torch / CUDA | `595.71.05` / `2.11.0+cu130` / `13.0` |
| Data | 2,455,508 rows, 7,000 series, `201801..202544` |
| Data SHA-256 | `328f547d5eb0a50c80dc60dc7bb89c09799599f8f6b8677406a0e3cc4a3ef547` |
| Qualification train target | through `202517` |
| Validation target | `202518..202544`, 7,000 windows |
| Lookback / horizon / stride | `52 / 27 / 4` |
| PatchTST capacity | `d_model=128`, `n_layers=2`, `d_ff=512` |
| Patch length / stride | `13 / 6` |
| Supervised epochs | `40` |
| Seeds | `11 / 22 / 33` |
| Full SSL mask ratio | `0.3` |
| Full SSL configured Pretrain epochs | `12` |
| Training order | seed별 `sl_only/full` 순서 교대 |
| Checkpoint evaluation | public `load_predictor(strict=True)` |
| VRAM measurement | `nvidia-smi memory.used`, 0.25초 간격 |

실험은 운영 checkout과 분리된 다음 경로에서 실행했습니다.

```text
/home/leekwanhyeong/workspace/ts_forecaster_lib_ssl_846d3e2_20260724
/home/leekwanhyeong/workspace/ts_forecaster_lib_ssl_846d3e2_20260724/artifacts/patchtst_ssl_20260724
```

## Pilot Result

seed 42에서 Pretrain 40 epochs와 supervised 1 epoch를 실행했습니다.

| Item | Result |
|---|---:|
| Best Pretrain epoch | `9` |
| Best validation reconstruction loss | `6.517309754729337e-07` |
| Best 3-epoch rolling window | `7..9` |
| Pilot training time | `95.89 s` |
| Pilot peak VRAM delta | `880 MiB` |

Validation loss는 epoch 1의 `0.273184`에서 epoch 7의
`8.684289e-07`까지 급격히 감소했습니다. epoch 9 이후에는 안정적인
추가 개선이 없었기 때문에 formal 탐색 상한을 12로 제한했습니다. Formal
seed 11, 22, 33도 모두 epoch 9를 best Pretrain state로 선택했습니다.

## Multi-Seed Result

3-seed 평균입니다.

| Metric | `sl_only` | `full` | Full minus SL | Change |
|---|---:|---:|---:|---:|
| MAE | `8.9531` | `9.8754` | `+0.9222` | `+10.30%` |
| WAPE | `52.7392%` | `58.1716%` | `+5.4325%p` | `+10.30%` |
| sMAPE | `138.9467%` | `138.8538%` | `-0.0929%p` | `-0.07%` |
| Training time | `95.49 s` | `119.10 s` | `+23.61 s` | `+24.73%` |
| Peak training VRAM delta | `848 MiB` | `882 MiB` | `+34 MiB` | `+4.01%` |
| Inference time | `0.3170 s` | `0.3125 s` | `-0.0045 s` | `-1.41%` |

두 checkpoint의 최종 supervised architecture와 parameter count는
같습니다. 따라서 작은 inference 시간 차이는 모델 구조의 차이가 아니라
측정 변동으로 봅니다.

| Seed | SL MAE | Full MAE | Full change | SL best epoch | Full best epoch | Pretrain best |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | `8.7184` | `9.7365` | `+11.68%` | `14` | `15` | `9` |
| 22 | `8.0382` | `9.7009` | `+20.69%` | `23` | `31` | `9` |
| 33 | `10.1028` | `10.1887` | `+0.85%` | `1` | `5` | `9` |

`full`의 MAE·WAPE 승수는 `0/3`입니다. sMAPE는 `2/3` seed에서
소폭 낮았지만 평균 차이는 `-0.0929%p`이며, 현재 qualification target의
많은 zero-demand cell에 민감한 지표 특성을 고려하면 운영 승격 근거로
사용하지 않습니다.

## Overlap Shortcut

현재 Pretrain은 `patch_len=13`, `stride=6`을 사용합니다. 길이 52에서
7개 patch가 생성되며 인접 patch가 7개 시점을 공유합니다.

독립 Bernoulli mask ratio `0.3`의 모든 mask 조합을 정확히 열거하면:

- masked patch 값의 기대 노출 비율: `64.2308%`
- 모든 값이 다른 unmasked patch에 노출되는 masked patch 비율: `35.0%`

즉 patch token 자체는 mask token으로 바뀌지만, 같은 원시 시점 상당수가
겹치는 이웃 patch를 통해 encoder에 남습니다. DSIO 데이터의 높은
zero-demand 비율과 결합하면 reconstruction objective가 너무 쉽게
풀리고 forecasting에 유용한 표현을 만들지 못했을 가능성이 있습니다.
이는 결과에 기반한 원인 가설이며 별도 ablation으로 검증해야 합니다.

공식 PatchTST self-supervised entrypoint의 기본값은
[`patch_len=12`, `stride=12`, `mask_ratio=0.4`, Pretrain 10 epochs](https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_self_supervised/patchtst_pretrain.py)입니다.
현재 설정은 이 기본 Pretrain 계약과 다릅니다.

## Next Qualification Candidate

다음 5090 비교 후보는 supervised 설정을 유지한 채 Pretrain patching만
분리합니다.

| Item | Frozen baseline | Next candidate |
|---|---:|---:|
| `patch_len` | `13` | `13` |
| supervised `stride` | `6` | `6` |
| Pretrain `stride` | `6` | `13` |
| `mask_ratio` | `0.3` | `0.4` |
| Pretrain patch count | `7` | `4` |

후보 경로는 versioned Pretrain checkpoint 계약과 backbone restore
검증까지 구현됐습니다. 정확도·속도·VRAM은 아직 측정하지 않았으므로 이
문서의 기존 `sl_only` 대 `full` 결과와 섞지 않습니다. 다음 실험은 동일
split·capacity·seed 11/22/33 조건에서 새 후보를 다시 qualification해야
합니다.

## Artifacts

- [Pilot curve and overlap diagnostic](PatchTSTSSL5090Pilot-20260724.json)
- [Multi-seed aggregate](PatchTSTSSL5090Comparison-20260724.json)
- [Per-seed cases](PatchTSTSSL5090ComparisonCases-20260724.csv)

모든 six-case checkpoint는 strict load와 7,000-series inference를
통과했습니다.

## Source Snapshot

로컬 기준 branch는 `exogenous-models`, base HEAD는
`846d3e2a1adfb453931a0e79850a9a9cd3b36865`입니다. 실험은 해당
HEAD 위의 현재 미커밋 작업 트리를 격리 복사해 실행했습니다.

| File | SHA-256 |
|---|---|
| `patchtst_pretrain.py` | `adef21d65b88e0d049270c6f0ad6e9b99a5d3c38bc8c0b99f3261976608f7d68` |
| `patchtst_finetune.py` | `250b8daf58f1437ea624da85f260662238aac7596ac6fe83ec15c86a3e31a3a2` |
| `total_train.py` | `d6a0e58be791f803b8f528bbf1ef684e14ef248fef28ab94a26e9872ed104d1a` |
| `dsio_total_running.py` | `79c01e238742629e877bf80cebbcac56a5565b74f4183047f878abae7592e1f3` |
| `benchmark_patchtst_ssl_5090.py` | `d160a0b00df60eca537e1f90a7b9cc44b3a4fad9ce55caa2e24db3b544da30f8` |

로컬과 5090의 위 파일 SHA-256은 일치합니다.

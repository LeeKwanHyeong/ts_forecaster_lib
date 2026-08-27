# DSIO 202545 Endogenous Production Refit

## Status

- Date: 2026-07-23
- Branch: `exogenous-models`
- Production-refit support commit: `c2135a3`
- Forecast origin: `202545`
- Decision: fix five endogenous checkpoints at their qualified epoch counts

## Shared Data Contract

| Item | Value |
|---|---:|
| Source | `raw_data/master/tb_master_target.parquet` |
| Source rows | 2,455,508 |
| Series | 7,000 |
| Source range | `201801..202544` |
| Source SHA-256 | `328f547d5eb0a50c80dc60dc7bb89c09799599f8f6b8677406a0e3cc4a3ef547` |
| Lookback / horizon | `52 / 27` |
| Window stride | `4` |
| Training windows | 480,072 |
| Validation windows | 0 |
| Seed | 42 |

Every run used `training_mode=production_refit`. No validation loader, early
stopping, or best-state restoration was used; each artifact stores the final
configured epoch.

## Artifact Inventory

| Model key | Epochs | Parameters | Checkpoint | SHA-256 |
|---|---:|---:|---|---|
| `patchtst_base` | 8 | 403,099 | `weekly_PatchTST_L52_H27.pt` | `2674a5b01a882a7d3bf36af598d787136d2c15181879307989a8206a43fa2d78` |
| `patchtst_quantile` | 3 | 8,941,265 | `weekly_PatchTSTQuantile_L52_H27.pt` | `e99ca9d843b1303529d06a8ea6996fba7e477fe21f1e502e33b9e5840c98ee0c` |
| `patchmixer` | 3 | 129,584 | `weekly_PatchMixer_L52_H27.pt` | `727e77b8e3bf72e9bcb42fb48c9f22fc29e9fa11439ef5413305f9327cc9b677` |
| `nhits_base` | 31 | 276,682 | `weekly_NHITSBase_L52_H27.pt` | `391a559d81a0202c4f230a92d6ce1d5e13bf907c230b3db4674611b28caec861` |
| `timemixer` | 33 | 21,548 | `weekly_TimeMixer_L52_H27.pt` | `50a67e0ff3d45b3ca146d3419625a292f13178cb6582557980329d78cd68a184` |

The PatchTST Small artifact was trained at source commit `5894dfe`; the other
four artifacts were trained at `c2135a3`. All five are strictly loadable by
the production-refit contract at `c2135a3`.

## RTX 5090 Verification

Each checkpoint was loaded with `strict=True` in a fresh process and generated
all 189,000 rows for 7,000 series over `202545..202619`.

| Model key | Forecast SHA-256 | Negative raw rows | Minimum raw point | Inference | Series/s |
|---|---|---:|---:|---:|---:|
| `patchtst_base` | `84110bfa70b2bf8a6d56d687d49642d75587ba92ea79442cd886c9a3ab16f2a1` | 38,966 | -165.782 | 0.201 s | 34,907 |
| `patchtst_quantile` | `8672fb4cd8d0b5f51a5c41e87827dae0ce73cad068399e4a1620955ce94fc01d` | 55,845 | -68.689 | 0.223 s | 31,349 |
| `patchmixer` | `978e29fdc53037b5f096f9b3d82c71e3cbdfdc48e03c6a2a103718d6a16075da` | 88,796 | -158.278 | 0.233 s | 30,029 |
| `nhits_base` | `9ac37b6eeab93b3ac5234b1e346aec0174f0e1afd9ce4d4d8adf3f0af563f123` | 46,737 | -10.737 | 0.115 s | 60,650 |
| `timemixer` | `5d1506b8640c0d8d07c9b3b8f9d2307c27da26a9bed5bce0c2ad106ac0b0fbc7` | 78,091 | -133.609 | 0.191 s | 36,635 |

Negative values are intentional raw model outputs. The library preserves
those values; Demand Engine owns the nonnegative demand projection at its
processed-output boundary.

## Artifact Roots

- PatchTST:
  `artifacts/dsio_202545_patchtst_small_production_refit_5894dfe`
- PatchTST Quantile:
  `artifacts/dsio_202545_patchtst_quantile_production_refit_c2135a3`
- PatchMixer:
  `artifacts/dsio_202545_patchmixer_production_refit_c2135a3`
- NHITS:
  `artifacts/dsio_202545_nhits_production_refit_c2135a3`
- TimeMixer:
  `artifacts/dsio_202545_timemixer_production_refit_c2135a3`

The canonical checkpoint is under each root's `endo_only/` directory. The
binary artifacts remain on the RTX 5090 host and are not committed to Git.

## Private Wheel Deployment

The production library artifact was built once from a clean detached
`c2135a343f0bd5ae84dfc49b45027af7c557da65` worktree:

| Item | Value |
|---|---|
| Wheel | `modeling_module-0.2.0-1c2135a3-cp312-none-any.whl` |
| Wheel SHA-256 | `65555c6e3ac6cad945761d4960fc2e0cab55e1e8416ca9f926feef17c2f40c8f` |
| Source wheel SHA-256 | `b28bb3973c393b49676b57981895c0a2c648fb1ef39395434db47301e4d46620` |
| Build tag | `1c2135a3` |
| Python tag | `cp312` |
| Builder worktree dirty | `false` |

The local retained copy is
`dist/private/modeling_module-0.2.0-1c2135a3-cp312-none-any.whl`; `dist/` remains
Git-ignored, while the artifact identity is committed in this document and the
Demand Engine production Registry.

The same byte-for-byte wheel was installed with `--no-deps` in:

- RTX 5080:
  `/home/leekwanhyeong/miniconda3/envs/demand_engine`
- RTX 5090:
  `/home/leekwanhyeong/.venvs/ts_forecaster_non_sellm_e53269e`

Both hosts reported `modeling-module==0.2.0`, PyTorch `2.11.0+cu130`, the
expected GPU, all five public model keys, and the embedded `c2135a3`
provenance. The RTX 5090 installation also strictly restored all five
production checkpoints with the parameter counts listed above.

The RTX 5090 Demand Engine runtime root is:

`/home/leekwanhyeong/workspace/DemandEngine-v2/data/models/weekly/production/202545`

Its `.env` sets `FORECAST_CHECKPOINT_ROOT` to that absolute path. All copied
checkpoint hashes match the sealed Demand Engine production Registry.

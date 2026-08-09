# ADR-0005: ts_forecaster_lib V2 lineage and Samsung GCS migration

## Status

Accepted on 2026-08-09.

## V2 baseline

- Branch: `ts_forecaster_lib_v2`
- Branch point: `5bafabab09565453c50ea3cc71e99d47d167ee3c`
- Source branch: `origin/exogenous-models`
- RTX 5090 training source: `c2135a343f0bd5ae84dfc49b45027af7c557da65`
- Ancestry check: `git merge-base --is-ancestor c2135a3 5bafaba` passed.

The V2 branch starts from the maintained model and public API lineage. The
historical `samsung_gcs` branch remains available as the original Samsung GCS
experiment record and is not used as the V2 base.

## RTX 5090 H26 compatibility

The five V100 checkpoints were trained with modeling-module 0.2.0 at
`c2135a3`, using `production_refit`, seed 42, lookback 52, horizon 26,
train end week 202509, and forecast origin 202510.

All checkpoints passed `load_predictor(..., strict=True)` on V2 and produced
exactly 26 public forecast rows with horizon steps W0 through W25.

| Model key | Restored class | State keys | Parameters | Checkpoint SHA-256 |
|---|---|---:|---:|---|
| `patchtst_base` | `PatchTSTEndogenousModel` | 39 | 402,970 | `38861c34df002ae3cb8198f9c8d845fe7dba594b3161b0c7fd712528107e949c` |
| `patchtst_quantile` | `PatchTSTQuantileEndogenousModel` | 41 | 426,190 | `b14eb08d06b92ac78ef3a13319c9509a1c8a4c54b68652a13b4b2ca2e6d2aa96` |
| `patchmixer` | `PatchMixerModel` | 94 | 124,866 | `98565c666b6b053b74f3318ff56ca743c8784d167e7330a6076ae7adf9d1c7db` |
| `nhits_base` | `NHITSModel` | 18 | 276,425 | `b45407b2cc5f7b90ebc8dd797f3d8b9be3b69eda2f71b413f9f5af94173aab9e` |
| `timemixer` | `TimeMixerModel` | 81 | 21,447 | `b63831f9b5c6cacb9e133b6ad123d10f3cd493aefe35a7d634f9536b0fd79bf0` |

The checkpoint files remain external runtime artifacts. They are not committed
to this source repository.

## Samsung GCS migration decision

The four Samsung commits were reviewed by file instead of cherry-picked as a
block.

| Commit | Decision |
|---|---|
| `52ce0f3` | Preserve the no-future versus token cross-attention experiment intent through the maintained public AB runner. Do not restore the legacy `head_flatten` model path or direct model edits. |
| `7cb4384` | Do not restore generated notebooks or legacy wrapper changes. Their maintained replacement is `run_exogenous_model_ab.py`. |
| `217d3b3` | Preserve the PatchTST future-exogenous reports. Do not restore the duplicated legacy training runner. |
| `c68aba3` | Preserve the architecture matrix and eight-revision FAR method through V2 modules using current long-format output and public runner contracts. |

Maintained V2 replacements:

- `src/model_test/exogenous_test/run_samsung_gcs_patchtst_sweep.py`
- `src/model_test/exogenous_test/far_metrics.py`
- `tests/test_samsung_gcs_v2_tools.py`

The following legacy behavior is intentionally excluded:

- direct construction of `MultiPartExoDataModule`
- direct calls to `run_total_train_weekly`
- `head_flatten` future-exogenous fusion
- hard-coded Windows data and artifact paths
- generated notebook output and duplicate experiment runners

## Repository policy

- Model papers and the PatchTST comparison reports are kept under each model's
  `docs` directory.
- PDF files are marked binary through `.gitattributes`.
- Checkpoints, benchmark output, local agent state, IDE state, archives, and OS
  metadata remain ignored.

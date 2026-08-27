# DSIO 202545 PatchTST Multi-Seed Capacity Decision

## Scope

This decision promotes a DSIO operational default for the endogenous
`patchtst` family. It does not rewrite the historical seed-42 capacity sweep or
the five-model qualification baseline.

The comparison uses two capacities:

| Capacity | `d_model` | Layers | `d_ff` | Parameters |
|---|---:|---:|---:|---:|
| Small | 128 | 2 | 512 | 403,099 |
| Current control | 384 | 5 | 1,536 | 8,891,931 |

All seed `11 / 22 / 33 / 42` artifacts use separate Python processes, the same
RTX 5090, canonical source SHA-256, temporal split, MAE loss, patch contract,
40-epoch qualification limit, and public strict-load evaluator described in
`DSIO202545PatchTSTCapacitySweep.md`.

## Overall Qualification

| Seed | Small MAE | Current MAE | Small improvement |
|---:|---:|---:|---:|
| 11 | 8.718399 | 9.877556 | 11.7353% |
| 22 | 8.038179 | 8.755594 | 8.1938% |
| 33 | 10.102836 | 9.836910 | -2.7034% |
| 42 | 9.143390 | 10.289115 | 11.1353% |

| Capacity | Mean MAE | MAE std | Worst MAE | Mean WAPE | Mean sMAPE |
|---|---:|---:|---:|---:|---:|
| **Small** | **9.000701** | 0.864331 | **10.102836** | **53.0193%** | **138.9124%** |
| Current | 9.689793 | **0.655442** | 10.289115 | 57.0785% | 138.9397% |

Small wins three of four seeds and improves mean MAE/WAPE by `7.1115%`.
Its seed variance is higher, but its worst observed seed is still better than
the Current control's worst seed. Mean sMAPE is effectively tied because the
qualification target remains dominated by zero-demand cells.

### Checkpoint identity

| Seed | Small SHA-256 | Current SHA-256 |
|---:|---|---|
| 11 | `88fa47ae8365858a0ac6eb683822691cdefaceaa18ca76dace4289db274bad8a` | `1abc5e5f4816a7b5c89d184c165004b1ac360ea657a85bd46edefa93dd5a5897` |
| 22 | `187112d9b237cf92a59175d0c19899af6ed78108482a2f780b8ca424986fbdfd` | `800b4449a7b48cc7ebdb9bf7a0a8d30bd94ecb5ce923cdff55b825324f0264c8` |
| 33 | `ec9e80d11ddff4399ced48aa34ee4140d0a30e5735d9e2ba676652475d9ac3fd` | `6092e3180ed0661558b4e4b8b526390edda0e0a6e6325ff74ff6fb20fb003537` |
| 42 | `0ac84618edaa46831a060da428b909b4f2eb09a71ab0f6773b60279a7b012cfa` | `f9949e740c491bf87556b647f7b21907f6cf8322e39a571ce43523d45d4f8bc8` |

## Demand Cohorts

Cohorts are computed only from observations through `202517`, so the
qualification target is not used for classification. The implementation
matches the Demand Engine Syntetos-Boylan contract:

- `ADI < 1.32`: smooth or erratic, grouped as `dense`
- `ADI >= 1.32`: intermittent or lumpy, grouped as `intermittent`
- `CV2` threshold: `0.49`
- zero epsilon / minimum periods: `0.0 / 10`

The resulting population is `6,524` dense and `476` intermittent series.

| Cohort | Small mean MAE | Current mean MAE | Small seed wins | Decision |
|---|---:|---:|---:|---|
| Dense | **8.615243** | 9.610421 | **4 / 4** | Small default |
| Intermittent | 14.283735 | **10.777659** | 1 / 4 | Small is not preferred |

Small improves dense MAE by `10.3552%` and is better on 25 of 27
cohort-horizon averages. For intermittent demand, Small is `32.5310%` worse on
mean MAE and Current is better on all 27 cohort-horizon averages. Seed 33 is
the principal Small failure case for this cohort.

This means one global PatchTST checkpoint should use Small for the current DSIO
population, but intermittent demand remains an explicit routing exception.
Current capacity is retained as a reproducible control until the Demand Engine
compares it with N-HiTS and intermittent statistical models. This experiment
does not introduce automatic capacity routing inside `ts_forecaster_lib`.

## Horizon Stability

Small has lower four-seed mean MAE on 22 of 27 horizons. The five exceptions
are H1, H2, H3, H5, and H6.

| Horizon band | Small mean MAE | Current mean MAE | Small change |
|---|---:|---:|---:|
| H01-04 | 3.018606 | **2.872479** | 5.0872% worse |
| H05-13 | **6.411558** | 6.708761 | 4.4301% better |
| H14-27 | **12.374319** | 13.553976 | 8.7034% better |

The Small default is therefore strongest for the medium and long portion of
the 27-week operating horizon. Near-term forecasting and intermittent demand
must remain visible in downstream model selection rather than being hidden by
the global micro average.

## Production Refit Epoch

Individual Small best-validation epochs are:

| Seed | Best epoch | Best validation loss |
|---:|---:|---:|
| 11 | 14 | 8.731728 |
| 22 | 23 | 8.020154 |
| 33 | 1 | 10.094755 |
| 42 | 12 | 9.149590 |

For one seed-independent production policy, validation loss is averaged across
the four isolated runs at every epoch. The earliest global minimum is:

- fixed production-refit epochs: **8**
- mean validation loss at epoch 8: `9.83644225`
- seed validation losses at epoch 8:
  `9.371605 / 10.102849 / 10.408137 / 9.463178`

Production refit must train Small from scratch on every target through
`202544`, retain the qualification scheduler horizon `t_max=40`, run exactly
eight epochs, save the final epoch state, and not early-stop or restore a
best-validation state using the consumed qualification holdout.

The epoch policy is fixed here. The existing qualification runner still
constructs a validation loader and restores best state, so a dedicated
production-refit execution mode is required before creating the promoted
`.pt` artifact.

## Default Decision

- DSIO `patchtst` default capacity: **Small (`128 / 2 / 512`)**
- fixed production-refit epochs: **8**
- Current (`384 / 5 / 1536`) remains available through explicit CLI overrides
- no checkpoint format, registry key, or public API change
- no automatic intermittent/dense capacity switching in the library

The default applies to `dsio_total_running.py`. Compact public
`PatchTSTArchitectureConfig` fields remain optional, and existing Current
checkpoints remain loadable.

## Reproducibility Artifacts

RTX 5090 artifact root:

`artifacts/dsio_202545_patchtst_capacity_multiseed_cbea159`.

The `analysis` directory contains:

- seed and capacity metrics
- cohort, horizon, horizon-band, and cohort-horizon metrics
- paired Small-versus-Current stability tables
- complete epoch curves and seed-specific best epochs
- `production_refit_policy.json`
- source and temporal contract summary

The aggregation entry point is
`tools/analyze_patchtst_capacity_multiseed.py`.

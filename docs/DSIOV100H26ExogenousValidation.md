# DSIO V100 H26 Exogenous Validation

## Scope

- Source commit: `9af7449`
- Checkpoint training commit: `f750d9d`
- Window: weekly L52/H26
- Validation target: `202436` through `202509`
- Source series: 7,000
- Eligible series: 6,952
- Excluded series: 48 with insufficient pre-validation history
- Forecast points per model: 180,752
- Seed: 42
- Checkpoint selection: best validation state within 40 epochs

This is a detailed comparison of the qualification validation set. It is not an
untouched test-set estimate because the same validation loss selected each
checkpoint's best epoch.

## Overall Results

The primary comparison applies the nonnegative demand rule `max(0, raw)`.
WAPE, sMAPE, and normalized bias are shown as percentages. Bias is the signed
mean error in demand units, where a positive value means overforecasting.

| Rank | Model | MAE | WAPE | sMAPE | Bias | Normalized bias |
|---:|---|---:|---:|---:|---:|---:|
| 1 | PatchTSTExogenous | 1.1878 | 25.2350% | 50.6668% | +0.1623 | +3.4475% |
| 2 | ExoTST | 1.2548 | 26.6591% | 52.7229% | -0.0606 | -1.2880% |
| 3 | TimeXer | 1.3430 | 28.5338% | 72.6657% | +0.3235 | +6.8736% |

PatchTSTExogenous improves MAE by 5.34% over ExoTST and 11.56% over
TimeXer. ExoTST has the smallest absolute bias, while TimeXer has both the
largest overall error and the strongest overforecast bias.

## Output Policy Effect

| Model | Raw MAE | Nonnegative MAE | Raw negative rate | Raw sMAPE | Nonnegative sMAPE |
|---|---:|---:|---:|---:|---:|
| PatchTSTExogenous | 1.2673 | 1.1878 | 20.5918% | 88.3272% | 50.6668% |
| ExoTST | 1.3864 | 1.2548 | 23.9527% | 93.7404% | 52.7229% |
| TimeXer | 1.3967 | 1.3430 | 8.8547% | 88.0647% | 72.6657% |

The nonnegative rule improves every model and does not change the overall
ranking. Raw negative output is frequent enough that raw and processed metrics
must remain separate in future comparisons.

## Horizon Segments

| Model | Segment | MAE | WAPE | sMAPE | Bias |
|---|---|---:|---:|---:|---:|
| PatchTSTExogenous | W0-W8 | 1.0331 | 17.9802% | 39.0891% | -0.1564 |
| PatchTSTExogenous | W9-W17 | 1.1751 | 24.3877% | 49.2779% | -0.0670 |
| PatchTSTExogenous | W18-W25 | 1.3760 | 40.3236% | 65.2542% | +0.7787 |
| ExoTST | W0-W8 | 1.0575 | 18.4047% | 43.8354% | -0.2483 |
| ExoTST | W9-W17 | 1.3769 | 28.5762% | 56.7954% | -0.2676 |
| ExoTST | W18-W25 | 1.3394 | 39.2502% | 58.1398% | +0.3834 |
| TimeXer | W0-W8 | 1.1612 | 20.2097% | 62.7687% | +0.0659 |
| TimeXer | W9-W17 | 1.4205 | 29.4807% | 76.0364% | +0.1392 |
| TimeXer | W18-W25 | 1.4604 | 42.7977% | 80.0079% | +0.8207 |

PatchTSTExogenous wins 16 of 26 horizons, covering W1, W3, and W6-W19.
ExoTST wins the other 10 horizons, including every horizon from W20 through
W25. TimeXer does not win an individual horizon. The late-horizon result means
PatchTSTExogenous is the best aggregate checkpoint, but ExoTST is more stable at
the far end of H26.

## Horizon MAE

| Horizon | PatchTSTExogenous | ExoTST | TimeXer |
|---|---:|---:|---:|
| W0 | 0.8049 | **0.7110** | 0.8393 |
| W1 | **0.9355** | 1.0269 | 1.0606 |
| W2 | 0.9409 | **0.8779** | 1.1210 |
| W3 | **0.9435** | 0.9478 | 1.1679 |
| W4 | 1.1019 | **1.0128** | 1.0863 |
| W5 | 1.1089 | **1.0817** | 1.2312 |
| W6 | **1.1565** | 1.3157 | 1.3358 |
| W7 | **1.0169** | 1.2106 | 1.2393 |
| W8 | **1.2890** | 1.3332 | 1.3695 |
| W9 | **1.1607** | 1.1657 | 1.3931 |
| W10 | **1.0881** | 1.4042 | 1.3713 |
| W11 | **1.2080** | 1.4410 | 1.4033 |
| W12 | **1.3780** | 1.5196 | 1.5429 |
| W13 | **1.1196** | 1.2752 | 1.3911 |
| W14 | **1.0935** | 1.4600 | 1.4255 |
| W15 | **1.2238** | 1.4428 | 1.4168 |
| W16 | **1.2382** | 1.4598 | 1.4851 |
| W17 | **1.0660** | 1.2241 | 1.3554 |
| W18 | **1.0547** | 1.3565 | 1.2886 |
| W19 | **1.2097** | 1.3492 | 1.3471 |
| W20 | 1.3675 | **1.2894** | 1.4738 |
| W21 | 1.3622 | **1.2154** | 1.4439 |
| W22 | 1.4709 | **1.4482** | 1.4654 |
| W23 | 1.4038 | **1.3834** | 1.5010 |
| W24 | 1.5869 | **1.3930** | 1.5686 |
| W25 | 1.5524 | **1.2798** | 1.5950 |

## Evidence

Remote artifact root:
`/home/leekwanhyeong/artifacts/exogenous-h26-qualification-f750d9d-seed42-e40/validation-evaluation`

- `validation-overall.csv`: `79f6d70391b697c402cf60ec615339b4f048928a47df35c49ad100c883966c43`
- `validation-by-horizon.csv`: `e02f86360cc72f51ac873fc0bbbf90aedfaf188e1ddaa5b414048dacb9733d4d`
- `validation-metrics.json`: `7aa7503b99c41f0e8451fd2ac662d6c9c2730d1adbfe2e404a36e4d20086a243`

The JSON receipt also seals the source dataset, input manifest, qualification
receipt, and all three checkpoint SHA-256 values.

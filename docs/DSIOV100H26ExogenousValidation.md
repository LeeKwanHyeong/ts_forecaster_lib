# DSIO V100 H26 Exogenous Validation

## Scope

- Initial evaluation commit: `9af7449`
- Multi-seed qualification commit: `ee5d18b`
- Seed 42 checkpoint training commit: `f750d9d`
- Window: weekly L52/H26
- Validation target: `202436` through `202509`
- Source series: 7,000
- Eligible series: 6,952
- Excluded series: 48 with insufficient pre-validation history
- Forecast points per model: 180,752
- Initial baseline seed: 42
- Stability seeds: 11, 22, 33
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

At seed 42, PatchTSTExogenous improves MAE by 5.34% over ExoTST and 11.56%
over TimeXer. ExoTST has the smallest absolute bias, while TimeXer has both the
largest overall error and the strongest overforecast bias. The multi-seed
results below supersede this single-seed ranking for model strategy decisions.

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

At seed 42, PatchTSTExogenous wins 16 of 26 horizons, covering W1, W3, and
W6-W19. ExoTST wins the other 10 horizons, including every horizon from W20
through W25. TimeXer does not win an individual horizon.

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

## Multi-seed Stability

Seeds 11, 22, and 33 retrain ExoTST and PatchTSTExogenous from scratch with
the same data, split, architecture, loss, 40-epoch limit, and best-validation
state selection. The table uses nonnegative point output.

| Seed | ExoTST MAE | PatchTSTExogenous MAE | Winner |
|---:|---:|---:|---|
| 11 | **1.2630** | 1.4099 | ExoTST |
| 22 | 1.3062 | **1.2175** | PatchTSTExogenous |
| 33 | **1.2053** | 1.2235 | ExoTST |
| Mean | **1.2582** | 1.2836 | ExoTST |
| Population standard deviation | **0.0413** | 0.0893 | ExoTST |

ExoTST wins two of three stability seeds and has less than half the MAE
variation of PatchTSTExogenous. PatchTSTExogenous therefore does not retain its
seed 42 aggregate advantage consistently and must not be promoted as the sole
default model.

Across seeds 11, 22, 33, and the supplemental seed 42 baseline, W18 favors
PatchTSTExogenous in three of four seeds, while W19 favors ExoTST in three of
four. ExoTST wins W22-W25 in all four seeds. This places the stable contiguous
handoff boundary between W18 and W19.

## Routing Comparison

A cutoff of `k` means PatchTSTExogenous supplies W0 through W(k-1), and
ExoTST supplies Wk through W25. The primary comparison below uses seeds 11,
22, and 33.

| Strategy | Mean MAE | MAE std | Worst-seed MAE | Mean WAPE | Mean sMAPE | Mean bias |
|---|---:|---:|---:|---:|---:|---:|
| ExoTST only | 1.2582 | 0.0413 | 1.3062 | 26.7308% | 56.9503% | -0.1911 |
| Cutoff W18 | 1.2275 | **0.0368** | **1.2793** | 26.0792% | 55.2406% | +0.0325 |
| Cutoff W19 | **1.2259** | 0.0409 | 1.2838 | **26.0462%** | 55.2769% | +0.0563 |
| PatchTSTExogenous only | 1.2836 | 0.0893 | 1.4099 | 27.2720% | 57.8789% | +0.2419 |

The W19 cutoff has the lowest mean MAE among all 27 contiguous routing
candidates. It improves mean MAE by 2.56% over ExoTST-only and 4.50% over
PatchTSTExogenous-only. The same W19 cutoff remains the best mean-MAE candidate
when seed 42 is included.

## TimeXer Multi-seed Qualification

TimeXer was subsequently qualified from scratch with seeds 11, 22, and 33
under the same L52/H26 data, split, architecture, loss, 40-epoch limit, and
best-validation state selection. TimeXer remains an independent full-H26 model
and uses the 12 past continuous calendar features without future exogenous
features.

| Seed | Best epoch | Best validation loss | MAE | WAPE | sMAPE | Bias |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 29 | 1.386854 | 1.339047 | 28.4491% | 72.5298% | +0.359800 |
| 22 | 25 | 1.399899 | 1.356807 | 28.8265% | 71.9633% | +0.405909 |
| 33 | 10 | 1.408210 | 1.331664 | 28.2923% | 70.3282% | +0.289316 |
| Mean | 21.33 | 1.398321 | 1.342506 | 28.5226% | 71.6071% | +0.351675 |
| Population standard deviation | 8.18 | 0.008790 | 0.010552 | 0.2242% | 0.9334% | 0.047944 |

Each seed evaluates 6,952 eligible series and 180,752 forecast points. Its
horizon CSV contains all 26 steps for both raw and nonnegative output policies.
The qualification artifacts and their receipt seals are recorded below. This
section fixes qualification evidence only; it does not yet choose the TimeXer
production-refit epoch.

## Model Interpretation

The W19 cutoff comparison is diagnostic evidence only. It shows how the two
models differ by forecast distance, but it does not define a hybrid routing
policy and must not be used to join their horizon blocks.

PatchTSTExogenous and ExoTST remain independent full-H26 models. Each model is
trained, stored, selected, called, and evaluated separately for W0-W25. ExoTST
has the lower multi-seed mean MAE and variation, while PatchTSTExogenous retains
useful seed- and horizon-specific results. Model selection must compare their
complete outputs rather than combine partial horizons.

Both independent model outputs use the existing `max(0, raw)` postprocessing
contract. Production refit, artifact promotion, and Demand Engine registration
remain separate implementation and approval steps.

## Evidence

Remote artifact root:
`/home/leekwanhyeong/artifacts/exogenous-h26-qualification-f750d9d-seed42-e40/validation-evaluation`

- `validation-overall.csv`: `79f6d70391b697c402cf60ec615339b4f048928a47df35c49ad100c883966c43`
- `validation-by-horizon.csv`: `e02f86360cc72f51ac873fc0bbbf90aedfaf188e1ddaa5b414048dacb9733d4d`
- `validation-metrics.json`: `7aa7503b99c41f0e8451fd2ac662d6c9c2730d1adbfe2e404a36e4d20086a243`

The JSON receipt also seals the source dataset, input manifest, qualification
receipt, and all three checkpoint SHA-256 values.

Multi-seed evaluation roots:

- Seed 11: `/home/leekwanhyeong/artifacts/exogenous-h26-multiseed-ee5d18b-seed11-e40/validation-evaluation`
  - Evaluation JSON SHA-256: `fc111a09e1588a69557986992a54f5bc767e842001864b17c4ff0ae6da342eaf`
  - ExoTST checkpoint: `9d3d1c30769f982c3235c78c1fe686c988cafaa9e13fa5799fa74281528fe7dd`
  - PatchTSTExogenous checkpoint: `fcd0c437ca624c0c23df33218fe53aec4e3c0992ce8233205064dd5747225086`
- Seed 22: `/home/leekwanhyeong/artifacts/exogenous-h26-multiseed-ee5d18b-seed22-e40/validation-evaluation`
  - Evaluation JSON SHA-256: `5edd2334d9edd92ae6fe3cab73aab5574b65ca0d2e0021525fafe23a9ebf68f0`
  - ExoTST checkpoint: `84fddf647b73cbce29e5cbe05d251d0673a74575bf4a15e33fb5c19e167927ee`
  - PatchTSTExogenous checkpoint: `cd30a1e94e63b8791274770514e8d6c7841de489447aa497936835dc39d9bf3b`
- Seed 33: `/home/leekwanhyeong/artifacts/exogenous-h26-multiseed-ee5d18b-seed33-e40/validation-evaluation`
  - Evaluation JSON SHA-256: `a47fc61016304af02bdfc0e6c555e8558009e2abd1c54d8d74a00193aea4e338`
  - ExoTST checkpoint: `be5b8d736c9912ad07d7a4914beb3575723e21497c9e3869331ac4ba59f50b39`
  - PatchTSTExogenous checkpoint: `996a5bbe7ac2320500e066a9cf3858dcf991428b941699176de1ec363008e7a7`

All three evaluation receipt seals were recomputed successfully.

TimeXer multi-seed roots at source commit `bdea737`:

- Seed 11: `/home/leekwanhyeong/artifacts/exogenous-h26-timexer-multiseed-bdea737-seed11-e40`
  - Checkpoint: `a3a531d0b99e996b17be60bbfa4dc00c714c76362ec3b166abf93f980f03db58`
  - Evaluation receipt: `f82d0d76d2883396564c489af3e6729e0ddc13b8f72d1accd0f560317346095c`
- Seed 22: `/home/leekwanhyeong/artifacts/exogenous-h26-timexer-multiseed-bdea737-seed22-e40`
  - Checkpoint: `cc8156c974262c586095dfd4d239aae16aff4e0a9da4c65bf0dfe964ccc87fda`
  - Evaluation receipt: `6e177832b05787b0517ed1730659a4a5bd9fe215196e51033d2b12d0933305f9`
- Seed 33: `/home/leekwanhyeong/artifacts/exogenous-h26-timexer-multiseed-bdea737-seed33-e40`
  - Checkpoint: `ed9b78ce95a2ad3bbfa872ea70732622fca16fdfabf377866db208a3cb01477e`
  - Evaluation receipt: `ffba8eebe5e888a5d666f89429f64285a7767da103a4741bb3396186731bba8f`

The aggregate qualification, model, and evaluation receipt seals for all three
TimeXer seeds were independently recomputed successfully.

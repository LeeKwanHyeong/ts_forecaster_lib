# Legacy distribution checkpoint fixtures

These are frozen, full PyTorch checkpoint artifacts built with the repository's historical model and writer code. They are intentionally tiny and contain primitive config dictionaries plus complete model `state_dict` values.

- `v1_*`: generated from commit `97b1eb8`, the last unversioned writer before the formal checkpoint version was introduced.
- `v2_*`: generated from commit `e038bbf`, whose writer emits `modeling_module.ckpt.v2`.
- Each model was initialized with `lookback=4`, `horizon=2`, small CPU dimensions, and either the historical `Normal` or `StudentT` `DistributionLoss`.
- `v2_*_point.pt` files are negative controls proving that fail-closed distribution detection still accepts ordinary point artifacts from every supported family.
- No training data or production weights are included.

The manifest records the immutable SHA-256, intended distribution, saved head width, and supported/rejected boundary. A successful legacy restore means structural recovery only: distribution family, parameter order, output multiplier, head architecture, and all saved state values. Loss options discarded by the old writer, such as `num_samples`, cannot be recovered exactly and use the documented legacy defaults.

Known rejected boundaries are deliberate:

- v1 PatchTST/PatchMixer configs use field names removed before the public API's current config schema.
- Titan v1/v2 saved a distribution-shaped head without distribution identity in its config.
- ExoTST v1/v2 overwrote its distribution head with a point head while retaining the distribution-loss state entry.

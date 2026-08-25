# AutoTimes integration

`autotimes_base` is the product adapter for the official THUML AutoTimes model.
The upstream baseline is fixed to commit
`9ff9aac5083e24c233404c35d7b7a3c0643f2c70` under the MIT license. Reviewed
file hashes are recorded in `../upstream_manifest.json`.

## Product contract

- Input and output follow the public library contract: `[B, L, C] -> [B, H, C]`.
- The lookback must be divisible by `token_len`.
- The LLM backbone is always frozen. Only the numeric tokenizer, detokenizer,
  and timestamp mixing scale are trainable.
- Forecasting is autoregressive by numeric token. H26 and H27 both return the
  exact requested horizon even when the last token is only partially used.
- Timestamp embeddings are accepted as an explicit tensor or as a SHA256
  verified artifact. No timestamp values are inferred inside the model.
- ICL execution accepts a sealed Episode artifact. Endogenous episodes contain
  only target history. Exogenous episodes additionally bind ordered observed-past
  and known-future feature schemas to the checkpoint by SHA256.
- Past and future exogenous features have separate roles and may use different
  widths. AutoTimes places them in disjoint numeric-token channels so future
  values cannot be read as observed history.
- Known-future exogenous values are rolled into each autoregressive step. A
  checkpoint trained without the sealed exogenous schema cannot accept them.

The official benchmark CLI, dataset loader, and experiment runners are not
vendored into the package.

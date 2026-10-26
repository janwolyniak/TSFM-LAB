# Changelog

## Next feature release in development

- Add Chronos-Bolt Base to the curated catalog and terminal setup using the
  existing adapter, without requiring an individual Base inference test.
- Separate adapter representative checks from selected checkpoint evidence,
  expose variant/runtime metadata, and snapshot project verification in new
  pretrained run provenance while preserving old checkpoint fingerprints.
- Document the runtime compatibility gates for the planned Chronos-2, TiRex-2,
  and TTM integrations. Those three adapters are not implemented by this change.

## 0.1.0 — Unreleased

The first local release candidate provides:

- A Python numerical core for one regular univariate series, CSV/optional Parquet
  input, fixed-context rolling origins, last-value/seasonal-naive forecasts, and
  pooled/per-lead MAE, RMSE, and MASE.
- Safe YAML configuration, UUID experiment records, checked artifacts, compatible
  comparisons, history, and CSV/JSON/Parquet forecast exports.
- A persistent `tsfm` terminal prompt with local slash commands, inline setup,
  command history/completion, progress, and managed process cancellation.
- An optional CPU Chronos-Bolt adapter with curated checkpoint metadata, explicit
  downloads, exact revision identities, and cached/local-only experiment loading.
- A second optional CPU adapter, `timesfm_2p5`, with pinned TimesFM 2.5 checkpoint
  identities, genuine median output, strict local loading, and the same shell,
  guided setup, comparison, and saved-record workflows.
- A bundled original seasonal demo, an educational direct-Python adapter example,
  installation/evaluation/extension guides, and source/wheel verification tools.
- A separate lead-1, stride-1 OHLC finance extension with explicit latency,
  long/flat/short signals, two-leg costs, short financing, costed benchmarks,
  self-contained checked JSON reports, and CSV equity exports.

The verified real model integrations are Chronos-Bolt Tiny/Small and TimesFM 2.5
on CPU. Other families, GPU/MPS, panels, covariates, probability
scoring, training, broker execution, and live trading remain outside this
candidate. Synthetic examples establish arithmetic and integration behavior,
without a forecasting-advantage claim.

See the [release verification record](docs/milestone5-verification.md) for local
evidence and outstanding platform/publication gates. No package release has been
published.
The [second-family record](docs/milestone6-verification.md) adds current CPU
integration evidence. The [finance record](docs/milestone7-verification.md)
documents the separately agreed accounting extension; the bounded
[fev study](docs/benchmarking-extension.md) does not replace core scoring.

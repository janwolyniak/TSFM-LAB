# Changelog

## Next feature release in development

- Add optional natural-language experiment drafts and clarifications, preserving
  prior choices and the selected local dataset through normal guided review.
- Add explanations of catalog models, checked experiments and saved finance reports
  with bounded, path-free evidence and separately displayed authoritative facts.
  Every request requires explicit preview/confirmation; no generated code or actions
  can start a run, change artifacts, or select another billing route.

- Add four explicit assistant connection routes, hidden credential prompts,
  session/native-vault storage, discovery/selection, and a confirmed fixed
  connection check. Qualification uses fake credentials and simulated providers;
  natural-language setup and explanations remain P7.
- Isolate managed numerical workers from arbitrary parent environment variables
  before interpreter startup, preserving progress, cancellation, and saved records.

- Add Generic, Finance, Energy, Temperature, and Weather usage profiles to
  configuration, both guided terminal interfaces, checked saved metadata,
  history, and result wording. Profile-only changes preserve numerical identity.
- Add original domain examples, explicit measurement/units questions, and an
  OHLC report command handoff preserving all existing execution inputs/rules.
- Preserve full native-point output labels in CLI and screen comparisons.

- Add Granite TTM R2 with three pinned exact-context profiles, native 96-step
  prefix forecasts, causal external standardization, and terminal selection.
- Add explicit native-point comparisons preserving actual output labels and
  legacy median records. OHLC reports continue to require the median task.
- Qualify the shared Transformers 4.56 runtime with cached representative TTM
  and five existing-checkpoint CPU checks. Larger TTM profiles remain
  compatibility checked without separate inference. See the P4 record.

- Add Chronos-Bolt Base to the curated catalog and terminal setup using the
  existing adapter, without requiring an individual Base inference test.
- Separate adapter representative checks from selected checkpoint evidence,
  expose variant/runtime metadata, and snapshot project verification in new
  pretrained run provenance while preserving old checkpoint fingerprints.
- Add the optional Chronos-2 adapter with curated Small/standard checkpoints,
  strict CPU float32 local loading, independent native median forecasts,
  terminal/YAML selection, and saved-result compatibility. Small receives a
  bounded real CPU check; standard remains metadata/API checked without inference.
- Add TiRex-2 with four curated training profiles, per-profile limits, strict
  Safetensors loading, CPU median forecasts, and terminal/YAML selection.
- Expose variant-specific bounds and separate optional-runtime notices; include
  the NXAI attribution and mLSTM Kernels Community License copy in distributions.
- Coordinate checkpoint runtimes on Hub 0.35.3 while retaining Torch 2.8 and
  Transformers 4.55. P3 qualification records the representative TiRex check
  and existing-family regressions. P4 subsequently qualifies the TTM runtime.

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

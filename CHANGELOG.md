# Changelog

## 0.3.0 — Unreleased

- Extend distribution audits and exact installed-wheel checks to the expanded
  catalog, seven fake assistant routes, onboarding preferences, guided historical
  and future baseline workers, detached exports and old saved records. Keep
  Windows-only runtime requirements optional under their explicit extra.
  See [development qualification](docs/v0.3.0-qualification.md) for local evidence
  and the remaining release gates.

- Add Windows drive-path and terminal quoting support, portable locking/build
  checks and a Linux/macOS/Windows Python 3.11–3.13 CI matrix. Preserve detached
  cross-platform records without changing numerical identities; prevent record-only
  configurations from starting work. Native Windows qualification remains pending.
  See [Windows support and limits](docs/windows.md).

- Add S7 curated static documentation generation with a metadata-only model
  matrix, five category examples, separate historical/future semantics, citation
  guidance, crawl files and consistent canonical/structured metadata. Require an
  explicit candidate HTTPS base URL and a new output directory. Existing private
  documentation, datasets and experiments are not copied or deployed. See the
  [public documentation workflow](docs/public-documentation.md).

- Add S6 TTM R2 native horizons 192/336/720 and R2.1 weekly/daily profiles with
  strict config/cadence validation and recorded frequency tokens. Add the TOTO 2
  4M family on Python 3.12/3.13 with explicit project bounds, strict safe loading,
  native median output and refusal of nonfinite inverse outputs. Old identities
  remain unchanged. Installation/import, real inference and native-platform
  qualification for new behavior remain pending; see the
  [TTM](docs/s6-ttm-verification.md) and [TOTO](docs/s6-toto2-verification.md) records.

- Add S5 DeepSeek, local installed Ollama and explicit compatible endpoint
  connections through existing reviewed drafting/explanation services. Keep
  credentials bound to each endpoint, bounded nonstreaming protocols and no
  automatic cloud fallback or local model pull. Fake-provider checks pass; live
  routes and native credential-vault qualification remain pending. See the
  [S5 record](docs/s5-connections-verification.md).

- Add S4 future point forecasting from the final observation in guided/assisted,
  YAML/CLI and separate Python APIs. Enforce native/operational limits, causal
  context, UTC grids and actual output labels; save eight validated artifacts
  with unavailable accuracy. Reuse reviewed preparation and managed cancellation,
  preserve historical schemas, and refuse future comparison/finance reports.
  Explanations retain metadata-only disclosure. See
  [future forecasting](docs/future-forecasting.md).

- Accept checked historical guided runs in finance reports, retaining the report's
  original five parent-artifact identities after validating all seven guided
  artifacts. Preserve report schema and accounting results; reject damaged
  chart/provenance outputs before accounting.

- Add S3 guided checkpoint review/preparation with exact-revision/file consent,
  offline cache reuse, runtime and integrity checks, managed stage/byte/window
  progress, and durable cancellation/failure records. Save checked SVG/provenance
  outputs atomically with guided schema-2 manifests while preserving old records
  and script defaults. Real pretrained qualification remains pending. See
  [guided preparation](docs/guided-preparation.md).

- Add S2 model-specific setup with friendly family/generation/variant choices,
  applicable options before dataset selection, original-unit previews, and
  advanced checkpoint details. Share settings validation across guided/assisted,
  YAML, and Python entry points; preserve compatible edits and refresh dependent
  choices on back navigation. Historical semantics and valid record identities
  remain unchanged. See [model-specific setup](docs/model-setup.md).

- Begin S1 with optional first-launch onboarding, `/setup`, `/settings`, and
  atomic nonsecret directory/category/export preferences. Reuse optional assistant
  connection management and the offline baseline demo; retain explicit option and
  YAML precedence. Report runtime package compatibility without installing or
  importing inference frameworks. See [onboarding](docs/onboarding.md).

## 0.2.0 — 2026-10-27

Released through GitHub with source and wheel assets. Hosted CI is skipped under
the owner's explicit release exception; local verification and prior bounded
model evidence are recorded in [the v0.2.0 release record](docs/v0.2.0-verification.md).
Assistant routes are implemented and fake-provider tested; live provider and OS
credential-vault behavior remain unqualified. No PyPI package is published.

- Add release checks binding source archives to a committed revision and installed
  package bytes to the selected wheel, with retained pre-profile compatibility
  fixtures. Include project citation guidance in source distributions.

- Add optional natural-language experiment drafts and clarifications, preserving
  prior choices and the selected local dataset through normal guided review.
- Add explanations of catalog models, checked experiments and saved finance reports
  with bounded, path-free evidence and separately displayed authoritative facts.
  Every request requires explicit preview/confirmation; no generated code or actions
  can start a run, change artifacts, or select another billing route.

- Add four explicit assistant connection routes, hidden credential prompts,
  session/native-vault storage, discovery/selection, and a confirmed fixed
  connection check. Qualification uses fake credentials and simulated providers;
  natural-language setup and explanations use the reviewed P7 workflow above.
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
evidence and the platform/publication gates recorded at that stage. The v0.1.0
candidate was not published separately.
The [second-family record](docs/milestone6-verification.md) adds current CPU
integration evidence. The [finance record](docs/milestone7-verification.md)
documents the separately agreed accounting extension; the bounded
[fev study](docs/benchmarking-extension.md) does not replace core scoring.

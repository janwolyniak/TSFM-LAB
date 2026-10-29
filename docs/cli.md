# CLI and experiment configuration

The shell workflow runs local forecasting experiments synchronously. It provides
validated YAML input, saved records, history, controlled comparisons, and exports.
Normal experiments do not download models. Explicit `models pull` manages curated
checkpoints. `tsfm` without arguments opens the optional [interactive terminal](tui.md)
when both input and output are terminals, and prints help otherwise. Explicit
`tsfm ui --workspace results` opens it on a terminal. Named shell commands remain
noninteractive.

## Commands and output

| Command | Behavior |
| --- | --- |
| `tsfm models list` | List baselines and Chronos-Bolt, Chronos-2, TimesFM 2.5, and TiRex-2 without loading weights. |
| `tsfm models info IDENTIFIER` | Show offline catalog metadata for an adapter or curated checkpoint repository. |
| `tsfm models pull REPOSITORY --revision REF` | Explicitly resolve and download a curated checkpoint; optional `--cache-dir` selects its Hub cache. |
| `tsfm models cache --cache-dir PATH` | Inspect complete/incomplete curated snapshots without network access or deletion. |
| `tsfm datasets inspect FILE --timestamp-column timestamp --target-column demand --frequency 1h` | Validate a single CSV/Parquet series and report its range, units, frequency, and fingerprints. Add `--timezone` for naive datetimes. |
| `tsfm run CONFIG.yaml` | Dispatch strict historical schema 1 or future schema 2, validate before inference, and save mode-specific checked artifacts. |
| `tsfm experiments list --workspace results` | Scan local manifests and show completed, active/incomplete, failed, cancelled, and corrupt records. |
| `tsfm experiments show ID --workspace results` | Show a manifest; completed scores are displayed only after artifact validation. |
| `tsfm compare ID1 ID2 --workspace results` | Refuse mismatched forecasting tasks and compare validated completed records. |
| `tsfm export ID --workspace results --output forecasts.csv` | Export checked predictions as CSV, JSON, or Parquet. Existing files are never replaced. |
| `tsfm finance backtest ID --source-config CONFIG --ohlc CSV ... --output REPORT.json` | Derive an OHLC report from checked saved forecasts with explicit price, timing, capital, and cost assumptions. |
| `tsfm finance show REPORT.json` | Recompute a self-contained finance report without its original sources or model runtime. |
| `tsfm finance export REPORT.json --output equity.csv` | Export strategy, cash, and buy-and-hold equity at each target close. |
| `tsfm connections list --json` | Read nonsecret connection metadata without vault access or network requests. |

All these commands support `--json` for plain JSON on stdout. Progress and error
messages go to stderr, including during a JSON run. There are no interactive
prompts. Exit codes are 0 for success, 2 for invalid configuration/arguments or an
incompatible comparison, 1 for execution/storage failures, and 130 for cancellation.
An existing export destination is a failed file operation with exit code 1.
Finance input/report contract errors use exit code 2. See the
[finance guide](finance-extension.md) for its required options and strict
lead-1/stride-1 timing policy. Finance commands are shell commands; they have not
been added to the interactive prompt.

`tsfm connections manage` is a separate interactive menu requiring terminal input
and output for hidden credential prompts. It is also available as `/connections`
in the terminal and during `/new`. See [connections](connections.md) for explicit
authentication, billing, storage, discovery, and checks.

Natural-language setup (`/new describe`) and checked explanations (`/explain`)
are available in the interactive terminal. They require an explicit outgoing preview
and confirmation, and cannot bypass normal experiment review. See
[language assistance](language-assistance.md); ordinary shell commands remain offline.

Model entries include checkpoint variants and declared runtime requirements.
Each variant adds `max_context` and `max_horizon`; repository information refines
its `capabilities` to those bounds. Adapter information retains its maximum
across available variants. `runtime_notices` is a separate list for optional
dependency terms, empty where no additional notice is recorded. TiRex-2 exposes
the NXAI attribution and mLSTM Kernels Community License terms without changing
the Apache-2.0 model-weight license.
Repository information adds `selected_checkpoint` while preserving the existing
adapter fields. Human pull/cache output describes project verification; their
JSON checkpoint-record shapes remain unchanged. See the
[catalog guide](model-catalog.md) for exact reference/selected-file status and
historical verification snapshots.

Human-readable tables use [Rich](https://rich.readthedocs.io/en/stable/tables.html),
with [Typer](https://typer.tiangolo.com/tutorial/testing/) handling shell arguments.
The application functions return Python values and emit optional stage callbacks;
they do not print or depend on terminal rendering.

## Configuration boundary

The v0.3.0 development terminal adds [onboarding preferences](onboarding.md).
Shell commands ignore those preferences and never enter onboarding. An explicit
`tsfm ui --workspace` overrides the saved interactive results directory.

S2 adds [shared model-specific validation](model-setup.md). Known curated
revisions receive exact variant bounds during YAML/Python configuration validation;
unsupported tasks, context/horizons, or cadence fail before preparation. Valid
schema-version-1 configuration fields and resolved numerical identities are unchanged.

S3 [guided preparation](guided-preparation.md) adds a separately confirmed
download/run path in the interactive guide. `/run` and `tsfm run` do not opt into
it. Guided runs use manifest schema 2 for atomically checked chart/provenance
outputs; existing schema-1 records retain their strict original contract.

The runnable examples are [last-value.yaml](../examples/configurations/last-value.yaml)
and [seasonal-naive.yaml](../examples/configurations/seasonal-naive.yaml). Their
dataset/output paths are relative to the configuration file, so moving to another
working directory does not change the selected input or output workspace.
History commands instead use their explicit `--workspace`, defaulting to `results`
relative to the current working directory.

Schema version 1 requires `model`, `dataset`, and `evaluation`. A missing name
defaults to the configuration filename stem; output defaults to `./results`.
Unknown fields, duplicate keys, YAML aliases, unsafe object tags, unsupported
schema versions, and configurations larger than 1 MiB fail. Numeric parameters
require integers: booleans, floats, and numeric strings are not coerced.
The parser is based on [PyYAML SafeLoader](https://pyyaml.org/wiki/PyYAMLDocumentation)
with duplicate/alias rejection and ISO timestamps preserved as strings.

`model.source: builtin` selects adapter `last_value` or
`seasonal_naive`. Last-value has no options. Seasonal-naive requires
`options.seasonal_period` and a context long enough for that season.
`huggingface` and `local` select `chronos_bolt`, `chronos_2`, `timesfm_2p5`, or `tirex_2` through the strict
[Bolt](chronos.md), [Chronos-2](chronos2.md), [TimesFM](timesfm.md), or [TiRex-2](tirex2.md) checkpoint configuration.
Each adapter accepts only its own curated repositories or a compatible local
architecture. Arbitrary repositories, derived identity fields, and remote-code
options are unsupported.

| Adapter | Curated repositories | Supported context / horizon |
| --- | --- | --- |
| `chronos_bolt` | `amazon/chronos-bolt-tiny`, `amazon/chronos-bolt-small`, `amazon/chronos-bolt-base` | 1–2,048 / 1–64 |
| `timesfm_2p5` | `google/timesfm-2.5-200m-pytorch` | 1–16,256 / 1–128 |
| `chronos_2` | `autogluon/chronos-2-small`, `amazon/chronos-2` | 1–8,192 / 1–1,024 |
| `tirex_2` | `NX-AI/TiRex-2`, `NX-AI/TiRex-2-fevbench` | 1–2,048 / 1–320 |
| `tirex_2` | `NX-AI/TiRex-2-gifteval-zs`, `NX-AI/TiRex-2-gifteval-pretrain` | 1–8,192 / 1–928 |

These are adapter limits. TimesFM's larger native context metadata does not
permit a request above its supported bound or an undocumented horizon rollout.

Dataset fields select a local `.csv`, `.parquet`, or `.pq` file, timestamp/target
columns, and fixed frequency. Units default to the explicit string `unspecified`;
the series defaults to `series_0`. Extra source columns are not requested model
covariates. No panel, multivariate, covariate, or imputation configuration is
accepted. Parquet requires the optional extra; typed datetimes retain their
precision, while boolean/string/decimal target schemas are rejected.

Evaluation uses the [published numerical semantics](evaluation.md): complete
rolling horizons, fixed historical context, per-case aggregation, and a fixed
pre-evaluation MASE denominator. The initial point task is median; deterministic
baselines are valid for it. The model's seasonal period and MASE's lag are
separate explicit settings.

Runtime defaults are CPU, batch size 1, and seed 0. Baselines use float64;
All four pretrained adapters use float32 model arithmetic and return normalized
float64 forecasts. All score genuine median predictions. Chronos-2, TimesFM,
and TiRex-2 require Torch's existing global default dtype to be float32, with an
explicit failure otherwise; they do not modify that setting.
Other device/dtype combinations or batch sizes fail. A nonnegative seed is
recorded; the current deterministic point methods do not use it, which is stated
in resolved runtime metadata.
Chronos-2 records `cross_learning: false` and `limit_prediction_length: true`
under resolved `forecast_options`; these fixed settings are validated on reload.
TiRex-2 records its profile's sign-flip, differencing, and context-padding flags
and disables compilation/Flex Attention; reload binds these settings to the
saved native profile. All optional checkpoint runtimes now share the Hub 0.35.3
pin. Earlier saved records retain their historical environment versions.

## Saved artifacts and failure behavior

Each run has a new UUID and a private directory:

```text
results/experiments/<uuid>/
    manifest.json
    input.yaml
    configuration.json
    result.json
    metrics.json
    predictions.csv
```

`input.yaml` preserves the original text. `configuration.json` preserves validated
user selections and absolute paths. `result.json` contains resolved model/task
metadata, scores, provenance, counts, and stage timing, with large prediction
rows stored separately. `metrics.json` supports direct inspection. CSV preserves
float64 precision, UTC timestamps, and string identities such as `NA` and `001`.

Files are written through temporary files in the same directory and atomic
replacement. The manifest becomes completed only after all artifacts have been
read back and checked. It records fixed artifact names, sizes, and SHA-256 hashes.
Reload additionally validates result schemas, expected origin/lead timestamps,
model and series identities, counts, and scores recomputed from predictions and
the recorded MASE scale. These checks detect inconsistent records; they do not
authenticate files against a person deliberately rewriting the whole workspace.

Invalid configuration or data creates no run. Failures after run creation retain
the failure stage and error. Ctrl-C records cancellation when the process can
handle it. A crash or forced termination can leave an active stage; that record
is incomplete and is ineligible for export/comparison. Corrupt directories stay
visible in history. There is no automatic resumption, SQLite index, or inference
cache. Listing history reads and validates local records, which is intended for
small workspaces initially.

Exports use saved predictions and never rerun inference. Their output format is
inferred from `.csv`, `.json`, or `.parquet`, or explicitly chosen with `--format`.
Writes publish a new complete file and refuse existing paths, including symlinks.

## Comparison policy

`controlled_point_v1` requires equal canonical dataset/import identities, target
and units, frequency, evaluation bounds and cases, context length, horizon,
stride, requested point statistic, selected metric definitions/version,
aggregation, and MASE scaling metadata. Actual values must also match exactly.
Different context lengths are refused in this initial controlled view.

Model and runtime metadata remain visible in the comparison. Deterministic
baselines can use different forecasting rules while sharing the same task.
An undefined MASE retains its reason rather than being printed as zero. The
synthetic example scores verify arithmetic and make no pretrained-model or
trading-performance claim.

## Usage workflow metadata

Schema-1 configuration accepts an optional `workflow` block with a schema version,
profile, and exact profile options. The five choices are Generic, Finance,
Energy, Temperature, and Weather; see [usage profiles](usage-profiles.md).
Non-generic profiles require declared dataset units. Finance close-price data
must be positive; other legitimate negative observations remain unchanged.

Resolved workflow metadata is checked in initial configuration and result, with
agreement required before completion and on reload. It does not enter dataset
fingerprints or numerical comparison keys. Profile-only changes retain forecast
values/scores and comparison eligibility. Existing files without metadata display
as Generic without rewritten JSON or hashes. Text history/show/compare includes
profile and measurement context; JSON keeps its existing result/manifest shapes
with the optional workflow block inside the appropriate configuration.

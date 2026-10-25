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
| `tsfm models list` | List baseline, Chronos-Bolt, and TimesFM 2.5 adapters without loading weights. |
| `tsfm models info IDENTIFIER` | Show offline catalog metadata for an adapter or curated checkpoint repository. |
| `tsfm models pull REPOSITORY --revision REF` | Explicitly resolve and download a curated checkpoint; optional `--cache-dir` selects its Hub cache. |
| `tsfm models cache --cache-dir PATH` | Inspect complete/incomplete curated snapshots without network access or deletion. |
| `tsfm datasets inspect FILE --timestamp-column timestamp --target-column demand --frequency 1h` | Validate a single CSV/Parquet series and report its range, units, frequency, and fingerprints. Add `--timezone` for naive datetimes. |
| `tsfm run CONFIG.yaml` | Validate configuration/data before creating a run, execute one experiment, and save its artifacts. |
| `tsfm experiments list --workspace results` | Scan local manifests and show completed, active/incomplete, failed, cancelled, and corrupt records. |
| `tsfm experiments show ID --workspace results` | Show a manifest; completed scores are displayed only after artifact validation. |
| `tsfm compare ID1 ID2 --workspace results` | Refuse mismatched forecasting tasks and compare validated completed records. |
| `tsfm export ID --workspace results --output forecasts.csv` | Export checked predictions as CSV, JSON, or Parquet. Existing files are never replaced. |

All these commands support `--json` for plain JSON on stdout. Progress and error
messages go to stderr, including during a JSON run. There are no interactive
prompts. Exit codes are 0 for success, 2 for invalid configuration/arguments or an
incompatible comparison, 1 for execution/storage failures, and 130 for cancellation.
An existing export destination is a failed file operation with exit code 1.

Human-readable tables use [Rich](https://rich.readthedocs.io/en/stable/tables.html),
with [Typer](https://typer.tiangolo.com/tutorial/testing/) handling shell arguments.
The application functions return Python values and emit optional stage callbacks;
they do not print or depend on terminal rendering.

## Configuration boundary

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
`huggingface` and `local` select `chronos_bolt` or `timesfm_2p5` through the strict
[Chronos](chronos.md) or [TimesFM](timesfm.md) checkpoint configuration.
Each adapter accepts only its own curated repositories or a compatible local
architecture. Arbitrary repositories, derived identity fields, and remote-code
options are unsupported.

| Adapter | Curated repositories | Supported context / horizon |
| --- | --- | --- |
| `chronos_bolt` | `amazon/chronos-bolt-tiny`, `amazon/chronos-bolt-small` | 1–2,048 / 1–64 |
| `timesfm_2p5` | `google/timesfm-2.5-200m-pytorch` | 1–16,256 / 1–128 |

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
Chronos-Bolt and TimesFM 2.5 use float32 model arithmetic and return normalized
float64 forecasts. Both score genuine median predictions. TimesFM additionally
requires Torch's existing global default dtype to be float32, with an explicit
failure otherwise; it does not modify that setting.
Other device/dtype combinations or batch sizes fail. A nonnegative seed is
recorded; the current deterministic point methods do not use it, which is stated
in resolved runtime metadata.

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

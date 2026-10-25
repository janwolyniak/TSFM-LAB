# Core contracts and extension design

The numerical core implements `Dataset`, `EvaluationSpec`, `ForecastRequest`,
`ForecastOutput`, `EvaluationWindow`, model metadata/capabilities, and an
`ExperimentResult`. Built-in model constructors supply their options directly.
Milestone 2 adds validated YAML selections, recorded baseline runtime settings,
and persistent results through `run_from_config`. Verified checkpoint identities
support the Chronos-Bolt and TimesFM 2.5 adapters. Probabilistic outputs below
remain a design for later milestones.

Milestone 4 adds `config_from_mapping(mapping, base_directory=...)` for the same
strict validation of in-memory selections; relative paths use the explicit base
directory and no configuration file is created. Both entry points delegate to
`run_config(ExperimentConfig)`. Optional stage, completed-window, cancellation,
and record-created callbacks support the managed terminal process without
changing prediction or metric semantics. See the [terminal guide](tui.md).

The first numerical core handles one regular series, one numeric target, fixed
context length, one forecast horizon, and point forecasts. It preserves explicit
identities and time alignment so later adapters do not change scoring semantics.
Boundary validation rejects unknown options and unsupported features.

## Dataset

| Field | Contract |
| --- | --- |
| `series_id` | Nonempty string, explicitly supplied or defaulted to `series_0`. One series per initial dataset. |
| `timestamps` | One-dimensional datetime index of length N; unique, strictly ascending, and on the declared regular grid. |
| `values` | One-dimensional numeric float64 array of length N with finite values. |
| `target_name` | Source target column name. No implicit conversion from prices to returns or other domain units. |
| `units` | User-provided description, or the explicit string `unspecified`. Metrics retain these units. |
| `frequency` | Explicit fixed duration such as `1h`; positive and checked against every adjacent timestamp. Calendar frequencies are deferred. |
| `timezone` | Aware inputs normalize to UTC; naive inputs require a declared source timezone. Mixed awareness and ambiguous/nonexistent local times fail. |
| `provenance` | Source URI/path, source-byte SHA-256, selected columns, parsing policy, and canonical-content SHA-256. |

CSV and optional Parquet loaders implement this contract. Format-specific parser
policy is included in the canonical identity; importing equivalent values from
different formats does not automatically establish the same comparison task.
The CSV parser accepts ISO dates or timestamps with optional UTC offsets;
relative expressions and ambiguous slash-formatted dates are rejected.
Rows arrive in timestamp order; the first version rejects unordered rows rather
than silently sorting. Duplicate columns, duplicate timestamps, missing or
nonfinite targets, missing observations, and irregular intervals fail with row
or column context. No imputation, resampling, inferred seasonality, or learned
normalization is applied in the first slice.

The canonical fingerprint covers a versioned serialization of UTC timestamps,
float64 target values, series/target identity, frequency, units, and import policy.
Paths alone are not identities. Specify byte order, timestamp resolution, and
metadata key order when implementing the serialization, then test stability.
Keep raw and canonical fingerprints: identical content can come from different
file formats without making provenance identical.

## EvaluationSpec

| Field | Contract |
| --- | --- |
| `context_length` | Integer L greater than zero; booleans are rejected. |
| `horizon` | Integer H greater than zero, measured in observations. |
| `stride` | Integer S greater than zero, separating consecutive first-target positions. |
| `evaluation_start` | Timestamp of the first permitted forecast target; must exist on the dataset grid. |
| `evaluation_end` | Exclusive target boundary; a grid timestamp or exactly one grid step beyond the final observation. |
| `seasonal_period` | Integer m greater than zero, in observations; also fixes the MASE scale convention. Default 1. |
| `metrics` | Initial allowed names: `mae`, `rmse`, `mase`. Reject unknown names. |
| `point_statistic` | Requested point task, initially `median`. Deterministic baselines are admissible; a model's derived point statistic must match. |
| `aggregation` | Fixed identifier `forecast_case_micro_v1`; no configurable weighting in the first slice. |

The first origin must have L preceding observations. The entire H-step target
must be inside the evaluation interval. No complete windows is a validation error.
Insufficient pre-evaluation history for MASE yields an undefined metric with a
reason; invalid bounds or lengths fail before calling a model.

## Model identities and capabilities

The current configuration boundary validates adapter/source/options as
dictionaries. `ModelSpec` below is a conceptual grouping rather than a current
dataclass. Curated Hub and compatible standalone local checkpoints are supported
for Chronos-Bolt and TimesFM 2.5; there is no arbitrary-repository loader.
Each adapter has its own repository allowlist and validated architecture profile.

Input configuration supplies adapter/source/options and, for a Hub checkpoint,
the requested repository/revision. The catalog derives family and adapter version;
the resolver supplies checkpoint SHA or local file manifest. Runtime resolution
similarly supplies the actual device/dtype. Derived fields are authoritative run
metadata, not user-overridable configuration. Reject attempts to spoof them at
the YAML boundary.

`ModelSpec` separates the following fields:

- `family`: a catalog grouping such as `baseline`, `chronos`, or `timesfm`.
- `adapter_id` and `adapter_version`: implementation identity, for example
  `last_value`, `seasonal_naive`, `chronos_bolt`, or `timesfm_2p5`.
- `source`: `builtin`, `huggingface`, or `local`.
- Source-specific identity: repository plus requested/resolved revision, or
  local checkpoint path plus a file/checksum manifest. Built-ins have neither.
- Validated adapter-specific options. Reject irrelevant source fields.

`ModelCapabilities` reports supported target counts, covariate types, output
representations, context/horizon bounds, and verified devices/dtypes. Unimplemented
features are false, rather than speculative future support. Metadata is available
before model loading. Known checkpoint limits refine the implementation metadata
after inspecting configuration, before expensive inference.

`RuntimeSpec` records requested and resolved device, dtype, batch size, and seed.
The first built-ins run on CPU. Adapter validation rejects unsupported options;
it never silently truncates history, changes the horizon, removes a covariate, or
changes the requested device. The later `auto` policy may resolve to CPU, but
records that choice before execution.

Both checkpoint adapters currently require CPU, float32 model arithmetic, and
batch size 1. Normalized outputs retain float64 target-unit values. TimesFM 2.5
supports context lengths 1–16,256 and horizons 1–128; its stored native context
limit is 16,384, while the adapter reserves space for the rounded output patch.
It requires the process's default Torch dtype to be float32 and fails explicitly
otherwise; it does not change that global setting. See the
[TimesFM guide](timesfm.md) for its pinned runtime and genuine median semantics.

## ForecastRequest

Each request contains `request_id`, `series_id`, `origin`, historical timestamps
and values, and requested target timestamps. Historical arrays have shape
`[L]`; requested timestamps have shape `[H]`. `origin` equals the last historical
timestamp. All requested target timestamps follow the origin on the same grid.

A batch is an ordered collection of independent requests. The initial runner
can invoke one request at a time. When batching is introduced, outputs retain
request IDs; they must not rely solely on position in an array.

Future actual values are absent. Supply detached, read-only historical buffers
so adapters cannot mutate the dataset or discover future values through a view's
underlying full-series buffer. This boundary prevents accidental leakage by
conforming adapters; it is not a security boundary for hostile local Python code.
Ground truth belongs to a separate evaluator-only window record.

## ForecastOutput

A point output contains:

- `request_id`, `series_id`, and `origin` matching the request.
- `target_timestamps`, exactly equal to the requested timestamps and order.
- `point`, a finite float64 array of shape `[H]` in the original target units.
- `point_statistic`: `deterministic`, `mean`, `median`, or an explicitly identified
  alternative supported by the adapter.
- Adapter identity and checkpoint identity matching the resolved experiment.

The evaluator rejects missing/duplicate outputs, shifted timestamps, incorrect
series or origin, wrong length, and nonfinite values before scoring. No implicit
broadcasting, timestamp intersection, or dropped failures is allowed.
Validated outputs own detached immutable buffers, so an adapter reusing a work
array on a later request cannot change an earlier prediction snapshot.

Probability representations are reserved for a later schema version:
quantiles `[H,Q]` with ordered levels, or samples `[K,H]` with sample count and
generation settings. Neither is fabricated from a point output. Preserve the
native representation and explicitly record any reduction to a point estimate.
For batches, the corresponding dense shapes are `[B,H]`, `[B,H,Q]`, and `[B,K,H]`.

Keep the requested point task separate from output provenance. A deterministic
baseline is valid for a median or mean task because it has no distribution to
reduce. A probabilistic model must use the requested statistic, with its actual
reduction recorded; it cannot substitute a median for a requested mean.

## Adapter lifecycle

The proposed lifecycle is `metadata → validate → load → predict → unload`.
The implemented baseline Protocol uses `metadata()`, `configuration()`,
`validate(spec)`, `load()`, `predict(requests)`, and `unload()`. There are no
checkpoint or runtime arguments until a real adapter needs them. Its later
checkpoint-aware extension is:

```text
metadata() -> ModelMetadata
validate(model_spec, evaluation_spec, runtime_spec) -> None
load(resolved_model_spec, runtime_spec) -> None
predict(requests, inference_options) -> sequence[ForecastOutput]
unload() -> None
```

`validate` and `load` may report unsupported configuration, missing optional
dependencies, missing local files, or unavailable device. `predict` returns one
complete output per request or raises a typed inference error. `unload` is
idempotent and runs in a `finally` path even after inference failure. Built-ins
implement inexpensive load/unload operations without requiring torch or Hub code.
Use ordinary module-level exceptions initially; introduce a richer hierarchy only
when callers need to distinguish error classes.

## ExperimentConfig

Python callers can construct validated objects directly. The implemented YAML
boundary has `schema_version: 1`, a name, a `ModelSpec`, a dataset import spec,
an `EvaluationSpec`, a `RuntimeSpec`, and an output directory. Resolve relative
paths against the configuration file's directory, preserve the input snapshot,
and also persist resolved configuration.

This YAML illustrates the shell schema when saved in `examples/configurations`:

```yaml
schema_version: 1
name: seasonal-demo
model:
  source: builtin
  adapter_id: seasonal_naive
  options:
    seasonal_period: 4
dataset:
  path: ../datasets/seasonal.csv
  timestamp_column: timestamp
  target_column: demand
  timezone: UTC
  frequency: 1h
  units: synthetic_units
evaluation:
  context_length: 8
  horizon: 4
  stride: 4
  evaluation_start: "2026-01-01T16:00:00Z"
  evaluation_end: "2026-01-02T08:00:00Z"
  seasonal_period: 4
  metrics: [mae, rmse, mase]
  point_statistic: median
runtime:
  device: cpu
  batch_size: 1
  seed: 0
output:
  directory: ../../results
```

The baseline's seasonal period and MASE's seasonal period have different roles.
Store both; the guided UI may propose the same value without hiding either.

## ExperimentResult and records

Return resolved configuration, aligned prediction rows, overall metrics,
per-lead metrics, metric definitions/status, window/case/unique-target counts,
elapsed stages, and provenance. An undefined metric is JSON `null` plus a reason;
NaN and Infinity are not permitted in serialized JSON. Counts of attempted,
successful, and failed windows remain visible. Initial behavior fails the run on
any failed window and excludes incomplete runs from controlled comparisons.

Use a random UUID for run identity. A repeated run gets a new UUID even when its
deterministic forecasts match. Reproducibility tests compare numerical outputs and
method metadata, excluding elapsed time, creation time, paths, and run IDs.

Persisted baseline experiments record source Git revision and dirty state, package
version, Python and dependency versions, operating system, hardware/backend,
seed/determinism settings, dataset fingerprints, adapter
version, metric version, and file checksums. An unversioned/dirty source tree must
be identified explicitly, not represented as an exact clean source snapshot.
Checkpoint adapters record consumed file hashes, native capabilities, license
when available, and requested/resolved Hub revisions or a standalone local
identity. The schema-1 key set and existing Bolt fingerprints remain unchanged;
architecture and pipeline fields distinguish the supported families.
Optional runtime/environment hooks give direct Python and CLI callers the same
model dtype and dependency provenance. Reload validates identity without weights
or optional inference libraries; see the [Chronos](chronos.md) and
[TimesFM](timesfm.md) checkpoint rules.

`run_experiment` currently returns an in-memory numerical result with JSON
conversion through `to_dict()`. It validates dataset identity and snapshots
configuration before loading the adapter. `run_from_config` adds UUIDs, artifact
files, recorded seed settings, and persisted lifecycle states. It validates data
before creating a record, retains failures/cancellations after creation, and
uses the same numerical runner. See [CLI records](cli.md#saved-artifacts-and-failure-behavior)
for the current files and validation rules.

Use states `pending`, `preparing`, `loading`, `running`, `evaluating`, `saving`,
`completed`, `failed`, and `cancelled`. Permit forward stage transitions and
failure/cancellation from active stages; terminal states cannot become completed
later. See the [evaluation rules](evaluation.md) and
[implementation acceptance criteria](roadmap.md).

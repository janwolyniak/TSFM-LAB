# S4 future-forecast contract

Status: **approved for implementation on 29 October 2026**. The owner's
continuation instruction accepts the reviewed proposal from commit `b574c4d`.
Contract review date: 28 October 2026. This is the review required by the
[v0.3.0 release plan](release-v0.3.0-plan.md#s4-future-forecasting).
Historical behavior remains governed by [evaluation](evaluation.md),
[architecture](architecture.md), and [finance](finance-extension.md).

## Approved decisions

| Decision | Agreed behavior |
| --- | --- |
| Origin | The final observation only; `latest` or an explicit timezone-aware timestamp equal to it. No implicit truncation or replay. |
| Resource bounds | Future context at most 100,000 rows and horizon at most 10,000 rows, further restricted by the selected adapter. Reject excess, never clamp. |
| Configuration | Separate `FutureForecastConfig`/`FutureForecastSpec`; configuration schema 2 with `mode: future_forecast`. Historical schema 1 stays unchanged. |
| Persistence | Separate strict future result/CSV validators; result schema 2 and manifest schema 3 with an explicit future kind, in the existing experiments directory. Eight required completed artifacts. |
| Execution | A small typed mode dispatcher at the managed-worker and saved-record boundaries; reuse checkpoint preparation and process cleanup. No dummy historical evaluation. |
| Evaluation | Accuracy explicitly unavailable because future actuals are absent. Later actual import/scoring remains deferred. |

Approval accepts these interfaces and bounds for implementation. It does not
authorize dependency installation, real weight downloads/inference, live assistant
calls, publication, or qualification claims. S4 acceptance boxes remain open until
implementation and verification.

## Scope and input

One regular univariate series produces one CPU point forecast beyond its final
observation. Reuse the existing built-in and curated adapters, fixed UTC cadence,
usage categories, original-unit rules, and checkpoint/runtime requirements.
No covariates, panels, calendar resampling, probability scoring, training, live
feed, trading policy, or earlier-origin replay is added.

Let the loaded dataset contain N observations, the requested context be L, horizon
be H, declared fixed duration be delta, and final observed timestamp be T.

- Require integers `1 <= L <= 100000`, `1 <= H <= 10000`, and N >= L. Boolean
  lengths are invalid. Future mode requires no holdout observations.
- `forecast.origin` defaults to `latest`. An explicit timestamp must be aware,
  valid, and equal to T after UTC normalization. Earlier/later origins, naive
  timestamps, and NaT fail; no rows are silently discarded.
- Resolve `latest` during review. Freeze T, the full source identity, final-L
  context identity, model/settings, future grid, and exact checkpoint/file plan
  with every available expected hash. Missing weights need not have known hashes.
  Recheck before execution; appended or changed source data invalidates consent.
- Consume exactly the final L observations, ending at T inclusively. Targets are
  exactly T + k * delta for k = 1 through H. Never supply actual future values.
- Check UTC nanosecond representability of T + H * delta using Python integer
  arithmetic before allocating the timestamp grid. Reject overflow, unsupported
  cadence, nonfinite context, and invalid model-specific numerics before weights.

The declared cutoff describes the dataset supplied by the user. It does not prove
wall-clock availability, point-in-time data vintage, or absence of revisions.
Changing unused earlier observations cannot change the numerical request, but
does change full-source provenance and invalidate an already reviewed plan.

## Configuration and numerical request

Future YAML has exactly the common top-level `name`, `model`, `dataset`, optional
`runtime`, `output`, and optional `workflow` fields, plus `schema_version: 2`,
`mode: future_forecast`, and `forecast`. Reuse current common-field defaults,
path resolution, safe YAML, units, and strict unknown-field checks.
`forecast` admits only `context_length`, `horizon`, `point_statistic`, and `origin`.
L/H are required; the point task defaults to the selected descriptor's supported
default; origin defaults to `latest`. Reject `evaluation`, historical dates,
stride, metrics, MASE settings, supplied actuals, and caller-supplied target grids.
The adapter's seasonal period remains a model option where applicable.

Example YAML, using paths relative to `examples/configurations/`; see the
[future guide](future-forecasting.md) for implementation usage and limits.

```yaml
schema_version: 2
mode: future_forecast
name: future-baseline
model:
  source: builtin
  adapter_id: last_value
dataset:
  path: ../datasets/seasonal.csv
  timestamp_column: timestamp
  target_column: demand
  frequency: 1h
  units: synthetic_units
forecast:
  context_length: 8
  horizon: 4
  point_statistic: median
  origin: latest
output:
  directory: ../../results
```

`FutureForecastSpec` carries L/H, requested point task, and origin. Share the
adapter settings interface with `EvaluationSpec`; do not add fictitious
evaluation dates/metrics to satisfy historical validation. Historical configuration
serialization, defaults, fingerprints, and valid numerical results remain unchanged.
Saved future `configuration.json` resolves origin to T; `input.yaml` retains the
original input including `latest` where entered.

Use the existing `ForecastRequest` with a detached final-L context and generated
targets. Snapshot expected identity and timestamps independently before adapter
execution: read-only value arrays alone do not protect mutable timestamp buffers.
Require exactly one output matching request ID, series ID, adapter, origin, target
grid, H, and the permitted actual point label, with finite original-unit values.
Preserve the requested task separately from the actual output label; requesting
`native_point` cannot relabel a median or deterministic forecast.

Domain-separate the numerical request identity as `future_point_request_v1`.
It includes series/target/units/cadence, consumed timestamps/values, T/L/H/target
grid, requested task, adapter/checkpoint identity, and numerical runtime and
preprocessing choices. Use deterministic canonical serialization and SHA-256;
do not include paths, display names, category-only metadata, unused source prefix,
or preparation status. Record the full source identity separately. The consumed
context hash is reproducible from saved observations; it cannot prove the whole
source file's hash when N > L.

The pre-confirmation plan identity binds exact repository/revision, file manifest,
available expected hashes, data, and settings. The final numerical request ID is
assigned after approved preparation verifies the consumed checkpoint/configuration
bytes and before inference. It is the lowercase 64-character SHA-256 above;
`ForecastRequest.request_id`, output request ID, result, and CSV must agree.
Keep reviewed-plan identity and final request identity separate. Where upstream
weight hashes are unavailable, record observed-byte identity without claiming
new upstream authenticity. Built-ins require no checkpoint preparation.

### Existing-adapter compatibility review

These are current source-level limits, not new future-inference qualification.
The curated catalog/settings remain authoritative for a selected variant.
The operational L/H bounds above are additional guards, not native capacities.

| Adapter | Native context/horizon constraints | Actual output and numerical requirements |
| --- | --- | --- |
| Last value | L >= 1; no native H cap | Deterministic; final observed value |
| Seasonal naive | L >= seasonal period; no native H cap | Deterministic; repeat the last complete season |
| Chronos-Bolt | Per-checkpoint bounds, up to L=2048/H=64 | Median; finite float32-representable raw context |
| Chronos-2 | L <= 8192/H <= 1024 | Median; finite float32-representable raw context |
| TimesFM 2.5 | Effective L <= 16256/H <= 128 | Median; retain 32-observation internal padding and float32 context checks |
| TiRex-2 | Default/FEV L <= 2048/H <= 320; GiftEval L <= 8192/H <= 928 | Median; preserve profile preprocessing and float32 context checks |
| TTM R2 | Exact L=512, 1024, or 1536; H <= 96 | Native point only; cadence 1 minute–1 hour; required causal scaling/inverse |

TTM checks safe context normalization rather than imposing a blanket raw-float32
bound. Keep constant-context handling, native scaling, original-unit inversion,
and overflow/finiteness checks. Model-specific preflight rejects incompatible
numerics before weight preparation where determinable without inference.

## Future record and artifact contract

Use `output.directory/experiments/<canonical UUID>/` for both modes. Dispatch on
manifest schema/kind before reading a result. Historical manifest schemas 1/2
retain their current validators; schema 2 already means S3 guided historical
outputs and must never be reinterpreted as future output.

Future manifest schema 3 has `kind: future_forecast` from creation, including
pending, failed, and cancelled records. Retain the current run ID, name, UTC
creation/update timestamps, status, artifact identity map, and optional error
metadata. Unknown versions/kinds fail. Allow only forward active transitions
`pending -> preparing -> loading -> running -> saving`, then a terminal status;
stages may be skipped, and failure/cancellation can end any active stage.
There is no evaluating stage. Terminal records cannot be resumed or overwritten.

Creation requires exactly `configuration.json` and `input.yaml`. Incomplete
records may list only a subset of the eight allowed artifacts including those
two; they are never loadable as completed results. Completed records require
exactly these eight files, each with SHA-256 and byte size:

| Artifact | Required content |
| --- | --- |
| `configuration.json` | Strict resolved future configuration, with concrete UTC origin |
| `input.yaml` | Original safe configuration text |
| `result.json` | Strict future result schema 2, described below |
| `predictions.csv` | Exactly H future predictions; no actual column |
| `observations.csv` | Exactly L consumed observed context rows ending at T |
| `evaluation.json` | Explicit unavailable-evaluation object below |
| `forecast.svg` | Deterministic, safe future chart with origin boundary |
| `provenance.json` | Versioned reproducibility data consistent with the result |

Future result schema 2 requires these top-level fields; reject unknown fields:

| Field | Content |
| --- | --- |
| `schema_version`, `mode`, `run_id`, `status` | `2`, `future_forecast`, matching UUID, `completed` |
| `configuration` | Same resolved object as `configuration.json` |
| `request` | Request ID, series ID, origin/cutoff T, context start/end, L/H, cadence, exact target timestamps, context identity |
| `output` | Adapter ID, requested task, actual point statistic, target name and declared units |
| `counts` | Source observations N, context observations L, forecast requests 1, forecast rows H |
| `timing` | UTC start/completion and finite nonnegative elapsed seconds |
| `provenance` | Source, consumed-context, model, numerical settings, runtime, and implementation identities |
| `evaluation` | Exactly the unavailable object below |

`predictions.csv` uses this exact ordered header:

```text
request_id,series_id,adapter_id,origin,target_timestamp,lead,prediction,point_statistic
```

Require one request, H rows ordered by leads 1..H, identical origin T, exact target
grid, matching identities/labels, and finite original-unit predictions.
`observations.csv` has exact header `series_id,timestamp,observed_value`, exactly
L ordered fixed-grid finite context rows ending at T, and matching series identity.
Saved observations cover the consumed context, not necessarily the whole source.
All stored times use lossless aware UTC ISO strings; normalize equivalents when
validating, without losing subsecond precision. CSV writing/reload must round-trip
finite float64 values exactly so consumed-context/request hashes remain reproducible.

`evaluation.json` and the result's `evaluation` must equal:

```json
{
  "schema_version": 1,
  "mode": "future_forecast",
  "status": "unavailable",
  "reason": "future_actuals_absent",
  "metrics": null
}
```

No `metrics.json`, actuals, substituted historical scores, or financial returns
are permitted. Accuracy remains unavailable even if source timestamps happen to
precede today's date. Importing later observations requires a separate policy.

Provenance records full-source raw/canonical fingerprints and N; separately
recomputable consumed-context identity; entered origin policy and resolved cutoff;
requested/resolved model and revision plus verified weight identities where
applicable; actual point semantics; preprocessing/inverse policy; effective
runtime/device/dtype/batch/seed choices; package/code revision. Built-ins have no
invented checkpoint. Secrets never enter artifacts. A source fingerprint records
what was checked during execution; detached reload does not authenticate unsaved
source rows or external data vintage.

Use a new versioned renderer contract, `future_point_outputs_v1`, preserving the
existing historical renderer. Show observed context through T and predictions
strictly after T, with an origin line, timestamp axis, target/units, point label,
and unavailable-accuracy reason. No invented actuals, uncertainty bands, or scores.
Disclose any chart sampling; retain full CSVs. Reload validates checksums, strict
schemas, all cross-file identities/grids/counts, context hash/request identity,
and deterministic chart/provenance consistency without source data or model
weights. Recomputed checksums alone cannot make inconsistent artifacts valid.

Publish completed status last, only after all eight artifacts validate. Reuse
regular-file/symlink protections and atomic writes. Failure/cancellation during
saving leaves a truthful incomplete record; partial files cannot imply success.

## Integration and compatibility

Add a strict `load_future_result` alongside historical-only `load_result`.
The latter explicitly refuses future kinds before historical validation.
Use one small tagged dispatcher for manifest/history/show/summary/output-path/
configuration/export and worker reconciliation. Historical output shapes remain
unchanged; future summaries include mode, origin, H, and unavailable accuracy.
Future CSV/JSON export reads validated artifacts without inference; optional
Parquet export retains its installed-dependency requirement.

`tsfm run` and `/run` accept either strict configuration schema via a new
`load_run_config` dispatcher. Existing Python `load_config`, `config_from_mapping`,
`run_from_config`, and `run_config` stay historical-only, including their return
types; schema 2 receives an explicit future-API error there. Add
`load_future_config`, `future_config_from_mapping`, `run_future_from_config`, and
`run_future_config` for Python future use. Shared command/worker callers dispatch
the two known types explicitly, rather than assuming a returned `.evaluation`.

Historical comparison and finance reject future records before trying to access
actuals, metrics, or accounting. Finance may remain the target's usage category;
that metadata does not authorize a return calculation. Preserve old fixtures,
fingerprints, scores, comparison eligibility, and finance report schemas without
migration or rewriting. The S3 guided-finance regression found during this review
is repaired separately; see the [verification follow-up](s3-preparation-verification.md#follow-up-guided-finance-compatibility).

The managed execution seam accepts only the two explicitly validated configuration
types and dispatches to their worker/finalizer. Reuse trusted `worker_entry`,
credential-free worker environment, progress transport, cancellation, and process
cleanup. Use a separate future application service/plan, future persistence
methods, and real `FutureForecastSpec`, never a dummy `EvaluationSpec`.

Reuse S3 exact checkpoint/file review, runtime-before-transfer checks, cached
offline path, byte-bound download consent, dataset revalidation, and scriptable
no-download default. A future preparation record uses a new schema 2 with mode,
resolved origin, plan identity, and linked run ID; historical preparation schema 1
is unchanged. Preparation/download/verification progress remains distinct from
the future numerical stages above. Late completion wins only if the complete
future artifact set validates; IPC loss must retain recoverable IDs/status.

### Manual, assisted, and scriptable setup

Keep model options before dataset selection. After dataset mapping/preview, ask
the human to choose historical evaluation or future forecasting, then apply
mode-specific data bounds. Do not reject future N=L data with historical holdout
rules before that choice. Revalidate native limits against the selected mode.

Switching mode retains compatible category/model/checkpoint/dataset/mapping/
units/L/H. Clear historical dates/stride/MASE settings on entering future mode;
clear future origin on leaving it. Collect required historical bounds again when
switching back. Any switch invalidates prepared plans and requires final review.

Human-selected mode is authoritative in assisted setup. Introduce draft schema 2
for future fields including origin, retain historical schema-1 decoding, and
route both through ordinary validated configuration. Provider intent must agree
with the selected mode; conflicting suggestions require human mode selection and
a fresh preview. Do not merely remove the current future-intent regex or silently
convert future requests into historical evaluation. Guided, assisted, YAML, and
Python choices must produce the same resolved configuration and numerical request.

Future explanations use metadata-only, path-free evidence: target/units/cadence,
N/L/H, context start/end, origin, first/last target timestamps, requested/actual
point labels, selected model/checkpoint and numerical settings, and the exact
unavailable-accuracy reason. Preserve the existing omission of observed/predicted
values, row tables, and paths; do not add value summaries or a sampled prediction
table. The preview names these metadata fields and omissions truthfully. Preserve
outgoing-context consent, connection bounds, and the separation between assistant
and forecasting models; assistants cannot start downloads/runs.

## Verification required before S4 acceptance

| Area | Retained offline evidence required |
| --- | --- |
| Configuration | Strict mixed-mode/unknown-field rejection; equivalent YAML/Python/manual/assisted choices; legacy schema/fingerprint fixtures unchanged |
| Data/origin | N=L including N=L=H=1; UTC-equivalent explicit origin; reject earlier/later/naive/NaT; changed/appended source invalidates review |
| Grid/resources | Exact fixed-duration grid through DST/midnight/subseconds; reject timestamp/duration overflow and huge/bool/zero/negative L/H before allocation or model access |
| Adapters | Every current native boundary and actual label; TTM exact contexts, constant/negative data, scaling/inverse failures; no future-actual input |
| Isolation | Perturb unused prefix, including finite values outside float32 range, without changing predictions or consumed-context preflight; prior source consent still invalidates; detached context/grid survives adapter mutation; historical window/scoring functions must not be called |
| Outputs | Reject wrong identity/origin/grid/label, short/extra/nonfinite output; no fabricated actuals/metrics; chart visibly separates origin and future |
| Persistence | Strict eight-artifact completion; tampering rejected even after checksum updates; unavailable reason exact; reload/export without source or weights |
| Lifecycle | Cancellation/failure at preparation/loading/inference and each save; no false completion; late valid completion and lost IPC reconcile truthfully |
| Compatibility | Mixed-mode history/show/export; explicit future comparison/finance refusal; old records and guided-v2 finance remain valid |
| Interfaces | Back/mode changes retain only compatible state; fake assistant conflicting/malformed drafts and privacy checks; no hidden download or paid fallback |
| Delivery | Offline baseline from built wheel, terminal walkthrough and chart inspection; configured formatting/lint plus relevant and broadened compatibility checks |

Real pretrained future inference, live assistant/vault, native Windows/Linux, and
the full Python matrix remain separate authorized qualification work under S9.
Source inspection and fake adapters establish contract compatibility, not those
runtime/platform claims.

## Review record and implementation order

Read-only parallel reviews covered adapter/settings and numerical limits;
configuration/terminal/assistant/managed-worker integration; and persistence,
comparison, finance, and backwards compatibility. They identified the holdout
coupling, mutable timestamp buffers, unbounded baseline allocations, managed-worker
type/finalizer coupling, historical assistant gates, and guided-finance regression.
The proposal incorporated those findings and subsequently received owner approval.
Final review also made public configuration/run dispatch explicit, retained
metadata-only assistant evidence, and separated pre-confirmation plan identity
from the verified-byte numerical request ID. No material review findings remain.

Proposal checks: all **19 local links/anchors** across this document and the
updated release plan resolve; its YAML and JSON examples parse and their proposed
mode/availability fields agree; table columns are consistent; `git diff --check`
passes. The YAML is illustrative and has not been executed through a future
loader. These documentation checks do not establish S4 implementation acceptance.

The common experiments root with an explicit discriminator is preferred over a
second root because history/export already share that directory. Dedicated future
validators preserve historical refusal behavior. Saving consumed context rather
than the entire source provides a reloadable chart with less copied data, while
recording the limit of full-source verification. Dummy historical configurations
are rejected because they would mix evaluation semantics into future inference.

Implementation order: (1) strict configuration/request/numerical tests; (2) future
storage/validation/export and historical refusal; (3) managed preparation/lifecycle;
(4) both terminal flows and assistant evidence; (5) integrated verification and
acceptance updates. Branch `codex/v0.3.0-future-contract` depends on S3 commit
`30d6ac8`, S2 `d2eeab9`, and S1 `f35e069`; integrate prerequisites first through the
[contribution workflow](../CONTRIBUTING.md). No branch is published or merged by
this review.

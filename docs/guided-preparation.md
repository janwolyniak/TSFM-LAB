# Guided preparation, progress, and saved outputs

Status: implemented in the **unreleased v0.3.0 S3 checkout**. The published v0.2.0
package retains its prior workflow. See [model setup](model-setup.md),
[verification evidence](s3-preparation-verification.md), and the
[release plan](release-v0.3.0-plan.md).

## Review and confirmation

`/new guided` and reviewed `/new describe` suggestions share the same preparation
service. Select a curated family/generation/variant and its supported settings,
then validate the dataset, units, cadence, history, and historical evaluation
bounds. Pretrained sources offer three choices:

| Source | Behavior |
| --- | --- |
| `automatic` | Check the cache first. If files are missing, review public Hub metadata and a bounded configuration before asking permission to download weights and run. |
| `cached` | Inspect and verify the exact local snapshot offline. Missing files fail clearly; no Hub metadata request or weight transfer is made. |
| `local` | Use a compatible local checkpoint directory, retaining its checked identity. |

Runtime package metadata is checked before Hub preparation. Real runtime imports
are checked in the managed worker before weight transfer. Neither check installs
packages. Baselines and the bundled `/demo` require no assistant, Hub, model
runtime, or network connection.

The final review shows the exact repository/revision, cache status, missing-file
transfer size when known, and output location. A missing checkpoint asks
**Download missing checkpoint files and run experiment?**, default **no**. A
cached/local/baseline task retains **Run experiment?**, also default **no**.
Declining starts no inference or weight download and creates no run/preparation
record. Small configuration metadata may already be cached by automatic review.
Back navigation changes the reviewed plan and requires renewed confirmation.

Preparation uses the exact reviewed SHA and safe file manifest, without resolving
a moving branch again. Remote code, unsafe serialized weights, arbitrary Hub
repositories, and automatic runtime installation remain unsupported. Another
connection's credentials are never needed for forecasting; public Hub calls use
explicit unauthenticated access.

## Progress and interruption

The terminal displays cache checking, download, verification, data preparation,
loading, forecasting, evaluation, and saving. Elapsed time uses a monotonic clock;
forecast progress uses completed/total windows. Byte counts are measured rather
than estimated percentages. The pinned Hub API exposes completed-file transfer
counts rather than a per-byte callback, so a large file can remain at the download
stage with elapsed time until it finishes. Verification can report hashing bytes.
Unknown transfer totals remain unknown.

Ctrl+C requests cancellation through the existing managed worker, waits for it to
stop, and uses bounded termination/kill cleanup when needed. Failures and
cancellations retain a separate preparation record at
`RESULTS_DIRECTORY/preparations/OPERATION_ID.json`, including stage, status, elapsed time,
and linked run ID once available. No incomplete numerical run is promoted to
completed. A late cancellation cannot undo an already validated completed result.
Retry is an explicit new action, which reviews cache integrity again.

Review captures configuration/data identities. Execution rechecks them before
preparation and again when data is loaded. The prepared checkpoint fingerprint is
checked when building the adapter and again by its existing loader. Changed inputs
or checkpoint bytes require renewed review. Previously unverified copied caches
can establish observed-byte identity offline, but cannot establish an absent
upstream digest; newly prepared downloads retain integrity receipts for reuse.

## Output contract

Guided `/new` and `/demo` save these files in
`RESULTS_DIRECTORY/experiments/RUN_ID/`, where `RESULTS_DIRECTORY` is the configured
`output.directory`, normally `results`:

| File | Contents |
| --- | --- |
| `predictions.csv` | Full aligned historical forecasts, actuals, origins, leads, and point semantics. |
| `result.json` | Numerical results, metrics, configuration, provenance, and timing. |
| `metrics.json` | Evaluation metrics, including unavailable values and their reasons. |
| `configuration.json`, `input.yaml` | Validated resolved configuration and original request snapshot. |
| `forecast.svg` | Offline vector chart of evaluated actuals and separate forecast windows. |
| `provenance.json` | Recorded numerical/model/runtime provenance and chart sampling metadata. |

The chart preserves declared units and UTC timestamps, shows evaluation bounds,
and retains legitimate negative values. Actuals appear once per target timestamp;
forecast windows remain separate trajectories. It uses evaluated observations,
not an unavailable saved history prefix. Large charts use disclosed bounded
sampling; the full CSV remains unchanged. Missing/unrequested metrics have reasons.

Chart/provenance generation, validation, and hashing occur before the completed
manifest is atomically published. Guided records use manifest schema 2 with exactly
seven checked artifacts; numerical result schema and forecast identity stay
unchanged. Legacy schema-1 records retain their original strict five-artifact
contract and remain readable without rewriting. The
`guided_historical_outputs_v1` renderer is part of the saved contract: later style
changes require a new renderer contract while preserving this reader.

Completion and `/show ID` expose exact checked file paths. Viewing or exporting
saved results needs no model weights, inference framework, assistant package, or
original dataset. CSV/JSON exports remain available, with Parquet where its
optional dependency is installed. Existing files are never silently overwritten.

## Script and Python boundaries

`/run CONFIG.yaml`, `tsfm run CONFIG.yaml`, and the standalone pull command retain
their prior behavior. Configuration alone does not authorize a guided download.
The Python `review_guided` service returns a detached immutable plan; callers
explicitly invoke `execute_guided` to authorize its displayed preparation/run.
Ordinary `run_config` defaults to no guided preparation and schema-1 outputs;
`include_outputs=True` requests the saved chart/provenance without adding downloads.

S4 [future forecasting](future-forecasting.md) reuses review, exact consent, and
managed cancellation with its own schema-2 preparation records and schema-3
experiment manifests. Future mode omits metric evaluation and saves eight checked
artifacts. Historical preparation/records keep their previous schemas.

Scoring saved future predictions later, new model families, new assistant
endpoints, and native Windows qualification retain their separate release
stages. Real pretrained download/inference qualification remains pending
authorization; routine tests use fake Hub/runtime transports.

The pinned API behavior was checked against official
[Hub 0.35.3 download source](https://github.com/huggingface/huggingface_hub/blob/v0.35.3/src/huggingface_hub/file_download.py)
and [metadata source](https://github.com/huggingface/huggingface_hub/blob/v0.35.3/src/huggingface_hub/hf_api.py).

# Chronos Bolt integration verification record

Date: 25 October 2026. Scope: Milestone 3 following the owner's continuation
request. Local branch: `feat/chronos-bolt-integration`, dependent on the unmerged
CLI/experiment-management branch.

## Implemented workflow

The optional adapter supports CPU float32 median forecasts from curated cached
or compatible standalone local Chronos-Bolt checkpoints. The model catalog,
metadata inspection, explicit pulls, cache inventory, exact revision resolution,
configuration validation, normalized forecasts, and persisted checkpoint/runtime
identity are implemented. Normal experiments do not download weights.

Local checks used Python 3.13.6 on macOS. The model runtime is already installed:
Chronos forecasting 2.2.0, PyTorch 2.8.0, Transformers 4.55.0, Hugging Face Hub
0.34.4, Safetensors 0.6.2, and Accelerate 1.12.0. Importing this runtime succeeded.
No dependency installation or upgrade was needed.

| Check | Outcome |
| --- | --- |
| Full routine suite | 678 passed; the opt-in real CPU smoke test skipped in this routine command. |
| Marked real CPU smoke | Explicitly enabled after the approved download: 1 passed. |
| Black and Ruff | Passed. |
| Package and entry point | Wheel built without dependency resolution and installed with `--no-deps --no-index` in a disposable environment reusing existing libraries; installed help/catalog/info/empty-cache commands passed. |
| Optional boundaries | Installed catalog lookup passed with all Hub/model runtime imports blocked. Prior Milestone 2 records remained readable. |
| Checkpoint tests | Fake Hub clients cover exact revisions, moving-reference races, offline cache-only behavior, metadata-first pulls, missing files, local mutation, native limits, and safe file/path policies. |
| Adapter tests | Fake runtimes verify strict loading, incomplete weights, evaluation/inference mode, q0.5 shape, finite outputs, dtype limits, identities, and resource release. |
| Application and records | Fake Chronos workflow checks local/cached selection, model dtype/provenance, baseline comparison, corrupt identity rejection, and reload without checkpoint files. Direct Python results include the same model-runtime provenance as CLI results. |

Independent review found and repaired model identity bypasses through alternate
weight/config directives and PEFT sidecars, as well as missing direct-Python
runtime provenance. The subsequent independent checks found no outstanding
actionable issues in those reviewed boundaries.

## Real checkpoint execution

The pinned tiny checkpoint is
`amazon/chronos-bolt-tiny@a0e552de83495b5c28c14c71c374f3e33280b340`.
Read-only Hub metadata reports `model.safetensors` size 34,622,352 bytes and
SHA-256 `75068728d376d2bec670379eeef4bfb4d24c0cfe24d957451f8d19b447030a32`.
It was not initially present in the configured local Hub cache.

The owner authorized the approximately 35 MB download and CPU test. The actual
`models pull` command downloaded only the pinned configuration, Safetensors,
and model card into a temporary Hub cache. The consumed weights matched the
Hub's expected SHA-256. Additional consumed hashes were:

- `config.json`: `278f0086733031635fb1c861cb01c1bad6477420c7fcb19381a2993e335785e0`.
- `README.md`: `cc7f60bc7290b5e6146589dae9842a3c9b59d254e9d1d65fc2ef127f3c2498cd`.
- Checkpoint fingerprint: `fbef61a1da54789e66df624baf0c6afab471c753e698c83beaae6b13a3b47f75`.

Strict parameter loading succeeded. CPU inference used the original synthetic
`demand` target in `synthetic_units`, hourly frequency, context 8, horizon 4,
stride 4, and four origins covering 16 forecast cases. Outputs were finite and
aligned, with model arithmetic in float32 and normalized median points in float64.

An immediate cached repeat produced exactly equal prediction tables with all
network connections blocked. A standalone local copy of the consumed files
produced the same forecasts. Recorded source identities correctly distinguished
Hub revisions from local files. A real baseline-versus-Chronos comparison passed
the controlled-task compatibility checks; no accuracy advantage is claimed.

The explicitly enabled CPU smoke test separately used context 32 and horizon 4
for one origin on an original smooth synthetic series; it passed. Model commands
and real inference also passed through the installed wheel's `tsfm` entry point.
CSV, JSON, and Parquet exports reloaded successfully. After temporarily making
both weight directories unavailable and blocking all model-library imports,
actual model records still reloaded, compared, and exported in another process.

Runs from the source checkout correctly reported a dirty source revision during
implementation; installed-wheel runs reported the package version with no Git
checkout identity. These test records do not claim an exact clean source snapshot.

## Limits

Remote CI and a fresh dependency installation have not been run. Tiny was the
only downloaded/tested checkpoint; Small remains unexecuted here. GPU/MPS,
quantization, PEFT loading, other Chronos variants, probability scoring, and
forecasting superiority are outside this verification. No scientific advantage
claim follows from a synthetic CPU integration test.

The original untracked `PLAN.md` is unchanged. No branch was pushed, pull request
opened, or merge performed. See [the Chronos guide](chronos.md) for the supported
configuration and loading rules.

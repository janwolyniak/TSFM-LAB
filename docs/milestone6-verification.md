# Second-family and expanded CPU verification record

Date: 25 October 2026. Scope: the owner-selected TimesFM second family and the
separately approved Chronos-Bolt Small download/brief CPU integration. Branch:
`feat/timesfm-integration`. No package release is published by this milestone.

## Implemented boundaries

The new `timesfm_2p5` adapter uses the explicit TimesFM 2.5 Torch class through
runtime 3.0.2. It supports CPU float32 median forecasts, one native 128-step
output patch, and historical context up to 16,256. The library and weights are
optional; neither is imported/downloaded by catalog lookup or saved-result reload.

Configuration, factory, per-family checkpoint inspection, catalog, guided prompt,
and retained Textual forms recognize the second family. Old schema-1 Bolt record
bytes and fingerprints remain unchanged. Wrong adapter/repository/architecture
pairings fail, and imported model files keep their existing safe-loading rules.
Saved TimesFM records validate their family, supported runtime options, native
versus supported limits, and required version provenance without filesystem/model
access. Comparison continues to require identical tasks and cases.

The fixed TimesFM flags record input normalization and flip invariance, with
positivity inference, continuous quantile head, quantile-crossing repair, backcast,
and Torch compilation disabled. Channel 5 supplies the median; channel 0 is not
used as a median. Load/predict reject a non-float32 default Torch dtype because
upstream uses that default for internal caches; process-wide settings are not
changed.

## Environment and exact checkpoint identities

Local checks used Python 3.13.6 on macOS arm64, NumPy 2.1.3 and pandas 2.3.3.
The separately authorized TimesFM 3.0.2 wheel was installed without dependency
resolution into a temporary environment reusing existing libraries. The global
editable TimesFM 2.0.0 installation was preserved. Its 125,052-byte official wheel
has SHA-256 `cfa3d22ddc3f019f0d9d95f19055428c4a4713526dafd7e5d1a7769782e7613f`.

| Model | Resolved revision | Weight bytes | Weight SHA-256 |
| --- | --- | ---: | --- |
| Chronos-Bolt Small | `772f3d25d38aec6d914c8949dab4462e2d46f5d8` | 190,888,824 | `06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd` |
| TimesFM 2.5 Torch | `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` | 925,181,104 | `2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8` |

Small was explicitly downloaded into a temporary Hub cache. TimesFM weights were
already present; no TimesFM weight download was needed. Both weight hashes matched
the inspected official identities. TimesFM's cached directory lacked a model
card, so its consumed-file license is correctly `null` although the catalog/card
declares Apache-2.0. The Small record includes its consumed model-card license.

The TimesFM runtime was 3.0.2 with Torch 2.8.0, Hub 0.34.4, and Safetensors 0.6.2.
Small used Chronos forecasting 2.2.0, Torch 2.8.0, Transformers 4.55.0, Hub 0.34.4,
Safetensors 0.6.2, and Accelerate 1.12.0. Runtime package version 3.0.2 does not
register the separate TimesFM 3.0 model or its different license.

## Executed checks

| Check | Observed outcome |
| --- | --- |
| Full routine suite | **1,022 passed, 2 skipped** in 46.07 seconds, using the already installed Textual 8.2.8 environment. The skips are the two explicit real-model smoke tests. |
| Black/Ruff | Passed for all source, tests, examples, and scripts; 86 Python files checked. |
| Checkpoint compatibility | Frozen old-Bolt serialization/fingerprint fixture passes alongside family-specific safety and metadata-first pull tests. |
| Adapter tests | Median-versus-mean selection, complete output shapes, finite inputs/outputs, native bounds, no truncation, fixed compile flags, instance ownership, mutation resistance, dtype/version refusal, and failure cleanup pass with inert weights. |
| Small cached/local workflow | Four origins / 16 finite aligned medians, exact network-blocked cached replay, standalone local smoke, saved reload, baseline comparison, and CSV/JSON/Parquet export reload pass. |
| TimesFM cached workflow | Four origins / 16 finite aligned medians; a second full cached run is exactly equal with zero audited network attempts. |
| Real current-source smokes | The opt-in TimesFM test passed in 2.05 seconds; the Small test passed in 2.75 seconds after family-aware checkpoint changes. |
| Cross-family comparison | Real TimesFM, Small, and last-value results pass the same-task/case comparison. |
| No-runtime replay | Real saved model records compare and TimesFM CSV/JSON/Parquet exports reload exactly in a fresh process with all model-library imports and checkpoint file access blocked. |
| Setup/interfaces | Tests cover per-family repository selection, revision reset on family changes, CPU/float32 defaults, native-bound validation, shell catalog, prompt completion, and retained forms. |
| Source/wheel packaging | A source archive and wheel rebuilt from it passed content/metadata checks, including the TimesFM guide, example, runtime constraints, and two-library numerical-core boundary. |
| Clean installed package | The exact wheel installed from cached dependencies into a fresh environment without system site packages. `pip check` and all eight installed verifier stages passed, covering four catalog entries and both missing-runtime boundaries. |
| Installed TimesFM | A separate isolated-Python run imported the package from the installed wheel and completed one real cached TimesFM origin with four finite medians and no network attempts. Its temporary model runtime reused system dependencies; this is separate from the clean baseline installation. |

The numerical input is the original bundled seasonal fixture: hourly demand in
`synthetic_units`, context 8, horizon 4, stride 4, four windows and 16 cases,
with MASE lag 4 and fixed pre-evaluation scaling. Its arithmetic is not a financial
dataset or a model-selection benchmark.

| Method | MAE | RMSE | MASE |
| --- | ---: | ---: | ---: |
| Last value | 2.5 | 2.915475947 | 2.5 |
| Seasonal naive, period 4 | 1.0 | 1.0 | 1.0 |
| Chronos-Bolt Small | 2.557179928 | 3.110861829 | 2.557179928 |
| TimesFM 2.5 | 0.402282715 | 0.517457747 | 0.402282715 |

These values only identify the exercised workflow on one constructed series.
They do not establish out-of-sample superiority, distributional calibration,
financial value, or general performance. The tiny inference timings recorded
outside the repository are execution evidence, not a speed benchmark.

## Source, artifact evidence, and remaining gates

The initial Small check used a frozen source snapshot at `0e3eb702` to avoid
concurrent implementation edits. It was followed by a current-source real Small
smoke and cross-family comparison. TimesFM checks exercised the new feature files;
their initial records correctly identify a dirty checkout based on `0e3eb702`.
Final committed source and distribution identities are recorded separately by
the build helper; this document does not embed the containing archive's own hash.

The configured release build must include the TimesFM guide, example, runtime
constraints, and model modules. The installed-wheel verifier now checks four
catalog adapters and both missing-runtime boundaries without installing weights.
The checks above passed for the preparation tree. Final artifacts are rebuilt
from its committed revision and identified in external manifests. Any subsequent
remote matrix/integration is a separate gate recorded in repository history.

The pre-feature main matrix passed all six Linux/macOS Python 3.11–3.13 jobs at
[run 40233446539](https://github.com/janwolyniak/TSFM-LAB/actions/runs/40233446539).
That result is not a claim that this new feature's remote matrix has run.
GPU/MPS, Windows, other runtimes/checkpoints, probability scoring, and finance
remain outside this verification. No new training, paid compute, or package
publication was performed. `PLAN.md`, downloaded weights, environments, full
experiment outputs, and temporary evidence remain outside Git.

See the [TimesFM guide](timesfm.md), [Chronos guide](chronos.md), and
[release workflow](releasing.md) for maintained commands and publication rules.

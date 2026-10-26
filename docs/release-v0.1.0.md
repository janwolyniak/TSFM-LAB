# TSFM Lab v0.1.0

TSFM Lab's first release provides a local workflow for evaluating time-series
forecasts, from simple baselines to optional Chronos-Bolt and TimesFM 2.5 models.
Use the Python core, scriptable shell commands, or the interactive terminal to
run and inspect reproducible experiments.

**Preparation status:** these notes describe the v0.1.0 candidate. A GitHub
Release, release tag, and PyPI publication have not been created by this work.

## Included in this release

- Strict CSV and optional Parquet import for one regular numeric time series,
  with explicit timestamps, frequency, timezone, and units.
- Rolling-origin evaluation with last-value and seasonal-naive baselines,
  pooled/per-lead MAE, RMSE, and MASE, plus aligned prediction records.
- YAML experiment configuration, saved run history, artifact validation,
  comparison of compatible runs, and CSV/JSON/optional Parquet export.
- The `tsfm` command and a persistent interactive prompt with slash commands,
  guided setup, progress, cancellation, and a bundled offline demonstration.
- An optional CPU Chronos-Bolt adapter for the supported Tiny/Small checkpoints
  and compatible standalone local Bolt weights. Downloads are explicit;
  experiments use cached/local files and record checkpoint identities (resolved
  revisions or local-file hashes) and runtime provenance.
- An optional TimesFM 2.5 CPU adapter with explicit checkpoint identities,
  a pinned 3.0.2 runtime, median forecasts, and the same configuration, guided
  setup, saved-result, comparison, and export contracts.
- Source-to-wheel builds, archive-content validation, and verification of an
  ordinary installation outside the source checkout.
- A separate hypothetical OHLC report workflow from saved lead-1 price
  forecasts, with explicit latency/costs, long/flat/short accounting, cash and
  costed buy-and-hold benchmarks, and checked report reload/equity export.

## Requirements and installation

Python 3.11 or newer is required. The core depends on NumPy and pandas; command,
Parquet, and model runtimes are optional extras. The configured CI matrix covers
Python 3.11, 3.12, and 3.13 on Linux and macOS.

The intended release assets are:

| Asset | Purpose |
| --- | --- |
| `tsfm_lab-0.1.0-py3-none-any.whl` | Installable Python package. |
| `tsfm_lab-0.1.0.tar.gz` | Source distribution with documentation and examples. |
| `SHA256SUMS` | SHA-256 checksums for the two distribution files. |

After downloading the wheel, install it in an activated virtual environment:

```sh
python -m pip install './tsfm_lab-0.1.0-py3-none-any.whl[cli,parquet]'
tsfm --version
tsfm ui --workspace results
```

Use `/demo` in the terminal to run the bundled offline example. Install the wheel
without extras for the Python core, or use `[cli]` when Parquet is unnecessary.
Model weights are not included in the assets.

See the [installation guide](https://github.com/janwolyniak/TSFM-LAB/blob/main/docs/installation.md),
[terminal walkthrough](https://github.com/janwolyniak/TSFM-LAB/blob/main/docs/tui.md),
[CLI reference](https://github.com/janwolyniak/TSFM-LAB/blob/main/docs/cli.md), and
[Chronos-Bolt guide](https://github.com/janwolyniak/TSFM-LAB/blob/main/docs/chronos.md)
and [TimesFM guide](https://github.com/janwolyniak/TSFM-LAB/blob/main/docs/timesfm.md)
for complete setup and usage.

## Scope and verification limits

- The initial interface evaluates one regular univariate series. Missing values,
  irregular grids, duplicate timestamps, and unsorted input are rejected rather
  than silently repaired.
- Chronos-Bolt Tiny/Small and TimesFM 2.5 have bounded CPU integration evidence.
  Baseline installation checks do not verify every optional runtime or checkpoint.
- GPU/MPS execution, panel data, covariates, and probabilistic scoring remain
  outside this release's verified scope.
- Windows and Python/platform combinations outside the configured matrix remain
  unverified. Synthetic example scores check arithmetic and do not establish a
  model advantage on real data.
- The [finance extension](finance-extension.md) is a bounded simulation with
  explicit next-open/target-close fills; it supplies no broker execution or
  live trading. Additional model families, model training, and a hosted service
  are outside this release.
- Project code uses Apache License 2.0. Checkpoints, model implementations, and
  datasets retain their own licenses.

The [verification record](https://github.com/janwolyniak/TSFM-LAB/blob/main/docs/milestone5-verification.md)
documents the preparation and its limits. Final source revision, artifact hashes,
local checks, and current CI evidence must accompany the publication decision
under the [release guide](https://github.com/janwolyniak/TSFM-LAB/blob/main/docs/releasing.md).

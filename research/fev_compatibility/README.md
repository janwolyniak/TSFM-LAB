# Isolated fev compatibility study

This research companion executes seven original synthetic fixture groups against
`fev==0.10.0`. It is a feasibility study, not a production adapter, alternative
scoring backend, or supported panel/probabilistic interface. It adds no dependency
to TSFM Lab's core or routine CI.

The recorded run used Python 3.13.6 on macOS arm64. All seven groups passed;
see [results.json](results.json) for numerical results and core-source hashes.
[upstream.json](upstream.json) records the exact PyPI wheel identity and source
tag `v0.10.0`, commit `ec833b9865fac9a021e8ec9cc56b67bb4c515427`.
The installed `task.py`, `metrics.py`, `utils.py`, `__init__.py`, and
`__about__.py` matched that commit byte for byte.

## Reproduce

Use Python 3.13 and an isolated environment after obtaining dependency-installation
authorization. From the repository root, a clean environment can be prepared with:

```sh
python3 -m venv /tmp/tsfm-fev-environment
/tmp/tsfm-fev-environment/bin/python -m pip install -r research/fev_compatibility/requirements-study.txt
PYTHONDONTWRITEBYTECODE=1 /tmp/tsfm-fev-environment/bin/python research/fev_compatibility/run_study.py --output /tmp/tsfm-fev-output
```

Choose fresh environment/output paths; the runner refuses an existing output
directory. Installation needs PyPI access or equivalent locally available wheels.
The fixture run uses generated local Parquet files, sets Hub/dataset offline
flags, and blocks IP socket connections while permitting local process IPC.
No model weights, external datasets, credentials, or paid compute are needed.

The recorded installation instead used a fresh `--system-site-packages` venv,
reusing available libraries without modifying them. A size inspection preceded
installation: the five new wheels in [requirements-overlay.txt](requirements-overlay.txt)
totaled **888,930 bytes**. That file pins their hashes; it assumes the other exact
versions in [requirements-study.txt](requirements-study.txt) are already present.
A clean reproduction may need substantially more downloaded dependencies.

The inherited environment has an unrelated `mlxtend 0.24.0` requirement for
NumPy >=2.3.5 while NumPy 2.1.3 is installed. Whole-environment `pip check` therefore
fails. The study separately checked all active requirements in its own
43-distribution dependency closure, including requested extras; those passed.
It neither imports `mlxtend` nor repairs global packages. The recorded run is not
a hermetic environment or cross-platform compatibility claim.

## What the fixtures establish

| Group | Observed result |
| --- | --- |
| Bundled arithmetic | Both baselines have the same 16 keyed predictions as TSFM. Last-value MAE/RMSE/MASE are 2.5/2.915476/2.5; seasonal-naive scores are 1/1/1. |
| Scaling and pooling | Changing-history MASE is 1.611111 in native fev versus TSFM's fixed-scale 3.0. Native window-averaged RMSE is 6 versus pooled 6.218253. |
| Unequal series | Native equal-window MAE 2.5 differs from case-weighted 3.333333 and series-weighted 5.0; a short item disappears from one native window. |
| Undefined/nonfinite | Native undefined MASE can be NaN; TSFM preserves null and reasons. Native fev rejects nonfinite forecasts, but its scorer omits a NaN actual. |
| Overlap | Nine origin/target/lead cases retain five unique target timestamps. |
| Covariates | A properly classified past-only feature is absent from future input. A deliberately misclassified feature is exposed by fev. |
| Quantiles | MQL 0.3 equals twice ordinary mean pinball 0.15. fev sorts levels and rejects a missing level column, but accepts crossing quantiles. |

The fixture scripts contain the independent arithmetic and temporal assertions.
The prototype's complete-item, declared-availability, and quantile-crossing guards
are **user-defined policies**, not guarantees supplied by fev. Classifying a
feature as known-future does not prove its real publication time. No real release
schedule, probabilistic TSFM adapter, multivariate task, accelerator, significance
test, or forecasting advantage was verified.

The full output directory contains generated inputs, strict-JSON detailed traces,
dependency requirements, and a concise summary. Native undefined values are
represented as null with explicit status in the checked-in summary; the native
NaN representation is recorded separately. fev's `num_forecasts` counts item/window
forecasts here (four), not horizon cases (16).

## Files and decision

- [run_study.py](run_study.py): bounded entry point and evidence summarization.
- [study_support.py](study_support.py): isolated local-data conversion and guards.
- [fixtures_baselines.py](fixtures_baselines.py): matching baseline, scaling,
  and panel-weighting fixtures.
- [fixtures_policies.py](fixtures_policies.py): undefined values, overlap,
  covariate availability, and quantile fixtures.

The supported decision is to keep fev optional and require an explicit semantic
bridge before product integration. The study demonstrates both useful conversion
paths and differences that preclude silently substituting the native scorer.
See the [benchmarking proposal](../../docs/benchmarking-extension.md) for the
remaining product agreement boundary.

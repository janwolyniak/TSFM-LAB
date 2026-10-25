# Shell workflow verification record

Date: 25 October 2026. Scope: Milestone 2, following the owner's continuation
request. Work is local on `feat/cli-experiment-management`, based on the unmerged
numerical-core branch. That prerequisite must be integrated first when publication
and merging are authorized.

## Implemented workflow

The optional `tsfm` command runs validated YAML baseline configurations, inspects
CSV/Parquet datasets, lists and reloads experiment records, compares compatible
completed runs, and exports predictions as CSV, JSON, or Parquet. Application
functions share the existing numerical runner; terminal output is separate from
evaluation and persistence.

Run directories preserve the original YAML and resolved selections, numerical
results, metrics, and predictions. Atomic manifests mark completion last.
Reload verifies artifact hashes/sizes, schemas, model/series identity, complete
configured forecast grids, counts, and scores recalculated from predictions with
the recorded MASE scale. Invalid data/configuration creates no run; later failures
and cancellations retain their stage and do not become comparison candidates.

## Executed checks

Environment: macOS, Python 3.13.6, NumPy 2.1.3, pandas 2.3.3, Typer 0.24.1,
Rich 14.3.2, PyYAML 6.0.2, PyArrow 23.0.1, pytest 9.0.2, Black 26.1.0,
Ruff 0.15.4, and setuptools 80.9.0. Third-party dependencies were already present;
none were downloaded, installed, or upgraded during implementation.

| Check | Outcome |
| --- | --- |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest` | 450 tests passed. |
| `python3 -m black --check src tests examples` | Passed. |
| `python3 -m ruff check src tests examples` | Passed. |
| Wheel build without dependency resolution | Passed; package metadata includes the `tsfm` console entry point and optional dependency groups. |
| Installed console command | Built project wheel installed with `--no-deps --no-index` in a disposable virtual environment using existing system libraries; help/version, both CSV baselines, history, show, and comparison passed. |
| Reload without source data | Renamed the temporary original input file; separate `tsfm` processes still showed, compared, and exported completed records. |
| Formats | CSV/JSON/Parquet exports reloaded 16 forecast cases; Parquet input/inspection reproduced the expected last-value MAE of 2.5. |
| File preservation | Export to an existing destination failed and preserved its original bytes. |
| Terminal rendering | Inspected model, experiment, and metric tables at 80 columns; JSON stdout contained no progress formatting. |
| Optional boundaries | Core import passed with terminal/Parquet modules blocked; a missing CLI extra produced a useful exit-2 message. |

Independent review identified saved schema/model/grid/score inconsistency and
validation error classification gaps. Regression tests now reject these cases.
The final read-only review found no outstanding actionable issues in that scope.

The original synthetic example still yields last-value MAE/MASE 2.5 and
RMSE sqrt(8.5), and seasonal-naive MAE/RMSE/MASE 1. These are constructed
pipeline checks, not pretrained-model benchmark results.

## Limits and continuation

The configured Linux/macOS Python 3.11/3.12/3.13 CI matrix has not run remotely.
The disposable installation tested this project's wheel and console entry point
against existing dependencies; it did not test a fresh dependency installation or
establish compatibility across the whole advertised matrix.

Experiments remain synchronous and local. There is no model download, TUI,
SQLite index, automatic resumption, or inference cache. Large workspace performance
has not been benchmarked. The next milestone is the explicitly downloaded,
revision-pinned Chronos-Bolt adapter; actual dependency/model downloads still
require the relevant owner authorization.

The original untracked `PLAN.md` is preserved. No push, pull request, or merge was
performed. See [CLI usage](cli.md) and [roadmap](roadmap.md) for the next steps.

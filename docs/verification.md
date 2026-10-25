# Numerical core verification record

Date: 25 October 2026. Scope: Milestone 1, agreed after the architecture review.

## Implemented behavior

The Python package loads one regular ISO-timestamped numeric series from CSV,
creates detached historical rolling requests, runs last-value or seasonal-naive
adapters, validates forecast identity/alignment, and returns predictions with
pooled/per-lead MAE, RMSE, and fixed-history MASE. Results expose a JSON-compatible
representation with configuration, counts, scale history, source/environment
provenance, and stage timings.

The architecture, contracts, methodology, and roadmap describe later terminal,
checkpoint, and persistence work separately. No pretrained-model scores were
generated. The existing local product brief was preserved without staging it.

## Local checks

Environment: macOS, Python 3.13.6, NumPy 2.1.3, pandas 2.3.3, pytest 9.0.2,
Ruff 0.15.4, Black 26.1.0, and setuptools 80.9.0. Dependencies were already
available; none were installed or upgraded.

| Check | Outcome |
| --- | --- |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest` | 155 tests passed. |
| `python3 -m black --check src tests examples` | Passed. |
| `python3 -m ruff check src tests examples` | Passed. |
| Offline example | Both baseline scores match independent arithmetic; each returns four windows and 16 aligned cases. |
| Wheel build | Built using the existing backend with no dependency resolution or installation. |
| Packaged-code smoke | Wheel contents inspected; extracted package imports and executes the example using the available NumPy/pandas environment. |
| Core dependency boundary | Importing the runner does not import torch, Hub, Typer, Rich, or Textual. |
| Documentation | Local links/anchors, YAML syntax, Python README example, and rolling-window arithmetic checked. |

The tests cover invalid timestamps/grids and numeric inputs, DST ambiguity,
fingerprints, complete windows and overlap, baseline seasonal wrap, known pooled
and per-lead scores, undefined MASE, large finite errors, adapter failures,
cleanup, future-value perturbation, detached buffers, and output identities.
Review findings about mutable request dates, mutable adapter options, late
fingerprint validation, and unrequested MASE arithmetic were repaired with
regression tests. The subsequent independent review reported no outstanding
findings in that scope.

## Verification limits and continuation

The CI workflow is configured for Linux/macOS with Python 3.11, 3.12, and 3.13;
remote CI has not run. No fresh environment with dependency installation was
tested. Packaged-code execution reuses the existing numerical dependencies and
therefore does not establish clean-install or cross-platform compatibility.

The next milestone adds YAML validation, shell commands, Parquet, local run
records, history, and comparison compatibility checks. Model weights, terminal
UI behavior, accelerators, checkpoint loading, persistent failure recovery,
and financial evaluations are outside the checks performed here.

Work remains local on `feat/numerical-evaluation-core`, based on the existing
contribution-workflow commits. Publication and integration into `main` have not
been performed. Follow [CONTRIBUTING.md](../CONTRIBUTING.md) when those actions
are authorized.

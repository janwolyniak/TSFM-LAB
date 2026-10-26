# TSFM Lab

**Local time-series forecast evaluation with reproducible experiments.**

[![Source version: 0.1.0](https://img.shields.io/badge/source-v0.1.0-2563eb)](pyproject.toml)
[![Release: in preparation](https://img.shields.io/badge/release-in_preparation-d97706)](docs/release-v0.1.0.md)
[![Python checks](https://github.com/janwolyniak/TSFM-LAB/actions/workflows/checks.yml/badge.svg?branch=main&event=push)](https://github.com/janwolyniak/TSFM-LAB/actions/workflows/checks.yml)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](docs/installation.md)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-15803d)](LICENSE)

[Quick start](#run-the-offline-example) ·
[Terminal walkthrough](docs/tui.md) ·
[Chronos-Bolt](docs/chronos.md) ·
[TimesFM](docs/timesfm.md) ·
[OHLC backtests](docs/finance-extension.md) ·
[Release notes](docs/release-v0.1.0.md) ·
[Contributing](CONTRIBUTING.md)

TSFM Lab is a local forecasting evaluation project with a domain-independent
Python core, scriptable commands, an interactive terminal, and optional
Chronos-Bolt and TimesFM 2.5 adapters. A separate finance extension evaluates
hypothetical OHLC strategies from saved price forecasts.

## Project status

- **Evaluate:** import one regular numeric series from CSV or optional Parquet,
  configure rolling forecasts in YAML, and score pooled/per-lead MAE, RMSE, and MASE.
- **Reproduce:** save experiment records, inspect history, compare compatible
  runs, and export aligned predictions as CSV, JSON, or optional Parquet.
- **Work locally:** use the Python core, scriptable `tsfm` commands, or a
  persistent terminal prompt with inline setup, progress, and cancellation.
  Output stays in terminal scrollback; terminal dependencies remain optional.
- **Add pretrained forecasts:** use optional Chronos-Bolt or TimesFM 2.5 adapters with
  curated checkpoints, explicit downloads, exact revisions, local/cache-only
  experiments, and recorded runtime provenance. Additional model families remain
  later work.
- **Evaluate price decisions:** derive long/flat/short signals from saved lead-1
  forecasts, require explicit OHLC timing and costs, and inspect verified
  strategy, cash, and buy-and-hold reports without rerunning a model.

### Release status

The source version is **0.1.0**. The first GitHub Release is being prepared;
no GitHub or PyPI package release has been published. See the
[draft release notes](docs/release-v0.1.0.md),
[release verification record](docs/milestone5-verification.md), and
[publication checklist](docs/releasing.md). The CI badge above tracks checks on
`main`; it does not indicate that a package has been published.

## Run the offline example

From a clone or unpacked source archive, using Python 3.11 or newer, create an
environment and install the command interface:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.txt '.[cli,parquet]'
python examples/baseline.py
```

`requirements-dev.txt` records exact development constraints; the library allows
the compatible dependency ranges declared in `pyproject.toml`. No model weights,
credentials, or torch are required. Install only `.[cli]` for shell commands,
add `[parquet]` for Parquet input/export, or use the base package for Python-only
CSV evaluation. See the [installation guide](docs/installation.md) for built
wheels, optional runtimes, and the generated `tsfm` command.

The same installation provides the interactive terminal:

```sh
tsfm ui --workspace results
```

In a terminal, `tsfm` without arguments also opens the application. With redirected
input or output it prints help; named shell commands remain scriptable. See the
[command-session walkthrough](docs/tui.md) for `/demo`, `/new`, history, and cancellation.

If NumPy and pandas are already available, the example can run from source
without an installation:

```sh
PYTHONPATH=src python3 examples/baseline.py
```

It prints JSON containing both experiments and all aligned predictions. Expected
scores for this constructed 32-observation series are:

| Baseline | MAE | RMSE | MASE |
| --- | --- | --- | --- |
| Last value | 2.5 | 2.915476 | 2.5 |
| Seasonal naive with period 4 | 1.0 | 1.0 | 1.0 |

Each run has context 8, horizon 4 observations, stride 4, four windows, and 16
forecast cases in `synthetic_units`. The scaling history precedes evaluation.
These are arithmetic checks on [original synthetic data](examples/datasets/README.md),
not results for pretrained models or real-world demand.

## Use the shell workflow

```sh
tsfm models list
tsfm run examples/configurations/last-value.yaml --json
tsfm run examples/configurations/seasonal-naive.yaml --json
tsfm experiments list --workspace results --json
```

The two run commands return new UUIDs and write under `results/experiments/`.
Use those UUIDs in place of `RUN_ID`, `FIRST_ID`, and `SECOND_ID` below:

```sh
tsfm experiments show RUN_ID --workspace results --json
tsfm compare FIRST_ID SECOND_ID --workspace results --json
tsfm export RUN_ID --workspace results --output predictions.parquet --json
```

Exports support CSV, JSON, and optional Parquet and refuse existing destinations.
Comparisons require the same dataset/import identity, forecast cases, context,
horizon, stride, scaling convention, and metric definitions. Saved artifacts
are checked before showing completed scores, comparing, or exporting.

Relative dataset/output paths resolve against the YAML file's directory.
`--workspace` on history/show/compare/export resolves against the current working
directory; the examples use `results` from the repository root. Use
`python -m tsfm_lab` in place of `tsfm`, or `PYTHONPATH=src python3 -m tsfm_lab` when the CLI
dependencies are already available and you are working without installation.
See [CLI and configuration](docs/cli.md) for validation, records, and exit codes.

## Use Chronos Bolt

The optional CPU adapter supports the curated Tiny, Small, and Base checkpoints and
compatible standalone local Bolt weights. `tsfm models info chronos_bolt` shows
catalog metadata without downloading or importing the model runtime. Downloading
requires an explicit `tsfm models pull` command; normal experiments require
already cached or local files.

See the [Chronos guide](docs/chronos.md) for the pinned runtime, exact revisions,
cache paths, local checkpoints, median semantics, and the optional CPU test.
[chronos-bolt-tiny.yaml](examples/configurations/chronos-bolt-tiny.yaml) uses the
same numerical task as the baseline examples. Routine tests need no model weights.
Base is available after metadata/API compatibility checks; its weights were not
downloaded or individually tested with inference. The
[catalog guide](docs/model-catalog.md) distinguishes reference and selected-file
verification and documents the pinned Base choice.

## Use TimesFM 2.5

The optional `timesfm_2p5` adapter provides CPU float32 median forecasts with
context up to 16,256 observations and horizon up to 128. It uses the explicit
TimesFM 2.5 model through the pinned `timesfm` 3.0.2 runtime; model and library
versions are separate identities. Model imports and downloads remain explicit.

See the [TimesFM guide](docs/timesfm.md) for installation, exact cached revisions,
safe local loading, and the [matching example](examples/configurations/timesfm-2p5.yaml).
The [verification record](docs/milestone6-verification.md) covers the real CPU
workflow and expanded Chronos-Bolt Small checks. It makes no real-data advantage
claim.

## Evaluate a hypothetical price strategy

Run the [synthetic OHLC example](examples/configurations/finance-seasonal-naive.yaml)
and use its returned UUID as `RUN_ID`:

```sh
tsfm run examples/configurations/finance-seasonal-naive.yaml --json
tsfm finance backtest RUN_ID --workspace results \
  --source-config examples/configurations/finance-seasonal-naive.yaml \
  --ohlc examples/datasets/ohlc.csv --price-target \
  --initial-capital 1000 --threshold-bps 100 --fee-bps 5 --slippage-bps 2 \
  --short-financing-bps 3 --signal-latency-seconds 60 \
  --output results/finance/report.json
tsfm finance show results/finance/report.json --json
tsfm finance export results/finance/report.json --output results/finance/equity.csv
```

The strategy enters at the next bar's open and exits at that bar's close.
Entry must be strictly later than the origin close plus declared latency;
touching continuous bars fail even with zero latency. All assumptions above
are required inputs, and the numbers are illustrative. This is a constructed
accounting demonstration. See the [finance contract](docs/finance-extension.md)
for data requirements, exact formulas, and limitations.

## Use the Python core

```python
import pandas as pd

from tsfm_lab.contracts import EvaluationSpec
from tsfm_lab.data import load_csv
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.models.baselines import LastValue

dataset = load_csv(
    "examples/datasets/seasonal.csv",
    timestamp_column="timestamp",
    target_column="demand",
    frequency="1h",
    units="synthetic_units",
)
spec = EvaluationSpec(
    context_length=8,
    horizon=4,
    stride=4,
    evaluation_start=dataset.timestamps[16],
    evaluation_end=dataset.timestamps[-1] + pd.Timedelta("1h"),
    seasonal_period=4,
)
result = run_experiment(dataset, LastValue(), spec)
print(result.scores.overall)
print(result.predictions)
```

CSV timestamps must be ISO dates or timestamps; Parquet also accepts typed Arrow
datetimes. Supply `timezone="UTC"` (or an
explicit source timezone) for naive timestamps. Aware timestamps normalize to
UTC. Input must be strictly ordered, unique, finite, and follow the declared
fixed frequency. Gaps, ambiguous DST timestamps, and missing values fail without
sorting or filling. MASE is `null` with a reason for zero or insufficient scaling
history. See [evaluation.md](docs/evaluation.md) for complete semantics.

## Checks

```sh
python3 -m black --check src tests examples scripts
python3 -m ruff check src tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest
python3 scripts/build_distribution.py --out-dir /tmp/tsfm-lab-distributions
```

Install development checks when needed with:

```sh
python -m pip install -c requirements-dev.txt setuptools
python -m pip install --no-build-isolation -c requirements-dev.txt -c requirements-tui.txt '.[dev,cli,parquet,tui]'
```

The distribution builder uses already available setuptools 77+ and pip. It makes
a source archive, checks its contents, and builds the wheel from that archive
without resolving dependencies. Use an empty output directory. `[tui]` supplies
the original headless Textual regression tests; those skip when Textual is
unavailable. The active prompt uses `[cli]`; its tests include real pseudo-terminal
interactions. CI installs both sets without model downloads. The configured CI matrix covers Python 3.11,
3.12, and 3.13 on Linux and macOS. Local verification used Python 3.13.6 on macOS;
the live matrix status is available in the CI results linked above. Release checks include an installation into a
clean environment; see the [Milestone 5 record](docs/milestone5-verification.md).
See the [Milestone 1 record](docs/verification.md) and
[Milestone 2 record](docs/milestone2-verification.md), plus the
[Chronos CPU record](docs/milestone3-verification.md) and
[original terminal record](docs/milestone4-verification.md), plus the
[prompt rework record](docs/terminal-rework.md), for evidence and limits.

The [release guide](docs/releasing.md) describes artifact verification and the
remaining publication gates. The repository has not published a package release.

## Design and implementation plan

- [Architecture and bounded MVP](docs/architecture.md): critique, proposed stack,
  component boundaries, terminal execution, persistence, and model integration.
- [Core contracts](docs/contracts.md): datasets, experiments, adapters, and
  normalized forecast outputs.
- [Evaluation methodology](docs/evaluation.md): temporal boundaries, metric
  formulas, aggregation, and comparison requirements.
- [Development roadmap](docs/roadmap.md): review decisions, implementation tasks,
  and milestone acceptance criteria.
- [Adapter extension guide](docs/adapters.md): the current protocol and a runnable
  direct-Python example, including limits on shell/catalog integration.

The agreed implementation sequence starts with Python, NumPy, and pandas;
uses Typer and Rich for shell commands and prompt_toolkit for the persistent
interactive prompt. Model runtimes remain optional. Chronos-Bolt is the first integrated
family, followed by TimesFM 2.5; other Chronos variants and TimesFM 3.0 remain
unsupported. The [finance contract](docs/finance-extension.md) defines the separate
OHLC extension; the [benchmarking study](docs/benchmarking-extension.md) records
the bounded compatibility findings before further benchmark integration.

## Contributing

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for development, validation, commits,
review, and merging. Architecture and evaluation changes require discussion
before implementation. Dependency installation and publication need owner
authorization within the scope of those actions.

## License

Project code is licensed under [Apache License 2.0](LICENSE). Model implementations,
checkpoints, and datasets retain their own licenses.

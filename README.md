# TSFM Lab

TSFM Lab is a local forecasting evaluation project with a domain-independent
Python core, scriptable commands, an interactive terminal, and an optional
Chronos-Bolt adapter. Financial analysis remains a later extension.

## Project status

Milestones 1 and 2 provide an offline Python and shell workflow: load one regular
numeric series from CSV or Parquet, configure rolling baseline forecasts in YAML,
save experiment records, inspect history, compare compatible runs, and export
predictions. Scores include pooled/per-lead MAE, RMSE, and MASE. The `tsfm` command
uses optional terminal dependencies; the numerical core remains independent.
Milestone 3 adds curated checkpoint selection, explicit pulls, exact revisions,
verified local/cache-only Chronos-Bolt inference, and model-runtime provenance.
The interactive workflow uses a persistent local prompt, slash commands, inline
setup questions, progress/cancellation, saved results, comparison, and export.
Output stays in terminal scrollback. Other model families remain later work.

## Run the offline example

From a clone, using Python 3.11 or newer, create an environment and install the
core with development tools when you are ready to install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.txt -e '.[dev,cli,parquet]'
python examples/baseline.py
```

`requirements-dev.txt` records exact development constraints; the library allows
the compatible dependency ranges declared in `pyproject.toml`. No model weights,
credentials, or torch are required. Install only `.[cli]` for shell commands,
add `[parquet]` for Parquet input/export, or use the base package for Python-only
CSV evaluation. The commands above install the development checks as well.

For the interactive terminal, add the optional UI dependencies:

```sh
python -m pip install -c requirements-dev.txt -e '.[cli,parquet]'
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

The optional CPU adapter supports the official tiny/small checkpoints and
compatible standalone local Bolt weights. `tsfm models info chronos_bolt` shows
catalog metadata without downloading or importing the model runtime. Downloading
requires an explicit `tsfm models pull` command; normal experiments require
already cached or local files.

See the [Chronos guide](docs/chronos.md) for the pinned runtime, exact revisions,
cache paths, local checkpoints, median semantics, and the optional CPU test.
[chronos-bolt-tiny.yaml](examples/configurations/chronos-bolt-tiny.yaml) uses the
same numerical task as the baseline examples. Routine tests need no model weights.

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
python3 -m black --check src tests examples
python3 -m ruff check src tests examples
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest
python3 -m pip wheel --no-deps --no-build-isolation --no-index --wheel-dir /tmp/tsfm-lab-wheels .
```

The wheel command uses an already available setuptools 77+ build backend; it
does not install dependencies. Install `[tui]` with the constraints above to
include the original headless Textual regression tests; those skip when Textual is
unavailable. The active prompt uses `[cli]`; its tests include real pseudo-terminal
interactions. CI installs both sets without model downloads. The configured CI matrix covers Python 3.11,
3.12, and 3.13 on Linux and macOS. Local verification used Python 3.13.6 on macOS;
remote CI and a fresh dependency installation have not been run. See the
[Milestone 1 record](docs/verification.md) and
[Milestone 2 record](docs/milestone2-verification.md), plus the
[Chronos CPU record](docs/milestone3-verification.md) and
[original terminal record](docs/milestone4-verification.md), plus the
[prompt rework record](docs/terminal-rework.md), for evidence and limits.

## Design and implementation plan

- [Architecture and bounded MVP](docs/architecture.md): critique, proposed stack,
  component boundaries, terminal execution, persistence, and model integration.
- [Core contracts](docs/contracts.md): datasets, experiments, adapters, and
  normalized forecast outputs.
- [Evaluation methodology](docs/evaluation.md): temporal boundaries, metric
  formulas, aggregation, and comparison requirements.
- [Development roadmap](docs/roadmap.md): review decisions, implementation tasks,
  and milestone acceptance criteria.

The agreed implementation sequence starts with Python, NumPy, and pandas;
uses Typer and Rich for shell commands and prompt_toolkit for the persistent
interactive prompt. Model runtimes remain optional. Chronos-Bolt is the first integrated
family; other Chronos variants have separate APIs and remain unsupported.

## Contributing

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for development, validation, commits,
review, and merging. Architecture and evaluation changes require discussion
before implementation. Dependency installation and publication need owner
authorization within the scope of those actions.

## License

Project code is licensed under [Apache License 2.0](LICENSE). Model implementations,
checkpoints, and datasets retain their own licenses.

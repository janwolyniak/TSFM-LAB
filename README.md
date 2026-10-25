# TSFM Lab

TSFM Lab is a local forecasting evaluation project with a domain-independent
Python core. The planned terminal application will evaluate pretrained
time-series models on user datasets. Financial analysis will be a later extension.

## Project status

Milestone 1 implements an offline Python workflow: load one regular numeric series
from CSV, generate rolling forecasts with last-value and seasonal-naive baselines,
and return aligned predictions, pooled/per-lead MAE, RMSE, and MASE, configuration,
and provenance. The repository includes packaging, tests, and a small synthetic
example. CLI/TUI, Parquet, persistence, and pretrained adapters are later milestones;
there is no `tsfm` command yet.

## Run the offline example

From a clone, using Python 3.11 or newer, create an environment and install the
core with development tools when you are ready to install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.txt -e '.[dev]'
python examples/baseline.py
```

`requirements-dev.txt` records exact development constraints; the library allows
the compatible dependency ranges declared in `pyproject.toml`. No model weights,
credentials, torch, or terminal frameworks are required.

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

CSV timestamps must be ISO dates or timestamps. Supply `timezone="UTC"` (or an
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
does not install dependencies. The configured CI matrix covers Python 3.11,
3.12, and 3.13 on Linux and macOS. Local verification used Python 3.13.6 on macOS;
remote CI and a fresh dependency installation have not been run. See the
[verification record](docs/verification.md) for evidence and limits.

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
introduces Typer and Rich for shell commands; and adds Textual for the interactive
application. Model runtimes remain optional. Chronos-Bolt Tiny is the selected
first adapter candidate, pending its runtime/dependency validation.

## Contributing

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for development, validation, commits,
review, and merging. Architecture and evaluation changes require discussion
before implementation. Dependency installation and publication need owner
authorization within the scope of those actions.

## License

Project code is licensed under [Apache License 2.0](LICENSE). Model implementations,
checkpoints, and datasets retain their own licenses.

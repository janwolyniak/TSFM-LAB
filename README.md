# TSFM Lab

**Local time-series forecast evaluation with reproducible experiments.**

[![Source version: 0.3.1](https://img.shields.io/badge/source-v0.3.1-2563eb)](pyproject.toml)
[![GitHub release: v0.3.1](https://img.shields.io/badge/GitHub_release-v0.3.1-15803d)](https://github.com/janwolyniak/TSFM-LAB/releases/tag/v0.3.1)
[![Python checks](https://github.com/janwolyniak/TSFM-LAB/actions/workflows/checks.yml/badge.svg?branch=main&event=push)](https://github.com/janwolyniak/TSFM-LAB/actions/workflows/checks.yml)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](docs/installation.md)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-15803d)](LICENSE)

[Quick start](#run-the-offline-example) ·
[Terminal walkthrough](docs/tui.md) ·
[Chronos-Bolt](docs/chronos.md) ·
[Chronos-2](docs/chronos2.md) ·
[TimesFM](docs/timesfm.md) ·
[TiRex-2](docs/tirex2.md) ·
[TTM](docs/ttm.md) ·
[TOTO 2](docs/toto2.md) ·
[Usage profiles](docs/usage-profiles.md) ·
[Windows target](docs/windows.md) ·
[OHLC backtests](docs/finance-extension.md) ·
[Release notes](docs/release-v0.3.1.md) ·
[Public guide](https://janwolyniak.github.io/TSFM-LAB/) ·
[Research origin and citation](#research-origin-and-citation) ·
[Contributing](CONTRIBUTING.md)

TSFM Lab is a local forecasting evaluation project with a domain-independent
Python core, scriptable commands, an interactive terminal, and optional
Chronos-Bolt, Chronos-2, TimesFM 2.5, TiRex-2, Granite TTM R2/R2.1 and TOTO 2 adapters.
The TTM additions and TOTO 2 have bounded macOS CPU evidence; optional runtimes
on other platforms remain subject to qualification.
A separate finance extension evaluates
hypothetical OHLC strategies from saved price forecasts.

TSFM Lab extends the smaller, finance-focused model-testing pipeline developed
for [Zeroshot_forecasting](https://github.com/janwolyniak/Zeroshot_forecasting)
into a broader, domain-independent evaluation platform. The projects remain in
separate repositories: Zeroshot_forecasting contains the financial research
study and its supporting experiments; TSFM Lab provides reusable tooling for
new forecasting evaluations. See [research origin and citation](#research-origin-and-citation)
to acknowledge the study when using TSFM Lab in your work.

## Project status

- **Evaluate:** import one regular numeric series from CSV or optional Parquet,
  configure rolling forecasts in YAML, and score pooled/per-lead MAE, RMSE, and MASE.
- **Forecast ahead:** select separate future mode from the final observation,
  retain original-unit forecasts and consumed history, and show accuracy as
  unavailable while future actuals are absent. This is 0.3.0 behavior;
  see the [future guide](docs/future-forecasting.md).
- **Reproduce:** save experiment records, inspect history, compare compatible
  runs, and export aligned predictions as CSV, JSON, or optional Parquet.
- **Work locally:** use the Python core, scriptable `tsfm` commands, or a
  persistent terminal prompt with inline setup, progress, and cancellation.
  Output stays in terminal scrollback; terminal dependencies remain optional.
- **Add pretrained forecasts:** choose Chronos-Bolt, Chronos-2, TimesFM 2.5, TiRex-2, TTM or TOTO 2 with
  curated checkpoints, explicit downloads, exact revisions, local/cache-only
  experiments, and recorded runtime provenance. TTM uses explicit context profiles
  and a native-point evaluation task. TOTO 2 requires Python 3.12/3.13;
  configuration/fake-runtime checks do not establish installed or real inference support.
- **Choose a workflow:** select Generic, Finance, Energy, Temperature, or Weather
  in `/new`. Profile-specific questions preserve declared units and measurement
  semantics; history and reports show the selected usage without changing scores.
- **Use optional assistance:** describe an experiment for editable setup choices,
  or request explanations of checked models, results, and finance reports. Seven
  connection routes keep account/model/usage selection explicit; outgoing content
  and execution require review. See the [assistant guide](docs/language-assistance.md).
  DeepSeek, local Ollama and compatible endpoints are 0.3.0 additions;
  live provider and native-vault qualification remain pending.
- **Evaluate price decisions:** derive long/flat/short signals from saved lead-1
  forecasts, require explicit OHLC timing and costs, and inspect verified
  strategy, cash, and buy-and-hold reports without rerunning a model.

### Release status

Version **0.3.0** adds onboarding/preferences, model-specific setup, guided
preparation and saved outputs, future forecasting, additional connections and
model profiles, Windows portability, and
[static documentation preparation](docs/public-documentation.md).
The **0.3.1 qualification patch** repairs the setup, offline assistant protocol
and Windows defects found during follow-up checks. The original 0.3.0 tag and
assets remain unchanged. The [GitHub release](https://github.com/janwolyniak/TSFM-LAB/releases/tag/v0.3.1)
provides a source archive, installable wheel, verification manifest and checksums.
The curated guide uses the owner-approved GitHub Pages URL. Source remains
private; PyPI publication remains separate.

The release follows local review, offline tests and exact installed-package
verification. The five new TTM profiles passed real CPU checks on macOS ARM64
Python 3.12; TOTO passed on Python 3.12 and 3.13 using SciPy 1.14.1. See the
[model evidence](docs/v0.3.0-model-qualification.md) for exact wheel identities,
native output agreement, causal replay, recovery and detached-export limits.
**Windows 11/Windows Terminal, live providers, native vaults and browser sign-in
remain unqualified by owner decision.** TOTO 2 supports Python 3.12–3.13 only.
Hosted core CI is separate from those deferred interactive and optional-runtime
checks. See the [release notes](docs/release-v0.3.1.md),
[patch verification record](docs/v0.3.1-verification.md), and
[release procedure](docs/releasing.md). The CI badge tracks `main` and does not
certify a release whose jobs could not start.

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

## Use Chronos-2

The optional `chronos_2` adapter offers Small and standard checkpoints through
the existing Chronos runtime. It supports independent univariate CPU float32
median forecasts, context up to 8,192 observations, and horizon up to 1,024.
Use `/new chronos_2` in the terminal to choose either variant, or start from the
[Small YAML example](examples/configurations/chronos2-small.yaml).

See the [Chronos-2 guide](docs/chronos2.md) for explicit downloads, pinned
revisions, safe loading, and saved-result behavior. Small has a bounded real CPU
check; standard has metadata/API compatibility checks without individual inference.
The [P2 record](docs/p2-chronos2-verification.md) describes the evidence and limits.

## Use Granite TTM R2 and R2.1

The optional `ttm_r2` adapter retains its three original R2 MSE profiles and adds
512-context native horizons 192/336/720 plus R2.1 weekly 52/16 and daily 90/30.
Each profile requires its exact context and supported cadence. R2 accepts
one-minute through hourly series; R2.1 requires fixed `7D` or `1D` respectively.
The five additions have offline configuration/fake-runtime checks, with real
inference pending in the [S6 record](docs/s6-ttm-verification.md). Select `/new ttm_r2`
and a checkpoint profile; guided setup supplies a longer hourly R2 example.
R2.1 needs your own dataset with the required daily/weekly cadence.
TTM uses the explicit `native_point` task and records causal external scaling
plus the native scaler. Existing models retain their actual median or
deterministic output labels when compared in that task. OHLC reports continue
to require the median task.

See the [TTM guide](docs/ttm.md), [approved contract](docs/ttm-contract.md), and
[P4 verification record](docs/p4-ttm-verification.md). Install the optional
runtime with `python -m pip install -c requirements-ttm.txt '.[cli,ttm]'`. Checkpoints are acquired
separately through an explicit pull.

## Use TOTO 2

The optional `toto_2` adapter curates the 4M TOTO 2 checkpoint on Python 3.12 or
3.13, with project context bounds 32–2,048 in multiples of 32 and horizon 1–128.
It uses strict local Safetensors, native median output and recorded causal
preprocessing. These bounds are integration choices, not upstream architecture
limits. Python 3.11 retains metadata and saved-result operations but cannot run
this family. See the [TOTO guide](docs/toto2.md),
[future YAML example](examples/configurations/toto2-future.yaml) and
[S6 verification limits](docs/s6-toto2-verification.md). Runtime installation/import
and real inference remain pending; routine checks use inert weights/fake tensors.

## Prepare static documentation

The [public documentation workflow](docs/public-documentation.md) builds a
publication-ready static artifact from curated safe text, styles, catalog
metadata and citation fields. The source repository stays private. An explicit
HTTPS base URL is a candidate until a host and deployment are confirmed.
No registration or deployment is part of this local build; see
[S7 evidence](docs/s7-public-docs-verification.md).

## Use TiRex-2

The optional `tirex_2` adapter offers four training profiles through the terminal
and YAML. Default/FEV checkpoints support context up to 2,048 and horizon up to
320; the two GiftEval profiles support 8,192/928. All use CPU float32 inference
and the native median. Use `/new tirex_2` or the
[default YAML example](examples/configurations/tirex2-main.yaml).

**Built with technology from NXAI.** Model weights declare Apache-2.0;
the optional mLSTM Kernels dependency has separate NXAI Community License terms.
See the [TiRex-2 guide](docs/tirex2.md), [NOTICE](NOTICE), and
[P3 verification record](docs/p3-tirex2-verification.md) for pinned references,
runtime setup, checkpoint-specific status, and the qualification evidence.

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
3.12, and 3.13 on Linux, macOS and Windows. Local verification used Python 3.11–3.13 on macOS;
the live matrix status is available in the CI results linked above. Release checks include an installation into a
clean environment; see the [Milestone 5 record](docs/milestone5-verification.md).
See the [Milestone 1 record](docs/verification.md) and
[Milestone 2 record](docs/milestone2-verification.md), plus the
[Chronos CPU record](docs/milestone3-verification.md) and
[original terminal record](docs/milestone4-verification.md), plus the
[prompt rework record](docs/terminal-rework.md), for evidence and limits.

The [release guide](docs/releasing.md) describes artifact verification and
publication. The [GitHub release](https://github.com/janwolyniak/TSFM-LAB/releases/tag/v0.3.1)
provides packages; PyPI publication remains separate.
The [P8 preparation record](docs/p8-release-verification.md) preserves the earlier
qualification evidence. The [v0.3.1 verification record](docs/v0.3.1-verification.md)
describes patch checks and remaining limits. The
[v0.3.0 record](docs/v0.3.0-verification.md) retains its original release evidence. Earlier
[v0.2.0 evidence](docs/v0.2.0-verification.md) remains historical.

## Design and implementation plan

- [Architecture and bounded MVP](docs/architecture.md): critique, proposed stack,
  component boundaries, terminal execution, persistence, and model integration.
- [Core contracts](docs/contracts.md): datasets, experiments, adapters, and
  normalized forecast outputs.
- [Evaluation methodology](docs/evaluation.md): temporal boundaries, metric
  formulas, aggregation, and comparison requirements.
- [Development roadmap](docs/roadmap.md): review decisions, implementation tasks,
  and milestone acceptance criteria.
- [v0.3.0 plan](docs/release-v0.3.0-plan.md): staged release scope; development
  includes [first-launch onboarding](docs/onboarding.md), `/setup`, `/settings`,
  [model-specific guided setup](docs/model-setup.md), and
  [guided preparation and outputs](docs/guided-preparation.md), and
  [future forecasting](docs/future-forecasting.md).
- [Adapter extension guide](docs/adapters.md): the current protocol and a runnable
  direct-Python example, including limits on shell/catalog integration.

The agreed implementation sequence starts with Python, NumPy, and pandas;
uses Typer and Rich for shell commands and prompt_toolkit for the persistent
interactive prompt. Model runtimes remain optional. Chronos-Bolt is the first integrated
family, followed by TimesFM 2.5, Chronos-2, TiRex-2, and TTM. Further work follows the
[v0.3.0 plan](docs/release-v0.3.0-plan.md). The [finance contract](docs/finance-extension.md) defines the separate
OHLC extension; the [benchmarking study](docs/benchmarking-extension.md) records
the bounded compatibility findings before further benchmark integration.

## Research origin and citation

The related [Zeroshot_forecasting project](https://github.com/janwolyniak/Zeroshot_forecasting)
studies forecast quality and financial decision value in hourly Bitcoin. Its
working paper by Jan Wołyniak and Robert Ślepaczuk is
[*From Forecast Error to Trading Outcomes: A Decision-Layer Evaluation of Frozen
Models in Hourly Bitcoin*](https://github.com/janwolyniak/Zeroshot_forecasting/blob/main/main.tex).
It provides the research background for TSFM Lab's development.

If you use TSFM Lab in academic work, please cite this working paper and identify
the TSFM Lab repository and the version or commit used for your experiments.
The paper citation is also available in [CITATION.cff](CITATION.cff).

```bibtex
@unpublished{wolyniak_slepaczuk_forecast_decision,
  author = {Wo{\l}yniak, Jan and {\'S}lepaczuk, Robert},
  title = {From Forecast Error to Trading Outcomes: A Decision-Layer Evaluation of Frozen Models in Hourly {Bitcoin}},
  note = {Working paper},
  url = {https://github.com/janwolyniak/Zeroshot_forecasting/blob/main/main.tex}
}
```

The two projects have separate evaluation contracts. TSFM Lab's
[OHLC extension](docs/finance-extension.md) uses explicit next-open-to-target-close
execution, latency, and costs; it does not reproduce the paper's decision policy
or empirical results automatically. Cite the underlying models and datasets
separately where applicable.

## Contributing

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for development, validation, commits,
review, and merging. Architecture and evaluation changes require discussion
before implementation. Dependency installation and publication need owner
authorization within the scope of those actions.

## License

Project code is licensed under [Apache License 2.0](LICENSE). Model implementations,
checkpoints, and datasets retain their own licenses.

# TSFM Lab

**Evaluate time-series foundation models on your own data, locally.**

[![Source version: 0.1.0](https://img.shields.io/badge/source-v0.1.0-2563eb)](pyproject.toml)
[![Python checks](https://github.com/janwolyniak/TSFM-LAB/actions/workflows/checks.yml/badge.svg?branch=main&event=push)](https://github.com/janwolyniak/TSFM-LAB/actions/workflows/checks.yml)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](docs/installation.md)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-15803d)](LICENSE)

[Models](#supported-models) · [Install](#installation) ·
[Terminal UI](#terminal-ui) · [Next release](#next-release) ·
[Documentation](#documentation)

## What is TSFM Lab?

TSFM Lab is a local workspace for testing pretrained time-series foundation
models (TSFMs) and simple forecasting baselines under the same evaluation rules.
Bring a historical series, choose a model and forecast horizon, and inspect how
its predictions compare with what actually happened.

The workflow is straightforward:

1. **Import a series** from CSV or optional Parquet, with explicit timestamps,
   frequency, target column, and units.
2. **Run a historical evaluation.** At each rolling origin, the model sees only
   the preceding context and predicts the next observations.
3. **Inspect and compare.** View MAE, RMSE, and MASE overall and by forecast lead,
   compare compatible runs, and export aligned predictions.
4. **Keep the evidence.** Saved experiments record their configuration, scores,
   predictions, and model/checkpoint provenance for later inspection.

Use the guided terminal UI, script experiments with YAML and shell commands, or
work directly with the Python core. Generic, Finance, Energy, Temperature, and
Weather profiles adapt setup questions and result wording to your data while
keeping the same scoring rules.

The current scope is **one regular numeric series evaluated against historical
observations**. Missing values, gaps, and duplicate or unsorted timestamps require
explicit preparation before import. Live feeds, future forecasts without
observed targets, model training, panels, covariates, and probabilistic scoring
are outside the current workflow.

## Supported models

Five pretrained forecasting families are available alongside two offline
baselines. The table describes support in this development checkout; upstream
models may offer capabilities beyond the adapters listed here.

| Family | Available choices | Terminal setup |
| --- | --- | --- |
| [Chronos-Bolt](docs/chronos.md) | Tiny, Small, Base | `/new chronos_bolt` |
| [Chronos-2](docs/chronos2.md) | Small, Standard | `/new chronos_2` |
| [TimesFM](docs/timesfm.md) | TimesFM 2.5 200M | `/new timesfm_2p5` |
| [TiRex-2](docs/tirex2.md) | Default, FEV benchmark, GiftEval zero-shot, GiftEval pretrain | `/new tirex_2` |
| [IBM Granite TTM](docs/ttm.md) | TTM R2 profiles with 512, 1,024, or 1,536 context observations | `/new ttm_r2` |
| Baselines | Last value, Seasonal naive | `/new last_value` or `/new seasonal_naive` |

The pretrained adapters currently use **CPU float32 inference** with optional
runtimes and separately acquired weights. Chronos-Bolt, Chronos-2, TimesFM, and
TiRex-2 expose median forecasts. TTM uses its native point output, requires the
selected exact context length and a minute-to-hour frequency, and supports up
to 96 forecast steps. TiRex-2 choices are training profiles, with different
context/horizon limits.

Inspect the catalog without downloading weights or importing model runtimes:

```sh
tsfm models list
tsfm models info chronos_bolt
```

The [model catalog](docs/model-catalog.md) lists exact limits, pinned checkpoints,
and verification status. Each family has representative CPU inference evidence;
some variants have compatibility checks only. These checks establish integration
behavior, not a ranking of forecasting quality.

**Built with technology from NXAI.** TiRex-2's optional mLSTM Kernels dependency
has separate NXAI Community License terms; see [NOTICE](NOTICE) and the
[runtime guide](docs/tirex2.md#optional-runtime-and-notices).

## Installation

Use **Python 3.11 or newer**. From a source checkout on macOS or Linux:

```sh
git clone https://github.com/janwolyniak/TSFM-LAB.git
cd TSFM-LAB
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.txt '.[cli,parquet]'
tsfm --version
tsfm ui --workspace results
```

If you already have the source, start at the environment creation step.
`[cli]` includes the active terminal UI and shell commands; `[parquet]` adds
Parquet import/export and can be omitted for CSV/JSON workflows. The constraints
file selects recorded dependency versions without installing development tools.
The base package alone provides the NumPy/pandas Python core.

The first demo runs offline after installation: **no model weights, PyTorch,
API keys, or assistant connection are needed**. See the
[installation guide](docs/installation.md) for supplied wheels and platform
verification details. The configured baseline CI covers Python 3.11–3.13 on
Linux and macOS; optional model runtimes have separate qualification records.

### Add a pretrained model

Install only the runtime you need in your chosen environment. Each family guide
covers installation and an explicit checkpoint download:

| Family | Package extra | Runtime constraints |
| --- | --- | --- |
| Chronos-Bolt / Chronos-2 | `cli,chronos` | [requirements-chronos.txt](requirements-chronos.txt) |
| TimesFM 2.5 | `cli,timesfm` | [requirements-timesfm.txt](requirements-timesfm.txt) |
| TiRex-2 | `cli,tirex2` | [requirements-tirex2.txt](requirements-tirex2.txt) |
| TTM R2 | `cli,ttm` | [requirements-ttm.txt](requirements-ttm.txt) |

For example, to install the Chronos runtime:

```sh
python -m pip install -c requirements-chronos.txt '.[cli,chronos]'
```

Runtime installation and weight acquisition are separate steps. Use the chosen
family's guide for `tsfm models pull`, then select its cached exact revision or
a compatible local checkpoint during setup. Normal experiment runs use local
or cached weights and do not download them automatically.

## Terminal UI

The TUI is a persistent command prompt with guided questions, inline progress,
and results kept in terminal scrollback. Start it with `tsfm ui --workspace results`,
or simply `tsfm` in an interactive terminal.

### Your first session

Enter `/demo`, review the experiment summary, and answer `y` to
`Run experiment?`. Then explore the saved result:

```text
/demo
/history
/show latest
/export latest results/exports/forecasts.csv
```

The demo uses a bundled synthetic series and the last-value baseline. It checks
that the workflow works; its scores do not measure real-world model quality.
Exports preserve existing files, so choose a new destination for each export.

### Everyday commands

| Command | What it does |
| --- | --- |
| `/new` | Choose a usage profile, model, dataset, and evaluation settings through guided questions. |
| `/models` | Browse available models and checkpoint choices. |
| `/run CONFIG.yaml` | Run an existing experiment configuration. |
| `/history` | List saved experiments in the current workspace. |
| `/show ID` | Inspect verified scores, metrics by forecast lead, and a prediction preview. |
| `/compare ID ID` | Compare experiments that share a compatible evaluation task. |
| `/export ID PATH` | Export saved predictions as CSV, JSON, or optional Parquet. |
| `/workspace DIRECTORY` | Change the workspace used by the session. |
| `/help` / `/exit` | Show help or leave the application. |

Use `latest`, a full run UUID, or an unambiguous ID prefix. Tab completes
commands and supported paths; Up/Down recall commands. During setup, `/back`
revisits a step and `/cancel` abandons it. Execution requires confirmation.
Ctrl+C requests cancellation during a run.

Start `/new` to use your own data or a domain example. Profiles preserve declared
units and measurement semantics without silently resampling, filling gaps, or
converting values. Finance also supports a separate
[hypothetical OHLC backtest](docs/finance-extension.md) from eligible saved price
forecasts, with explicit execution timing and costs.

See the [terminal walkthrough](docs/tui.md) and
[usage-profile guide](docs/usage-profiles.md) for the full interaction flow.

### Script the same workflow

From the repository root, run a supplied YAML configuration and inspect history:

```sh
tsfm run examples/configurations/last-value.yaml --json
tsfm experiments list --workspace results --json
```

Runs return IDs for `tsfm experiments show`, `tsfm compare`, and `tsfm export`.
YAML dataset/output paths resolve relative to the configuration file;
`--workspace` resolves relative to the current working directory. See the
[CLI reference](docs/cli.md) for configuration and command examples.

## Next release

The working target is **v0.2.0**. Package metadata still reports **0.1.0**, and
the first GitHub Release remains in preparation. This README describes the
current development checkout; implementation progress and publication are
separate milestones.

The [next-release plan](docs/next-release-plan.md) tracks what is already
implemented and what remains:

| Area | Status and expected behavior |
| --- | --- |
| Model selection | Implemented: all five pretrained families, curated checkpoint choices, and explicit compatibility/inference status. |
| Domain setup | Implemented: Generic, Finance, Energy, Temperature, and Weather profiles in guided setup and saved results. |
| Optional assistant connections | Planned: ChatGPT sign-in and OpenAI, Anthropic, or Gemini API-key connections, with explicit connection and usage selection. |
| Natural-language setup | Planned: describe a goal to produce an editable experiment draft, then validate and confirm it through the normal workflow. |
| Result explanations | Planned: explain models, settings, and validated saved results in plain language. |
| Release qualification | Pending: final documentation, distribution builds, installed-workflow checks, and CI qualification. |

The current terminal uses explicit commands and guided questions. Assistant
features are not available yet. Manual workflows will remain usable offline
without an assistant connection. Large comparative benchmarks and model-ranking
claims are deferred beyond this release plan.

## Documentation

| Start here | Details |
| --- | --- |
| [Installation](docs/installation.md) | Environments, extras, wheels, and runtime setup. |
| [Terminal guide](docs/tui.md) · [CLI reference](docs/cli.md) | Interactive and scripted workflows. |
| [Model catalog](docs/model-catalog.md) | Checkpoints, limits, licenses, and verification evidence. |
| [Evaluation methodology](docs/evaluation.md) | Temporal boundaries, metrics, and comparison rules. |
| [Usage profiles](docs/usage-profiles.md) · [Finance reports](docs/finance-extension.md) | Domain setup and hypothetical strategy evaluation. |
| [Architecture](docs/architecture.md) · [Core contracts](docs/contracts.md) · [Adapter guide](docs/adapters.md) | Python interfaces and extension boundaries. |
| [Next release](docs/next-release-plan.md) · [Release notes](docs/release-v0.1.0.md) | Planned work and the prepared v0.1.0 candidate. |
| [Contributing](CONTRIBUTING.md) | Development, validation, review, and merging. |

## Development checks

Install development tools in your chosen environment when authorized:

```sh
python -m pip install -c requirements-dev.txt setuptools
python -m pip install --no-build-isolation -c requirements-dev.txt -c requirements-tui.txt '.[dev,cli,parquet,tui]'
```

Run the repository checks:

```sh
python -m black --check src tests examples scripts
python -m ruff check src tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
python scripts/build_distribution.py --out-dir /tmp/tsfm-lab-distributions
```

Use an empty build output directory. `[tui]` supplies the retained Textual
regression tests; `[cli]` is sufficient for the active prompt. Routine checks
need no model downloads. See the [contribution guide](CONTRIBUTING.md) and
[release guide](docs/releasing.md) for validation and publication requirements.

## License

Project code is licensed under [Apache License 2.0](LICENSE). Model runtimes,
checkpoints, and datasets retain their own licenses.

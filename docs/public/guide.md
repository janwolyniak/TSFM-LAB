## Overview

TSFM Lab is a local tool for running time-series forecasts and inspecting their
evaluation, outputs and reproducibility information. Use its Python core,
scriptable commands or persistent terminal prompt. Forecasting models and
optional assistant models are separate selections.

Start with a baseline. Add a curated pretrained checkpoint when its runtime and
weights are available. A model appearing in the catalog does not establish its
accuracy, resource requirements or native platform support.

This guide describes the current development checkout, including **0.3.0
unreleased features**. The source package version remains **{{SOURCE_VERSION}}**. The source
repository is currently private; installation and repository links require
authorized access. Public documentation preparation does not change access to
the repository, datasets or experiment outputs. No PyPI publication is claimed.

## Quickstart

From an authorized source checkout, create your chosen environment and install
the command interface. These commands require the ordinary package installation
step; the baseline itself uses no model weights, assistant server or API key.
For a Linux/macOS shell, activate the environment before installing:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -c requirements-dev.txt '.[cli]'
```

Other shells need their environment's activation command or explicit executable
paths. Native Windows commands remain subject to platform qualification.
In an actual terminal, launch:

```sh
tsfm ui --workspace results
```

First launch offers workspace, results and model-cache directories, runtime
diagnostics, optional assistant connection setup and a bundled demonstration.
Setup can be skipped or cancelled. `/setup` revisits onboarding; `/settings`
changes nonsecret preferences. Directory changes apply to subsequent work and
do not move existing results or weights. Explicit command/configuration choices
override defaults. Named shell commands never enter interactive onboarding.

Use `/demo` for the bundled baseline, then `/new` for guided setup. Choose a
category, model family/generation/variant, applicable model options, dataset and
mode. Preview timestamps and declared units; review bounds, checkpoint identity,
cache/download requirements and output location before confirming. Compatible
choices survive back navigation. Progress and cancellation use the shared
managed worker, with completed results retained in terminal scrollback.

For a scripted baseline from the repository root:

```sh
tsfm run examples/configurations/last-value.yaml --json
tsfm experiments list --workspace results --json
```

The optional assistant routes include the four existing connections plus
DeepSeek, a local installed Ollama server/model, and an explicit compatible API
endpoint. Requests show destination and usage. Keys are connection-specific;
endpoint changes do not forward another route's credentials. Ollama never pulls
a model or falls back to cloud inference. Assistant drafts remain editable and
must pass ordinary configuration and execution review. Live provider, protected
credential-vault and endpoint-specific qualification remain pending.

## Model matrix

The table is generated from the maintained catalog. **Profile compatibility**
and **reference inference** describe separate evidence. A reference check does
not certify another revision or larger variant, and none of these checks ranks
forecast accuracy. Inspect exact selected files and runtime provenance for a run.

On a narrow screen, scroll the table horizontally to see the compatibility and
inference-status columns. The table also supports keyboard focus.

{{MODEL_MATRIX}}

Pretrained integrations use CPU float32 and one series per call. Baselines
preserve deterministic output labels. Chronos, TimesFM, TiRex and TOTO use median
outputs; TTM uses its native point output without asserting a calibrated mean
or median. The explicit `native_point` task permits disclosed mixed-output
comparisons; ordinary median-task comparisons keep their stricter eligibility.

TTM R2 requires minute-to-hour cadence. Its R2.1 daily/weekly profiles require
exactly `1D`/`7D` fixed UTC intervals and exact context. They are distinct
architecture/frequency profiles, not model sizes. TOTO 2 requires **Python
3.12 or 3.13**, context multiples of 32, and project integration bounds of
32–2,048 context / 1–128 horizon. These are not upstream architecture limits.
Its installed-runtime and real inference qualification remain pending.

Runtime installation and checkpoint acquisition are explicit separate actions.
Safe loading keeps exact revisions, strict parameters, verified local
Safetensors and disabled remote code. Normal scripted runs do not download
weights. Guided preparation displays missing files before its separate download
and run confirmation; cached offline use rechecks integrity.

## Five category examples

These bundled examples use original synthetic data and baseline models. Category
questions preserve declared measurements and units; category labels do not add
forecast accuracy or change the metric definitions. Run from the checkout root:

| Category | Declared target | Example command |
| --- | --- | --- |
| Generic | Synthetic numeric series | `tsfm run examples/configurations/usage-generic.yaml --json` |
| Finance | Toy close price, `toy_quote_per_asset` | `tsfm run examples/configurations/usage-finance.yaml --json` |
| Energy | Power in `kW` | `tsfm run examples/configurations/usage-energy.yaml --json` |
| Temperature | Temperature in `degC` | `tsfm run examples/configurations/usage-temperature.yaml --json` |
| Weather | Synthetic wind speed in `m/s` | `tsfm run examples/configurations/usage-weather.yaml --json` |

Relative dataset/output paths resolve against each YAML file. The default
examples create separate UUID run directories under `results/experiments/`.
Use your own local data only when its access and usage terms permit it.

## Historical and future modes

Historical evaluation creates chronological rolling forecasts using only each
origin's preceding context. Actual values enter evaluation separately. MAE, RMSE
and MASE retain their recorded case aggregation and scaling conventions. Missing
or ineligible metrics include reasons. Comparisons require matching datasets,
forecast cases, context, horizon and methodology; category names alone do not
make comparisons controlled.

Future forecasting uses observations available at the declared final-row origin
and creates the requested horizon on the same fixed UTC grid. It records
observations separately from predictions, preserving original units and numerical
provenance. Its operational context/horizon bounds and each adapter's narrower
limits apply. No holdout is required:

```sh
tsfm run examples/configurations/future-baseline.yaml --json
```

**Future accuracy is unavailable because future actuals are absent.** Zero
errors, invented actuals, historical scores or financial returns are not
substituted. Saved future forecasts cannot enter historical comparison or
finance reports. Importing later actuals and rescoring saved future forecasts is
deferred; old historical records remain readable without migration.

## Outputs and reproducibility

Completed guided runs show exact paths to forecast CSV, saved SVG chart,
machine-readable results/provenance and applicable evaluation summary. Future
records also retain consumed observations and an explicit unavailable-evaluation
object. Atomic publication and validation prevent incomplete work from appearing
completed. Cancellation/failure records remain truthful; retries are explicit.

Saved artifacts can be inspected and exported without loading model weights or
requesting an assistant explanation. Replace `RUN_ID` with a completed run's UUID:

```sh
tsfm experiments show RUN_ID --workspace results --json
tsfm export RUN_ID --workspace results --output forecasts.csv --json
```

CSV and JSON are available; Parquet requires its optional dependency. Export
refuses existing destinations. Preserve configuration, dataset identity, exact
checkpoint/files, preprocessing, runtime, source identity and recorded metric
semantics when reproducing or comparing work. A completed user run does not add
a maintainer inference check to the public catalog.

## Finance and limitations

The separate finance extension evaluates explicit hypothetical OHLC policies
from eligible saved historical median-task price forecasts. It requires declared
signal timing, execution prices, transaction costs and accounting assumptions.
Forecast error, calibration and post-cost decision value are different questions;
neither good point accuracy nor a software integration check establishes profit.
This release adds no trading policy or market result.

The numerical contract remains one regular univariate series with finite values
and a fixed UTC interval. There is no hidden imputation, resampling, clipping or
unit conversion. Panels, covariates, irregular calendars, live feeds, training,
fine-tuning, optimization, hosted inference and GPU/MPS certification are outside
this release. Optional runtime and model-weight licenses are separate from the
project's Apache-2.0 code license; TiRex retains its NXAI runtime notices.

Retained baseline and earlier model checks are platform-specific; current local
checks use macOS. Linux/macOS and native Windows 11 x64 on Python 3.11–3.13 are
targets, with the full current CI matrix and platform, terminal, worker,
credential and optional-runtime qualification still pending. TOTO 2 excludes
Python 3.11. No unperformed platform or real-model check is represented as a pass.

## Release notes

**0.3.0 is unreleased.** S1 adds onboarding and persistent nonsecret preferences.
S2 adds model-specific setup; S3 adds reviewed preparation, progress and saved
outputs; S4 adds separate future forecasting. S5 adds DeepSeek, local Ollama and
compatible endpoints. S6 adds TTM daily/weekly/native-horizon profiles and TOTO 2
with its explicit Python restriction. S7 prepares this static documentation.

Real pretrained qualification for new behavior, live assistant and native-vault
checks, native Windows verification and final release qualification remain open.
Current source package metadata remains {{SOURCE_VERSION}}; this development content is not
claimed to ship in the published 0.2.0 artifacts. Versioning, package/repository
publication and documentation deployment are separate actions.

## Citation

{{CITATION}}

TSFM Lab extends tooling originating in the separate Zeroshot_forecasting study.
The repositories and evaluation contracts remain separate. This documentation
adds no empirical result and does not equate TSFM Lab's OHLC execution with that
study's decision policy or findings.

## Documentation publication

This is a locally prepared publication artifact. Its HTTPS base URL is an
explicit **candidate**, not a registered host or verified deployment. Confirm the
chosen host and URL before deployment, then check public reachability and crawl
behavior separately. Source access remains private.

The artifact provides text-accessible content, stable section links, consistent
metadata, a sitemap and crawl rules without running the forecasting application.
An optional `llms.txt` supplies navigation. These measures do not guarantee
indexing or AI citations; see [Google's guidance on AI search features](https://developers.google.com/search/docs/appearance/ai-features).

# Future forecasting

Future mode predicts beyond the final observation of one regular series.
Historical evaluation remains available when observed outcomes and forecast
accuracy are needed. This guide describes the v0.3.0 development checkout;
the published v0.2.0 package has no future mode.

## Start a future forecast

In `/new`, choose the category, forecasting model and its options, then select
your local dataset and review its columns, units and cadence. Choose **Future
forecasting** at the mode question. Review the origin, which is the final observed
timestamp, and the requested context and horizon before confirming the run.
Assisted setup also requires a human mode selection; the assistant cannot change
it or start a forecast without ordinary review and confirmation.

The bundled baseline example needs no checkpoint or API key:

```sh
tsfm run examples/configurations/future-baseline.yaml
```

The same file is accepted by `/run`. Scripts keep the existing no-download
default. Guided model preparation retains its separate, exact download/run
confirmation; optional runtime installation is a separate action.

Future YAML uses schema 2 and `mode: future_forecast`. The `forecast` section has
`context_length`, `horizon`, optional `point_statistic`, and optional `origin`.
The origin defaults to `latest`; an explicit aware timestamp must equal the final
observation after conversion to UTC. The original input and the resolved origin
are both retained. Historical schema-1 configurations and Python APIs keep their
prior behavior.

```python
from tsfm_lab.future_config import load_future_config
from tsfm_lab.future_application import run_future_config

config = load_future_config("examples/configurations/future-baseline.yaml")
summary = run_future_config(config)
print(summary["run_id"])
```

`load_run_config` in `tsfm_lab.config` accepts either configuration type when a
caller needs explicit dispatch. Future Python entry points also include
`future_config_from_mapping` and `run_future_from_config`. Historical
`load_config`, `config_from_mapping`, `run_config` and `run_from_config` remain
historical-only.

## Context, horizon and output meaning

Only the final L observations enter a model with context length L. Exactly L
observations are sufficient; no historical holdout is required. The H predictions
begin one declared interval after the origin. Cadence remains fixed in UTC,
including across daylight-saving transitions. There is no silent resampling,
imputation, clipping, unit conversion, or earlier-origin truncation.

Future context is limited to 100,000 observations and horizon to 10,000, with
stricter selected-model bounds taking precedence. Invalid origins, timestamp
overflow, impossible settings, insufficient history, or incompatible consumed
values fail before inference. Recheck and confirm a plan after changing its data
or settings. A reviewed source file's changed bytes invalidate that review even
when the parsed values happen to agree.

The output retains original units and actual point semantics. A baseline remains
deterministic; a median adapter remains median; TTM remains native point with its
required context-only scaling. Choosing a task does not relabel the actual output.
See the [model catalog](model-catalog.md) for each variant's requirements.

## Saved records and reuse

Completed future records use manifest schema 3 with `kind: future_forecast` under
the ordinary workspace's `experiments/<run-id>/` directory. All eight artifacts
must validate before completion:

- `configuration.json` and `input.yaml`: resolved and original configuration.
- `predictions.csv`: exactly the requested future rows, without an actual column.
- `observations.csv`: the consumed context, ending at the origin.
- `result.json` and `provenance.json`: checked identities and reproducibility data.
- `evaluation.json`: accuracy unavailable, reason `future_actuals_absent`.
- `forecast.svg`: observed context followed by predictions, with a marked origin.

The chart may disclose sampling for display; full CSVs are retained. Checkpoint
and consumed-context identities bind the final numerical request. The source
file's recorded fingerprint describes the original check; saved context does not
reconstruct unsaved earlier source rows or prove historical data availability.

Use `/history`, `/show ID`, `/export ID PATH`, or their scriptable equivalents to
inspect/export checked artifacts without model weights or the original dataset.
CSV and JSON need no optional export engine; Parquet requires installed support.
Failed and cancelled records remain explicitly identifiable as future records.

Accuracy is unavailable because there are no future actual observations. No zero
score, historical score, or hypothetical financial return stands in for it.
Historical comparison and finance reports reject future records. Importing later
actuals and scoring them is deferred pending a matching/revision policy.
Optional assistant explanations receive metadata only: no observation values,
prediction values, row tables, value summaries, or source paths.

See the [approved contract](future-forecast-contract.md) for the exact schema,
identity, lifecycle and validation boundaries, and the
[release plan](release-v0.3.0-plan.md) for remaining qualification work.

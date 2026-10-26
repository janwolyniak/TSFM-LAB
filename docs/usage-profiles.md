# Usage profiles

Start `/new` and choose Generic, Finance, Energy, Temperature, or Weather.
The selected profile guides measurement questions, examples, review, history,
and result wording. All profiles evaluate one regular historical series against
observed targets, using the same forecasting models and scoring rules.

| Profile | Questions and interpretation |
| --- | --- |
| Generic | Existing numeric-series setup. No domain assumptions. |
| Finance | Close price, return, or another explicitly named target; quote/measurement units. Close-price observations must be positive. |
| Energy | Power, energy, or another named measurement, with explicit units and interval. Negative values remain valid. |
| Temperature | Explicit Celsius, Kelvin, or other declared units. Legitimate negative readings remain valid. |
| Weather | One variable from one station, units, and whether values are instantaneous, interval means/totals/maxima, or another described measurement. |

Profiles never resample, fill, clip, convert units, infer trading rules, or change
metrics. MAE/RMSE use the input target units; MASE remains dimensionless. Units
are declared once in `dataset.units`. No physical unit conversion or range
inference is performed. Declared units are preserved; profiles do not check physical ranges.

## Guided setup and examples

The persistent terminal asks for usage before model selection, then the dataset
and relevant measurement details. The screen-based terminal keeps its four
setup stages and includes equivalent profile controls. Both use the same strict
configuration boundary, review, execution confirmation, and cancellation rules.

Each domain offers an original 1,600-row hourly synthetic example, long enough
for all three TTM contexts. Temperature includes negative readings. A custom
dataset path is preserved when changing profiles, and switching from a fictional
example to your data requires your units and measurement declarations. Returning
to the same example preserves explicit edits. The original quick `/demo` keeps
its existing data, numerical settings, and profile-absent serialization.

For hourly Energy, Temperature, and Weather data, setup shows 24 observations
per day and 168 per week as seasonal examples. These are suggestions; the user
chooses baseline and MASE lags explicitly. Existing numeric answers are retained.
Model-specific restrictions remain in force, including TTM's exact context,
minute-to-hour cadence, and native-point task.

Run an explicit baseline example from the source checkout or source archive:

```sh
tsfm run examples/configurations/usage-generic.yaml --json
tsfm run examples/configurations/usage-finance.yaml --json
tsfm run examples/configurations/usage-energy.yaml --json
tsfm run examples/configurations/usage-temperature.yaml --json
tsfm run examples/configurations/usage-weather.yaml --json
```

These examples establish workflow behavior, not model quality or real-world
domain performance. The [fixture description](../examples/datasets/README.md)
gives their formulas, units, and fictional measurement assumptions.

## Configuration and saved results

Schema 1 gains an optional, strictly validated `workflow` block:

```yaml
workflow:
  schema_version: 1
  profile: energy
  options:
    measurement: power
dataset:
  # Other required dataset fields remain explicit.
  units: kW
```

| Profile | Exact options |
| --- | --- |
| `generic` / `temperature` | `{}` |
| `finance` | `target_kind: close_price`, `return`, or `other`; `other` also requires `measurement` text. |
| `energy` | `measurement: power`, `energy`, or `other`; `other` also requires `quantity` text. |
| `weather` | Nonempty `variable`, `station`, and `measurement_semantics`; allowed semantics are `instantaneous`, `interval_mean`, `interval_total`, `interval_max`, or `other`. The last also requires `description`. |

Non-generic profiles require declared units. Unknown keys/options fail clearly.
The Python runner accepts the same block through `run_experiment(...,
workflow=...)`; profile metadata is never passed to the forecasting adapter.

The resolved block is stored in the initial configuration and numerical result.
Reload validates both, including their agreement, without accessing original
data or model runtimes. It cannot validate unavailable historical observations
again; normal artifact checks remain the integrity boundary, not authentication
of a deliberately rewritten workspace.

Old configurations and records omit the block and display as Generic. They are
not rewritten. Profile-only changes preserve dataset fingerprints, numerical
predictions, scores, and controlled-comparison eligibility. Actual target,
units, context, task, and evaluation-policy differences remain authoritative.

## Finance report handoff

A saved Finance close-price result offers the existing OHLC command when it has
a median task, stride 1, and positive selected lead-1 actuals/forecasts. The
template names every required execution input; it supplies no trading costs,
capital, threshold, or latency defaults. Supply the original matching dataset
configuration and separate OHLC file. A saved relative `input.yaml` may need its
original source location, so do not substitute it blindly.

The manual finance command keeps its existing explicit `--price-target`, timing,
cost, price, alignment, and accounting checks. A profile label does not grant
execution eligibility or alter that policy. Forecast-only finance work remains
available, and native-point runs remain outside OHLC report eligibility.
See the [finance guide](finance-extension.md) and the matching synthetic
`finance-demo-ohlc.csv` example. Report creation is a separate explicit command.

No LLM connections, future forecasting without actuals, weather grids, live
feeds, multiple-series panels, or covariates are added in P5. Connections and
natural-language assistance follow the [release plan](next-release-plan.md).

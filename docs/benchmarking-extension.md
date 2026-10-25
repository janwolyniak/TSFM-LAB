# Broader-benchmarking study and proposed integration

Status: the bounded compatibility study was approved and executed on 25 October
2026. Product integration remains pending agreement. No `fev` dependency,
benchmark backend, panel/covariate interface, or probabilistic result schema has
been added to the production package. Its maintained
[evaluation semantics](evaluation.md) remain unchanged.

## Executed study

The [research companion](../research/fev_compatibility/README.md) contains portable
fixture scripts, exact runtime pins, [results](../research/fev_compatibility/results.json),
and [upstream identities](../research/fev_compatibility/upstream.json). The tested
wheel was `fev==0.10.0`, source tag `v0.10.0` at
`ec833b9865fac9a021e8ec9cc56b67bb4c515427`; five installed Python files matched
that commit byte for byte. All seven synthetic fixture groups passed.

| Fixture | Observed result |
| --- | --- |
| Current arithmetic | Both baselines matched all 16 TSFM case keys and predictions. Last-value MAE/RMSE/MASE were 2.5/2.915476/2.5; seasonal-naive scores were 1/1/1. |
| Changing history | Native fev MASE was 1.611111 versus fixed-scale TSFM 3.0. Its mean of window RMSEs was 6 versus pooled TSFM RMSE 6.218253. |
| Unequal series | Native equal-window MAE 2.5 differed from case-weighted 3.333333 and series-weighted 5.0. A short item disappeared from a later native window. |
| Undefined/nonfinite | Constant and insufficient-history native MASE could be NaN; TSFM retained null with a reason. Nonfinite forecasts were rejected, but a NaN actual was omitted by native scoring. |
| Overlap | Nine origin/target/lead cases retained five unique target timestamps. |
| Covariates | A declared past-only future measurement did not affect earlier inputs. Deliberately declaring it known-future exposed it through fev. |
| Quantiles | Native MQL 0.3 was twice ordinary mean pinball 0.15. fev sorted levels and rejected a missing level column, but accepted crossing quantiles. |

The prototype rejects incomplete item coverage, incorrect declared availability,
and crossing quantiles under its explicit policies. These are user-defined
research guards, not fev guarantees or implemented TSFM product behavior. In this
fixture, fev's four `num_forecasts` are item/window forecasts, not 16 horizon cases.

Execution used Python 3.13.6 on macOS arm64 in a new venv that reused existing
system libraries read-only. The five new, hash-checked wheels totaled 888,930
bytes. All active requirements in the 43-distribution study closure were checked.
An unrelated inherited `mlxtend`/NumPy requirement conflict made whole-environment
`pip check` fail; it was recorded rather than repaired. This was not a hermetic
or cross-platform environment check. Fixture execution used local synthetic
Parquet data with offline flags and an IP socket guard. No pretrained-model or
external-dataset download, paid compute, or performance claim was involved.

## Proposed reuse decision

Keep fev optional and investigate an explicit semantic bridge rather than silently
replacing the core scorer. Its documented [task and benchmark interfaces](https://autogluon.github.io/fev/latest/tutorials/03-tasks-and-benchmarks/)
and [dataset format](https://autogluon.github.io/fev/latest/tutorials/02-dataset-format/)
are useful conversion candidates; the executed study does not establish a
production integration or compatibility with every native feature.

Preserve TSFM's checkpoint policy, adapter lifecycle, forecast identities,
complete-case validation, and independent records. The upstream
[dependency declaration](https://github.com/autogluon/fev/blob/v0.10.0/pyproject.toml)
includes `datasets`, Pydantic, NumPy, and SciPy. Making fev mandatory would change
the current NumPy/pandas numerical-core dependency boundary. No such change is
proposed by this study.

## Semantic contract required before integration

The following differences are supported by the pinned execution and
[Task/window source](https://github.com/autogluon/fev/blob/v0.10.0/src/fev/task.py)
and [metric source](https://github.com/autogluon/fev/blob/v0.10.0/src/fev/metrics.py).
Moving documentation is explanatory; recheck behavior before changing the tested
version.

| Area | Required policy |
| --- | --- |
| Scaling | Preserve the fixed pre-evaluation MASE history. A native denominator calculated from each window's supplied context is a different metric policy. |
| Undefined values | Preserve `zero_scale` and `insufficient_scale_history`, together with valid/excluded counts. Do not introduce an epsilon or omit undefined series silently. |
| Nonfinite values | Keep strict input/output validation before lower-level scoring. Native Task forecast rejection and NaN-aware metric aggregation concern different layers. |
| Weighting | Agree series, window, target, and lead weights explicitly. A mean of window RMSEs is not pooled RMSE, and different eligible-series sets can change window weighting. |
| Alignment | Preserve every expected origin/target/lead key. No implicit inner join, deduplication, or dropped short items may change a controlled evaluation. |
| Covariates | Classify known-future, past-only, and static inputs, and specify actual availability times. A declaration or future timestamp does not prove a value was available at the origin. |
| Forecasts | Keep mean, median, deterministic output, and quantiles distinct. Do not fabricate uncertainty or silently change the requested point statistic. |
| Quantile metrics | Specify ordered levels, shape, crossing policy, units, aggregation, and normalization; fev's quantile-loss factor of two must be named when comparing scores. |

Panel weighting, covariate availability, probability representation, dependency
policy, and persisted-schema changes need owner agreement before product code.
New representations must receive a versioned contract; existing
`point_metrics_v1` and `forecast_case_micro_v1` records must retain their meanings.

## Verification ceiling and next decision

The scripts exercise two baselines, changing/truncated scaling history, unequal
window counts, zero/insufficient scales, NaN actuals, NaN/Inf forecasts, overlap,
declared covariates, and manually chosen 0.1/0.5/0.9 quantiles. They verify the
reported arithmetic and explicit prototype guards, not all possible invalid
inputs or all library capabilities.

No real publication-time metadata, multivariate target, probabilistic TSFM
adapter, accelerator, uncertainty interval, significance test, or forecasting
advantage was verified. The next decision is whether to design an optional
product bridge under the explicit policies above, revise its scope, or retain
the current engine. Completing this research study does not approve those
production interfaces or turn broader library capabilities into supported TSFM
features.

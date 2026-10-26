# Evaluation methodology

This is the implemented numerical-core contract.

The core evaluates forecasts of the imported numeric target in its stated units
and sampling interval. A horizon H means H observations ahead. It does not imply
hours, returns, positions, or financial performance. The first population is all
complete rolling windows of one regular series inside an explicit evaluation
interval.

## Chronology and window construction

Let ordered observations be `(t[i], y[i])`, for `i = 0, ..., N-1`. Let L be context
length, H horizon, S stride, and `[a,b)` the evaluation target interval expressed
as positions. The last boundary b may equal N.

```text
first target position: o = a, a+S, a+2S, ... with o+H <= b
historical values:     y[o-L:o]
forecast origin:       t[o-1]
target timestamps:     t[o:o+H]
ground truth:          y[o:o+H]  (evaluator only)
```

Require `L > 0`, `H > 0`, `S > 0`, `L <= a < b <= N`, and `a+H <= b`.
The number of windows is `1 + floor((b-a-H)/S)`. Never move an invalid requested
start forward silently. Do not emit partial horizons at the end. Evaluation
timestamps must resolve exactly to the regular grid under the
[dataset contract](contracts.md#dataset).

For a small example, let `y = [1,2,3,4,5,6,7,8]`, L=3, H=2, S=1, a=4, b=8:

| First target o | Historical values | Origin position | Target positions |
| --- | --- | --- | --- |
| 4 | `[2,3,4]` | 3 | 4, 5 |
| 5 | `[3,4,5]` | 4 | 5, 6 |
| 6 | `[4,5,6]` | 5 | 6, 7 |

Later rolling origins may consume earlier evaluation observations once those
observations are historical. This is a sequential evaluation with observed
updates, not one forecast of the whole held-out interval from its first origin.
Setting a single complete window provides a fixed-origin holdout evaluation.

Forecast creation must never access its own target values. No full-series
normalization, interpolation through future observations, automatic seasonal
selection, or tuning against the final holdout is allowed. Future preprocessing
must fit on explicitly permitted history and record that history. A zero-shot
checkpoint can still have pretraining overlap with the test dataset; record known
overlap and do not equate correct rolling evaluation with proven pretraining
independence.

## Baselines

Last-value forecasts repeat `y[o-1]` for every lead.

For a seasonal period m and zero-based future step j, seasonal-naive predicts
`y[o-m+(j % m)]`. Require `m > 0` and `L >= m`; horizons longer than a season
repeat the last observed seasonal cycle. The adapter uses only its detached
historical request, not the full series.

Seasonality is a user-specified lag in observations. It is not inferred from the
whole dataset. These baselines verify the pipeline and provide reference scores;
their presence does not establish that a pretrained model will outperform them.

## Point metrics

For every origin k and one-based lead h, define error
`e[k,h] = forecast[k,h] - actual[k,h]`. For W complete windows:

```text
MAE  = sum(abs(e[k,h])) / (W*H)
RMSE = sqrt(sum(e[k,h]**2) / (W*H))
```

Both are in target units. RMSE is the square root of the pooled squared errors,
not the mean of per-window RMSEs. Per-lead metrics apply the same formulas across
origins at a fixed h.

MASE uses a fixed scale from all available observations strictly before the first
evaluation target. With T=a and seasonal lag m:

```text
scale = sum(abs(y[i] - y[i-m]) for i in range(m,T)) / (T-m)
MASE  = MAE / scale
```

This adopts the standard training-history scaled-error definition described by
the authors of [Forecasting Principles and Practice](https://otexts.com/fpp3/accuracy.html#scaled-errors).
TSFM Lab fixes that history before evaluation so later origins do not change the
denominator. Record the scale, its timestamp range, lag, observation count, and
definition version. It may use more history than the model's L-point context;
that distinction belongs in the result.

If T<=m, report MASE as undefined with `insufficient_scale_history`. If the scale
is zero, report `zero_scale`, including for a perfect forecast. Do not add an
epsilon, return infinity, or silently omit the metric. A nonfinite numerical
calculation is an explicit error. Later panel aggregation must report the number
of valid and excluded series before averaging any defined scales.
If MASE is not requested, no scale arithmetic is performed: scale metadata is
`null` with reason `not_requested` and a zero scale count.

MASE below one means a smaller test MAE than the seasonal-naive one-step error
on the scaling history. It does not guarantee a win against a seasonal-naive
forecast evaluated over the same test horizons. Report both baseline test scores
when making that comparison.

## Aggregation and overlap

`forecast_case_micro_v1` weights each `(series_id, origin, target_timestamp, lead)`
case equally. With S<H, several forecasts can target the same timestamp at
different origins/leads; these remain separate cases. Do not deduplicate them or
treat their errors as statistically independent observations for significance
tests. Record W, W*H, the number of unique target timestamps, and stride.

The initial release has one series and reports overall and per-lead MAE, RMSE,
and MASE. Cross-series macro/micro choices, probabilistic metrics, confidence
intervals, and significance testing need separate specifications. No financial
utility claims follow from these point scores.

## Alignment and fair comparison

Construct a complete expected key set from the evaluation specification before
scoring. Verify each output's model/request identity, origin, exact target
timestamps, finite values, and shape. Any missing, duplicate, or unexpected key
fails the run; there is no inner join that silently drops difficult forecasts.

A controlled comparison requires identical canonical dataset content and import
semantics, target/units, evaluation keys, L, H, S, preprocessing, scaling history,
metric versions, aggregation, and point statistic. Check provenance independently
of matching file names. Here point statistic means the requested forecasting task;
the default `median` task admits median and deterministic outputs. The explicit
`native_point` task compares declared point outputs while retaining every actual
label. It uses `controlled_native_point_v1`, per-run `output_statistics`, and
`mixed_point_statistics` to disclose different statistical targets. TTM retains
`native_point`, the other curated families retain `median`, and baselines retain
`deterministic`. A mixed native-point comparison makes no common mean/median
claim. Tasks cannot be mixed; legacy median comparisons keep their old policy
and representation. Different context lengths fail controlled comparison.
The numerical scores and chronological case-alignment rules are unchanged.
Record runtime and device differences beside timing comparisons. Incomplete runs are ineligible.

## Required regression evidence

The following tests precede a pretrained adapter:

- Hand-calculated trend and seasonal fixtures reproduce expected predictions,
  pooled metrics, and per-lead metrics, including H>m.
- Changing only observations at or after an origin's first target leaves that
  origin's forecast unchanged. Later rolling forecasts may change legitimately.
- A recording adapter sees no future values and cannot mutate the dataset through
  a shared historical buffer.
- Boundary origins, S<H, S>H, and an incomplete final horizon produce the
  specified window count and alignment.
- Invalid input grids, nonfinite observations/forecasts, shifted outputs, and
  missing or duplicate requests fail before a completed result is emitted.
- Constant scaling history and insufficient scaling history produce explicit
  undefined MASE statuses.
- Repeated runs match deterministic forecasts, metric values, and method metadata.

The numerical test suite covers these cases. See the
[verification record](verification.md) for executed checks and their limits, and
the [roadmap](roadmap.md) for implementation order.

## Usage labels and numerical identity

Usage profiles guide measurement terminology, units, examples, and reporting.
They are validated workflow metadata, outside numerical comparison keys. On the
same valid input with unchanged units/task/settings, changing only a profile
preserves individual predictions, pooled/per-lead scores, MASE scale, canonical
dataset fingerprint, and controlled-comparison eligibility. Domain checks can
refuse an invalid declaration such as nonpositive close-price observations;
they never transform a value to make it admissible. Existing measurement-unit
and task differences still prevent controlled comparison.

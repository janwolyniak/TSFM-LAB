# Synthetic seasonal example

`seasonal.csv` is original synthetic data distributed under the repository's
Apache-2.0 license. It contains 32 hourly UTC observations starting at
2026-01-01 00:00 UTC. Its target `demand` is in `synthetic_units`:

```text
y[i] = 10 + floor(i/4) + [0, 5, -3, 2][i mod 4], i = 0,...,31
```

The example uses context 8, horizon 4, stride 4, targets at positions 16 through
31, and MASE lag 4. It produces four complete windows and 16 forecast cases.
The pre-evaluation MASE scale is 1. Seasonal-naive has MAE/RMSE/MASE all equal
to 1; last-value has MAE=MASE=2.5 and RMSE=sqrt(8.5).

These are constructed arithmetic checks of the pipeline. They are not empirical
results about pretrained models or real demand forecasts.

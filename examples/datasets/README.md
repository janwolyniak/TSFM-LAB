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

## Synthetic OHLC example

`ohlc.csv` is an original, hand-constructed fixture under the same Apache-2.0
license. Its twelve closing prices are
`[100,105,102,108,103,110,104,112,105,113,106,114]`; opens are
`[99,103,104,105,107,106,109,108,111,109,112,110]`.
For every row, `high=max(open,close)+2` and `low=min(open,close)-1`.
Each fictional daily session opens at 09:00 UTC and closes at 16:00 UTC, including
weekends. This artificial calendar provides an explicit 17-hour observation gap;
it represents no real venue, asset, trading calendar, or market data source.

The [finance configuration](../configurations/finance-seasonal-naive.yaml) uses
context 4, seasonal-naive lag 2, horizon 1, stride 1, eight evaluation bars, and
`toy_quote_per_asset` units. It exercises the separate
[OHLC timing and accounting contract](../../docs/finance-extension.md).
Constructed signals and returns demonstrate implementation behavior and provide
no evidence of predictive ability or investment returns.

## Longer synthetic TTM example

`ttm-seasonal.csv` is original synthetic data under Apache-2.0, with 1,600
hourly UTC observations starting at 2026-01-01 00:00 UTC. It is duplicated as a
packaged resource for guided setup and supports all three exact-context profiles.
Values are rounded to eight decimal places:

```text
y[i] = 50 + 0.01*i + 5*sin(2*pi*i/24) + 2*cos(2*pi*i/168)
```

The [512-profile example](../configurations/ttm-512-96.yaml) scores positions
512–527 in four windows, with horizon/stride 4 and MASE lag 24. It checks the
workflow only; it provides no model ranking or domain-performance evidence.

## Domain usage examples

The four `*-demo.csv` files are original synthetic data under Apache-2.0, with
1,600 hourly UTC observations starting at 2026-01-01 00:00 UTC. Identical copies
are bundled for terminal setup. Each file has `timestamp` and one target column.
Values are rounded to eight decimal places, with `i=0,...,1599`:

| File / target | Declared units | Construction |
| --- | --- | --- |
| `finance-demo.csv` / `close` | `toy_quote_per_asset` | `100 + 0.02*i + 3*sin(2*pi*i/24)` |
| `energy-demo.csv` / `power` | `kW` | `50 + 0.005*i + 10*sin(2*pi*i/24) + 4*cos(2*pi*i/168)` |
| `temperature-demo.csv` / `temperature` | `degC` | `-3 + 6*sin(2*pi*i/24) + 2*cos(2*pi*i/168)` |
| `weather-demo.csv` / `wind_speed` | `m/s` | `5 + 1.5*sin(2*pi*i/24) + 0.4*cos(2*pi*i/168)` |

The weather example declares a fictional `synthetic-station` and instantaneous
wind speed. These files represent no measured station, energy installation,
financial asset, or empirical domain result. Temperature includes negative
values, which must survive import, forecasts, and exports unchanged.

The [usage configs](../../docs/usage-profiles.md) use domain context 48, targets
48–63, MASE lag 24, and 16 forecast cases. Finance uses seasonal-naive lag 24 with
horizon/stride 1; the other domain examples use last-value with horizon/stride 4.
Generic retains the original short seasonal fixture/settings.

`finance-demo-ohlc.csv` is separate original synthetic execution data. Its close
column exactly matches `finance-demo.csv` on the full grid. Row i opens 30 minutes
before its close, with `open=close-0.3+0.1*sin(i/7)`, `high=max(open,close)+0.5`,
and `low=min(open,close)-0.5`, all rounded to eight decimal places. Its explicit
30-minute interbar gap is fictional. Users must still supply every timing/cost
assumption; the fixture implies no actual exchange, fill model, or trading rule.

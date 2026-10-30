# Finance: saved forecasts and explicit OHLC execution

The first finance module evaluates completed forecast records under policy
`ohlc_next_open_to_target_close_v1`. It imports the original price series and a
separate OHLC file, checks their identities and timing, and builds a hypothetical
cash/share ledger. It does not construct a model, load weights, download prices,
or place orders. Existing price-forecast scores retain their meaning.

## Run the synthetic example

From a source checkout or unpacked source distribution with the CLI installed,
create the supplied stride-1 baseline experiment:

```sh
tsfm run examples/configurations/finance-seasonal-naive.yaml --json
```

Use its returned UUID in place of `RUN_ID`. The following cost and threshold
values are explicit demonstration assumptions, not a venue fee schedule or a
recommended trading rule:

```sh
tsfm finance backtest RUN_ID \
  --workspace results \
  --source-config examples/configurations/finance-seasonal-naive.yaml \
  --ohlc examples/datasets/ohlc.csv \
  --price-target \
  --initial-capital 1000 \
  --threshold-bps 100 \
  --fee-bps 5 \
  --slippage-bps 2 \
  --short-financing-bps 3 \
  --signal-latency-seconds 60 \
  --output results/finance/report.json \
  --json
```

The forecast command creates the parent experiment. The finance command consumes
that saved run; `--source-config` supplies only the original dataset import and
does not execute the configuration's model. The original dataset's canonical
identity must match the parent run, including its import semantics and units.
The supplied example is constructed data for checking accounting and integration.

Show a checked report or export its capital paths:

```sh
tsfm finance show results/finance/report.json --json
tsfm finance export results/finance/report.json \
  --output results/finance/equity.csv --json
```

The CSV has one row per selected close, with the signal, strategy equity/returns,
transaction costs, and both benchmark equities. `--json` selects machine-readable
command output. Report and export destinations must be new paths; existing files
and symlinks are refused. Report inspection and
export recalculate the embedded cases without opening the original dataset,
OHLC file, workspace, or model checkpoint.

## Required inputs and identity

The parent must be a completed experiment accepted by the existing
[artifact validator](../src/tsfm_lab/artifact_validation.py). Its forecast stride
must be 1. Every successful origin contributes its lead-1 forecast, even when the
parent horizon is longer; no origin is silently dropped. All cases use the saved
median or deterministic point statistic and the parent series identity.

`--price-target` explicitly declares that the imported target is a price. The
original dataset must declare quote units per asset unit; `unspecified` is
rejected. Origin prices, actual closes, selected forecasts, and all OHLC prices
must be finite and strictly positive. Negative forecasts are rejected, not clipped.

The OHLC CSV has exactly these six columns; their order may vary:

| Column | Meaning |
| --- | --- |
| `open_timestamp` | Actual opening time of the bar. |
| `close_timestamp` | Actual closing time, matching the forecast target grid. |
| `open` | Entry reference price. |
| `high` | Highest observed price during the bar. |
| `low` | Lowest observed price during the bar. |
| `close` | Closing reference price and original forecast target. |

Use explicit timezone-aware timestamps, preferably UTC. Naive OHLC timestamps
require `--ohlc-timezone TIMEZONE`; this is separate from the source
configuration's dataset timezone. Each bar must open strictly before it closes;
bars cannot overlap. Both timestamp columns must be
unique and ascending, and closing times must follow the declared fixed frequency.
Prices must satisfy `low <= open/close <= high`. Import never sorts, fills, or
infers opening times.

The entire OHLC closing grid and every close value must exactly match the
original dataset, including history outside the evaluated interval. Series,
frequency, and units must agree. An evaluation-only slice, revised source series,
or independently normalized price table does not satisfy this identity check.
See the [dataset contract](contracts.md#dataset) for canonical source identity.

## Signal, timing, and holding period

For origin close `C0`, the saved lead-1 target-close forecast `F` gives the signal
basis `F / C0 - 1`. With threshold `tau = threshold_bps / 10000`:

- A strict excess above `tau` selects long (`+1`).
- A strict excess below `-tau` selects short (`-1`).
- Threshold ties and values inside the band select flat (`0`).

Comparison uses policy `decimal_shortest_float64_v1`: imported float64 prices and
the threshold are converted through their shortest decimal strings, then compared
in price-difference basis points. For example, `F=101`, `C0=100`, and a 100-bps
threshold is a flat tie. This avoids a false signal caused by dividing binary
floating-point numbers and subtracting one. No broad tolerance widens the band.
Choose the fixed threshold before evaluating outcomes; the module does not tune it.

Enter at the next bar's explicit open `O` and liquidate at that same bar's close
`C`, which is the lead-1 target. Entry is allowed only when:

```text
open_timestamp > origin_close_timestamp + signal_latency
```

Equality fails, including with zero latency. Contiguous bars whose next open and
previous close share a timestamp are therefore outside this execution contract.
Do not fabricate a timestamp gap to admit them. The synthetic OHLC example
deliberately includes a declared opening gap; this does not establish realistic
latency or fills for a live market.

The future opening price never affects the signal. Its realized effect enters
position sizing and P&L. A close-to-close forecast-implied return is different
from the realized holding return `C / O - 1`; the report retains both bases.

Every nonflat bar has a separate entry and exit, including consecutive bars with
the same direction. The strategy holds cash between bars and does not net a new
signal against a carried position. No overnight or interbar price change earns
strategy P&L while flat.

## Cash, units, and costs

Every report requires initial capital, threshold, one-way fees, one-way slippage,
short financing, and latency. Explicit zero values are allowed; none is inferred.
Values must be finite and nonnegative, initial capital must be positive, and
combined one-way fee/slippage must be below 10,000 bps.

Let `E` be pre-entry equity, `z` the selected position, `c` the combined one-way
fee/slippage fraction, and `b` the per-held-bar short-financing fraction:

```text
q           = z * E / (O * (1 + c))
entry_cash  = E - q * O - c * abs(q) * O
next_equity = entry_cash + q * C - c * abs(q) * C
              - b * abs(q) * O * indicator(z == -1)
```

Here `q` is signed asset units. Sizing targets unit gross exposure against equity
after entry costs. Prices can subsequently change that exposure before the close;
there is no intrabar rebalancing. Entry and exit fees/slippage apply separately to
their actual absolute notionals, with each component recorded in the ledger.
Slippage is a proportional cash charge against the supplied reference price.
Short financing is charged on entry short notional for each held bar, not on
ending equity and not as an annual rate. Flat bars incur no trading costs.

Before completing a short bar, the ledger checks marked equity at the observed
bar high using its post-entry cash and signed shares. Nonpositive marked equity
is rejected, even if the closing price would restore a profit. Nonpositive ending
equity after exit costs/financing, nonfinite arithmetic, and unrepresentable
nonzero positions are also rejected. No partial or clipped capital path is saved
as a completed result. This check is not a broker margin or liquidation model.

At a constant price, with `E=100` and combined one-way costs `c=0.01`, either
direction ends at approximately `98.01980198` before short financing. With zero
costs and constant equity, each traded bar contributes turnover 2; two bars
contribute turnover 4 whether their directions repeat or flip.

## Benchmarks, diagnostics, and metrics

The report includes cash at zero interest and costed buy-and-hold. Buy-and-hold
buys once at the first selected open, marks those shares at each selected close,
and sells at the last selected close. It pays only that entry and final exit. It
retains exposure across interbar gaps, unlike the strategy. Both use the same
initial capital, quote units, calendar endpoints, and transaction-cost rates.
For a single bar, buy-and-hold and an all-long strategy have the same ledger.

Each capital path reports:

| Quantity | Definition |
| --- | --- |
| Cumulative return | `final_capital / initial_capital - 1`. |
| Period net return | Change in period-end equity divided by prior equity, with costs included. |
| Maximum drawdown | Largest loss from a preceding period-end peak, including initial capital as the first peak. |
| Total turnover | All entry/exit absolute traded notional divided by initial capital. |
| Row turnover | That row's traded notional divided by pre-entry equity; this differs from total turnover's denominator. |
| Volatility | Unannualized sample standard deviation of period net returns, `ddof=1`. |

Volatility is `null` with reason `insufficient_returns` for fewer than two returns.
A constant return path with at least two observations has zero volatility.
The report records fees, slippage, financing, cash, shares, notionals, and gross
and net P&L separately. Drawdown is measured at period ends, not at every intrabar
high/low. No annualization, Sharpe ratio, or risk-free curve is supplied.

Return diagnostics compare the saved forecast-implied simple return
`(F-C0)/C0` with the actual close-to-close return `(C-C0)/C0`. They report MAE,
RMSE, and three-category directional accuracy using exact negative/zero/positive
signs, with category counts and the full denominator. A zero forecast move is
correct only for an actual zero move. These diagnostics do not use the trading
threshold and do not replace the parent's price-unit forecast scores. A
transformed point forecast is not an expected-return or calibrated probability
estimate.

## Saved reports and verification limits

Reports are separate strict JSON artifacts under the finance policy and schema
version. They retain the parent run ID, copied parent result and artifact
identities, original-data and OHLC fingerprints, explicit rules, implementation
and environment provenance, raw selected cases, diagnostics, and capital paths.
The source experiment remains unchanged.

Report schema 1 embeds the five original parent-artifact identities. Creation
also accepts historical guided schema-2 experiments, validating all seven parent
artifacts before selecting those five identities. The guided chart and provenance
remain in the source experiment; their corruption prevents report creation.

Saving validates and recalculates the report, adds its SHA-256 content fingerprint,
and atomically publishes it without replacing an existing destination. Reload
rejects duplicate JSON keys, nonfinite values, unsupported schemas/policies,
inconsistent metadata, checksum failures, and arithmetic inconsistent with the
embedded cases. Selected lead-1 price errors must also reproduce the copied
parent's recorded per-lead scores. Recalculated floating-point values use relative
tolerance `1e-12` and zero absolute tolerance. The JSON limit is 64 MiB.
Reload and export need no model runtime or original source files. Stored input
identities describe what was checked at
creation; detached reload recalculates the embedded cases rather than rereading
the complete source series or parent predictions file. A copied predictions-file
hash alone cannot authenticate the selected rows without that file. These are
integrity and internal-consistency checks, not external data authentication.

This is a hypothetical reference-price simulation. It assumes fractional units
and supplied OHLC prices, without liquidity, order-book depth, partial fills,
market impact beyond the explicit slippage charge, taxes, dividends, corporate
actions, borrow availability, or venue-specific margin/funding rules. All input
price columns must use a consistent adjustment basis. The short high check cannot
establish realistic borrowing, margin, or liquidation behavior; real short sales
have additional obligations. See the [SEC short-selling explanation](https://www.sec.gov/investor/pubs/regsho.htm).

No forecast or strategy superiority follows from passing synthetic checks.
Threshold selection, dataset reuse, checkpoint pretraining overlap, and supplied
execution/cost assumptions limit any empirical interpretation. Backtested results
are hypothetical performance; see the [SEC performance guidance](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-47).
See [evaluation methodology](evaluation.md) for the separate forecasting contract.

## Finance workflow hints

The optional Finance usage profile records whether the target is a close price,
return, or another named measurement. A close-price profile requires positive
observations and explicit quote units. After a median-task, stride-1 result with
positive selected lead-1 actuals and forecasts, terminal result views offer the
existing OHLC command with every required input left explicit.

This offer changes no finance calculation or manual eligibility rule. The
`--price-target` declaration, separate aligned OHLC data, original dataset
identity, positive prices, timing, threshold, costs, capital, and accounting
checks remain required for every report. Generic and older runs retain the same
manual route; native-point runs remain ineligible. Forecast-only return/other
finance data is supported without inferred trading assumptions.

The [profile finance example](../examples/configurations/usage-finance.yaml)
uses an hourly synthetic close series and a lag-24 baseline. Its matching
[OHLC fixture](../examples/datasets/finance-demo-ohlc.csv) is fictional: each bar
opens 30 minutes before its close, leaving an explicit 30-minute gap after the
previous close. Use it only to exercise the contract, with the explicit inputs
from the command example above and the profile example as `--source-config`.
It represents no trading venue, realistic execution, or predictive result.

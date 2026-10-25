# Proposed first finance extension

Status: proposal pending owner agreement. This document specifies a possible next
stage; it does not add finance commands, reporting, or a trading engine.

The owner chose to agree trading rules before finance coding. The three pending
decisions are timing, permitted positions, and explicit cost/risk inputs. The
following is the concrete first-version proposal, not an approved strategy.

## Proposed first trading contract

- The forecast target is explicitly declared to be a positive price. Import the
  original dataset and require its canonical identity to match the saved run.
- Select lead 2 from a run with horizon at least 2 and stride 1. At origin `t`,
  form the signal from `forecast_price[t+2] / observed_price[t] - 1`. The future
  entry price must not affect that signal.
- Enter at the input price at `t+1` and exit or rebalance at `t+2`. Each holding
  interval spans one observation; consecutive intervals share a boundary but do
  not overlap. Require complete coverage and reject any different timing grid.
  These are idealized fills at the supplied observation prices, with a full
  observation of delay; they do not assert that an origin close was executable.
- Use a fixed, explicitly supplied threshold in basis points: a strict excess
  gives a long signal, a strict negative excess gives a short signal when
  permitted, and ties stay flat. Never choose thresholds on evaluated outcomes.
- Choose unit long/flat/short target exposures or long/flat only. No target has
  gross exposure above one. Mark-to-market drift is recorded; this is not a
  continuous intraperiod margin or liquidation model.
- Require initial capital, one-way fee basis points, one-way slippage basis
  points, and a per-observation short-financing rate for every report. A zero
  value must be explicitly supplied; there is no assumed exchange fee schedule.
- Account for actual cash and asset units. At each rebalance, proportional costs
  apply to the absolute traded notional. Solve for target units against equity
  after those costs, rather than charging only changes in a direction label.
  Charge short financing on the interval's entry short notional. Liquidate the
  final position and include its costs in the final period's return.
- Reject a nonpositive marked equity path rather than silently clipping losses
  or presenting a partial path as completed. This discrete model cannot detect
  an unobserved intraperiod margin event.
- Include cash at zero interest and a costed buy-and-hold position over exactly
  the same observation interval. Report cumulative return, period-end equity
  and maximum drawdown, traded-notional turnover, and sample volatility of net
  interval returns. Volatility is undefined with fewer than two returns. Calendar
  annualization, Sharpe, taxes, and a separate risk-free curve remain outside the
  first contract.

Acceptance must cover independent self-financing cash/unit calculations,
initial/final costs, a constant-price cost-free long-to-short turnover of two,
zero trading costs for a flat path, threshold ties, short financing, future-price
perturbations, benchmark alignment, and insufficient-volatility-history status.
The existing forecast metrics and artifacts keep their original meaning.

## First slice: saved-forecast diagnostics

Start with return and direction diagnostics computed from completed, validated
forecast records. Keep model execution and the generic price-unit scores
unchanged. Loading a financial diagnostic must require neither model weights nor
an inference runtime. The existing [evaluation contract](evaluation.md) and
[artifact validation](../src/tsfm_lab/artifact_validation.py) remain authoritative.

The proposed inputs are:

- One completed experiment loaded through the existing checked experiment store.
- An explicit declaration that its target represents positive prices, with the
  recorded series identity and target units. Do not infer this from a file name.
- Explicitly supplied original price data, imported with the run's recorded
  column, timezone, frequency, and format rules. Its canonical dataset fingerprint
  must match the saved run before extracting origin prices. A separate price
  table with unverified provenance is outside this first slice.
- One selected lead `h`, with `1 <= h <= horizon`; recommend `h = 1` initially.
  Keep every `(series_id, origin, target_timestamp, lead)` case for that lead.
- A selected return convention: recommend simple returns by default, with an
  explicit log-return option. Both conventions use the same origin-price basis.

For each case, let `P0` be the observed price at the forecast origin, `P1` the
actual price at the selected target timestamp, and `F1` its saved price forecast:

| Convention | Forecast-implied return | Actual return |
| --- | --- | --- |
| Simple | `F1 / P0 - 1` | `P1 / P0 - 1` |
| Log | `log(F1 / P0)` | `log(P1 / P0)` |

All three prices must be finite and strictly positive. Reject invalid values,
missing origins, duplicate keys, mismatched actuals, and incomplete coverage.
Do not clip negative forecasts, fill missing prices, silently reorder inputs, or
derive an origin price from a future target. A target-to-target return is a
different diagnostic and must not be substituted for an origin-to-target return.

Report MAE/RMSE of the implied returns, their convention and horizon, and
three-category directional accuracy. The recommended zero-move rule uses exact
signs `-1`, `0`, and `1`; a zero forecast is correct only for an actual zero move.
Report the count of each actual/forecast category and the full denominator.
Keep existing price-unit metrics beside these diagnostics. A transformed price
forecast is an implied return forecast; do not label it an expected return or a
new probabilistic output. Overlapping target periods are not independent samples.

Persist the parent run ID, model/checkpoint identity, imported price-data
fingerprint, selected lead, return/zero-move conventions, diagnostic version,
case counts, and derived values in a separate derived report. Its exact file or
command interface remains part of the implementation proposal, not an extension
of the current generic forecast schema.

## Acceptance before calling the first slice complete

- Independently check `P0=100`, `F1=110`, `P1=105`: simple returns are `0.10` and
  `0.05`, with absolute error `0.05`; log error is `log(110/105)`.
- Check positive, negative, and zero moves, including a flat forecast against a
  nonzero actual move and a nonzero forecast against a flat actual.
- Verify exact selected-lead coverage, timestamp alignment, units, and dataset
  identity. Reject missing/duplicate origins and a changed original dataset.
- Verify that changing observations beyond an earlier case's target cannot alter
  that earlier diagnostic. A changed actual at its target must change its error,
  while leaving the stored forecast unchanged.
- Preserve deterministic/median forecast provenance and original-unit scores;
  reject corrupted parent artifacts through the existing store checks.
- Reload the derived report without accessing weights, downloading data, or
  rerunning inference. No profitability claim follows from passing these tests.

## Trading requires a separate owner decision

Approve the diagnostic scope and conventions above before implementation. Before
adding an executable strategy, agree each of the following explicitly:

| Decision | Proposed simple starting point or required clarification |
| --- | --- |
| Signal | One fixed threshold, chosen before evaluation; decide whether its basis is the origin price or forecasted entry/exit prices. |
| Timing and fills | Specify forecast availability, execution delay, entry/exit prices, and required price columns; do not assume an origin close remains executable after inference. |
| Horizon and overlap | One selected lead and a written holding/rebalancing rule; reject ambiguous overlapping trades. |
| Positions and costs | Unit long/flat/short exposure without leverage; explicit fees, slippage, borrowing/funding assumptions, turnover, and initial/final liquidation. |
| Capital and risk | Specify compounding, loss handling, market calendar, annualization, and any risk-free return. |
| Benchmarks | Cash and buy-and-hold under the same price, capital, and cost conventions. |

Trading acceptance must additionally check independent capital-path arithmetic,
zero costs for a flat position, two units of turnover on a long-to-short flip,
causal signal timing, terminal liquidation, and explicitly undefined Sharpe for
zero variance. Such work belongs in a finance module consuming generic forecasts;
it must not change the meaning of forecasting accuracy or imply realistic fills
from diagnostic origin prices alone.

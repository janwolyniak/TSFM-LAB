"""Derived finance reports from saved forecasts; no inference or trading API."""

from pathlib import Path
from typing import Annotated

import typer

from tsfm_lab.application import load_dataset
from tsfm_lab.cli_support import emit, guard, table
from tsfm_lab.config import ConfigError, load_config
from tsfm_lab.finance.backtest import backtest_run
from tsfm_lab.finance.contracts import BacktestRules, FinanceError
from tsfm_lab.finance.exports import export_equity
from tsfm_lab.finance.ohlc import load_ohlc_csv
from tsfm_lab.finance.reports import load_report, save_report

finance = typer.Typer(
    help="Inspect hypothetical price strategies from saved forecasts.",
    no_args_is_help=True,
)
Output = Annotated[Path, typer.Option("--output", "-o")]
Json = Annotated[bool, typer.Option("--json")]


def _show(report):
    evaluation = report["evaluation"]
    rows = []
    for name, path in (
        ("Strategy", evaluation["strategy"]),
        ("Cash", evaluation["benchmarks"]["cash"]),
        ("Buy and hold", evaluation["benchmarks"]["buy_and_hold"]),
    ):
        metrics = path["metrics"]
        volatility = metrics["volatility_unannualized"]
        rows.append(
            [
                name,
                f"{metrics['cumulative_return']:.2%}",
                f"{metrics['max_drawdown']:.2%}",
                f"{metrics['turnover_initial_capital']:.6g}",
                (
                    f"{volatility['value']:.6g}"
                    if volatility["value"] is not None
                    else "undefined"
                ),
            ]
        )
    table(
        "Hypothetical OHLC backtest",
        ["Path", "Return", "Max drawdown", "Turnover", "Volatility"],
        rows,
    )
    typer.echo(f"{evaluation['cases']} bars; report {report['report_id']}")


@finance.command("backtest")
@guard
def backtest(
    run_id: str,
    output: Output,
    source_config: Annotated[Path, typer.Option("--source-config")],
    ohlc: Annotated[Path, typer.Option("--ohlc")],
    initial_capital: Annotated[float, typer.Option("--initial-capital")],
    threshold_bps: Annotated[float, typer.Option("--threshold-bps")],
    fee_bps: Annotated[float, typer.Option("--fee-bps")],
    slippage_bps: Annotated[float, typer.Option("--slippage-bps")],
    short_financing_bps: Annotated[float, typer.Option("--short-financing-bps")],
    signal_latency_seconds: Annotated[float, typer.Option("--signal-latency-seconds")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")] = Path("results"),
    ohlc_timezone: Annotated[str | None, typer.Option("--ohlc-timezone")] = None,
    price_target: Annotated[bool, typer.Option("--price-target")] = False,
    json_output: Json = False,
):
    """Evaluate lead-1, stride-1 close forecasts using explicit next-bar OHLC."""
    if not price_target:
        raise ConfigError("Declare a positive price target with --price-target")
    try:
        rules = BacktestRules(
            initial_capital,
            threshold_bps,
            fee_bps,
            slippage_bps,
            short_financing_bps,
            signal_latency_seconds,
        )
        dataset = load_dataset(load_config(source_config).dataset)
        bars = load_ohlc_csv(
            ohlc,
            frequency=dataset.frequency,
            timezone=ohlc_timezone,
            units=dataset.units,
            series_id=dataset.series_id,
        )
        report = backtest_run(
            run_id, workspace, dataset, bars, rules, target_is_price=True
        )
        saved = save_report(report, output)
    except (FinanceError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    if json_output:
        emit(report)
    else:
        _show(report)
        typer.echo(f"Saved to {saved['path']}")


@finance.command("show")
@guard
def show(report: Path, json_output: Json = False):
    """Verify a self-contained report and recompute its financial results."""
    try:
        value = load_report(report)
    except FinanceError as exc:
        raise ConfigError(str(exc)) from exc
    emit(value) if json_output else _show(value)


@finance.command("export")
@guard
def export(report: Path, output: Output, json_output: Json = False):
    """Export checked strategy and benchmark equity curves to a new CSV."""
    try:
        result = export_equity(load_report(report), output)
    except FinanceError as exc:
        raise ConfigError(str(exc)) from exc
    if json_output:
        emit(result)
    else:
        typer.echo(f"Exported {result['rows']} bars to {result['path']}")

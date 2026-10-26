"""Export independently checked finance curves without overwriting files."""

import csv
import io

from tsfm_lab.finance.reports import _publish_bytes, validate_report


def export_equity(report: dict, destination) -> dict:
    """Write one CSV row per target close for the strategy and both benchmarks."""
    report = validate_report(report)
    evaluation = report["evaluation"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "origin",
            "open_timestamp",
            "close_timestamp",
            "signal_position",
            "strategy_equity",
            "strategy_net_return",
            "strategy_traded_notional",
            "strategy_entry_fee",
            "strategy_exit_fee",
            "strategy_entry_slippage",
            "strategy_exit_slippage",
            "strategy_short_financing",
            "cash_equity",
            "buy_and_hold_equity",
        ],
    )
    writer.writeheader()
    for case, position, strategy, cash, hold in zip(
        report["cases"],
        evaluation["signal_positions"],
        evaluation["strategy"]["curve"],
        evaluation["benchmarks"]["cash"]["curve"],
        evaluation["benchmarks"]["buy_and_hold"]["curve"],
        strict=True,
    ):
        row = {
            name: case[name] for name in ("origin", "open_timestamp", "close_timestamp")
        }
        row.update(signal_position=position)
        for name in (
            "equity",
            "net_return",
            "traded_notional",
            "entry_fee",
            "exit_fee",
            "entry_slippage",
            "exit_slippage",
            "short_financing",
        ):
            row[f"strategy_{name}"] = strategy[name]
        row.update(cash_equity=cash["equity"], buy_and_hold_equity=hold["equity"])
        writer.writerow(row)
    return {
        **_publish_bytes(buffer.getvalue().encode("utf-8"), destination),
        "rows": len(report["cases"]),
    }

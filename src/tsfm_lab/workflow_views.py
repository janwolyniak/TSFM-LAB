"""Read-only usage-profile context for terminal views over checked results."""

import shlex

from tsfm_lab.repl.text import literal
from tsfm_lab.run_modes import read_completed
from tsfm_lab.workflows import (
    display_workflow,
    ohlc_workflow_eligible,
    workflow_summary,
)


def history_profile(store, record):
    """Describe a checked record without adding fields to its saved manifest."""
    if record["status"] == "corrupt":
        return "Unavailable"
    try:
        if record["status"] == "completed":
            configuration = read_completed(store, record["run_id"])[0]["configuration"]
        else:
            configuration = store.load_configuration(record["run_id"])
        return display_workflow(configuration.get("workflow"))
    except (OSError, ValueError, KeyError):
        return "Unavailable"


def profile_context(configuration):
    """Use declared measurements and units, without changing the saved result."""
    lines = [f"Usage profile: {workflow_summary(configuration.get('workflow'))}"]
    units = configuration["dataset"]["units"]
    if configuration.get("mode") == "future_forecast":
        return [literal(lines[0]), "Accuracy unavailable: future actuals are absent."]
    metrics = set(configuration["evaluation"]["metrics"])
    errors = "/".join(name.upper() for name in ("mae", "rmse") if name in metrics)
    if errors:
        lines.append(f"{errors} error units: {units}")
    if "mase" in metrics:
        lines.append("MASE is dimensionless.")
    return [literal(line) for line in lines]


def ohlc_handoff(result, predictions, workspace, run_id):
    """Offer the existing explicit report route, without inventing any inputs."""
    configuration = result["configuration"]
    if configuration.get("mode") == "future_forecast":
        return []
    workflow = configuration.get("workflow")
    evaluation = configuration["evaluation"]
    selected = predictions.loc[predictions.lead == 1]
    if (
        workflow is None
        or not ohlc_workflow_eligible(workflow)
        or evaluation["point_statistic"] != "median"
        or evaluation["stride"] != 1
        or selected.empty
        or not (selected[["actual", "prediction"]] > 0).all().all()
    ):
        return []
    command = (
        f"tsfm finance backtest {shlex.quote(run_id)} "
        f"--workspace {shlex.quote(str(workspace))} "
        "--source-config ORIGINAL_CONFIG.yaml --ohlc OHLC.csv --price-target "
        "--initial-capital CAPITAL --threshold-bps THRESHOLD_BPS "
        "--fee-bps FEE_BPS --slippage-bps SLIPPAGE_BPS "
        "--short-financing-bps FINANCING_BPS "
        "--signal-latency-seconds LATENCY_SECONDS --output NEW_REPORT.json"
    )
    return [
        "Optional OHLC report: use separate execution bars and explicit timing/cost inputs.",
        "Replace every named placeholder. ORIGINAL_CONFIG.yaml must import the original matching data; saved input.yaml may have relative paths.",
        literal(command),
    ]

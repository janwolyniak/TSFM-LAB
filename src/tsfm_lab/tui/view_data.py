"""Presentation-only formatting for verified experiment data."""

import pandas as pd


def metric_text(metric: dict) -> str:
    if metric["value"] is None:
        reason = str(metric.get("reason") or "unavailable").replace("_", " ")
        return f"undefined ({reason})"
    return f"{metric['value']:.6g}"


def task_summary(dataset: dict, evaluation: dict) -> str:
    frequency = dataset["frequency"]
    if isinstance(frequency, int):
        frequency = str(pd.Timedelta(frequency, unit="ns"))
    return (
        f"Target: {dataset['target_name']} · Units: {dataset['units']} · Frequency: {frequency}\n"
        f"Horizon: {evaluation['horizon']} observations · Context: {evaluation['context_length']} observations\n"
        f"Targets: {evaluation['evaluation_start']} to {evaluation['evaluation_end']} (end excluded)"
    )


def record_details(result: dict) -> str:
    configuration = result["configuration"]
    model, runtime = configuration["model"], configuration["runtime"]
    checkpoint = model["options"].get("checkpoint")
    lines = [
        f"Model: {model['adapter_id']} · Family: {model['family']}",
        f"Adapter version: {model['adapter_version']}",
        f"Runtime: {runtime['device']} · {runtime['dtype']}",
    ]
    if checkpoint:
        lines.extend(
            f"{label}: {checkpoint.get(key) or 'not recorded'}"
            for label, key in (
                ("Repository", "repository"),
                ("Revision", "resolved_revision"),
                ("Checkpoint fingerprint", "fingerprint"),
                ("Model license", "license"),
            )
        )
    source = result["provenance"]["source"]
    lines.append(f"Source revision: {source.get('git_revision') or 'unversioned'}")
    if source.get("git_dirty"):
        lines.append("Source checkout had local changes during execution.")
    return "\n".join(lines)


def error_text(error) -> str:
    if isinstance(error, dict):
        stage = f" ({error['stage']})" if error.get("stage") else ""
        return f"{error.get('message') or error.get('type') or 'No details recorded'}{stage}"
    return str(error or "No error details recorded")


def preview_rows(frame) -> list[tuple[str, ...]]:
    return [
        (
            row.origin.isoformat(),
            row.target_timestamp.isoformat(),
            str(row.lead),
            f"{row.actual:.6g}",
            f"{row.prediction:.6g}",
        )
        for row in frame.head(12).itertuples()
    ]

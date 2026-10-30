"""Presentation of checked future records without suggesting observed accuracy."""

import json


def summary_lines(result):
    config, request, output = (
        result["configuration"],
        result["request"],
        result["output"],
    )
    targets = request["target_timestamps"]
    return [
        "Future forecast",
        f"Target: {output['target_name']} · Units: {output['units']} · Frequency: {config['dataset']['frequency']}",
        f"Origin: {request['origin']} · Context: {request['context_length']} observations · Horizon: {request['horizon']}",
        f"Future targets: {targets[0]} to {targets[-1]} (inclusive)",
        f"Forecast output: {output['point_statistic']}",
        "Accuracy unavailable: future actuals are absent.",
    ]


def details(result):
    return json.dumps(result["provenance"], indent=2, ensure_ascii=False)

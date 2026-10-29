"""Versioned deterministic future charts and provenance; historical renderer stays intact."""

import json
import textwrap
from xml.etree import ElementTree as ET

from tsfm_lab.artifact_validation import StoreError
from tsfm_lab.output_artifacts import _display, _element, _indices

_NAMES = {"forecast.svg", "provenance.json"}


def _label(parent, value, x, y, size=12, **attributes):
    return _element(parent, "text", value, x=x, y=y, font_size=size, **attributes)


def _line(parent, x1, y1, x2, y2, color="#e2e8f0", **attributes):
    return _element(
        parent, "line", x1=x1, x2=x2, y1=y1, y2=y2, stroke=color, **attributes
    )


def _sampling(predictions, observations):
    observed = observations.iloc[list(_indices(len(observations), 400))]
    predicted = predictions.iloc[list(_indices(len(predictions), 400))]
    sampling = {
        "policy": "uniform_including_endpoints_v1",
        "context_rows_total": len(observations),
        "context_rows_plotted": len(observed),
        "forecast_rows_total": len(predictions),
        "forecast_rows_plotted": len(predicted),
        "applied": len(observed) != len(observations)
        or len(predicted) != len(predictions),
    }
    return observed, predicted, sampling


def _coordinates(observations, predictions):
    first, last = observations.timestamp.iloc[0], predictions.target_timestamp.iloc[-1]
    low = min(
        float(observations.observed_value.min()), float(predictions.prediction.min())
    )
    high = max(
        float(observations.observed_value.max()), float(predictions.prediction.max())
    )
    scale = max(abs(low), abs(high), 1.0)
    low, high = low / scale, high / scale
    if low == high:
        low, high = min(low, 0), max(high, 0)
        if low == high:
            low, high = -1, 1

    def x(value):
        return 100 + 920 * (int(value.value) - int(first.value)) / (
            int(last.value) - int(first.value)
        )

    def y(value):
        return 440 - 280 * (float(value) / scale - low) / (high - low)

    return x, y, first, last, low, high, scale


def _axes(parent, first, last, low, high, scale):
    import pandas as pd

    for index in range(5):
        fraction = index / 4
        x, y = 100 + 920 * fraction, 440 - 280 * fraction
        _line(parent, 100, y, 1020, y)
        _label(
            parent,
            f"{((low + (high - low) * fraction) * scale):.6g}",
            90,
            y + 4,
            text_anchor="end",
        )
        nanoseconds = (
            int(first.value) + (int(last.value) - int(first.value)) * index // 4
        )
        timestamp = pd.Timestamp(nanoseconds, unit="ns", tz="UTC")
        _label(parent, timestamp.strftime("%Y-%m-%d"), x, 465, 11, text_anchor="middle")
        _label(
            parent,
            timestamp.isoformat().split("T")[1].removesuffix("+00:00"),
            x,
            481,
            11,
            text_anchor="middle",
        )
    _label(parent, "Timestamp · UTC", 560, 507, 13, text_anchor="middle")


def _trajectory(parent, times, values, x, y, role, color):
    points = [(x(time), y(value)) for time, value in zip(times, values, strict=True)]
    _element(
        parent,
        "polyline",
        data_role=role,
        points=" ".join(f"{a:.3f},{b:.3f}" for a, b in points),
        fill="none",
        stroke=color,
        stroke_width=2,
    )
    for a, b in points:
        _element(
            parent,
            "circle",
            data_role=f"{role}-point",
            cx=f"{a:.3f}",
            cy=f"{b:.3f}",
            r="2.3",
            fill=color,
        )


def _chart(result, predictions, observations, sampled, forecast, sampling):
    import pandas as pd

    config, output = result["configuration"], result["output"]
    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": "0 0 1120 760",
            "width": "1120",
            "height": "760",
            "role": "img",
            "aria-labelledby": "future-title future-description",
        },
    )
    _element(root, "title", config["name"], id="future-title")
    _element(
        root,
        "desc",
        "Consumed observed context through the forecast origin and point predictions strictly after it. Future accuracy is unavailable because actual future observations are absent.",
        id="future-description",
    )
    _element(root, "rect", x=0, y=0, width=1120, height=760, fill="white")
    labels = _element(root, "g", font_family="sans-serif", fill="#0f172a")
    for index, line in enumerate(
        textwrap.wrap(_display(config["name"], 150), width=80)[:2]
    ):
        _label(labels, line, 50, 34 + 25 * index, 23, font_weight=600)
    subtitle = _display(
        f"{output['target_name']} · original units: {output['units']}", 170
    )
    for index, line in enumerate(textwrap.wrap(subtitle, width=110)[:2]):
        _label(labels, line, 50, 91 + 17 * index, 15)
    _label(
        labels,
        f"Requested task: {output['requested_task']} · actual point label: {output['point_statistic']}",
        50,
        132,
        14,
    )
    x, y, first, last, low, high, scale = _coordinates(observations, predictions)
    _axes(labels, first, last, low, high, scale)
    origin = pd.Timestamp(result["request"]["origin"]).tz_convert("UTC")
    boundary = x(origin)
    _line(
        labels,
        boundary,
        150,
        boundary,
        444,
        "#b45309",
        stroke_dasharray="5 4",
        data_role="origin-boundary",
    )
    _label(
        labels,
        "Forecast origin / last observed timestamp",
        boundary,
        150,
        11,
        text_anchor="start" if boundary < 560 else "end",
        fill="#92400e",
    )
    trajectories = _element(root, "g")
    _trajectory(
        trajectories,
        sampled.timestamp,
        sampled.observed_value,
        x,
        y,
        "observations",
        "#0f172a",
    )
    _trajectory(
        trajectories,
        forecast.target_timestamp,
        forecast.prediction,
        x,
        y,
        "predictions",
        "#0369a1",
    )
    _line(labels, 50, 536, 85, 536, "#0f172a", stroke_width=2)
    _label(labels, "Observed context (through origin)", 97, 541, 13)
    _line(labels, 620, 536, 655, 536, "#0369a1", stroke_width=2)
    _label(labels, "Future point predictions (after origin)", 667, 541, 13)
    lines = [
        "Future accuracy unavailable: future_actuals_absent. No scores or future actuals are supplied.",
        f"Origin: {origin.isoformat()} · Forecast through: {last.isoformat()}.",
        "Observed CSV/chart context contains only the consumed L rows; earlier source observations are omitted.",
        f"Displayed: {sampling['context_rows_plotted']}/{sampling['context_rows_total']} context rows and {sampling['forecast_rows_plotted']}/{sampling['forecast_rows_total']} future rows.",
        (
            "Sampled display: uniform selection including endpoints; full rows remain in both CSV files."
            if sampling["applied"]
            else "All consumed observations and future predictions are displayed; full rows remain in both CSV files."
        ),
        "Lines join displayed points. Long labels may be shortened; full metadata remains in provenance.json.",
    ]
    for index, line in enumerate(lines):
        _label(labels, line, 50, 579 + index * 27)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _outputs(result, predictions, observations):
    observed, predicted, sampling = _sampling(predictions, observations)
    document = {
        "schema_version": 1,
        "artifact_contract": "future_point_outputs_v1",
        **{
            key: result[key]
            for key in (
                "mode",
                "run_id",
                "configuration",
                "request",
                "output",
                "counts",
                "provenance",
                "evaluation",
            )
        },
        "chart": {
            "scope": "consumed_context_and_future_predictions",
            "sampling": sampling,
        },
    }
    return {
        "forecast.svg": _chart(
            result, predictions, observations, observed, predicted, sampling
        ),
        "provenance.json": (
            json.dumps(document, allow_nan=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
    }


def build_outputs(result, predictions, observations):
    """Render already checked future observations/predictions without inference."""
    return _outputs(result, predictions, observations)


def validate_outputs(artifacts, result, predictions, observations):
    """Require exact future_point_outputs_v1 bytes; preserve this renderer version."""
    if type(artifacts) is not dict or set(artifacts) != _NAMES:
        raise StoreError("Future outputs require chart and provenance only")
    expected = _outputs(result, predictions, observations)
    if any(
        type(artifacts[key]) is not bytes or artifacts[key] != expected[key]
        for key in _NAMES
    ):
        raise StoreError("Future chart/provenance disagree with the validated record")

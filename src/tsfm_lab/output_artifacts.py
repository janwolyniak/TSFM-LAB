"""Deterministic offline outputs from checked historical prediction records."""

import json
import textwrap
from xml.etree import ElementTree as ET

from tsfm_lab.artifact_validation import StoreError

OUTPUT_NAMES = frozenset({"forecast.svg", "provenance.json"})
_OBSERVATIONS = 400
_WINDOWS = 32
_WINDOW_POINTS = 128
_SVG = "http://www.w3.org/2000/svg"
_SECRETS = {
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "credentials",
    "credential",
    "secret",
    "client_secret",
    "token",
    "hf_token",
    "aws_session_token",
    "aws_secret_access_key",
    "aws_access_key_id",
    "google_application_credentials",
    "headers",
}


def _recorded(value):
    """Copy numerical provenance without adding arbitrary credential fields."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key = str(key)
            normalized = key.casefold().replace("-", "_")
            if normalized in _SECRETS or normalized.endswith(
                ("_api_key", "_access_token", "_refresh_token", "_secret", "_password")
            ):
                continue
            result[key] = _recorded(item)
        return result
    if isinstance(value, list):
        return [_recorded(item) for item in value]
    return value


def _indices(size, limit):
    if size <= limit:
        return range(size)
    return [index * (size - 1) // (limit - 1) for index in range(limit)]


def _display(value, limit=180):
    # Result text is data: replace unsupported XML characters and bound labels.
    text = " ".join(str(value).split())
    text = "".join(
        (
            char
            if 32 <= ord(char) <= 0xD7FF
            or 0xE000 <= ord(char) <= 0xFFFD
            or 0x10000 <= ord(char) <= 0x10FFFF
            else "\ufffd"
        )
        for char in text
    )
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _element(parent, tag, text=None, **attributes):
    node = ET.SubElement(
        parent,
        tag,
        {key.replace("_", "-"): str(value) for key, value in attributes.items()},
    )
    if text is not None:
        node.text = _display(text)
    return node


def _sample(frame):
    observed = frame.sort_values("target_timestamp", kind="stable").drop_duplicates(
        "target_timestamp"
    )
    if (frame.groupby("target_timestamp")["actual"].nunique() != 1).any():
        raise StoreError("Observed values disagree for a shared target timestamp")
    origins = sorted(frame.origin.unique())
    selected = {origins[index] for index in _indices(len(origins), _WINDOWS)}
    windows = []
    for origin, window in frame.loc[frame.origin.isin(selected)].groupby(
        "origin", sort=True
    ):
        window = window.sort_values("lead", kind="stable")
        windows.append(
            (origin, window.iloc[list(_indices(len(window), _WINDOW_POINTS))])
        )
    sampled_observed = observed.iloc[list(_indices(len(observed), _OBSERVATIONS))]
    plotted = sum(len(window) for _, window in windows)
    sampling = {
        "policy": "uniform_including_endpoints_v1",
        "observed_timestamps_total": len(observed),
        "observed_timestamps_plotted": len(sampled_observed),
        "forecast_windows_total": len(origins),
        "forecast_windows_plotted": len(windows),
        "forecast_cases_total": len(frame),
        "forecast_points_plotted": plotted,
        "points_per_window_limit": _WINDOW_POINTS,
        "applied": len(observed) != len(sampled_observed) or len(frame) != plotted,
    }
    return sampled_observed, windows, sampling


def _metric_lines(result):
    lines = []
    metrics = result["metrics"]["overall"]
    for name in ("mae", "mase", "rmse"):
        if name not in metrics:
            lines.append(f"{name.upper()} unavailable: not_requested")
            continue
        metric = metrics[name]
        if metric["value"] is None:
            lines.append(f"{name.upper()} unavailable: {metric['reason']}")
        else:
            lines.append(f"{name.upper()} = {metric['value']:.8g}")
    return lines


def _chart(result, frame, observed, windows, sampling):
    config = result["configuration"]
    dataset, evaluation = config["dataset"], config["evaluation"]
    import pandas as pd

    start, end = (
        pd.Timestamp(evaluation[key]).tz_convert("UTC")
        for key in ("evaluation_start", "evaluation_end")
    )
    low = float(frame[["actual", "prediction"]].min().min())
    high = float(frame[["actual", "prediction"]].max().max())
    scale = max(abs(low), abs(high), 1.0)
    low, high = low / scale, high / scale
    if low == high:
        low, high = min(low, 0.0), max(high, 0.0)
        if low == high:
            low, high = -1.0, 1.0

    def x(timestamp):
        return 100 + 920 * (timestamp.value - start.value) / (end.value - start.value)

    def y(value):
        return 470 - 300 * (float(value) / scale - low) / (high - low)

    root = ET.Element(
        "svg",
        {
            "xmlns": _SVG,
            "viewBox": "0 0 1120 860",
            "width": "1120",
            "height": "860",
            "role": "img",
            "aria-labelledby": "chart-title chart-description",
        },
    )
    title = result.get(
        "name", f"{dataset['target_name']} · {config['model']['adapter_id']}"
    )
    _element(root, "title", title, id="chart-title")
    _element(
        root,
        "desc",
        "Historical evaluation: observed actuals at evaluated target timestamps and separate forecast window trajectories. Original units; UTC timestamps. History prefix is not saved.",
        id="chart-description",
    )
    _element(root, "rect", x=0, y=0, width=1120, height=860, fill="white")
    header = _element(root, "g", font_family="sans-serif", fill="#0f172a")
    for index, line in enumerate(textwrap.wrap(_display(title, 150), width=80)[:2]):
        _element(
            header, "text", line, x=50, y=35 + 25 * index, font_size=23, font_weight=600
        )
    subtitle = _display(
        f"{dataset['target_name']} · original units: {dataset['units']}", 170
    )
    for index, line in enumerate(textwrap.wrap(subtitle, width=110)[:2]):
        _element(header, "text", line, x=50, y=92 + 17 * index, font_size=15)
    _element(
        header,
        "text",
        f"Requested task: {evaluation['point_statistic']} · saved statistic: {', '.join(sorted(set(frame.point_statistic)))}",
        x=50,
        y=132,
        font_size=14,
    )
    _axes(header, start, end, low, high, scale)
    _element(header, "line", x1=100, x2=1020, y1=170, y2=170, stroke="#e2e8f0")
    for origin, window in windows:
        group = _element(
            root, "g", data_role="forecast-window", data_origin=origin.isoformat()
        )
        _element(
            group,
            "title",
            f"Forecast origin {origin.isoformat()} · {len(window)} displayed target points",
        )
        _trajectory(group, window.target_timestamp, window.prediction, x, y, "#0369a1")
    actual = _element(root, "g", data_role="observations")
    _trajectory(actual, observed.target_timestamp, observed.actual, x, y, "#0f172a")
    for row in observed.itertuples():
        _element(
            actual,
            "circle",
            cx=f"{x(row.target_timestamp):.3f}",
            cy=f"{y(row.actual):.3f}",
            r="2.2",
            fill="#0f172a",
            data_role="observed-point",
            data_timestamp=row.target_timestamp.isoformat(),
        )
    _captions(header, result, sampling, start, end)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _trajectory(group, timestamps, values, x, y, color):
    points = [
        (x(timestamp), y(value))
        for timestamp, value in zip(timestamps, values, strict=True)
    ]
    _element(
        group,
        "polyline",
        points=" ".join(f"{a:.3f},{b:.3f}" for a, b in points),
        fill="none",
        stroke=color,
        stroke_width="1.7",
        stroke_opacity="0.7",
    )
    if len(points) == 1:
        _element(
            group,
            "circle",
            cx=f"{points[0][0]:.3f}",
            cy=f"{points[0][1]:.3f}",
            r="3",
            fill=color,
        )


def _axes(parent, start, end, low, high, scale):
    for index in range(5):
        fraction = index / 4
        y = 470 - 300 * fraction
        value = (low + (high - low) * fraction) * scale
        _element(parent, "line", x1=100, x2=1020, y1=y, y2=y, stroke="#e2e8f0")
        _element(
            parent,
            "text",
            f"{value:.6g}",
            x=90,
            y=y + 4,
            font_size=12,
            text_anchor="end",
        )
        timestamp = start + (end - start) * fraction
        x = 100 + 920 * fraction
        _element(
            parent,
            "text",
            timestamp.strftime("%Y-%m-%d"),
            x=x,
            y=493,
            font_size=11,
            text_anchor="middle",
        )
        _element(
            parent,
            "text",
            timestamp.isoformat().split("T")[1].removesuffix("+00:00"),
            x=x,
            y=509,
            font_size=11,
            text_anchor="middle",
        )
    for x, label in (
        (100, "Evaluation start (inclusive)"),
        (1020, "Evaluation end (exclusive)"),
    ):
        _element(
            parent,
            "line",
            x1=x,
            x2=x,
            y1=160,
            y2=476,
            stroke="#b45309",
            stroke_dasharray="5 4",
        )
        _element(
            parent,
            "text",
            label,
            x=x,
            y=151,
            font_size=11,
            text_anchor="start" if x == 100 else "end",
            fill="#92400e",
        )
    _element(
        parent,
        "text",
        "Target timestamp · UTC",
        x=560,
        y=534,
        font_size=13,
        text_anchor="middle",
    )


def _captions(parent, result, sampling, start, end):
    _element(
        parent, "line", x1=50, x2=85, y1=555, y2=555, stroke="#0f172a", stroke_width=2
    )
    _element(
        parent,
        "text",
        "Observed actuals · each evaluated timestamp once",
        x=97,
        y=560,
        font_size=13,
    )
    _element(
        parent, "line", x1=620, x2=655, y1=555, y2=555, stroke="#0369a1", stroke_width=2
    )
    _element(
        parent,
        "text",
        "Separate forecast window trajectories",
        x=667,
        y=560,
        font_size=13,
    )
    lines = [
        "Evaluated target timestamps only. The historical context prefix is not saved in these artifacts.",
        f"Evaluation: {start.isoformat()} ≤ target timestamp < {end.isoformat()}.",
        f"Displayed: {sampling['observed_timestamps_plotted']}/{sampling['observed_timestamps_total']} observed timestamps; {sampling['forecast_windows_plotted']}/{sampling['forecast_windows_total']} forecast windows; {sampling['forecast_points_plotted']}/{sampling['forecast_cases_total']} forecast points.",
        (
            "Sampled display: uniformly selected timestamps, windows, and window points; endpoints included."
            if sampling["applied"]
            else "All evaluated observations and forecast points are displayed."
        ),
        "Lines join displayed points; omitted points are not reconstructed. Full forecasts remain in predictions.csv.",
        "Historical accuracy over forecast cases (overlapping windows retain their evaluation weights):",
        *_metric_lines(result),
        "Long text labels may be shortened for display; full recorded metadata remains in provenance.json.",
    ]
    for index, line in enumerate(lines):
        _element(parent, "text", line, x=50, y=596 + 26 * index, font_size=12)


def _artifacts(result, predictions):
    observed, windows, sampling = _sample(predictions)
    provenance = {
        "schema_version": 1,
        "artifact_contract": "guided_historical_outputs_v1",
        "result_schema_version": result["schema_version"],
        **{
            key: _recorded(result[key])
            for key in (
                "run_id",
                "name",
                "configuration",
                "provenance",
                "counts",
                "metrics",
                "timing_seconds",
            )
            if key in result
        },
        "chart": {
            "scope": "evaluated_target_timestamps_only",
            "history_prefix_available": False,
            "sampling": sampling,
        },
    }
    return {
        "forecast.svg": _chart(result, predictions, observed, windows, sampling),
        "provenance.json": (
            json.dumps(provenance, allow_nan=False, sort_keys=True, indent=2) + "\n"
        ).encode(),
    }


def build_output_artifacts(result: dict, predictions) -> dict[str, bytes]:
    """Build original-unit chart/provenance from already validated saved values."""
    return _artifacts(result, predictions)


def validate_output_artifacts(artifacts, result, predictions) -> None:
    """Require exact v1 outputs; preserve this renderer when adding new versions."""
    if not isinstance(artifacts, dict) or set(artifacts) != OUTPUT_NAMES:
        raise StoreError("Guided outputs require forecast.svg and provenance.json")
    expected = _artifacts(result, predictions)
    if any(
        type(artifacts[name]) is not bytes or artifacts[name] != expected[name]
        for name in OUTPUT_NAMES
    ):
        raise StoreError(
            "Guided output artifacts disagree with checked predictions or provenance"
        )

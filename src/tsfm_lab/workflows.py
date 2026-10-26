"""Validated usage metadata, separate from forecast and dataset identities."""

import math

PROFILE_NAMES = ("generic", "finance", "energy", "temperature", "weather")
_LABELS = {
    "generic": "Generic",
    "finance": "Finance",
    "energy": "Energy",
    "temperature": "Temperature",
    "weather": "Weather",
}
_WEATHER_SEMANTICS = (
    "instantaneous",
    "interval_mean",
    "interval_total",
    "interval_max",
    "other",
)


class WorkflowError(ValueError):
    """A usage profile or its declared measurement is unsupported."""


def _fields(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise WorkflowError(f"{label} requires exactly: {', '.join(sorted(expected))}")


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise WorkflowError(f"{label} must be a nonempty string")
    return value


def _choice(value, choices, label):
    if type(value) is not str or value not in choices:
        raise WorkflowError(f"{label} must be one of: {', '.join(choices)}")
    return value


def _options(profile, value):
    if type(value) is not dict:
        raise WorkflowError("workflow.options must be an object")
    if profile in {"generic", "temperature"}:
        _fields(value, (), f"{profile} options")
        return {}
    if profile in {"finance", "energy"}:
        key = "target_kind" if profile == "finance" else "measurement"
        choices = (
            ("close_price", "return", "other")
            if profile == "finance"
            else (
                "power",
                "energy",
                "other",
            )
        )
        choice = _choice(value.get(key), choices, f"{profile} {key}")
        detail = "measurement" if profile == "finance" else "quantity"
        required = {key, detail} if choice == "other" else {key}
        _fields(value, required, f"{profile} options")
        return {
            key: choice,
            **(
                {detail: _text(value[detail], f"{profile} {detail}")}
                if choice == "other"
                else {}
            ),
        }
    semantics = _choice(
        value.get("measurement_semantics"),
        _WEATHER_SEMANTICS,
        "weather measurement_semantics",
    )
    required = {"variable", "station", "measurement_semantics"}
    if semantics == "other":
        required.add("description")
    _fields(value, required, "weather options")
    result = {
        "variable": _text(value["variable"], "weather variable"),
        "station": _text(value["station"], "weather station"),
        "measurement_semantics": semantics,
    }
    if semantics == "other":
        result["description"] = _text(value["description"], "weather description")
    return result


def normalize_workflow(value):
    """Snapshot explicit profile metadata; absence preserves legacy serialization."""
    if value is None:
        return None
    _fields(value, {"schema_version", "profile", "options"}, "workflow")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise WorkflowError("workflow.schema_version must be 1")
    profile = _choice(value["profile"], PROFILE_NAMES, "workflow.profile")
    return {
        "schema_version": 1,
        "profile": profile,
        "options": _options(profile, value["options"]),
    }


def display_workflow(value):
    """Display historical records as Generic without adding metadata to them."""
    workflow = normalize_workflow(value)
    return _LABELS[workflow["profile"]] if workflow else "Generic"


def workflow_summary(value):
    """Describe declared usage only; units remain in the dataset declaration."""
    workflow = normalize_workflow(value)
    if workflow is None:
        return "Generic"
    profile, options = workflow["profile"], workflow["options"]
    label = _LABELS[profile]
    if profile == "finance":
        measurement = options.get(
            "measurement", options["target_kind"].replace("_", " ")
        )
        return f"{label} · {measurement}"
    if profile == "energy":
        measurement = options.get("quantity", options["measurement"])
        return f"{label} · {measurement}"
    if profile == "weather":
        semantics = options.get(
            "description", options["measurement_semantics"].replace("_", " ")
        )
        return f"{label} · {options['variable']} · {options['station']} · {semantics}"
    return label


def validate_workflow_units(value, units):
    """Require an explicit measurement unit without copying it into usage metadata."""
    workflow = normalize_workflow(value)
    if workflow is None or workflow["profile"] == "generic":
        return
    if (
        type(units) is not str
        or not units.strip()
        or units.strip().casefold() == "unspecified"
    ):
        raise WorkflowError(
            f"{display_workflow(workflow)} requires explicit dataset units"
        )


def validate_workflow_dataset(value, dataset):
    """Check declared measurement constraints without transforming any values."""
    workflow = normalize_workflow(value)
    validate_workflow_units(workflow, getattr(dataset, "units", None))
    if workflow is None:
        return
    if (
        workflow["profile"] == "finance"
        and workflow["options"]["target_kind"] == "close_price"
    ):
        values = getattr(dataset, "values", ())
        try:
            positive = len(values) > 0 and all(
                math.isfinite(item) and item > 0 for item in values
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise WorkflowError(
                "Finance close-price targets require finite positive observations"
            ) from exc
        if not positive:
            raise WorkflowError(
                "Finance close-price targets require finite positive observations"
            )


def ohlc_workflow_eligible(value):
    """Offer guided OHLC help only for an explicit close-price workflow.

    This metadata hint never replaces the manual report's price declaration,
    timing, cost, stride, or forecast-task validation.
    """
    workflow = normalize_workflow(value)
    return workflow is not None and (
        workflow["profile"] == "finance"
        and workflow["options"]["target_kind"] == "close_price"
    )

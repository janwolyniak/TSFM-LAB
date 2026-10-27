"""Bounded explanation facts projected from local, authoritative validators."""

import copy
import json
import math

from tsfm_lab.models.catalog import CURATED_REPOSITORIES, catalog_info
from tsfm_lab.workflows import normalize_workflow

_LEADS = 12
_CAPABILITIES = (
    "min_context",
    "max_context",
    "max_horizon",
    "target_count",
    "output_types",
    "devices",
)
_EVALUATION = (
    "context_length",
    "horizon",
    "evaluation_start",
    "evaluation_end",
    "stride",
    "seasonal_period",
    "metrics",
    "point_statistic",
    "aggregation",
)
_FORECAST_OPTIONS = (
    "normalize_inputs",
    "force_flip_invariance",
    "infer_is_positive",
    "use_continuous_quantile_head",
    "fix_quantile_crossing",
    "return_backcast",
    "cross_learning",
    "limit_prediction_length",
    "tta_sign_flip",
    "tta_diff",
    "pad_context",
    "output",
    "horizon_policy",
    "external_scaling",
    "constant_scale",
    "native_scaling",
    "native_inverse",
)
_VARIANT = (
    "repository",
    "variant",
    "display_name",
    "reference_revision",
    "weight_bytes",
    "parameter_count",
    "min_context",
    "max_context",
    "max_horizon",
    "project_inference_status",
    "verification_label",
)
_COUNTS = (
    "attempted_windows",
    "successful_windows",
    "failed_windows",
    "forecast_cases",
    "unique_target_timestamps",
)
_FINANCE_METRICS = (
    "final_capital",
    "cumulative_return",
    "max_drawdown",
    "volatility_unannualized",
    "total_traded_notional",
    "turnover_initial_capital",
    "fees",
    "slippage",
    "short_financing",
    "total_costs",
)
_RULES = (
    "initial_capital",
    "threshold_bps",
    "fee_bps",
    "slippage_bps",
    "short_financing_bps",
    "signal_latency_seconds",
)
_COMMON_OMITTED = {
    "names_and_free_descriptions": True,
    "filesystem_paths": True,
    "raw_observations_and_predictions": True,
    "source_yaml_and_environment": True,
}
_GLOSSARY = {
    "context_length": "Past observations supplied for each forecast window.",
    "horizon": "Number of future observations evaluated per forecast origin.",
    "evaluation_end": "The evaluation end is excluded from the target period.",
    "mae": "Mean absolute error, in the declared target units; lower is better.",
    "rmse": "Root mean squared error, in the declared target units; lower is better.",
    "mase": "Mean absolute error divided by a causal seasonal-naive scale; dimensionless.",
    "native_point": "A model-native point output does not establish a median or mean.",
    "verification": "Integration checks are not accuracy benchmarks or model rankings.",
}


class EvidenceError(ValueError):
    """Explanation context cannot be safely represented within its local bound."""


def _native(value):
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise EvidenceError("Explanation object keys must be strings")
        for item in value.values():
            _native(item)
    elif type(value) is list:
        for item in value:
            _native(item)
    elif type(value) not in {str, int, float, bool, type(None)}:
        raise EvidenceError("Explanation context requires native JSON values")
    elif type(value) is float and not math.isfinite(value):
        raise EvidenceError("Explanation context cannot contain nonfinite values")


def bounded_json(context: dict, max_bytes: int = 16000) -> str:
    """Serialize exact JSON facts, refusing oversize context without truncation."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise EvidenceError("Explanation byte limit must be a positive integer")
    try:
        if type(context) is not dict:
            raise EvidenceError("Explanation context must be an object")
        _native(context)
        text = json.dumps(
            context,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(text.encode("utf-8")) > max_bytes:
            raise EvidenceError("Explanation context exceeds the local byte limit")
        return text
    except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError) as exc:
        if isinstance(exc, EvidenceError):
            raise
        raise EvidenceError("Explanation context is invalid JSON") from exc


def fact_map(context: dict) -> dict:
    """Map data/glossary scalar leaves to stable RFC 6901 JSON-pointer IDs.

    Object keys escape tilde and slash; list indices start at zero. Values stay
    native JSON scalars, including booleans and null; envelope metadata is absent.
    """
    bounded_json(context)
    facts = {}

    def visit(value, pointer):
        if type(value) is dict:
            for key, item in value.items():
                escaped = key.replace("~", "~0").replace("/", "~1")
                visit(item, f"{pointer}/{escaped}")
        elif type(value) is list:
            for index, item in enumerate(value):
                visit(item, f"{pointer}/{index}")
        else:
            facts[pointer] = value

    for key in ("data", "glossary"):
        if key in context:
            visit(context[key], f"/{key}")
    return facts


def _pick(value, keys):
    return {key: copy.deepcopy(value[key]) for key in keys if key in value}


def _check_summary(check):
    result = {"revision": check["revision"]}
    if check["repository"] in CURATED_REPOSITORIES:
        result["repository"] = check["repository"]
    for key, choices in (
        ("device", {"cpu", "mps", "cuda"}),
        ("dtype", {"float32", "float64", "float16", "bfloat16"}),
    ):
        if check[key] in choices:
            result[key] = check[key]
    return result


def _envelope(kind, data, *, omitted=None, glossary=None, limitations=()):
    context = {
        "schema_version": 1,
        "kind": kind,
        "data": data,
        "omitted": {**_COMMON_OMITTED, **(omitted or {})},
        "glossary": copy.deepcopy(_GLOSSARY if glossary is None else glossary),
        "limitations": [
            "Use only these local facts; no benchmark rankings are established.",
            "Assistant annotations are untrusted prose and cannot change saved values.",
            *limitations,
        ],
    }
    # Detach every returned object and exercise the same bound used for preview/send.
    return json.loads(bounded_json(context))


def model_evidence(identifier: str, variant: str | None = None) -> dict:
    """Explain one supported catalog entry without files, network, or inference."""
    entry = catalog_info(identifier, variant)
    data = _pick(
        entry,
        (
            "adapter_id",
            "family",
            "display_name",
            "source",
            "license",
            "dtypes",
            "point_statistic",
            "supported_tasks",
            "frequency_policy",
            "runtime_requirements",
            "runtime_notices",
        ),
    )
    data["capabilities"] = _pick(entry["capabilities"], _CAPABILITIES)
    data["variants"] = [_pick(item, _VARIANT) for item in entry["variants"]]
    if "repository" in entry:
        data["selected_variant"] = _pick(entry, _VARIANT)
    checks = entry["variants"][0]["representative_checks"] if entry["variants"] else []
    data["adapter_representative_checks"] = [_check_summary(item) for item in checks]
    return _envelope(
        "model",
        data,
        limitations=(
            "Weight-file bytes do not measure runtime RAM.",
            "Representative adapter checks do not establish inference for another variant.",
        ),
    )


def _model(configuration, provenance):
    model = configuration["model"]
    try:
        entry = catalog_info(model["adapter_id"])
    except ValueError:
        return {"catalog_supported": False}
    data = {
        "catalog_supported": True,
        **_pick(
            entry,
            (
                "adapter_id",
                "family",
                "display_name",
                "point_statistic",
                "supported_tasks",
            ),
        ),
    }
    if entry["adapter_id"] == "seasonal_naive":
        period = model["options"].get("seasonal_period")
        if type(period) is int and period > 0:
            data["seasonal_period"] = period
    checkpoint = model["options"].get("checkpoint")
    if checkpoint:
        data["checkpoint"] = _pick(
            checkpoint,
            (
                "source",
                "resolved_revision",
                "native_context_length",
                "native_prediction_length",
                "point_forecast",
            ),
        )
        if checkpoint["repository"] in CURATED_REPOSITORIES:
            data["checkpoint"]["repository"] = checkpoint["repository"]
        verification = provenance.get("project_verification")
        data["project_verification"] = None
        if verification is not None:
            data["project_verification"] = {
                "status": verification["project_inference_status"],
                "compatibility": copy.deepcopy(verification["compatibility"]),
                "selected_checkpoint_check_count": len(
                    verification["selected_checkpoint_checks"]
                ),
                "adapter_representative_checks": [
                    _check_summary(item)
                    for item in verification["adapter_representative_checks"]
                ],
            }
    return data


def _configuration(result):
    configuration = result["configuration"]
    workflow = normalize_workflow(configuration.get("workflow"))
    profile = {"profile": workflow["profile"] if workflow else "generic"}
    if workflow:
        profile["options"] = _pick(
            workflow["options"],
            (
                "target_kind",
                "measurement",
                "measurement_semantics",
            ),
        )
        # Finance's `measurement` is a free description when target_kind=other.
        if profile["profile"] != "energy":
            profile["options"].pop("measurement", None)
    runtime = configuration["runtime"]
    safe_runtime = _pick(runtime, ("batch_size", "seed"))
    for key in ("seed_used", "deterministic"):
        if type(runtime.get(key)) is bool:
            safe_runtime[key] = runtime[key]
    for key, choices in (
        ("device", {"cpu", "mps", "cuda"}),
        ("dtype", {"float32", "float64", "float16", "bfloat16"}),
    ):
        if runtime[key] in choices:
            safe_runtime[key] = runtime[key]
    if configuration["model"]["adapter_id"] in {
        "chronos_bolt",
        "chronos_2",
        "timesfm_2p5",
        "tirex_2",
        "ttm_r2",
    }:
        safe_runtime["forecast_options"] = _pick(
            runtime.get("forecast_options", {}), _FORECAST_OPTIONS
        )
        if type(runtime.get("torch_compile")) is bool:
            safe_runtime["torch_compile"] = runtime["torch_compile"]
    return {
        "dataset": _pick(configuration["dataset"], ("units", "frequency")),
        "workflow": profile,
        "model": _model(configuration, result["provenance"]),
        "evaluation": _pick(configuration["evaluation"], _EVALUATION),
        "runtime": safe_runtime,
    }


def _result(result, statistics):
    metrics = result["metrics"]
    leads = sorted(metrics["by_lead"], key=int)
    units = result["configuration"]["dataset"]["units"]
    return {
        "configuration": _configuration(result),
        "counts": _pick(result["counts"], _COUNTS),
        "output_statistics": sorted(set(statistics)),
        "metrics": {
            **_pick(metrics, ("overall", "scale", "scale_reason", "scale_count")),
            "by_lead": {
                lead: _pick(metrics["by_lead"][lead], ("mae", "rmse", "mase"))
                for lead in leads[:_LEADS]
            },
            "included_leads": min(len(leads), _LEADS),
            "total_leads": len(leads),
            "units": {
                name: "dimensionless" if name == "mase" else units
                for name in metrics["overall"]
            },
        },
        "scale_history": _pick(
            result["provenance"]["scale_history"],
            (
                "start",
                "end_exclusive",
                "observations",
                "seasonal_period",
            ),
        ),
    }


def experiment_evidence(store, run_id: str) -> dict:
    """Project only a completed result accepted by full artifact validation."""
    result, predictions = store.load_result(run_id)
    data = _result(result, predictions["point_statistic"].unique().tolist())
    return _envelope(
        "experiment",
        data,
        omitted={
            "by_lead_count": max(data["metrics"]["total_leads"] - _LEADS, 0),
        },
    )


def finance_evidence(path) -> dict:
    """Explain a saved finance report only after fingerprint and arithmetic checks."""
    from tsfm_lab.finance.reports import load_report

    report = load_report(path)
    evaluation = report["evaluation"]
    diagnostics = evaluation["diagnostics"]
    units = report["parent"]["result"]["configuration"]["dataset"]["units"]
    paths = {"strategy": evaluation["strategy"], **evaluation["benchmarks"]}
    parent = _result(
        report["parent"]["result"],
        (case["point_statistic"] for case in report["cases"]),
    )
    data = {
        "parent_forecast": parent,
        "policy": report["policy"],
        "rules": _pick(report["rules"], _RULES),
        "resolved_policy": copy.deepcopy(report["resolved_policy"]),
        "cases": evaluation["cases"],
        "diagnostics": _pick(
            diagnostics,
            (
                "basis",
                "cases",
                "forecast_return_mae",
                "forecast_return_rmse",
                "directional_accuracy_3way",
                "zero_move_rule",
                "forecast_direction_counts",
                "actual_direction_counts",
            ),
        ),
        "capital_metrics": {
            name: _pick(value["metrics"], _FINANCE_METRICS)
            for name, value in paths.items()
        },
        "rule_units": {
            key: (
                units
                if key == "initial_capital"
                else "seconds" if key == "signal_latency_seconds" else "basis_points"
            )
            for key in _RULES
        },
        "metric_units": {
            key: (
                "dimensionless"
                if key
                in {
                    "cumulative_return",
                    "max_drawdown",
                    "volatility_unannualized",
                    "turnover_initial_capital",
                }
                else units
            )
            for key in _FINANCE_METRICS
        },
        "diagnostic_units": "Simple-return fractions and directional-accuracy fractions are dimensionless.",
    }
    return _envelope(
        "finance",
        data,
        omitted={
            "by_lead_count": max(parent["metrics"]["total_leads"] - _LEADS, 0),
            "finance_cases_curves_positions_and_return_arrays": True,
        },
        limitations=(
            "Historical report values depend on the explicitly declared execution and cost rules.",
            "Volatility is an unannualized sample standard deviation; drawdown uses period-end equity including initial capital.",
        ),
    )

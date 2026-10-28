"""Resolve safe suggestions against local references and normal form validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tsfm_lab.config import ConfigError
from tsfm_lab.contracts import EvaluationSpec
from tsfm_lab.evaluation.windows import evaluation_bounds
from tsfm_lab.models.catalog import (
    CHECKPOINT_REPOSITORIES,
    catalog_info,
    checkpoint_info,
)
from tsfm_lab.models.settings import model_settings, validate_settings
from tsfm_lab.tui.forms import config_from_fields, default_fields, inspect_data
from tsfm_lab.workflows import PROFILE_NAMES

if TYPE_CHECKING:
    from tsfm_lab.assistant.drafts import ExperimentDraft

_REQUIRED = {
    "usage_profile",
    "adapter",
    "timestamp_column",
    "target_column",
    "frequency",
    "units",
    "context_length",
    "horizon",
    "stride",
    "mase_period",
    "evaluation_start",
    "evaluation_end",
    "point_statistic",
}


@dataclass(frozen=True)
class DraftResolution:
    state: str
    seed_fields: dict[str, str]
    explicit_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    errors: tuple[str, ...]
    assumptions: tuple[str, ...]


def _profile_required(fields):
    profile = fields.get("usage_profile")
    if profile == "finance":
        required = {"finance_target_kind"}
        if fields.get("finance_target_kind") == "other":
            required.add("finance_measurement")
        return required
    if profile == "energy":
        required = {"energy_measurement"}
        if fields.get("energy_measurement") == "other":
            required.add("energy_quantity")
        return required
    if profile == "weather":
        required = {
            "weather_variable",
            "weather_station",
            "weather_measurement_semantics",
        }
        if fields.get("weather_measurement_semantics") == "other":
            required.add("weather_description")
        return required
    return set()


def _model(fields, missing, errors):
    adapter = fields.get("adapter", "")
    if not adapter:
        return None
    try:
        entry = catalog_info(adapter)
    except ValueError as exc:
        errors.append(str(exc))
        return None
    if entry["adapter_id"] != adapter:
        errors.append("adapter must be a catalog adapter ID, not a repository")
        return None
    repository = fields.get("repository", "")
    variant = fields.get("checkpoint_variant", "")
    if adapter not in CHECKPOINT_REPOSITORIES:
        if repository or variant:
            errors.append("Built-in baselines do not accept checkpoint selections")
        if (
            adapter == "seasonal_naive"
            and not fields.get("seasonal_period", "").strip()
        ):
            missing.add("seasonal_period")
        return entry
    if not repository:
        missing.add("repository")
        return entry
    if repository not in entry["repositories"]:
        errors.append("Choose a curated repository belonging to the requested adapter")
        return entry
    choices = [item for item in entry["variants"] if item["repository"] == repository]
    if not variant and len(choices) > 1:
        missing.add("checkpoint_variant")
        return entry
    try:
        selected = checkpoint_info(repository, variant or None)
        fields.update(source="huggingface", revision=selected["reference_revision"])
        return catalog_info(repository, selected["variant"])
    except ValueError as exc:
        errors.append(str(exc))
        return entry


def _capabilities(fields, entry, errors):
    if entry is None:
        return
    try:
        settings = model_settings(
            fields["adapter"],
            repository=fields.get("repository") or None,
            variant=fields.get("checkpoint_variant") or None,
            revision=fields.get("revision") or None,
        )
    except ValueError as exc:
        if str(exc) not in errors:
            errors.append(str(exc))
        return
    numbers = {}
    for key in ("context_length", "horizon", "seasonal_period"):
        value = fields.get(key, "")
        if value and (not value.isascii() or not value.isdecimal()):
            errors.append(f"{key} must be a positive whole number")
            return
        if value:
            numbers[key] = int(value)
    try:
        task = fields.get("point_statistic") or settings.supported_tasks[0]
        if task not in settings.supported_tasks:
            raise ValueError("The requested forecast task is unsupported by this model")
        spec = EvaluationSpec(
            numbers.get("context_length", settings.default_context),
            numbers.get("horizon", settings.default_horizon),
            "2000-01-01T00:00:00Z",
            "2000-01-02T00:00:00Z",
            point_statistic=task,
        )
        validate_settings(
            {
                "adapter_id": fields["adapter"],
                **{
                    key: fields[key]
                    for key in ("repository", "revision")
                    if fields.get(key)
                },
                "options": {"seasonal_period": numbers.get("seasonal_period", 1)},
            },
            spec,
            fields.get("frequency") or None,
        )
    except ValueError as exc:
        errors.append(str(exc))


def resolve_draft(
    draft: ExperimentDraft, *, dataset_refs: Mapping[str, Path]
) -> DraftResolution:
    """Resolve local metadata only; guided preparation and confirmation remain required."""
    fields = dict(draft.fields)
    # Every proposal is retained and protected, even when its provenance was omitted.
    protected = set(fields)
    missing, errors = set(), []
    if draft.intent == "future_forecast":
        errors.append(
            "Future forecasts without observed actuals are unsupported; choose a historical evaluation explicitly"
        )
    elif draft.intent == "unclear":
        missing.add("intent")
    reference = draft.dataset_ref
    if reference is None or reference not in dataset_refs:
        missing.add("dataset_ref")
        if reference is not None:
            errors.append("Choose a dataset reference supplied by local setup")
    else:
        fields["dataset_path"] = str(
            Path(dataset_refs[reference]).expanduser().resolve()
        )
    required = _REQUIRED | _profile_required(fields)
    missing.update(key for key in required if not fields.get(key, "").strip())
    if fields.get("units", "").strip().casefold() == "unspecified":
        missing.add("units")
    if fields.get("usage_profile") and fields["usage_profile"] not in PROFILE_NAMES:
        errors.append("Choose a supported usage profile")
    entry = _model(fields, missing, errors)
    _capabilities(fields, entry, errors)
    if not missing and not errors:
        try:
            complete = {
                **default_fields(fields["adapter"]),
                **{
                    key: value
                    for key, value in fields.items()
                    if key != "checkpoint_variant"
                },
            }
            arguments = {"workspace": Path.cwd(), "base_directory": Path.cwd()}
            config = config_from_fields(complete, **arguments)
            dataset = inspect_data(complete, **arguments)
            evaluation_bounds(dataset, config.evaluation)
        except (ConfigError, ValueError, OSError, ImportError) as exc:
            errors.append(str(exc))
    protected.update(fields)
    state = "ready"
    if missing or errors:
        state = "needs_input"
    if draft.intent == "future_forecast":
        state = "unsupported"
    return DraftResolution(
        state,
        fields,
        tuple(sorted(protected)),
        tuple(sorted(missing)),
        tuple(errors),
        draft.assumptions,
    )

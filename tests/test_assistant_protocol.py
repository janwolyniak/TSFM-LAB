"""Private metadata stays local; annotation checks have explicit factual limits."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from finance_helpers import price_fixture

from tsfm_lab.assistant.evidence import (
    bounded_json,
    experiment_evidence,
    finance_evidence,
    model_evidence,
)
from tsfm_lab.assistant.protocol import (
    AssistanceError,
    decode_annotations,
    user_goal,
    validate_context,
)
from tsfm_lab.contracts import Dataset, EvaluationSpec
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.finance.backtest import backtest_run
from tsfm_lab.finance.reports import save_report
from tsfm_lab.models.baselines import LastValue
from tsfm_lab.models.catalog import catalog_entries

PRIVATE_PATHS = (
    "/Users/jan/private",
    "/123/private",
    "/żółw/private",
    "units:/Users/jan/private",
    "units=/123/private",
    r"\\server\share",
    r"units:\\server\share",
    "//server/share",
    r"C:\Users\private",
    "~/private",
    "../private",
    "file:///Users/jan/private",
    "private.csv",
    "settings.env",
)
FAKE_CREDENTIALS = (
    "sk-proj-fake-letters-only-credential",
    "sk-ant-api03-fake-letters-only-credential",
    "AIzaFakeLettersOnlyCredentialValue",
    "ya29.FakeLettersOnlyCredentialValue",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmYWtlIn0.ZmFrZXNpZ25hdHVyZQ",
    "Bearer fake-letters-only-credential",
    "api_key=fake-value",
    "password: fake-value",
    "client_secret=fake-value",
)


def context_with(value):
    context = model_evidence("last_value")
    context["data"]["declared_units"] = value
    return context


@pytest.mark.parametrize("value", [*PRIVATE_PATHS, *FAKE_CREDENTIALS])
def test_user_goals_reject_private_paths_and_obvious_foreign_credentials(value):
    with pytest.raises(AssistanceError) as error:
        user_goal(f"Explain {value}")
    assert error.value.code == "input"
    assert value not in str(error.value)


@pytest.mark.parametrize("value", [*PRIVATE_PATHS, *FAKE_CREDENTIALS])
def test_private_content_inside_allowed_metadata_cannot_be_previewed_or_rendered(value):
    with pytest.raises(AssistanceError) as error:
        validate_context(context_with(value))
    assert error.value.code == "input"
    assert value not in str(error.value)


@pytest.mark.parametrize(
    "value", ["m/s", "kg/m³", "W/m²", "kWh/day", "USD/asset", "°C", "s^-1", "1.25e-12"]
)
def test_legitimate_units_and_exact_numeric_text_are_preserved(value):
    source = context_with(value)
    checked = validate_context(source)
    assert checked == source
    assert checked["data"]["declared_units"] == value
    assert user_goal(f"Explain an error in {value}") == f"Explain an error in {value}"
    checked["data"]["declared_units"] = "changed locally"
    assert source["data"]["declared_units"] == value


def test_every_catalog_model_context_and_static_license_url_remain_supported():
    for entry in catalog_entries():
        context = model_evidence(entry["adapter_id"])
        assert validate_context(context) == context
        for variant in entry["variants"]:
            selected = model_evidence(variant["repository"], variant["variant"])
            assert validate_context(selected) == selected
    url = "https://github.com/NX-AI/mlstm_kernels/blob/v2.0.6/LICENSE"
    assert url in validate_context(model_evidence("tirex_2"))["data"]["runtime_notices"]
    assert user_goal(
        "Explain amazon/chronos-bolt-small and ibm-granite/granite-timeseries-ttm-r2"
    )


def test_checked_experiment_and_finance_contexts_keep_values_reasons_and_periods(
    tmp_path,
):
    times = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    dataset = Dataset(
        times, np.arange(8, dtype=float) * 0.12345678901234567, "1h", units="m/s"
    )
    result = run_experiment(
        dataset,
        LastValue(),
        EvaluationSpec(3, 2, times[4], times[-1], seasonal_period=8),
    )
    store = ExperimentStore(tmp_path / "workspace")
    run_id = store.create("local name", {}, "local source")
    store.complete(run_id, result.to_dict(), result.predictions)
    context = experiment_evidence(store, run_id)
    checked = validate_context(context)
    assert checked == context
    assert checked["data"]["metrics"]["overall"]["mase"] == {
        "value": None,
        "reason": "insufficient_scale_history",
    }
    assert (
        checked["data"]["configuration"]["evaluation"]["evaluation_start"]
        == times[4].isoformat()
    )
    fixture = price_fixture(tmp_path / "prices")
    report = backtest_run(
        fixture.run_id,
        fixture.workspace,
        fixture.dataset,
        fixture.ohlc,
        fixture.rules,
        target_is_price=True,
    )
    path = tmp_path / "saved.json"
    save_report(report, path)
    context = finance_evidence(path)
    assert validate_context(context) == context
    assert (
        validate_context(context)["data"]["capital_metrics"]["strategy"]
        == report["evaluation"]["strategy"]["metrics"]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"kind": "unvalidated"},
        {"kind": []},
        {"data": []},
        {"data": {}},
        {"extra": "field"},
        {"omitted": []},
        {"omitted": {"count": -1}},
        {"omitted": {"count": "1"}},
        {"glossary": []},
        {"glossary": {"meaning": 1}},
        {"limitations": "text"},
        {"limitations": [1]},
    ],
)
def test_context_requires_the_expected_versioned_envelope(changes):
    context = model_evidence("last_value")
    context.update(changes)
    with pytest.raises(AssistanceError) as error:
        validate_context(context)
    assert error.value.code == "input"


@pytest.mark.parametrize(
    "value",
    [
        "line\nbreak",
        "\x1b[31m",
        "invisible\u202econtrol",
        "x" * 513,
        float("nan"),
        {1: "bad key"},
    ],
)
def test_unsafe_or_oversize_context_values_fail_with_a_generic_local_error(value):
    with pytest.raises(AssistanceError) as error:
        validate_context(context_with(value))
    assert error.value.code == "input"


def test_missing_envelope_fields_unsafe_keys_and_total_size_are_rejected():
    context = model_evidence("last_value")
    del context["omitted"]
    with pytest.raises(AssistanceError):
        validate_context(context)
    for key in ("/123/private", "bad\x00key", "sk-proj-fake-letters-only-credential"):
        context = model_evidence("last_value")
        context["data"][key] = "safe value"
        with pytest.raises(AssistanceError):
            validate_context(context)
    context = model_evidence("last_value")
    context["data"]["many_fields"] = ["x" * 500] * 40
    with pytest.raises(AssistanceError):
        validate_context(context)


def annotation_context():
    context = model_evidence("last_value")
    context["data"] = {
        "error": {"value": 0.125, "reason": None},
        "units": "m/s",
        "horizon": 4,
    }
    return context


def response(text="The error is 0.125 m/s.", references=None, **changes):
    item = {
        "text": text,
        "evidence_ids": (
            references
            if references is not None
            else ["/data/error/value", "/data/units"]
        ),
    }
    value = {"schema_version": 1, "annotations": [item], **changes}
    return json.dumps(value)


def test_annotations_accept_exact_facts_and_equivalent_exponent_notation():
    context = annotation_context()
    assert decode_annotations(response(), context)[0].text == "The error is 0.125 m/s."
    assert (
        decode_annotations(response("The error is 1.25e-1 m/s."), context)[0].text
        == "The error is 1.25e-1 m/s."
    )
    assert decode_annotations(
        response("MAE is in target units.", ["/glossary/mae"]), context
    )
    assert json.loads(bounded_json(context))["data"]["error"]["value"] == 0.125


@pytest.mark.parametrize(
    "text,references",
    [
        ("The error is 0.13 m/s.", ["/data/error/value", "/data/units"]),
        ("The error is 12.5 percent.", ["/data/error/value"]),
        ("There are 4 observations.", ["/data/error/value"]),
        ("The model is accurate.", ["/data/unknown"]),
        ("The model is accurate.", []),
        ("The model is accurate.", ["/data/units", "/data/units"]),
        ("This is the best model.", ["/data/units"]),
        ("The model outperforms alternatives.", ["/data/units"]),
        ("Results are guaranteed.", ["/data/units"]),
        ("You should buy assets.", ["/data/units"]),
        ("Read /123/private.", ["/data/units"]),
        ("sk-proj-fake-letters-only-credential", ["/data/units"]),
        ("Hidden\u202econtrol", ["/data/units"]),
        ("Escape\x1bsequence", ["/data/units"]),
    ],
)
def test_unsupported_annotations_are_rejected_without_echoing_the_response(
    text, references
):
    with pytest.raises(AssistanceError) as error:
        decode_annotations(response(text, references), annotation_context())
    assert error.value.code == "response"
    assert text not in str(error.value)


@pytest.mark.parametrize(
    "value",
    [
        '{"schema_version":1,"schema_version":1,"annotations":[]}',
        '{"schema_version":1,"annotations":[{"text":"safe","evidence_ids":["/data/units"],"tool":"run"}]}',
        response(schema_version=True),
        response(annotations=[]),
        response(extra="unknown"),
    ],
)
def test_annotation_structure_cannot_add_actions_or_unknown_fields(value):
    with pytest.raises(AssistanceError):
        decode_annotations(value, annotation_context())


def test_valid_references_and_numbers_do_not_certify_arbitrary_prose_meaning():
    # A cited scalar can still be described incorrectly: these checks are not a
    # semantic proof, so the UI must continue to label wording as untrusted.
    source = annotation_context()
    before = copy.deepcopy(source)
    annotations = decode_annotations(
        response("MAE measures direction perfectly.", ["/glossary/mae"]), source
    )
    assert annotations[0].text == "MAE measures direction perfectly."
    assert source == before

"""Usage declarations remain explicit and never transform numerical data."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.contracts import Dataset
from tsfm_lab.data import canonical_fingerprint
from tsfm_lab.workflows import (
    PROFILE_NAMES,
    WorkflowError,
    display_workflow,
    normalize_workflow,
    ohlc_workflow_eligible,
    validate_workflow_dataset,
    validate_workflow_units,
    workflow_summary,
)


def declaration(profile="generic", **options):
    return {"schema_version": 1, "profile": profile, "options": options}


EXPLICIT = (
    declaration(),
    declaration("finance", target_kind="close_price"),
    declaration("finance", target_kind="return"),
    declaration("finance", target_kind="other", measurement="traded volume"),
    declaration("energy", measurement="power"),
    declaration("energy", measurement="energy"),
    declaration("energy", measurement="other", quantity="meter register"),
    declaration("temperature"),
    declaration(
        "weather",
        variable="wind speed",
        station="Station A",
        measurement_semantics="instantaneous",
    ),
)


@pytest.fixture
def dataset():
    return Dataset(
        pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
        np.array([4, 5, 3, 6, 7, 5, 8, 9]),
        "1h",
        target_name="observed",
        units="declared_units",
    )


def test_absent_workflow_preserves_legacy_metadata_and_generic_display():
    assert normalize_workflow(None) is None
    assert display_workflow(None) == workflow_summary(None) == "Generic"
    assert ohlc_workflow_eligible(None) is False
    assert PROFILE_NAMES == ("generic", "finance", "energy", "temperature", "weather")


@pytest.mark.parametrize("value", EXPLICIT)
def test_explicit_declarations_roundtrip_without_inventing_options(value):
    original = copy.deepcopy(value)
    normalized = normalize_workflow(value)
    assert normalized == json.loads(json.dumps(original))
    assert normalized is not value and normalized["options"] is not value["options"]
    assert value == original
    normalized["options"]["test_mutation"] = "local only"
    assert value == original


def test_measurement_text_and_station_identity_are_preserved_exactly():
    value = declaration(
        "weather",
        variable="Rainfall [measured]",
        station=" Site A / Sensor 7 ",
        measurement_semantics="other",
        description=" 24-hour bucket, local instrument label ",
    )
    assert normalize_workflow(value) == value
    summary = workflow_summary(value)
    assert value["options"]["variable"] in summary
    assert value["options"]["station"] in summary
    assert value["options"]["description"] in summary


def test_source_mutation_cannot_rewrite_a_normalized_declaration():
    source = declaration("finance", target_kind="other", measurement="volume")
    snapshot = normalize_workflow(source)
    source["profile"] = "generic"
    source["options"]["measurement"] = "revised interpretation"
    assert snapshot == declaration("finance", target_kind="other", measurement="volume")


@pytest.mark.parametrize(
    "value",
    [
        {},
        [],
        "temperature",
        True,
        {"schema_version": 1, "profile": "generic"},
        {**declaration(), "units": "K"},
        {**declaration(), "schema_version": True},
        {**declaration(), "schema_version": 1.0},
        {**declaration(), "schema_version": 2},
        {**declaration(), "profile": "Temperature"},
        {**declaration(), "profile": "humidity"},
        {**declaration(), "profile": None},
        {**declaration(), "options": None},
        {**declaration(), "options": []},
        declaration("generic", target_kind="close_price"),
        declaration("temperature", units="Celsius"),
        declaration("temperature", station="station A"),
        declaration("finance"),
        declaration("finance", target_kind="price"),
        declaration("finance", target_kind="other"),
        declaration("finance", target_kind="other", measurement=" "),
        declaration("finance", target_kind="close_price", measurement="close"),
        declaration("energy", measurement="watts"),
        declaration("energy", measurement="other"),
        declaration("energy", measurement="other", quantity=3),
        declaration("energy", measurement="power", quantity="power"),
        declaration("weather"),
        declaration("weather", variable="rain", station="A"),
        declaration(
            "weather", variable="rain", station="A", measurement_semantics="sum"
        ),
        declaration(
            "weather", variable="", station="A", measurement_semantics="interval_total"
        ),
        declaration(
            "weather",
            variable="rain",
            station=" ",
            measurement_semantics="interval_total",
        ),
        declaration(
            "weather", variable="rain", station="A", measurement_semantics="other"
        ),
        declaration(
            "weather",
            variable="rain",
            station="A",
            measurement_semantics="other",
            description=[],
        ),
        declaration(
            "weather",
            variable="rain",
            station="A",
            measurement_semantics="instantaneous",
            description="extra",
        ),
    ],
)
def test_missing_extra_or_malformed_profile_fields_are_refused(value):
    with pytest.raises(WorkflowError):
        normalize_workflow(value)


def test_native_json_types_are_required_and_cycles_fail_without_recursion():
    value = declaration("energy", measurement=np.str_("power"))
    with pytest.raises(WorkflowError):
        normalize_workflow(value)
    value = declaration()
    value["options"] = value
    with pytest.raises(WorkflowError):
        normalize_workflow(value)


@pytest.mark.parametrize("value", (None, *EXPLICIT))
def test_usage_validation_preserves_every_data_value_and_canonical_identity(
    dataset, value
):
    before = canonical_fingerprint(dataset)
    values, timestamps, units, target, frequency = (
        dataset.values.copy(),
        dataset.timestamps.copy(),
        dataset.units,
        dataset.target_name,
        dataset.frequency,
    )
    assert validate_workflow_dataset(value, dataset) is None
    np.testing.assert_array_equal(dataset.values, values)
    assert dataset.timestamps.equals(timestamps)
    assert (dataset.units, dataset.target_name, dataset.frequency) == (
        units,
        target,
        frequency,
    )
    assert canonical_fingerprint(dataset) == before


@pytest.mark.parametrize(
    "value, units",
    [
        (declaration("temperature"), "Celsius"),
        (declaration("temperature"), "Kelvin"),
        (declaration("temperature"), "sensor_native_units"),
        (declaration("energy", measurement="power"), "kW"),
        (declaration("energy", measurement="energy"), "kWh"),
        (declaration("finance", target_kind="return"), "log_return"),
        (
            declaration("finance", target_kind="other", measurement="signed flow"),
            "units",
        ),
        (EXPLICIT[-1], "m/s"),
    ],
)
def test_legitimate_negative_and_zero_values_remain_unmodified(value, units):
    dataset = Dataset(
        pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        [-12, 0, -1.5, 7],
        "1h",
        units=units,
    )
    original = dataset.values.copy()
    validate_workflow_dataset(value, dataset)
    np.testing.assert_array_equal(dataset.values, original)
    assert dataset.units == units


@pytest.mark.parametrize(
    "units", [None, "", " ", "unspecified", "UNSPECIFIED", " unspecified "]
)
@pytest.mark.parametrize("value", EXPLICIT[1:])
def test_domain_profiles_require_explicit_units(value, units):
    dataset = SimpleNamespace(values=[1, 2, 3], units=units)
    with pytest.raises(WorkflowError, match="explicit dataset units"):
        validate_workflow_dataset(value, dataset)
    with pytest.raises(WorkflowError, match="explicit dataset units"):
        validate_workflow_units(value, units)


@pytest.mark.parametrize("value", [None, declaration()])
def test_generic_workflow_keeps_unspecified_units_admissible(value, dataset):
    generic = Dataset(**{**vars(dataset), "units": "unspecified"})
    validate_workflow_dataset(value, generic)
    assert validate_workflow_units(value, generic.units) is None


@pytest.mark.parametrize(
    "values", [[], [0, 1], [-1, 2], [float("nan")], [float("inf")], ["12"]]
)
def test_close_price_declaration_refuses_nonpositive_or_nonfinite_observations(values):
    dataset = SimpleNamespace(values=values, units="quote_units_per_asset")
    with pytest.raises(WorkflowError, match="finite positive"):
        validate_workflow_dataset(
            declaration("finance", target_kind="close_price"), dataset
        )


def test_ohlc_route_depends_on_explicit_measurement_in_new_workflows():
    assert ohlc_workflow_eligible(None) is False
    assert ohlc_workflow_eligible(declaration("finance", target_kind="close_price"))
    for value in EXPLICIT:
        if value != declaration("finance", target_kind="close_price"):
            assert ohlc_workflow_eligible(value) is False
    with pytest.raises(WorkflowError):
        ohlc_workflow_eligible(declaration("finance", target_kind="price"))


def test_usage_labels_and_summaries_expose_declared_target_details():
    assert display_workflow(declaration("temperature")) == "Temperature"
    assert (
        workflow_summary(declaration("finance", target_kind="close_price"))
        == "Finance · close price"
    )
    assert (
        workflow_summary(declaration("energy", measurement="power")) == "Energy · power"
    )
    assert "traded volume" in workflow_summary(EXPLICIT[3])
    assert "meter register" in workflow_summary(EXPLICIT[6])


def test_workflow_contract_import_needs_only_standard_library():
    root = Path(__file__).parents[1]
    code = """
import importlib.abc
import sys
class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','pandas','yaml','torch','transformers','textual','huggingface_hub'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Blocked())
from tsfm_lab.workflows import normalize_workflow, display_workflow
assert normalize_workflow(None) is None
assert display_workflow(None) == 'Generic'
print('stdlib only')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stdlib only"

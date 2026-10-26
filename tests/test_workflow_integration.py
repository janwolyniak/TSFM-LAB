"""Usage metadata is persistent and checked without changing forecast arithmetic."""

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from finance_helpers import price_fixture

import tsfm_lab.application as application
from tsfm_lab.application import load_dataset, run_config, run_from_config
from tsfm_lab.artifact_validation import validate_predictions
from tsfm_lab.comparison import ComparisonError, compare_runs
from tsfm_lab.config import ConfigError, config_from_mapping, load_config
from tsfm_lab.contracts import Dataset
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.exports import export_predictions
from tsfm_lab.finance.backtest import backtest_run
from tsfm_lab.finance.contracts import FinanceError
from tsfm_lab.models.baselines import LastValue
from tsfm_lab.workflows import display_workflow

WORKFLOWS = (
    {"schema_version": 1, "profile": "generic", "options": {}},
    {
        "schema_version": 1,
        "profile": "finance",
        "options": {"target_kind": "close_price"},
    },
    {"schema_version": 1, "profile": "energy", "options": {"measurement": "power"}},
    {"schema_version": 1, "profile": "temperature", "options": {}},
    {
        "schema_version": 1,
        "profile": "weather",
        "options": {
            "variable": "wind speed",
            "station": "station A",
            "measurement_semantics": "interval_mean",
        },
    },
)


def resnapshot_artifact(store, run_id, filename, value):
    """Expose semantic tampering independently of the file checksum boundary."""
    directory = store.root / run_id
    raw = (json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    (directory / filename).write_bytes(raw)
    manifest = store.get_manifest(run_id)
    manifest["artifacts"][filename] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def direct_result(config, workflow=None):
    return run_experiment(
        load_dataset(config.dataset), LastValue(), config.evaluation, workflow=workflow
    )


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda value: value["profile"])
def test_each_profile_flows_from_yaml_through_run_reload_and_exports(
    config_factory, tmp_path, workflow, monkeypatch
):
    config = load_config(config_factory(workflow=workflow))
    assert config.workflow == workflow
    assert config.resolved_dict()["workflow"] == workflow
    summary = run_config(config)
    store = ExperimentStore(config.output_directory)
    saved, frame = store.load_result(summary["run_id"])
    assert saved["configuration"]["workflow"] == workflow
    assert store.load_configuration(summary["run_id"])["workflow"] == workflow
    assert saved["counts"]["successful_windows"] == 4
    assert len(frame) == 16
    assert saved["metrics"]["overall"]["mae"]["value"] == 2.5
    assert (
        display_workflow(saved["configuration"].get("workflow"))
        == workflow["profile"].title()
    )
    source = tmp_path / "seasonal.csv"
    source.rename(tmp_path / "source-unavailable.csv")
    monkeypatch.setattr(
        application,
        "build_model",
        lambda *args: pytest.fail("Saved reads must not build models"),
    )
    for format in ("json", "csv", "parquet"):
        target = tmp_path / f"forecasts.{format}"
        response = export_predictions(summary["run_id"], store.workspace, target)
        assert response["rows"] == 16
        if format == "json":
            exported = pd.DataFrame(json.loads(target.read_text()))
        elif format == "csv":
            exported = pd.read_csv(target, float_precision="round_trip")
        else:
            exported = pd.read_parquet(target)
        np.testing.assert_array_equal(exported.prediction, frame.prediction)
        assert "workflow" not in exported.columns


def test_absent_profile_retains_legacy_configuration_and_numerical_bytes(
    config_factory,
):
    config = load_config(config_factory())
    assert config.workflow is None
    assert set(config.resolved_dict()) == {
        "schema_version",
        "name",
        "model",
        "dataset",
        "evaluation",
        "runtime",
        "output",
    }
    assert "workflow" not in json.loads(json.dumps(config.resolved_dict()))
    absent = run_experiment(
        load_dataset(config.dataset), LastValue(), config.evaluation
    )
    explicit_none = direct_result(config, None)
    assert set(absent.configuration) == {"model", "dataset", "evaluation", "runtime"}
    assert absent.configuration == explicit_none.configuration
    assert absent.scores == explicit_none.scores
    assert (
        absent.predictions.to_csv(index=False).encode()
        == explicit_none.predictions.to_csv(index=False).encode()
    )
    summary = run_config(config)
    store = ExperimentStore(config.output_directory)
    directory = store.root / summary["run_id"]
    assert "workflow" not in json.loads((directory / "configuration.json").read_bytes())
    assert "workflow" not in store.load_result(summary["run_id"])[0]["configuration"]
    assert (
        display_workflow(store.load_configuration(summary["run_id"]).get("workflow"))
        == "Generic"
    )


@pytest.mark.parametrize("task", ["median", "native_point"])
def test_profile_only_changes_keep_comparison_and_all_prediction_metric_bytes(
    config_factory, task
):
    configurations = [
        load_config(
            config_factory(
                **({"workflow": workflow} if workflow is not None else {}),
                evaluation={"point_statistic": task},
            )
        )
        for workflow in (None, *WORKFLOWS)
    ]
    summaries = [run_config(config) for config in configurations]
    store = ExperimentStore(configurations[0].output_directory)
    identifiers = [summary["run_id"] for summary in summaries]
    first, first_predictions = store.load_result(identifiers[0])
    first_directory = store.root / identifiers[0]
    for identifier in identifiers[1:]:
        result, predictions = store.load_result(identifier)
        assert (
            result["provenance"]["dataset"]["canonical_sha256"]
            == first["provenance"]["dataset"]["canonical_sha256"]
        )
        assert result["metrics"] == first["metrics"]
        assert result["counts"] == first["counts"]
        pd.testing.assert_frame_equal(predictions, first_predictions)
        directory = store.root / identifier
        assert (directory / "predictions.csv").read_bytes() == (
            first_directory / "predictions.csv"
        ).read_bytes()
        assert (directory / "metrics.json").read_bytes() == (
            first_directory / "metrics.json"
        ).read_bytes()
    comparison = compare_runs(identifiers, store.workspace)
    assert comparison["compatible"]
    assert comparison["comparison_policy"] == (
        "controlled_point_v1" if task == "median" else "controlled_native_point_v1"
    )


def test_profile_does_not_hide_changed_units_or_forecast_tasks(config_factory):
    first = load_config(config_factory(workflow=WORKFLOWS[2]))
    changed_units = replace(first, dataset={**first.dataset, "units": "other_units"})
    first_id = run_config(first)["run_id"]
    changed_id = run_config(changed_units)["run_id"]
    with pytest.raises(ComparisonError, match="different task settings"):
        compare_runs([first_id, changed_id], first.output_directory)
    native = replace(
        first, evaluation=replace(first.evaluation, point_statistic="native_point")
    )
    native_id = run_config(native)["run_id"]
    with pytest.raises(ComparisonError, match="different task settings"):
        compare_runs([first_id, native_id], first.output_directory)


@pytest.mark.parametrize("workflow", WORKFLOWS[1:], ids=lambda value: value["profile"])
def test_missing_domain_units_fail_before_dataset_read_or_model_construction(
    config_factory, tmp_path, monkeypatch, workflow
):
    path = config_factory(workflow=workflow, dataset={"units": "unspecified"})
    monkeypatch.setattr(
        application,
        "load_dataset",
        lambda *args: pytest.fail("Units must fail at config validation"),
    )
    monkeypatch.setattr(
        application,
        "build_model",
        lambda *args: pytest.fail("No model may be constructed"),
    )
    with pytest.raises(ConfigError, match="explicit dataset units"):
        run_from_config(path)
    assert not (tmp_path / "workspace").exists()


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_close_history_fails_before_model_and_run_artifacts(
    config_factory, tmp_path, monkeypatch, value
):
    config = load_config(config_factory(workflow=WORKFLOWS[1]))
    source = tmp_path / "seasonal.csv"
    frame = pd.read_csv(source)
    frame.loc[0, "demand"] = value
    frame.to_csv(source, index=False)
    monkeypatch.setattr(
        application,
        "build_model",
        lambda *args: pytest.fail(
            "Close-price data must fail before model construction"
        ),
    )
    with pytest.raises(ConfigError, match="finite positive"):
        run_config(config)
    assert not config.output_directory.exists()


def test_direct_runner_refuses_invalid_workflow_before_model_load(config_factory):
    config = load_config(config_factory())
    data = load_dataset(config.dataset)
    data = Dataset(**{**vars(data), "values": np.full(len(data.values), -2.0)})

    class Trap(LastValue):
        def metadata(self):
            pytest.fail("Workflow validation must precede model inspection")

        def load(self):
            pytest.fail("Workflow validation must precede model loading")

    with pytest.raises(ValueError, match="finite positive"):
        run_experiment(data, Trap(), config.evaluation, workflow=WORKFLOWS[1])


def test_negative_temperature_values_and_unit_identity_are_preserved(config_factory):
    config = load_config(
        config_factory(workflow=WORKFLOWS[3], dataset={"units": "Celsius"})
    )
    dataset = load_dataset(config.dataset)
    dataset = Dataset(**{**vars(dataset), "values": dataset.values - 30})
    generic = run_experiment(dataset, LastValue(), config.evaluation)
    temperature = run_experiment(
        dataset, LastValue(), config.evaluation, workflow=config.workflow
    )
    assert temperature.predictions.prediction.lt(0).all()
    pd.testing.assert_frame_equal(temperature.predictions, generic.predictions)
    assert temperature.scores == generic.scores
    assert temperature.provenance["dataset"] == generic.provenance["dataset"]
    assert temperature.configuration["dataset"]["units"] == "Celsius"


def test_config_and_runner_snapshot_caller_workflow_without_mutation(config_factory):
    config = load_config(config_factory(workflow=WORKFLOWS[4]))
    mapping = config.resolved_dict()
    parsed = config_from_mapping(mapping, base_directory=config.source_path.parent)
    mapping["workflow"]["options"]["station"] = "changed caller mapping"
    assert parsed.workflow["options"]["station"] == "station A"
    supplied = copy.deepcopy(parsed.workflow)
    result = direct_result(parsed, supplied)
    supplied["options"]["station"] = "changed caller object"
    assert result.configuration["workflow"]["options"]["station"] == "station A"
    parsed.resolved_dict()["workflow"]["options"][
        "station"
    ] = "changed exported mapping"
    assert parsed.workflow["options"]["station"] == "station A"


@pytest.mark.parametrize("bad", [[], None, "wrong"])
def test_malformed_saved_configuration_uses_store_error(config_factory, bad):
    config = load_config(config_factory(workflow=WORKFLOWS[0]))
    payload = direct_result(config, config.workflow).to_dict()
    payload["configuration"] = bad
    store = ExperimentStore(config.output_directory)
    run_id = store.create(config.name, config.resolved_dict(), config.source_yaml)
    with pytest.raises(StoreError):
        store.complete(
            run_id, payload, direct_result(config, config.workflow).predictions
        )
    assert store.get_manifest(run_id)["status"] == "pending"


@pytest.mark.parametrize(
    "workflow",
    [
        {**WORKFLOWS[0], "schema_version": True},
        {**WORKFLOWS[0], "profile": "wrong"},
        {**WORKFLOWS[2], "options": {}},
        {**WORKFLOWS[3], "options": {"convert_to": "Kelvin"}},
        {
            **WORKFLOWS[4],
            "options": {
                "variable": "rainfall",
                "station": "A",
                "measurement_semantics": "other",
            },
        },
    ],
)
def test_saved_usage_schema_is_checked_before_accepting_predictions(
    config_factory, workflow
):
    config = load_config(config_factory())
    result = direct_result(config)
    payload = result.to_dict()
    payload["configuration"]["workflow"] = workflow
    with pytest.raises(StoreError):
        validate_predictions(result.predictions, payload)


def test_saved_close_price_actuals_cannot_be_nonpositive(config_factory):
    config = load_config(config_factory(workflow=WORKFLOWS[1]))
    result = direct_result(config, config.workflow)
    frame = result.predictions.copy()
    frame.loc[0, "actual"] = 0
    with pytest.raises(StoreError, match="strictly positive"):
        validate_predictions(frame, json.loads(json.dumps(result.to_dict())))


def test_close_price_profile_does_not_clip_negative_forecasts(config_factory):
    config = load_config(config_factory(workflow=WORKFLOWS[1]))

    class NegativeForecast(LastValue):
        adapter_id = "negative_point_fixture"

        def metadata(self):
            return replace(super().metadata(), family="test_fixture")

        def predict(self, requests):
            return tuple(
                replace(output, point=-np.abs(output.point))
                for output in super().predict(requests)
            )

    result = run_experiment(
        load_dataset(config.dataset),
        NegativeForecast(),
        config.evaluation,
        workflow=config.workflow,
    )
    assert result.predictions.prediction.lt(0).all()
    store = ExperimentStore(config.output_directory)
    run_id = store.create(
        "Negative forecast fixture", result.configuration, "# Forecast accuracy only"
    )
    store.complete(run_id, result.to_dict(), result.predictions)
    pd.testing.assert_frame_equal(store.load_result(run_id)[1], result.predictions)


@pytest.mark.parametrize(
    "initial, final",
    [(None, WORKFLOWS[0]), (WORKFLOWS[0], None), (WORKFLOWS[0], WORKFLOWS[3])],
)
def test_initial_and_result_profile_mismatch_fails_before_writes_or_status_change(
    config_factory, initial, final
):
    config = load_config(
        config_factory(**({"workflow": initial} if initial is not None else {}))
    )
    result = direct_result(config, final)
    store = ExperimentStore(config.output_directory)
    run_id = store.create(config.name, config.resolved_dict(), config.source_yaml)
    manifest_before = (store.root / run_id / "manifest.json").read_bytes()
    with pytest.raises(StoreError, match="initial configuration"):
        store.complete(run_id, result.to_dict(), result.predictions)
    assert (store.root / run_id / "manifest.json").read_bytes() == manifest_before
    assert {file.name for file in (store.root / run_id).iterdir()} == {
        "manifest.json",
        "configuration.json",
        "input.yaml",
    }


@pytest.mark.parametrize("mutation", ["replace", "remove", "malformed_configuration"])
def test_resigned_saved_profile_tampering_is_rejected_against_initial_config(
    config_factory, mutation
):
    config = load_config(config_factory(workflow=WORKFLOWS[3]))
    summary = run_config(config)
    store = ExperimentStore(config.output_directory)
    saved, _ = store.load_result(summary["run_id"])
    if mutation == "replace":
        saved["configuration"]["workflow"] = copy.deepcopy(WORKFLOWS[0])
    elif mutation == "remove":
        saved["configuration"].pop("workflow")
    else:
        saved["configuration"] = None
    resnapshot_artifact(store, summary["run_id"], "result.json", saved)
    with pytest.raises(StoreError):
        store.load_result(summary["run_id"])
    assert store.list_runs()[0]["status"] == "corrupt"


@pytest.mark.parametrize("status", ["pending", "failed", "cancelled", "completed"])
def test_checked_configuration_is_detached_and_available_through_lifecycle(
    config_factory, status
):
    config = load_config(config_factory(workflow=WORKFLOWS[4]))
    store = ExperimentStore(config.output_directory)
    run_id = store.create(config.name, config.resolved_dict(), config.source_yaml)
    if status == "completed":
        result = direct_result(config, config.workflow)
        store.complete(run_id, result.to_dict(), result.predictions)
    elif status != "pending":
        store.set_status(run_id, status, {"message": "disposable test"})
    first = store.load_configuration(run_id)
    first["workflow"]["options"]["station"] = "changed caller snapshot"
    first["dataset"]["units"] = "changed caller units"
    second = store.load_configuration(run_id)
    assert second == config.resolved_dict()
    assert store.get_manifest(run_id)["status"] == status


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_configuration_checksum_failure_is_detected_without_source_data(
    config_factory, status
):
    config = load_config(config_factory(workflow=WORKFLOWS[2]))
    store = ExperimentStore(config.output_directory)
    run_id = store.create(config.name, config.resolved_dict(), config.source_yaml)
    if status == "failed":
        store.set_status(run_id, "failed")
    path = store.root / run_id / "configuration.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(StoreError, match="checksum|size"):
        store.load_configuration(run_id)


def test_resigned_malformed_pending_workflow_is_refused(config_factory):
    config = load_config(config_factory(workflow=WORKFLOWS[2]))
    store = ExperimentStore(config.output_directory)
    run_id = store.create(config.name, config.resolved_dict(), config.source_yaml)
    initial = config.resolved_dict()
    initial["workflow"]["options"] = {"measurement": "other"}
    resnapshot_artifact(store, run_id, "configuration.json", initial)
    with pytest.raises(StoreError, match="workflow"):
        store.load_configuration(run_id)


def test_legacy_pending_configuration_load_does_not_add_workflow_metadata(tmp_path):
    store = ExperimentStore(tmp_path / "workspace")
    initial = {"dataset": "legacy direct-Python metadata"}
    run_id = store.create("Old direct fixture", initial, "# Legacy setup")
    assert store.load_configuration(run_id) == initial
    assert "workflow" not in store.load_configuration(run_id)


@pytest.mark.parametrize(
    "workflow",
    (None, *WORKFLOWS),
    ids=lambda value: "legacy" if value is None else value["profile"],
)
def test_manual_ohlc_policy_is_unchanged_by_usage_labels(tmp_path, workflow):
    value = price_fixture(tmp_path)
    store = ExperimentStore(value.workspace)
    result = value.result.to_dict()
    if workflow is not None:
        result["configuration"]["workflow"] = copy.deepcopy(workflow)
    run_id = store.create(
        "Usage-labelled price fixture",
        result["configuration"],
        "# Explicit report inputs remain required",
    )
    store.complete(run_id, result, value.result.predictions)
    report = backtest_run(
        run_id,
        value.workspace,
        value.dataset,
        value.ohlc,
        value.rules,
        target_is_price=True,
    )
    original = backtest_run(
        value.run_id,
        value.workspace,
        value.dataset,
        value.ohlc,
        value.rules,
        target_is_price=True,
    )
    assert report["policy"] == original["policy"]
    assert report["cases"] == original["cases"]
    assert report["evaluation"] == original["evaluation"]
    with pytest.raises(FinanceError, match="Explicitly declare"):
        backtest_run(
            run_id,
            value.workspace,
            value.dataset,
            value.ohlc,
            value.rules,
            target_is_price=False,
        )


def test_close_price_usage_cannot_enable_ohlc_for_a_native_point_task(tmp_path):
    value = price_fixture(tmp_path)
    store = ExperimentStore(value.workspace)
    payload = value.result.to_dict()
    payload["configuration"]["workflow"] = copy.deepcopy(WORKFLOWS[1])
    payload["configuration"]["evaluation"]["point_statistic"] = "native_point"
    run_id = store.create(
        "Native close prices", payload["configuration"], "# Direct fixture"
    )
    store.complete(run_id, payload, value.result.predictions)
    with pytest.raises(FinanceError, match="median task"):
        backtest_run(
            run_id,
            value.workspace,
            value.dataset,
            value.ohlc,
            value.rules,
            target_is_price=True,
        )

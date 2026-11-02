"""Qualification orchestration with fixtures only; no pretrained inference."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_v030_models.py"
SPEC = importlib.util.spec_from_file_location("qualify_v030_models", SCRIPT)
QUALIFICATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QUALIFICATION
SPEC.loader.exec_module(QUALIFICATION)


@pytest.mark.parametrize("profile", QUALIFICATION.PROFILES)
def test_profiles_bind_exact_configurations_and_full_native_horizons(profile):
    chosen = QUALIFICATION.selection(profile)
    path = (
        ROOT / "tests/fixtures/toto2/config.json"
        if profile == "toto-4m"
        else ROOT / f"tests/fixtures/ttm_s6/{profile}.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == chosen["config_sha256"]
    assert len(chosen["weight_sha256"]) == 64
    assert len(chosen["revision"]) == 40
    if profile == "toto-4m":
        assert chosen["context"] == 64 and chosen["horizon"] == 65
    else:
        assert chosen["horizon"] == int(profile.split("-")[1])


@pytest.mark.parametrize("profile", QUALIFICATION.PROFILES)
def test_causal_fixture_changes_only_targets_and_future_retains_exact_context(
    tmp_path, profile
):
    from tsfm_lab.config import config_from_mapping
    from tsfm_lab.future_config import future_config_from_mapping

    chosen = QUALIFICATION.selection(profile)
    model = {
        "source": "huggingface",
        "adapter_id": chosen["adapter"],
        "repository": chosen["repository"],
        "revision": chosen["revision"],
        "offline": True,
    }
    raw, future = QUALIFICATION.fixture(tmp_path, chosen, model)
    historical = config_from_mapping(raw, base_directory=tmp_path)
    forecast = future_config_from_mapping(future, base_directory=tmp_path)
    assert historical.evaluation.horizon == forecast.forecast.horizon
    original = pd.read_csv(tmp_path / "input.csv")
    causal = pd.read_csv(tmp_path / "causal.csv")
    history = pd.read_csv(tmp_path / "history.csv")
    context = chosen["context"]
    pd.testing.assert_frame_equal(original.iloc[:context], causal.iloc[:context])
    pd.testing.assert_frame_equal(original.iloc[:context], history)
    np.testing.assert_allclose(
        causal.value.iloc[context:] - original.value.iloc[context:], 1000
    )


def test_missing_or_ambiguous_checkpoint_inputs_are_not_implicit(tmp_path):
    with pytest.raises(SystemExit):
        QUALIFICATION.main(
            [
                "--profile",
                "toto-4m",
                "--wheel",
                "candidate.whl",
                "--output-dir",
                str(tmp_path / "new"),
            ]
        )
    with pytest.raises(SystemExit):
        QUALIFICATION.main(
            [
                "--profile",
                "toto-4m",
                "--wheel",
                "candidate.whl",
                "--output-dir",
                str(tmp_path / "new"),
                "--cache-dir",
                "cache",
                "--checkpoint",
                "weights",
            ]
        )
    assert not (tmp_path / "new").exists()


def test_existing_evidence_directory_is_never_reused(tmp_path):
    sentinel = tmp_path / "existing.json"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        QUALIFICATION.main(
            [
                "--profile",
                "toto-4m",
                "--wheel",
                "candidate.whl",
                "--cache-dir",
                "cache",
                "--output-dir",
                str(tmp_path),
            ]
        )
    assert sentinel.read_text() == "keep"


def test_failed_qualification_retains_a_machine_readable_failure(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise ValueError("explicit checkpoint is unavailable")

    monkeypatch.setattr(QUALIFICATION, "qualify", failure)
    monkeypatch.setattr(
        QUALIFICATION, "installed_identity", lambda *args: {"fixture": True}
    )
    monkeypatch.setattr(QUALIFICATION, "os", SimpleNamespace(environ={}))
    directory = tmp_path / "new"
    result = QUALIFICATION.main(
        [
            "--profile",
            "toto-4m",
            "--wheel",
            "candidate.whl",
            "--checkpoint",
            "missing",
            "--output-dir",
            str(directory),
        ]
    )
    report = json.loads((directory / "qualification.json").read_text())
    assert result == 1 and report["status"] == "failed"
    assert report["error"]["message"] == "explicit checkpoint is unavailable"
    assert report["real_inference_requested"] is True
    assert "real_inference" not in report


def test_deadline_always_closes_the_managed_worker(tmp_path, monkeypatch):
    from tsfm_lab import execution

    closed = []

    class Worker:
        is_running = True
        log_path = None

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(execution, "ManagedRun", Worker)
    config = SimpleNamespace(name="deadline", output_directory=tmp_path / "workspace")
    with pytest.raises(ValueError, match="deadline"):
        QUALIFICATION.managed(config, -1)
    assert closed == [True]
    assert json.loads((tmp_path / "deadline-events.json").read_text()) == []


def test_detached_exports_need_no_input_and_preserve_records(tmp_path):
    from tsfm_lab.application import run_config
    from tsfm_lab.config import config_from_mapping
    from tsfm_lab.future_application import run_future_config
    from tsfm_lab.future_config import future_config_from_mapping

    chosen = {"context": 8, "horizon": 4, "frequency": "1h", "statistic": "median"}
    raw, future = QUALIFICATION.fixture(
        tmp_path, chosen, {"source": "builtin", "adapter_id": "last_value"}
    )
    raw["runtime"]["dtype"] = future["runtime"]["dtype"] = "float64"
    historical = run_config(config_from_mapping(raw, base_directory=tmp_path))
    forecast = run_future_config(
        future_config_from_mapping(future, base_directory=tmp_path)
    )
    for name in ("input.csv", "history.csv", "causal.csv"):
        (tmp_path / name).rename(tmp_path / f"unavailable-{name}")
    result = QUALIFICATION.detached(
        tmp_path / "workspace", [historical["run_id"], forecast["run_id"]], tmp_path
    )
    formats = ["csv", "json"]
    if importlib.util.find_spec("pyarrow") is not None:
        formats.append("parquet")
    assert result == {
        "records": 2,
        "exports": 2 * len(formats),
        "formats": formats,
        "saved_bytes_unchanged": True,
    }


def test_installed_probe_refuses_source_checkout_before_wheel_claims(monkeypatch):
    monkeypatch.setattr(
        QUALIFICATION,
        "sys",
        SimpleNamespace(
            flags=SimpleNamespace(isolated=1),
            prefix=str(ROOT / "src"),
            base_prefix="/different",
        ),
    )
    with pytest.raises(ValueError, match="Source checkouts"):
        QUALIFICATION.installed_identity("candidate.whl")


@pytest.mark.parametrize(
    "actual,target,accepted",
    [
        ("0.3.0", None, True),
        ("0.3.1", None, False),
        ("0.3.1", "0.3.1", True),
        ("0.3.0", "0.3.1", False),
    ],
)
def test_patch_wheel_qualification_requires_explicit_matching_target(
    tmp_path, monkeypatch, actual, target, accepted
):
    import verify_release_identity

    import tsfm_lab

    environment = tmp_path / "environment"
    package = environment / "lib/site-packages/tsfm_lab"
    monkeypatch.setattr(tsfm_lab, "__file__", str(package / "__init__.py"))
    monkeypatch.setattr(
        QUALIFICATION,
        "sys",
        SimpleNamespace(
            flags=SimpleNamespace(isolated=1),
            prefix=str(environment),
            base_prefix="/different",
        ),
    )
    calls = []

    def verify(wheel, origin):
        calls.append((wheel, origin))
        return {"version": actual}

    monkeypatch.setattr(verify_release_identity, "verify_installed", verify)
    options = {} if target is None else {"expected_version": target}
    if accepted:
        assert QUALIFICATION.installed_identity("exact.whl", **options) == {
            "version": actual
        }
    else:
        with pytest.raises(ValueError, match="qualification target"):
            QUALIFICATION.installed_identity("exact.whl", **options)
    assert calls == [("exact.whl", package)]


def test_registered_native_checks_match_retained_strict_schema_evidence():
    from tsfm_lab.models import verification
    from tsfm_lab.models._verification_evidence import PROJECT_CHECKS

    report = json.loads((ROOT / "docs/v0.3.0-model-qualification.json").read_text())
    recorded = [item for item in PROJECT_CHECKS if item["id"].startswith("v030-")]
    assert recorded == report["project_checks"] and len(recorded) == 10
    for item in recorded:
        verification._check(item, item["adapter_id"])
        assert item["source"]["package_version"] == "0.3.0"
        assert (
            item["source"]["git_revision"] == report["frozen_release"]["git_revision"]
        )
        if item["adapter_id"] == "toto_2":
            assert item["runtime"]["scipy"] == "1.14.1"
    assert report["clean_cli_exports"]["records"] == 57
    assert report["clean_cli_exports"]["exports"] == 171


def test_native_inverse_projection_matches_explicit_ttm_original_unit_arithmetic():
    from contextlib import nullcontext

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    observed = []

    def native(**arguments):
        observed.append(arguments)
        return SimpleNamespace(prediction_outputs=Tensor(np.array([[[1.0], [-1.0]]])))

    model = SimpleNamespace(
        adapter_id="ttm_r2",
        _model=native,
        _torch=SimpleNamespace(
            device=lambda _: nullcontext(),
            inference_mode=nullcontext,
            as_tensor=lambda value, **kw: np.asarray(value),
            float32="float32",
            long="int64",
        ),
        configuration=lambda: {
            "checkpoint": {"point_forecast": {"frequency_token": 9}}
        },
    )
    request = SimpleNamespace(
        context_values=np.array([10.0, 12.0]), target_timestamps=[1, 2]
    )
    point = QUALIFICATION.native_projection(model, request)
    np.testing.assert_array_equal(point, [12.0, 10.0])
    np.testing.assert_array_equal(observed[0]["past_values"][0, :, 0], [-1.0, 1.0])
    assert observed[0]["freq_token"].tolist() == [9]

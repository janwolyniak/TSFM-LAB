"""Assisted suggestions only prefill the ordinary inspected and confirmed guide."""

from collections import defaultdict, deque
from types import SimpleNamespace

import pandas as pd
import pytest

from tsfm_lab.config import ConfigError
from tsfm_lab.contracts import ModelCapabilities
from tsfm_lab.repl import wizard


class Questions:
    def __init__(self, answers=None):
        self.answers = defaultdict(deque)
        self.calls = []
        for label, values in (answers or {}).items():
            self.answers[label].extend(values if isinstance(values, list) else [values])

    def __call__(self, label, default="", choices=()):
        assert len(self.calls) < 200, "Setup did not reach a review or cancellation"
        self.calls.append((label, default, choices))
        return self.answers[label].popleft() if self.answers[label] else ""


def build(tmp_path, questions, emitted=None, **kwargs):
    return wizard.build_experiment(
        questions,
        (emitted if emitted is not None else []).append,
        workspace=tmp_path / "results",
        demo_path=tmp_path / "seasonal.csv",
        **kwargs,
    )


def seed(tmp_path, **updates):
    return {
        "name": "Requested evaluation",
        "usage_profile": "generic",
        "dataset_path": str(tmp_path / "seasonal.csv"),
        "units": "declared_units",
        "context_length": "6",
        "horizon": "2",
        "stride": "1",
        "mase_period": "3",
        "evaluation_start": "2026-01-01T10:00:00Z",
        "evaluation_end": "2026-01-02T06:00:00Z",
        **updates,
    }


def defaults(questions):
    return {label: default for label, default, _ in questions.calls}


def test_every_seed_survives_even_when_provider_marks_none_explicit(
    config_factory, tmp_path
):
    config_factory()
    questions = Questions({"Run experiment?": "yes"})
    config = build(tmp_path, questions, seed_fields=seed(tmp_path), explicit_fields=())
    assert config.name == "Requested evaluation"
    assert config.dataset["units"] == "declared_units"
    assert config.evaluation.context_length == 6
    assert config.evaluation.horizon == 2
    assert config.evaluation.stride == 1
    assert config.evaluation.seasonal_period == 3
    assert config.evaluation.evaluation_start == pd.Timestamp("2026-01-01T10:00Z")
    assert defaults(questions)["Forecast horizon"] == "2"
    assert not (tmp_path / "results").exists()


def test_seeded_demo_mapping_remains_editable_and_is_never_reset(
    config_factory, tmp_path
):
    config_factory()
    pd.DataFrame(
        {
            "when": pd.date_range("2026-01-01", periods=32, freq="30min"),
            "load": range(32),
        }
    ).to_csv(tmp_path / "seasonal.csv", index=False)
    questions = Questions({"Run experiment?": "yes"})
    config = build(
        tmp_path,
        questions,
        seed_fields=seed(
            tmp_path,
            timestamp_column="when",
            target_column="load",
            frequency="30min",
            timezone="UTC",
            evaluation_start="2026-01-01T05:00:00Z",
            evaluation_end="2026-01-01T15:00:00Z",
        ),
    )
    assert config.dataset["target_column"] == "load"
    assert config.dataset["timestamp_column"] == "when"
    assert config.dataset["frequency"] == "30min"
    assert defaults(questions)["Timestamp column"] == "when"
    assert defaults(questions)["Target column"] == "load"


def test_invalid_horizon_is_presented_for_correction_without_clamping(
    config_factory, tmp_path
):
    config_factory()
    questions = Questions({"Forecast horizon": ["", "2"], "Run experiment?": "yes"})
    emitted = []
    config = build(
        tmp_path, questions, emitted, seed_fields=seed(tmp_path, horizon="500")
    )
    horizon_calls = [call for call in questions.calls if call[0] == "Forecast horizon"]
    assert [call[1] for call in horizon_calls] == ["500", "500"]
    assert config.evaluation.horizon == 2
    assert any("Forecast horizon must be between" in line for line in emitted)


def test_ttm_context_conflict_requires_user_correction(
    config_factory, tmp_path, monkeypatch
):
    config_factory()
    dates = pd.date_range("2026-01-01", periods=1100, freq="1h")
    pd.DataFrame({"timestamp": dates, "demand": range(1100)}).to_csv(
        tmp_path / "seasonal.csv", index=False
    )

    class Model:
        def metadata(self):
            return SimpleNamespace(
                capabilities=ModelCapabilities(
                    min_context=1024, max_context=1024, max_horizon=96
                )
            )

        def validate(self, spec):
            assert spec.context_length == 1024

    monkeypatch.setattr(wizard, "build_model", lambda *_: Model())
    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", lambda *_: Model())
    questions = Questions({"Context length": ["", "1024"], "Run experiment?": "yes"})
    emitted = []
    config = build(
        tmp_path,
        questions,
        emitted,
        seed_fields=seed(
            tmp_path,
            adapter="ttm_r2",
            repository="ibm-granite/granite-timeseries-ttm-r2",
            checkpoint_variant="1024-96-r2",
            point_statistic="native_point",
            context_length="512",
            evaluation_start=dates[1024].tz_localize("UTC").isoformat(),
            evaluation_end=(dates[-1] + pd.Timedelta(hours=1))
            .tz_localize("UTC")
            .isoformat(),
        ),
    )
    context_calls = [call for call in questions.calls if call[0] == "Context length"]
    assert [call[1] for call in context_calls] == ["512", "512"]
    assert config.evaluation.context_length == 1024
    assert config.model["revision"] == "1a382618f3c5b63836281ad632e54724e55e85a5"
    assert any("between 1024 and 1024" in line for line in emitted)


def test_single_repository_variant_conflict_is_an_ordinary_question(
    config_factory, tmp_path, monkeypatch
):
    config_factory()

    class Model:
        def metadata(self):
            return SimpleNamespace(capabilities=ModelCapabilities(max_horizon=320))

        def validate(self, _spec):
            pass

    monkeypatch.setattr(wizard, "build_model", lambda *_: Model())
    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", lambda *_: Model())
    questions = Questions(
        {"Training profile": ["", "Default training profile"], "Run experiment?": "yes"}
    )
    config = build(
        tmp_path,
        questions,
        seed_fields=seed(
            tmp_path,
            adapter="tirex_2",
            repository="NX-AI/TiRex-2",
            checkpoint_variant="invented",
        ),
    )
    assert config.model["repository"] == "NX-AI/TiRex-2"
    assert defaults(questions)["Training profile"] == "invented"


def test_missing_units_require_input_and_final_confirmation_still_defaults_no(
    config_factory, tmp_path, monkeypatch
):
    config_factory()
    original = wizard.prepare_experiment
    prepared = []

    def inspect(*args, **kwargs):
        result = original(*args, **kwargs)
        prepared.append(result)
        return result

    monkeypatch.setattr(wizard, "prepare_experiment", inspect)
    fields = seed(tmp_path)
    fields.pop("units")
    questions = Questions({"Target units": ["", "kW"]})
    assert build(tmp_path, questions, seed_fields=fields) is None
    assert prepared and prepared[0].config.dataset["units"] == "kW"
    assert defaults(questions)["Run experiment?"] == "no"
    assert not (tmp_path / "results").exists()


def test_seeded_setup_can_cancel_without_preparation(
    config_factory, tmp_path, monkeypatch
):
    config_factory()
    monkeypatch.setattr(
        wizard, "prepare_experiment", lambda *_args, **_kwargs: pytest.fail("cancelled")
    )
    questions = Questions({"Usage profile": "/cancel"})
    assert build(tmp_path, questions, seed_fields=seed(tmp_path)) is None


def test_even_an_empty_assisted_seed_cannot_bypass_guided_setup(tmp_path):
    with pytest.raises(ConfigError, match="Assisted suggestions require guided setup"):
        build(tmp_path, Questions(), seed_fields={}, demo=True)

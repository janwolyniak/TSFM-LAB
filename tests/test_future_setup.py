"""Future setup uses final-row context and never requires historical holdouts."""

from collections import defaultdict, deque

import pandas as pd
import pytest

from tsfm_lab.future_config import FutureForecastConfig
from tsfm_lab.repl.wizard import build_experiment
from tsfm_lab.tui.forms import config_from_fields, default_fields, prepare_experiment


@pytest.mark.parametrize("change_after_plan", [False, True])
def test_screen_review_preserves_source_consent_until_managed_run(
    tmp_path, change_after_plan
):
    from tsfm_lab.guided import execute_guided

    path = source(tmp_path)
    fields = {
        **default_fields(),
        "mode": "future_forecast",
        "dataset_path": str(path),
        "units": "units",
    }
    prepared = prepare_experiment(
        fields, workspace=tmp_path / "results", base_directory=tmp_path
    )
    if not change_after_plan:
        frame = pd.read_csv(path)
        frame.loc[0, "demand"] = 999
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="changed during review"):
            prepared.review_plan()
        assert not (tmp_path / "results").exists()
    else:
        plan = prepared.review_plan()
        assert plan.config.forecast.origin == prepared.request.origin
        frame = pd.read_csv(path)
        frame.loc[len(frame)] = ["2026-01-01T08:00:00Z", 999]
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="final observed|changed"):
            execute_guided(plan)
        assert not (tmp_path / "results" / "experiments").exists()


class Questions:
    def __init__(self, answers=None):
        self.answers = defaultdict(deque)
        for label, values in (answers or {}).items():
            self.answers[label].extend(values if isinstance(values, list) else [values])
        self.calls = []

    def __call__(self, label, default="", choices=()):
        assert len(self.calls) < 120, self.calls[-8:]
        self.calls.append((label, default, choices))
        value = self.answers[label].popleft() if self.answers[label] else default
        if isinstance(value, BaseException):
            raise value
        return value


def source(tmp_path, count=8):
    path = tmp_path / "observations.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=count, freq="h", tz="UTC"),
            "demand": range(count),
        }
    ).to_csv(path, index=False)
    return path


def test_future_n_equals_context_has_no_holdout_or_horizon_restriction(tmp_path):
    path = source(tmp_path)
    questions = Questions(
        {"Forecast mode": "Future forecasting", "Run experiment?": "yes"}
    )
    config = build_experiment(
        questions, lambda _: None, workspace=tmp_path / "results", demo_path=path
    )
    assert isinstance(config, FutureForecastConfig)
    assert config.forecast.context_length == 8
    assert config.forecast.horizon == 4
    assert config.forecast.origin == "latest"
    labels = [label for label, _, _ in questions.calls]
    assert labels.index("Dataset path") < labels.index("Forecast mode")
    assert "First target timestamp (inclusive)" not in labels
    assert "MASE seasonal lag" not in labels
    assert not config.output_directory.exists()


def test_future_invalid_origin_reprompts_before_confirmation(tmp_path):
    path = source(tmp_path)
    questions = Questions(
        {
            "Forecast mode": "Future forecasting",
            "Forecast origin (latest or final observed timestamp)": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T08:00:00+01:00",
            ],
            "Run experiment?": "yes",
        }
    )
    emitted = []
    config = build_experiment(
        questions, emitted.append, workspace=tmp_path / "results", demo_path=path
    )
    assert config.forecast.origin == pd.Timestamp("2026-01-01T07:00Z")
    assert any("final observed timestamp" in value for value in emitted)


def test_future_one_observation_is_valid_with_explicit_context(tmp_path):
    path = source(tmp_path, 1)
    fields = {
        **default_fields(),
        "mode": "future_forecast",
        "context_length": "1",
        "horizon": "1",
        "dataset_path": str(path),
        "units": "units",
        "evaluation_start": "",
        "evaluation_end": "",
        "stride": "",
        "mase_period": "",
    }
    prepared = prepare_experiment(
        fields, workspace=tmp_path / "results", base_directory=tmp_path
    )
    assert prepared.request.origin == pd.Timestamp("2026-01-01T00:00Z")
    assert "future_actuals_absent" in prepared.summary()
    resolved = config_from_fields(
        fields, workspace=tmp_path / "results", base_directory=tmp_path
    ).resolved_dict()
    assert "evaluation" not in resolved
    assert set(resolved["forecast"]) == {
        "context_length",
        "horizon",
        "point_statistic",
        "origin",
    }


@pytest.mark.parametrize("ending", ["/cancel", KeyboardInterrupt(), EOFError()])
def test_cancel_future_mode_never_creates_results(tmp_path, ending):
    path = source(tmp_path)
    questions = Questions(
        {
            "Forecast mode": "Future forecasting",
            "Forecast origin (latest or final observed timestamp)": ending,
        }
    )
    if isinstance(ending, BaseException):
        with pytest.raises(type(ending)):
            build_experiment(
                questions,
                lambda _: None,
                workspace=tmp_path / "results",
                demo_path=path,
            )
    else:
        assert (
            build_experiment(
                questions,
                lambda _: None,
                workspace=tmp_path / "results",
                demo_path=path,
            )
            is None
        )
    assert not (tmp_path / "results").exists()


def test_mode_switch_rebuilds_plan_and_discards_only_incompatible_choices(
    tmp_path, monkeypatch
):
    import tsfm_lab.guided as guided

    path = source(tmp_path, 32)
    original = guided.review_guided
    plans = []

    def review(config, **kwargs):
        result = original(config, **kwargs)
        plans.append(result)
        return result

    monkeypatch.setattr(guided, "review_guided", review)
    questions = Questions(
        {
            "Forecast mode": ["Future forecasting", "Historical evaluation"],
            "Run experiment?": ["back", "yes"],
        }
    )
    result = build_experiment(
        questions,
        lambda _: None,
        workspace=tmp_path / "results",
        demo_path=path,
        guided_preparation=True,
    )
    assert len(plans) == 2
    assert isinstance(plans[0].config, FutureForecastConfig)
    assert "mode" not in result.config.resolved_dict()
    assert "forecast" not in result.config.resolved_dict()
    assert (
        result.config.evaluation.context_length
        == plans[0].config.forecast.context_length
    )
    assert result.config.evaluation.horizon == plans[0].config.forecast.horizon
    assert result.config.dataset == plans[0].config.dataset
    assert plans[0].configuration_sha256 != result.configuration_sha256
    assert not (tmp_path / "results").exists()


def test_future_short_context_correction_does_not_limit_horizon_to_holdout(tmp_path):
    path = source(tmp_path, 4)
    questions = Questions(
        {
            "Forecast mode": "Future forecasting",
            "Context length": ["8", "4"],
            "Forecast horizon": ["8", "8"],
            "Run experiment?": "yes",
        }
    )
    config = build_experiment(
        questions, lambda _: None, workspace=tmp_path / "results", demo_path=path
    )
    assert config.forecast.context_length == 4
    assert config.forecast.horizon == 8

"""Future assistant setup keeps human mode and metadata-only disclosure authoritative."""

import io
import json
from collections import defaultdict, deque

import pytest
from rich.console import Console
from test_assistant_drafts import payload
from test_assistant_service import KINDS, configured

from tsfm_lab.assistant.drafts import (
    DraftError,
    decode_draft,
    draft_contract,
    resolve_draft,
)
from tsfm_lab.assistant.evidence import experiment_evidence
from tsfm_lab.assistant.protocol import AssistanceError, validate_context
from tsfm_lab.assistant.service import AssistantService
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.future_application import run_future_config
from tsfm_lab.future_config import FutureForecastConfig
from tsfm_lab.repl.session import TerminalSession
from tsfm_lab.tui.forms import config_from_fields, default_fields

_REVIEW = "Review every setting, clarify with another message, use fresh guided setup, or cancel?"


def future_payload(**changes):
    value = payload(schema_version=2, intent="future_forecast")
    for key in ("stride", "mase_period", "evaluation_start", "evaluation_end"):
        value["fields"].pop(key)
    value["fields"]["origin"] = "latest"
    value["fields"].update(changes)
    return value


class Editor:
    def __init__(self, answers):
        self.answers = defaultdict(deque)
        self.calls = []
        self.lines = iter(["/new describe", "/exit"])
        for key, values in answers.items():
            self.answers[key].extend(values if isinstance(values, list) else [values])

    def ask(self, label, default="", choices=()):
        self.calls.append((label, default, choices))
        assert len(self.calls) < 100, self.calls[-5:]
        return self.answers[label].popleft() if self.answers[label] else default

    def read(self):
        return next(self.lines, "/exit")


@pytest.mark.parametrize("kind", KINDS)
def test_future_draft_schema_and_resolution_work_on_each_fake_route(
    tmp_path, config_factory, kind
):
    config_factory()
    manager, http = configured(tmp_path, kind, future_payload())
    service = AssistantService(manager)
    preview = service.preview_draft(
        "Forecast next week using the selected local observations.",
        mode="future_forecast",
    )
    sent = json.loads(preview["text"])
    assert sent["mode"] == "future_forecast"
    assert sent["draft_contract"]["properties"]["schema_version"]["const"] == 2
    assert str(tmp_path) not in preview["text"]
    draft = service.submit_draft(preview, confirmed=True)
    result = resolve_draft(
        draft,
        dataset_refs={"selected_dataset": tmp_path / "seasonal.csv"},
        mode="future_forecast",
    )
    assert result.state == "ready"
    assert result.seed_fields["mode"] == "future_forecast"
    assert result.seed_fields["forecast_origin"] == "latest"
    assert "evaluation_start" not in result.seed_fields
    assert len(http.requests) == 2
    assert not (tmp_path / "results").exists()


def test_future_schema_rejects_historical_or_authority_fields():
    assert draft_contract()["properties"]["schema_version"]["const"] == 1
    for key in (
        "evaluation_start",
        "stride",
        "mase_period",
        "mode",
        "dataset_path",
        "cache_dir",
    ):
        with pytest.raises(DraftError):
            decode_draft(json.dumps(future_payload(**{key: "invented"})))


@pytest.mark.parametrize("explicit", [False, True])
def test_mode_change_requires_explicit_local_authorization(tmp_path, explicit):
    manager, _ = configured(tmp_path, "openai_api", future_payload(), payload())
    service = AssistantService(manager)
    first = service.submit_draft(
        service.preview_draft("Forecast next week", mode="future_forecast"),
        confirmed=True,
    )
    preview = service.preview_draft(
        "Evaluate the historical observations instead",
        draft=first,
        **({"confirm_mode": "historical_evaluation"} if explicit else {}),
    )
    assert "origin" not in json.loads(preview["text"])["previous_proposal"]["fields"]
    if explicit:
        second = service.submit_draft(preview, confirmed=True)
        assert second.intent == "historical_evaluation"
        assert second.schema_version == 1
        assert second.fields["horizon"] == first.fields["horizon"]
    else:
        with pytest.raises(AssistanceError):
            service.submit_draft(preview, confirmed=True)


def test_assisted_future_reaches_ordinary_review_and_manual_configuration(
    tmp_path, config_factory
):
    config_factory()
    manager, _ = configured(tmp_path, "openai_api", future_payload())
    editor = Editor(
        {
            "Choose a local dataset (its path stays on this machine)": str(
                tmp_path / "seasonal.csv"
            ),
            "Forecast mode": "Future forecasting",
            "What would you like to forecast?": "Forecast next week",
            "Send this message?": "yes",
            _REVIEW: "review",
            "Run experiment?": "yes",
        }
    )
    received = []
    app = TerminalSession(
        tmp_path / "results",
        editor=editor,
        console=Console(file=io.StringIO()),
        connection_manager=manager,
        guided_runner=lambda plan, console: received.append(plan)
        or {"type": "cancelled", "run_id": None},
    )
    app.run()
    assert len(received) == 1
    config = received[0].config
    assert isinstance(config, FutureForecastConfig)
    manual = config_from_fields(
        {
            **default_fields(),
            **{
                key: value
                for key, value in future_payload()["fields"].items()
                if key != "origin"
            },
            "mode": "future_forecast",
            "forecast_origin": config.forecast.origin.isoformat(),
            "dataset_path": str(tmp_path / "seasonal.csv"),
        },
        workspace=tmp_path / "results",
        base_directory=tmp_path,
    )
    assert config.forecast == manual.forecast
    assert config.dataset == manual.dataset
    assert config.model == manual.model
    assert not (tmp_path / "results").exists()


def test_future_explanation_discloses_only_approved_metadata(tmp_path, config_factory):
    config_factory()
    config = config_from_fields(
        {
            **default_fields(),
            "mode": "future_forecast",
            "dataset_path": str(tmp_path / "seasonal.csv"),
            "units": "synthetic_units",
        },
        workspace=tmp_path / "results",
        base_directory=tmp_path,
    )
    result = run_future_config(config)
    context = validate_context(
        experiment_evidence(ExperimentStore(config.output_directory), result["run_id"])
    )
    assert context["kind"] == "future_forecast"
    assert context["data"]["evaluation"]["reason"] == "future_actuals_absent"
    assert context["omitted"]["observed_and_predicted_value_summaries"] is True
    assert str(tmp_path) not in json.dumps(context)
    assert "target_timestamps" not in context["data"]["request"]
    assert set(context["data"]) == {
        "mode",
        "target",
        "request",
        "counts",
        "model",
        "runtime",
        "preprocessing",
        "evaluation",
    }
    answer = {
        "schema_version": 1,
        "annotations": [
            {
                "text": "Accuracy is unavailable for these future forecasts.",
                "evidence_ids": ["/data/evaluation/reason"],
            }
        ],
    }
    manager, _ = configured(tmp_path, "openai_api", answer)
    service = AssistantService(manager)
    preview = service.preview_explanation("Explain the unavailable accuracy", context)
    assert "Future metadata only" in preview["data_included"]
    assert "No observed/predicted values" in preview["data_included"]
    assert (
        service.submit_explanation(preview, confirmed=True)[0].text
        == answer["annotations"][0]["text"]
    )

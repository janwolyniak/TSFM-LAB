"""Independent fake-provider checks of intent and the execution boundary."""

import copy
import io
import json
from collections import defaultdict, deque

import pytest
from rich.console import Console

from tsfm_lab.assistant.protocol import AssistanceError
from tsfm_lab.assistant.service import AssistantService
from tsfm_lab.repl.session import TerminalSession

_REVIEW = "Review every setting, clarify with another message, use fresh guided setup, or cancel?"


def proposal(**updates):
    value = {
        "schema_version": 1,
        "state": "ready",
        "intent": "historical_evaluation",
        "dataset_ref": "selected_dataset",
        "fields": {
            "usage_profile": "generic",
            "adapter": "last_value",
            "timestamp_column": "timestamp",
            "target_column": "demand",
            "frequency": "1h",
            "timezone": "UTC",
            "units": "kW",
            "context_length": "8",
            "horizon": "6",
            "stride": "2",
            "mase_period": "4",
            "evaluation_start": "2026-01-01T16:00:00Z",
            "evaluation_end": "2026-01-02T08:00:00Z",
            "point_statistic": "median",
        },
        "explicit_fields": ["horizon", "units"],
        "assumptions": [],
        "missing_fields": [],
    }
    value.update(updates)
    return value


class FakeManager:
    """Inert protocol double: no credentials, HTTP, account or vault access."""

    def __init__(self, *answers):
        self.answers = deque(json.dumps(value) for value in answers)
        self.previews, self.submissions = [], []

    def preview_message(self, text, **options):
        preview = {
            "id": f"fake-{len(self.previews)}",
            "connection_label": "Fake connection",
            "provider": "Fake provider",
            "usage": "No external service",
            "model": "fake-assistant",
            "account": None,
            "workspace_id": None,
            "text": text,
            "instructions": options["instructions"],
            "data_included": options["data_included"],
            "output_tokens": options["output_tokens"],
            "local_limits": {"seconds": 1, "received_bytes": 65536},
        }
        self.previews.append(copy.deepcopy(preview))
        return preview

    def submit_message(self, preview, *, confirmed, cancel=None):
        self.submissions.append((copy.deepcopy(preview), confirmed))
        return self.answers.popleft() if confirmed else ""


class Editor:
    def __init__(self, answers):
        self.answers = defaultdict(deque)
        self.calls = []
        for label, values in answers.items():
            self.answers[label].extend(values if isinstance(values, list) else [values])

    def ask(self, label, default="", choices=()):
        self.calls.append((label, default, choices))
        return self.answers[label].popleft() if self.answers[label] else default


def start(tmp_path, provider, *, goal="Evaluate historical demand", answers=None):
    editor = Editor(
        {
            "What would you like to evaluate?": goal,
            "Send this message?": "yes",
            _REVIEW: "review",
            **(answers or {}),
        }
    )
    output = io.StringIO()
    app = TerminalSession(
        tmp_path / "assisted-results",
        editor=editor,
        console=Console(file=output, width=120, color_system=None),
        connection_manager=provider,
        runner=lambda *_args, **_kwargs: pytest.fail("No run was confirmed"),
    )
    app.demo_path = tmp_path / "seasonal.csv"
    app.dispatch("/new describe")
    return app, editor, output.getvalue()


def test_ready_claim_without_dates_reaches_normal_questions_and_no_run(
    config_factory, tmp_path
):
    config_factory()
    value = proposal()
    value["fields"].pop("evaluation_start")
    value["fields"].pop("evaluation_end")
    provider = FakeManager(value)
    app, editor, text = start(tmp_path, provider)
    assert "Local draft status: needs input" in text
    assert "Still needed: evaluation end, evaluation start" in text
    labels = [call[0] for call in editor.calls]
    assert "First target timestamp (inclusive)" in labels
    assert "Target end timestamp (exclusive)" in labels
    assert ("Run experiment?", "no", ("yes", "no", "back")) in editor.calls
    assert len(provider.previews) == len(provider.submissions) == 1
    assert not app.workspace.exists()


def test_unknown_adapter_is_correctable_in_normal_guide(config_factory, tmp_path):
    config_factory()
    value = proposal()
    value["fields"]["adapter"] = "invented_adapter"
    provider = FakeManager(value)
    app, editor, text = start(
        tmp_path, provider, answers={"Model adapter": "last_value"}
    )
    model = next(call for call in editor.calls if call[0] == "Model adapter")
    assert model[1] == "invented_adapter"
    assert "Unsupported model" in text
    assert not app.workspace.exists()


def test_returned_future_intent_never_offers_seeded_historical_review(
    config_factory, tmp_path
):
    config_factory()
    provider = FakeManager(proposal(intent="future_forecast"))
    app, editor, text = start(tmp_path, provider, answers={_REVIEW: "cancel"})
    review = next(call for call in editor.calls if call[0] == _REVIEW)
    assert "review" not in review[2]
    assert "Future forecasts without observed actuals are unsupported" in text
    assert all(call[0] != "Model adapter" for call in editor.calls)
    assert not app.workspace.exists()


def test_explicit_future_goal_is_rejected_before_any_provider_request():
    provider = FakeManager()
    with pytest.raises(
        AssistanceError, match="Future forecasts need a separate feature"
    ):
        AssistantService(provider).preview_draft("Forecast next week")
    assert provider.previews == provider.submissions == []


def test_clarification_cannot_silently_replace_previous_explicit_numbers_or_units():
    original, changed = proposal(), proposal()
    changed["fields"].update(horizon="4", units="MW")
    provider = FakeManager(original, changed)
    service = AssistantService(provider)
    first = service.submit_draft(
        service.preview_draft("Use horizon 6 observations with units kW"),
        confirmed=True,
    )
    preview = service.preview_draft("Change only the experiment name", draft=first)
    outgoing = json.loads(preview["text"])
    assert outgoing["previous_proposal"]["fields"]["horizon"] == "6"
    try:
        second = service.submit_draft(preview, confirmed=True)
    except AssistanceError:
        return  # Refusing a conflicting response preserves the user's choice.
    assert second.fields["horizon"] == "6"
    assert second.fields["units"] == "kW"


def test_clarification_cannot_silently_turn_future_intent_into_history():
    provider = FakeManager(proposal(intent="future_forecast"), proposal())
    service = AssistantService(provider)
    first = service.submit_draft(
        service.preview_draft("Evaluate my demand goal"), confirmed=True
    )
    preview = service.preview_draft("Use the same goal and units", draft=first)
    try:
        second = service.submit_draft(preview, confirmed=True)
    except AssistanceError:
        return
    assert second.intent != "historical_evaluation"


def test_omitted_previous_fields_are_restored_after_clarification():
    original, incomplete = proposal(), proposal()
    incomplete["fields"].pop("horizon")
    incomplete["fields"].pop("units")
    incomplete["explicit_fields"] = []
    provider = FakeManager(original, incomplete)
    service = AssistantService(provider)
    first = service.submit_draft(
        service.preview_draft("Use horizon 6 observations with units kW"),
        confirmed=True,
    )
    second = service.submit_draft(
        service.preview_draft("Keep the same requested settings", draft=first),
        confirmed=True,
    )
    assert second.fields["horizon"] == "6"
    assert second.fields["units"] == "kW"


def test_switch_from_future_to_history_requires_explicit_local_authorization():
    provider = FakeManager(proposal(intent="future_forecast"), proposal())
    service = AssistantService(provider)
    first = service.submit_draft(
        service.preview_draft("Evaluate my demand goal"), confirmed=True
    )
    second = service.submit_draft(
        service.preview_draft(
            "Evaluate a historical period against actual observations",
            draft=first,
            confirm_historical=True,
        ),
        confirmed=True,
    )
    assert second.intent == "historical_evaluation"


def test_original_user_constraints_survive_clarification_context_without_prose():
    original = proposal(assumptions=["Untrusted assistant annotation"])
    provider = FakeManager(original)
    service = AssistantService(provider)
    goal = "Evaluate historical temperature. Do not clip negative observations."
    first = service.submit_draft(service.preview_draft(goal), confirmed=True)
    latest = "Clarify only the experiment name"
    preview = service.preview_draft(latest, draft=first)
    outgoing = json.loads(preview["text"])
    serialized = json.dumps(outgoing)
    assert goal in serialized
    assert latest in serialized
    assert "Untrusted assistant annotation" not in serialized


def test_selected_local_dataset_reference_cannot_be_lost_to_provider_null():
    service = AssistantService(FakeManager(proposal(dataset_ref=None)))
    draft = service.submit_draft(
        service.preview_draft("Evaluate historical demand"), confirmed=True
    )
    assert draft.dataset_ref == "selected_dataset"

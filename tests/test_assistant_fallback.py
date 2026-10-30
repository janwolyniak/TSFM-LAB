"""Independent fake-only checks for cancellation, consent, and local fallback."""

import json

import pytest
from test_assistant_drafts import payload
from test_connection_manager import (
    key_connection,
    prepared,
)
from test_connection_manager import (
    setup as manager_setup,
)

from tsfm_lab.assistant.drafts import decode_draft
from tsfm_lab.assistant.evidence import model_evidence
from tsfm_lab.assistant.protocol import AssistanceError
from tsfm_lab.assistant.service import AssistantService
from tsfm_lab.assistant.ui import AssistantFlow
from tsfm_lab.connections.contracts import ConnectionError

setup = manager_setup


class Editor:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def ask(self, label, default="", choices=()):
        self.calls.append((label, default, choices))
        if label == "Forecast mode":
            return default
        answer = next(self.answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class FakeService:
    def __init__(self, *, preview_error=None, submit_error=None):
        self.preview_error, self.submit_error = preview_error, submit_error
        self.previews, self.submissions, self.discards = [], [], []

    def preview_draft(self, goal, **kwargs):
        self.previews.append((goal, kwargs))
        if self.preview_error:
            raise self.preview_error
        return {
            "id": "fake-approval",
            "connection_label": "Chosen connection",
            "provider": "Fake provider",
            "usage": "Chosen usage",
            "model": "fake-model",
            "account": None,
            "workspace_id": None,
            "data_included": "Goal and safe local facts only",
            "instructions": "Return one supported draft",
            "text": "Safe bounded message",
            "output_tokens": 512,
            "local_limits": {"seconds": 30, "received_bytes": 262144},
        }

    def submit_draft(self, preview, **kwargs):
        self.submissions.append((preview, kwargs))
        if self.submit_error:
            raise self.submit_error
        return decode_draft(
            json.dumps(
                payload(state="needs_input", fields={"horizon": "8", "units": "kW"})
            )
        )

    def discard(self, preview):
        self.discards.append(preview)


@pytest.mark.parametrize("failure_stage", ["preview", "submission", "none"])
def test_guided_fallback_retains_the_locally_chosen_dataset(tmp_path, failure_stage):
    dataset = tmp_path / "owner-selected.csv"
    dataset.write_text("timestamp,demand\n2026-01-01T00:00:00Z,1\n")
    error = AssistanceError()
    service = FakeService(
        preview_error=error if failure_stage == "preview" else None,
        submit_error=error if failure_stage == "submission" else None,
    )
    answers = [str(dataset), "Evaluate the historical kW series."]
    if failure_stage != "preview":
        answers.append("yes")
    answers.append("guided")
    reviewed = []
    flow = AssistantFlow(
        Editor(answers), lambda _: None, service, connections=lambda: None
    )
    assert flow.describe(reviewed.append) is None
    assert len(reviewed) == 1 and reviewed[0] is not None
    assert reviewed[0].seed_fields["dataset_path"] == str(dataset.resolve())
    assert not (tmp_path / "results").exists()
    assert str(dataset) not in json.dumps(service.previews)


@pytest.mark.parametrize("answer", ["no", "/cancel", EOFError(), KeyboardInterrupt()])
def test_cancel_at_outgoing_consent_discards_approval_without_review(tmp_path, answer):
    dataset = tmp_path / "owner-selected.csv"
    dataset.write_text("timestamp,demand\n2026-01-01T00:00:00Z,1\n")
    service = FakeService()
    reviewed, output = [], []
    flow = AssistantFlow(
        Editor([str(dataset), "Evaluate this historical series.", answer]),
        output.append,
        service,
        connections=lambda: pytest.fail("Unexpected connection switch"),
    )
    assert flow.describe(reviewed.append) is None
    assert len(service.discards) == 1
    assert service.submissions == reviewed == []
    assert not (tmp_path / "results").exists()


def test_keyboard_interrupt_during_submission_does_not_enter_review(tmp_path):
    dataset = tmp_path / "owner-selected.csv"
    dataset.write_text("timestamp,demand\n2026-01-01T00:00:00Z,1\n")
    service = FakeService(submit_error=KeyboardInterrupt())
    reviewed, output = [], []
    flow = AssistantFlow(
        Editor([str(dataset), "Evaluate this historical series.", "yes"]),
        output.append,
        service,
        connections=lambda: pytest.fail("Unexpected connection switch"),
    )
    assert flow.describe(reviewed.append) is None
    assert len(service.submissions) == 1 and reviewed == []
    assert any("cancelled" in line.lower() for line in output)


@pytest.mark.parametrize("wrong_purpose", ["draft", "explanation"])
def test_service_purpose_mismatch_cannot_generate_or_replay(setup, wrong_purpose):
    prepared(setup)
    service = AssistantService(setup.manager)
    approval = (
        service.preview_draft("Evaluate a historical demand series.")
        if wrong_purpose == "explanation"
        else service.preview_explanation(
            "Explain the capabilities.", model_evidence("last_value")
        )
    )
    wrong_submit = (
        service.submit_explanation
        if wrong_purpose == "explanation"
        else service.submit_draft
    )
    with pytest.raises(AssistanceError) as error:
        wrong_submit(approval, confirmed=True)
    assert error.value.code == "state"
    for submit in (service.submit_draft, service.submit_explanation):
        with pytest.raises(AssistanceError):
            submit(approval, confirmed=True)
    assert setup.generation == []


def test_active_connection_change_after_preview_requires_new_consent(setup):
    prepared(setup)
    second = key_connection(setup, kind="gemini_api", value="other-fake-key-123")
    service = AssistantService(setup.manager)
    approval = service.preview_draft("Evaluate a historical demand series.")
    setup.manager.select(second)
    with pytest.raises(ConnectionError) as error:
        service.submit_draft(approval, confirmed=True)
    assert error.value.code == "state" and setup.generation == []
    with pytest.raises(AssistanceError):
        service.submit_draft(approval, confirmed=True)


@pytest.mark.parametrize("cancel_before_send", [True, False])
def test_cancelled_service_request_cannot_accept_valid_draft_or_retry(
    setup, monkeypatch, cancel_before_send
):
    prepared(setup)
    service = AssistantService(setup.manager)
    approval = service.preview_draft("Evaluate a historical demand series.")
    cancelled, calls = cancel_before_send, []

    def generate(kind, *args, **kwargs):
        nonlocal cancelled
        calls.append(kind)
        cancelled = True
        return json.dumps(payload())

    monkeypatch.setattr("tsfm_lab.connections.providers.generate", generate)
    with pytest.raises(ConnectionError) as error:
        service.submit_draft(approval, confirmed=True, cancel=lambda: cancelled)
    assert error.value.code == "cancelled"
    assert calls == ([] if cancel_before_send else ["openai_api"])
    with pytest.raises(AssistanceError):
        service.submit_draft(approval, confirmed=True)


def test_cancelled_service_preview_cannot_create_a_pending_request(setup):
    prepared(setup)
    service = AssistantService(setup.manager)
    with pytest.raises(ConnectionError) as error:
        service.preview_draft(
            "Evaluate a historical demand series.", cancel=lambda: True
        )
    assert error.value.code == "cancelled"
    assert service._pending == setup.manager._previews == {}
    assert setup.generation == []


def test_missing_provider_dataset_reference_stays_bound_to_local_selection(
    setup, monkeypatch, tmp_path
):
    prepared(setup)
    dataset = tmp_path / "owner-selected.csv"
    dataset.write_text("timestamp,demand\n2026-01-01T00:00:00Z,1\n")
    monkeypatch.setattr(
        "tsfm_lab.connections.providers.generate",
        lambda *args, **kwargs: json.dumps(
            payload(dataset_ref=None, state="needs_input", fields={"horizon": "8"})
        ),
    )
    reviewed = []
    flow = AssistantFlow(
        Editor([str(dataset), "Evaluate this historical series.", "yes", "review"]),
        lambda _: None,
        AssistantService(setup.manager),
        connections=lambda: pytest.fail("Unexpected connection switch"),
    )
    assert flow.describe(reviewed.append) is None
    assert len(reviewed) == 1
    assert reviewed[0].seed_fields["dataset_path"] == str(dataset.resolve())
    assert "dataset_ref" not in reviewed[0].missing_fields

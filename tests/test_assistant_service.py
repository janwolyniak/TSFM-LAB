"""All four assistant routes use real service boundaries with simulated HTTP only."""

import io
import json
import time
from types import SimpleNamespace

import pytest
from rich.console import Console
from test_assistant_drafts import payload
from test_connection_providers import FakeTransport, response, success_response

from tsfm_lab.assistant.drafts import resolve_draft
from tsfm_lab.assistant.evidence import experiment_evidence, model_evidence
from tsfm_lab.assistant.protocol import AssistanceError
from tsfm_lab.assistant.service import AssistantService
from tsfm_lab.connections.credentials import Credentials
from tsfm_lab.connections.manager import ConnectionManager
from tsfm_lab.connections.state import StateStore
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.repl.session import TerminalSession

FAKE = "fake-p7-connection-credential-123456789"
KINDS = ("chatgpt_plan", "openai_api", "anthropic_api", "gemini_api")


def configured(tmp_path, kind, *answers):
    pages = {
        "chatgpt_plan": {"models": [{"slug": "assistant-model", "visibility": "list"}]},
        "openai_api": {"data": [{"id": "assistant-model"}]},
        "anthropic_api": {"data": [{"id": "assistant-model"}], "has_more": False},
        "gemini_api": {
            "models": [
                {
                    "name": "models/assistant-model",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ]
        },
    }
    http = FakeTransport(
        response(pages[kind]),
        *(
            (
                success_response(kind, json.dumps(answer))
                if not isinstance(answer, Exception)
                else answer
            )
            for answer in answers
        ),
    )
    manager = ConnectionManager(
        StateStore(tmp_path.resolve() / "connections"), transport=http
    )
    if kind == "chatgpt_plan":
        item = manager.add_plan("Fake plan", storage="session")
        account = {
            "issuer": "https://auth.openai.com",
            "client_id": "fake-client",
            "subject": "fake-account",
            "email": "fake@example.invalid",
        }
        credentials = Credentials(
            {
                **account,
                "access_token": FAKE,
                "refresh_token": FAKE + "-refresh",
                "id_token": FAKE + "-identity",
                "scopes": ["chatgpt.tokens.use.direct"],
                "access_expires_at": time.time() + 3600,
                "identity_pending": False,
            }
        )
        manager.finish_sign_in(
            item["id"],
            SimpleNamespace(
                account=account, credentials=credentials, identity_pending=False
            ),
            manager.state.read()["generation"],
        )
    else:
        item = manager.add_key(kind, "Fake API", FAKE, storage="session")
    manager.select(item["id"])
    manager.models()
    manager.set_model("assistant-model")
    return manager, http


@pytest.mark.parametrize("kind", KINDS)
def test_confirmed_draft_then_explanation_one_request_each_and_no_paths(
    tmp_path, config_factory, kind
):
    config_factory()
    annotation = {
        "schema_version": 1,
        "annotations": [
            {
                "text": "This model uses the available context.",
                "evidence_ids": ["/data/adapter_id"],
            }
        ],
    }
    manager, http = configured(tmp_path, kind, payload(), annotation)
    service = AssistantService(manager)
    preview = service.preview_draft(
        "Evaluate historical demand with context 8, horizon 4 and stride 4."
    )
    assert len(http.requests) == 1
    assert str(tmp_path) not in preview["text"] and FAKE not in json.dumps(preview)
    draft = service.submit_draft(preview, confirmed=True)
    resolved = resolve_draft(
        draft, dataset_refs={"selected_dataset": tmp_path / "seasonal.csv"}
    )
    assert resolved.state == "ready" and resolved.seed_fields["horizon"] == "4"
    assert len(http.requests) == 2
    with pytest.raises(AssistanceError):
        service.submit_draft(preview, confirmed=True)
    context = model_evidence("last_value")
    preview = service.preview_explanation("Explain this model's context.", context)
    answer = service.submit_explanation(preview, confirmed=True)
    assert answer[0].text == annotation["annotations"][0]["text"]
    assert len(http.requests) == 3
    assert all(
        request.headers.get("Authorization") == f"Bearer {FAKE}"
        or request.headers.get("x-goog-api-key") == FAKE
        for request in http.requests
    )
    assert not (tmp_path / "workspace").exists()
    assert FAKE not in manager.state.path.read_text()


@pytest.mark.parametrize("kind", KINDS)
def test_invalid_response_stops_without_repair_or_switch(tmp_path, kind):
    manager, http = configured(
        tmp_path,
        kind,
        {
            "schema_version": 1,
            "code": "run something",
            "fields": {"dataset_path": "/private/secret"},
        },
    )
    service = AssistantService(manager)
    preview = service.preview_draft("Evaluate the historical demand series.")
    with pytest.raises(AssistanceError):
        service.submit_draft(preview, confirmed=True)
    assert len(http.requests) == 2
    assert manager.status()["kind"] == kind
    assert not (tmp_path / "results").exists()


class Editor:
    def __init__(self, answers, lines=("/new describe", "/exit")):
        self.answers, self.lines = iter(answers), iter(lines)
        self.asked = []

    def ask(self, label, default="", choices=()):
        self.asked.append(label)
        if label == "Forecast mode":
            return default
        answer = next(self.answers, default)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def read(self):
        return next(self.lines, "/exit")


def test_terminal_natural_goal_still_requires_complete_guided_review_and_confirmation(
    tmp_path, config_factory
):
    config_factory()
    manager, http = configured(tmp_path, "openai_api", payload())
    editor = Editor(
        [
            str(tmp_path / "seasonal.csv"),
            "Evaluate the historical demand series.",
            "yes",
            "review",
            *([""] * 30),
        ]
    )
    output = io.StringIO()
    app = TerminalSession(
        tmp_path / "results",
        console=Console(file=output, color_system=None),
        editor=editor,
        connection_manager=manager,
        runner=lambda *_: pytest.fail("Run default must remain no"),
    )
    # Default answers retain every proposed setting, including final no.
    base = editor.ask
    editor.ask = (
        lambda label, default="", choices=(): base(label, default, choices) or default
    )
    app.run()
    assert editor.asked.count("Run experiment?") == 1
    assert not app.last_run
    assert "Local draft status: ready" in output.getvalue()
    assert not (tmp_path / "results").exists()
    assert len(http.requests) == 2 and FAKE not in output.getvalue()


def test_terminal_confirmed_assisted_run_matches_manual_configuration(
    tmp_path, config_factory
):
    original = config_factory(dataset={"timezone": "UTC"})
    manager, http = configured(tmp_path, "openai_api", payload())
    editor = Editor(
        [
            str(tmp_path / "seasonal.csv"),
            "Evaluate the historical demand series.",
            "yes",
            "review",
        ]
    )
    base = editor.ask

    def ask(label, default="", choices=()):
        if label == "Run experiment?":
            return "yes"
        return base(label, default, choices) or default

    editor.ask = ask
    received = []

    def runner(config, console):
        received.append(config)
        return {"type": "cancelled", "run_id": None}

    app = TerminalSession(
        tmp_path / "results",
        console=Console(file=io.StringIO()),
        editor=editor,
        connection_manager=manager,
        runner=runner,
    )
    app.run()
    assert len(received) == 1
    from tsfm_lab.config import load_config

    manual = load_config(original)
    assisted = received[0]
    assert (
        assisted.dataset == manual.dataset and assisted.evaluation == manual.evaluation
    )
    assert assisted.model == manual.model
    assert len(http.requests) == 2


def test_saved_result_explanation_never_changes_artifacts(tmp_path, config_factory):
    from tsfm_lab.application import run_from_config

    result = run_from_config(config_factory())
    files = {
        path: path.read_bytes()
        for path in (tmp_path / "workspace").rglob("*")
        if path.is_file()
    }
    context = experiment_evidence(
        ExperimentStore(result["workspace"]), result["run_id"]
    )
    answer = {
        "schema_version": 1,
        "annotations": [
            {
                "text": "MAE is 2.5 in the target units.",
                "evidence_ids": [
                    "/data/metrics/overall/mae/value",
                    "/data/metrics/units/mae",
                ],
            }
        ],
    }
    manager, http = configured(tmp_path, "gemini_api", answer)
    service = AssistantService(manager)
    preview = service.preview_explanation("Explain these evaluation scores.", context)
    assert result["directory"] not in preview["text"]
    assert "predictions.csv" not in preview["text"]
    assert (
        service.submit_explanation(preview, confirmed=True)[0].text
        == answer["annotations"][0]["text"]
    )
    assert {path: path.read_bytes() for path in files} == files
    assert len(http.requests) == 2


def test_two_model_explanation_accepts_only_selected_catalog_facts(tmp_path):
    first, second = model_evidence("chronos_2"), model_evidence("timesfm_2p5")
    context = {
        **first,
        "kind": "model_comparison",
        "data": {"models": [first["data"], second["data"]]},
    }
    annotation = {
        "schema_version": 1,
        "annotations": [
            {
                "text": "These are separate model families with their own supported tasks.",
                "evidence_ids": [
                    "/data/models/0/adapter_id",
                    "/data/models/1/adapter_id",
                ],
            }
        ],
    }
    manager, http = configured(tmp_path, "openai_api", annotation)
    service = AssistantService(manager)
    preview = service.preview_explanation(
        "Explain the supported differences between these models.", context
    )
    assert service.submit_explanation(preview, confirmed=True)
    assert len(http.requests) == 2


@pytest.mark.parametrize(
    "assumption",
    [
        "text\u202espoof",
        "Use /private/local/data",
        "sk-proj-fakeforeigncredential123456789",
    ],
)
def test_unsafe_assumptions_are_rejected_before_terminal_display(tmp_path, assumption):
    manager, http = configured(
        tmp_path, "openai_api", payload(assumptions=[assumption])
    )
    service = AssistantService(manager)
    preview = service.preview_draft("Evaluate historical demand.")
    with pytest.raises(AssistanceError):
        service.submit_draft(preview, confirmed=True)
    assert len(http.requests) == 2

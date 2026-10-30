"""S5 protocols, destination binding and assistance use fake providers only."""

import json

import pytest
from test_assistant_drafts import payload
from test_connection_providers import FakeTransport, response

from tsfm_lab.assistant.protocol import AssistanceError
from tsfm_lab.assistant.service import AssistantService
from tsfm_lab.connections.contracts import ConnectionError, Limits
from tsfm_lab.connections.credentials import Credentials
from tsfm_lab.connections.endpoints import endpoint_for
from tsfm_lab.connections.manager import ConnectionManager
from tsfm_lab.connections.providers import generate
from tsfm_lab.connections.state import StateStore
from tsfm_lab.connections.transport import HTTPRequest

KINDS = ("deepseek_api", "compatible_api", "ollama_local")
SECRET = "fake-s5-private-key-123456789"
MODEL = "assistant:local"


def output(kind, text, *, finish="stop"):
    message = {"role": "assistant", "content": text}
    if kind == "ollama_local":
        return {"done": True, "done_reason": finish, "message": message}
    return {"choices": [{"finish_reason": finish, "message": message}]}


def setup(tmp_path, kind, *answers):
    discovery = (
        {"models": [{"name": MODEL}, {"name": "bad:cloud"}]}
        if kind == "ollama_local"
        else {"data": [{"id": MODEL}]}
    )
    responses = [response(discovery)]
    for answer in answers:
        if kind == "ollama_local":
            responses.append(
                response(
                    {
                        "model_info": {"general.architecture": "local"},
                        "capabilities": ["completion"],
                    }
                )
            )
        responses.append(answer)
    http = FakeTransport(*responses)
    manager = ConnectionManager(StateStore(tmp_path / "connections"), transport=http)
    item = manager.add_key(
        kind,
        "Test route",
        None if kind == "ollama_local" else SECRET,
        storage="session",
        **(
            {"endpoint": "https://example.invalid/v1"}
            if kind == "compatible_api"
            else {}
        ),
    )
    manager.select(item["id"])
    assert [m.identifier for m in manager.models()] == [MODEL]
    manager.set_model(MODEL)
    return manager, http


@pytest.mark.parametrize("kind", KINDS)
def test_draft_and_grounded_explanation_use_selected_protocol(tmp_path, kind):
    annotation = {
        "schema_version": 1,
        "annotations": [{"text": "Reason.", "evidence_ids": ["/data/adapter_id"]}],
    }
    manager, http = setup(
        tmp_path,
        kind,
        response(output(kind, json.dumps(payload()))),
        response(output(kind, json.dumps(annotation))),
    )
    service = AssistantService(manager)
    preview = service.preview_draft("Use the selected data to evaluate a baseline.")
    assert preview["destination"] == manager.status()["destination"]
    draft = service.submit_draft(preview, confirmed=True)
    assert draft.fields["adapter"] == "last_value"
    from tsfm_lab.assistant.evidence import model_evidence

    context = model_evidence("last_value")
    preview = service.preview_explanation("Explain this model.", context)
    service.submit_explanation(preview, confirmed=True)
    generated = [
        r for r in http.requests if r.url.endswith(("/chat", "/chat/completions"))
    ]
    assert len(generated) == 2
    for request in generated:
        assert request.json_body["stream"] is False
        assert set(request.json_body) <= {
            "model",
            "messages",
            "max_tokens",
            "options",
            "stream",
        }
        assert request.headers.get("Authorization") == (
            None if kind == "ollama_local" else f"Bearer {SECRET}"
        )
    assert SECRET not in manager.state.path.read_text()
    manager.disconnect(manager.status()["id"])
    with pytest.raises(ConnectionError):
        manager.models()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("finish", ["length", "tool_calls", None])
def test_truncated_or_unsupported_output_cannot_become_a_draft(tmp_path, kind, finish):
    manager, _ = setup(
        tmp_path, kind, response(output(kind, json.dumps(payload()), finish=finish))
    )
    service = AssistantService(manager)
    preview = service.preview_draft("Evaluate a baseline.")
    with pytest.raises((ConnectionError, AssistanceError)):
        service.submit_draft(preview, confirmed=True)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "answer", ["{}", "not JSON", '{"schema_version":1,"commands":["run"]}']
)
def test_complete_but_invalid_draft_remains_advisory(tmp_path, kind, answer):
    manager, _ = setup(tmp_path, kind, response(output(kind, answer)))
    service = AssistantService(manager)
    preview = service.preview_draft("Evaluate a baseline.")
    with pytest.raises(AssistanceError):
        service.submit_draft(preview, confirmed=True)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "status,code",
    [
        (401, "auth"),
        (403, "permission"),
        (429, "limit"),
        (408, "timeout"),
        (500, "provider"),
    ],
)
def test_errors_are_redacted_and_never_retry(tmp_path, kind, status, code):
    manager, http = setup(tmp_path, kind, response({"error": SECRET}, status))
    preview = manager.preview_test()
    before = len(http.requests)
    with pytest.raises(ConnectionError) as error:
        manager.submit_test(preview, confirmed=True)
    assert error.value.code == code
    assert SECRET not in str(error.value)
    assert len(http.requests) - before == (2 if kind == "ollama_local" else 1)


@pytest.mark.parametrize("kind", KINDS)
def test_decline_and_cancel_never_send(tmp_path, kind):
    manager, http = setup(tmp_path, kind)
    for confirmed, cancel in [(False, None), (True, lambda: True)]:
        preview = manager.preview_test()
        with pytest.raises(ConnectionError):
            manager.submit_test(preview, confirmed=confirmed, cancel=cancel)
    assert len(http.requests) == 1


def test_endpoint_change_requires_new_key_and_invalidates_preview(tmp_path):
    manager, http = setup(tmp_path, "compatible_api")
    preview = manager.preview_test()
    item = manager.status()
    manager.replace_key(
        item["id"], "new-fake-key-987654321", endpoint="https://second.invalid/v1"
    )
    with pytest.raises(ConnectionError):
        manager.submit_test(preview, confirmed=True)
    assert len(http.requests) == 1
    assert manager.status()["model"] is None
    # Editing nonsecret metadata cannot forward the saved old key to a new host.
    state = manager.state.read()
    state["profiles"][0]["endpoint"] = "https://third.invalid/v1"
    manager.state.write(state)
    with pytest.raises(ConnectionError, match="Credentials"):
        manager.models()
    assert len(http.requests) == 1


def test_local_route_survives_restart_without_a_secret_store(tmp_path):
    manager, _ = setup(tmp_path, "ollama_local")
    fresh = ConnectionManager(
        manager.state, transport=FakeTransport(response({"models": []}))
    )
    assert fresh.models() == []


@pytest.mark.parametrize(
    "root",
    [
        "https://user:key@host/v1",
        "http://remote.example/v1",
        "https://host/v1?key=secret",
        "https://host/v1#fragment",
        "https://host/../v1",
        "https://host/%2e/v1",
        "https://host/v1\n",
        "https://host:0/v1",
    ],
)
def test_endpoint_validation_rejects_unsafe_or_secret_bearing_destinations(root):
    with pytest.raises(ConnectionError):
        endpoint_for("compatible_api", root)


def test_request_cannot_escape_explicit_root_or_call_pull():
    for url in [
        "https://other.invalid/v1/models",
        "https://example.invalid/v1/models?x=1",
        "https://example.invalid/v1evil/models",
        "https://example.invalid/v1/api/pull",
    ]:
        with pytest.raises(ConnectionError):
            HTTPRequest("GET", url, endpoint="https://example.invalid/v1")


def test_disguised_ollama_cloud_model_fails_before_generation(tmp_path):
    manager, http = setup(tmp_path, "ollama_local")
    http.responses.append(
        response(
            {
                "remote_host": "https://ollama.com",
                "model_info": {"architecture": "local"},
            }
        )
    )
    preview = manager.preview_test()
    with pytest.raises(ConnectionError) as error:
        manager.submit_test(preview, confirmed=True)
    assert error.value.code == "model"
    assert http.requests[-1].url.endswith("/api/show")


@pytest.mark.parametrize("kind", KINDS)
def test_generation_byte_bounds_and_timeout_cancel(tmp_path, kind):
    manager, _ = setup(tmp_path, kind)
    credentials = Credentials(
        {"local": True} if kind == "ollama_local" else {"api_key": SECRET}
    )
    transport = FakeTransport()
    for kwargs in [{"cancel": lambda: True}, {"limits": Limits(max_input_bytes=1)}]:
        with pytest.raises(ConnectionError):
            generate(
                kind,
                credentials,
                transport,
                model=MODEL,
                instructions="",
                text="Hello",
                endpoint=manager.status()["destination"],
                **kwargs,
            )
    assert transport.requests == []


@pytest.mark.parametrize("name", ["assistant:8b", "assistant:local"])
def test_ollama_forces_local_source_and_rejects_remote_generation(tmp_path, name):
    from tsfm_lab.connections.additional import generate as local_generate
    from tsfm_lab.connections.providers import _Operation

    http = FakeTransport(
        response(
            {"model_info": {"architecture": "local"}, "capabilities": ["completion"]}
        ),
        response(
            {**output("ollama_local", "answer"), "remote_host": "https://ollama.com"}
        ),
    )
    with pytest.raises(ConnectionError):
        local_generate(
            "ollama_local",
            Credentials({"local": True}),
            http,
            _Operation(Limits(), None),
            "http://127.0.0.1:11434",
            name,
            "",
            "Explain.",
            32,
        )
    assert all(
        r.json_body["model"] == (name if name.endswith(":local") else name + ":local")
        for r in http.requests
    )


def test_ollama_embedding_only_model_never_receives_chat(tmp_path):
    manager, http = setup(tmp_path, "ollama_local")
    http.responses.append(
        response(
            {"model_info": {"architecture": "embedding"}, "capabilities": ["embedding"]}
        )
    )
    with pytest.raises(ConnectionError):
        manager.submit_test(manager.preview_test(), confirmed=True)
    assert http.requests[-1].url.endswith("/api/show")

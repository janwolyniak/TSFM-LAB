"""Provider contracts use fake credentials and an in-memory transport only."""

import base64
import json
import time
from urllib.parse import parse_qs, urlsplit

import pytest

from tsfm_lab.connections.contracts import ConnectionError, Limits, ModelChoice
from tsfm_lab.connections.credentials import Credentials
from tsfm_lab.connections.providers import (
    generate,
    list_models,
    reject_secret_reflection,
)
from tsfm_lab.connections.transport import HTTPResponse

SECRET = "fake-test-key-ABCD+/=123456"
KINDS = ("chatgpt_plan", "openai_api", "anthropic_api", "gemini_api")


def credentials(kind):
    if kind == "chatgpt_plan":
        return Credentials(
            {
                "access_token": SECRET,
                "scopes": ["chatgpt.tokens.use.direct"],
                "access_expires_at": time.time() + 3600,
            }
        )
    return Credentials({"api_key": SECRET})


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.closed = 0

    def send(self, request, *, limits, cancel=None):
        if cancel:
            cancel()
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        response._close = lambda: setattr(self, "closed", self.closed + 1)
        return response


def response(payload, status=200):
    return HTTPResponse(status, json.dumps(payload).encode())


def openai_output(text="A complete explanation."):
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text}],
            },
        ],
    }


def provider_output(kind, text="A complete explanation."):
    if kind in {"chatgpt_plan", "openai_api"}:
        return openai_output(text)
    if kind == "anthropic_api":
        return {
            "type": "message",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": text},
            ],
        }
    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "private", "thought": True},
                        {"text": text, "thoughtSignature": "opaque"},
                    ],
                },
            }
        ]
    }


def sse(*events, tail=b""):
    raw = (
        b"".join(
            b"event: "
            + event["type"].encode()
            + b"\r\ndata: "
            + json.dumps(event).encode()
            + b"\r\n\r\n"
            for event in events
        )
        + tail
    )
    return HTTPResponse(
        200,
        chunks=[raw[index : index + 7] for index in range(0, len(raw), 7)],
        headers={"Content-Type": "text/event-stream; charset=utf-8"},
    )


def success_response(kind, text="A complete explanation."):
    payload = provider_output(kind, text)
    if kind == "chatgpt_plan":
        return sse(
            {"type": "response.output_text.delta", "delta": "not returned"},
            {"type": "response.completed", "response": payload},
        )
    return response(payload)


def run(kind, transport, **kwargs):
    return generate(
        kind,
        credentials(kind),
        transport,
        model="selected-model",
        instructions="Explain clearly.",
        text="Explain this experiment.",
        **kwargs,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_one_generation_uses_selected_route_and_accepts_completed_text(kind):
    transport = FakeTransport(success_response(kind))
    assert run(kind, transport, output_tokens=32) == "A complete explanation."
    assert len(transport.requests) == transport.closed == 1
    request = transport.requests[0]
    assert request.method == "POST"
    assert SECRET not in request.url
    assert SECRET not in repr(request)
    assert SECRET not in json.dumps(request.json_body)
    if kind == "gemini_api":
        assert request.headers["x-goog-api-key"] == SECRET
        assert "Authorization" not in request.headers
        assert request.json_body["generationConfig"] == {
            "candidateCount": 1,
            "maxOutputTokens": 32,
        }
    else:
        assert request.headers["Authorization"] == f"Bearer {SECRET}"
        if kind == "anthropic_api":
            assert request.headers["anthropic-version"] == "2026-10-21"
            assert "x-api-key" not in request.headers
            assert request.json_body["max_tokens"] == 32
        else:
            assert request.json_body["store"] is False
            assert request.json_body["stream"] is (kind == "chatgpt_plan")
            if kind == "chatgpt_plan":
                assert request.json_body["input"] == [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Explain this experiment."}
                        ],
                    }
                ]
                assert "max_output_tokens" not in request.json_body
            else:
                assert request.json_body["input"] == "Explain this experiment."
                assert request.json_body["max_output_tokens"] == 32


def test_openai_text_is_aggregated_after_reasoning_without_assuming_first_item():
    payload = openai_output("First.")
    payload["output"].append(
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Second."}],
        }
    )
    assert run("openai_api", FakeTransport(response(payload))) == "First.Second."


@pytest.mark.parametrize("kind", KINDS)
def test_discovery_never_generates_or_chooses_automatically(kind):
    if kind == "chatgpt_plan":
        payload = {
            "models": [
                {
                    "slug": "visible-model",
                    "display_name": "Visible",
                    "visibility": "list",
                },
                {
                    "slug": "hidden-model",
                    "display_name": "Hidden",
                    "visibility": "hidden",
                },
            ]
        }
    elif kind == "gemini_api":
        payload = {
            "models": [
                {
                    "name": "models/visible-model",
                    "displayName": "Visible",
                    "supportedGenerationMethods": ["generateContent"],
                },
                {
                    "name": "models/embed",
                    "supportedGenerationMethods": ["embedContent"],
                },
            ]
        }
    else:
        payload = {"data": [{"id": "visible-model", "display_name": "Visible"}]}
        if kind == "anthropic_api":
            payload["has_more"] = False
    transport = FakeTransport(response(payload))
    assert list_models(kind, credentials(kind), transport) == [
        ModelChoice("visible-model", "Visible")
    ]
    assert len(transport.requests) == transport.closed == 1
    assert transport.requests[0].method == "GET"
    assert SECRET not in transport.requests[0].url


@pytest.mark.parametrize("kind", ("anthropic_api", "gemini_api"))
def test_discovery_follows_only_fixed_host_cursors(kind):
    if kind == "anthropic_api":
        pages = (
            {"data": [{"id": "first"}], "has_more": True, "last_id": "first"},
            {"data": [{"id": "second"}], "has_more": False},
        )
        expected = {"after_id": ["first"], "limit": ["100"]}
    else:
        pages = (
            {
                "models": [
                    {
                        "name": "models/first",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ],
                "nextPageToken": "https://untrusted.invalid/?x=1",
            },
            {
                "models": [
                    {
                        "name": "models/second",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            },
        )
        expected = {
            "pageToken": ["https://untrusted.invalid/?x=1"],
            "pageSize": ["1000"],
        }
    transport = FakeTransport(*(response(page) for page in pages))
    assert [
        choice.identifier for choice in list_models(kind, credentials(kind), transport)
    ] == ["first", "second"]
    first, second = (urlsplit(req.url) for req in transport.requests)
    assert (first.scheme, first.netloc, first.path) == (
        second.scheme,
        second.netloc,
        second.path,
    )
    assert parse_qs(second.query) == expected


def test_anthropic_workspace_is_explicit_nonsecret_metadata():
    transport = FakeTransport(success_response("anthropic_api"))
    run("anthropic_api", transport, workspace_id="wrkspc_test123")
    assert transport.requests[0].headers["anthropic-workspace-id"] == "wrkspc_test123"


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"scopes": []}, "permission"),
        ({"scopes": "chatgpt.tokens.use.direct"}, "permission"),
        ({"identity_pending": True}, "auth"),
        ({"refresh_in_progress": True}, "auth"),
        ({"access_expires_at": 0}, "auth"),
        ({"access_expires_at": None}, "auth"),
    ],
)
def test_plan_requires_direct_scope_complete_identity_and_unexpired_token(
    changes, code
):
    values = credentials("chatgpt_plan").snapshot()
    values.update(changes)
    transport = FakeTransport()
    with pytest.raises(ConnectionError) as error:
        list_models("chatgpt_plan", Credentials(values), transport)
    assert error.value.code == code
    assert transport.requests == []


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "auth"),
        (403, "permission"),
        (404, "model"),
        (429, "limit"),
        (500, "provider"),
    ],
)
@pytest.mark.parametrize("kind", KINDS)
def test_http_errors_never_retry_switch_routes_or_expose_provider_error(
    kind, status, code
):
    transport = FakeTransport(response({"error": SECRET + " private prompt"}, status))
    with pytest.raises(ConnectionError) as error:
        run(kind, transport)
    assert error.value.code == code
    assert SECRET not in str(error.value)
    assert "private prompt" not in str(error.value)
    assert len(transport.requests) == transport.closed == 1


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "encode",
    [
        lambda value: value,
        lambda value: base64.b64encode(value.encode()).decode(),
        lambda value: base64.urlsafe_b64encode(value.encode()).decode().rstrip("="),
        lambda value: value.encode().hex(),
        lambda value: "".join(f"\\u{ord(c):04x}" for c in value),
    ],
)
def test_successful_encoded_secret_reflection_is_rejected(kind, encode):
    transport = FakeTransport(success_response(kind, "Reflected " + encode(SECRET)))
    with pytest.raises(ConnectionError) as error:
        run(kind, transport)
    assert error.value.code == "response"
    assert SECRET not in str(error.value)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("text", ["", "  \n", "unsafe\x1b[31m", "unsafe\x00"])
def test_empty_or_terminal_control_output_is_rejected(kind, text):
    with pytest.raises(ConnectionError, match="unsupported response"):
        run(kind, FakeTransport(success_response(kind, text)))


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "incomplete", "output": []},
        {"status": "completed", "error": {"message": SECRET}, "output": []},
        {
            "status": "completed",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        },
        {
            "status": "completed",
            "output": [{"type": "function_call", "name": "dangerous"}],
        },
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "refusal", "refusal": "blocked"}],
                }
            ],
        },
    ],
)
@pytest.mark.parametrize("kind", ("chatgpt_plan", "openai_api"))
def test_openai_incomplete_refusal_and_tools_are_rejected(kind, payload):
    reply = (
        sse({"type": "response.completed", "response": payload})
        if kind == "chatgpt_plan"
        else response(payload)
    )
    with pytest.raises(ConnectionError):
        run(kind, FakeTransport(reply))


@pytest.mark.parametrize(
    "stop", ["max_tokens", "tool_use", "refusal", "pause_turn", None]
)
def test_anthropic_requires_complete_end_turn(stop):
    payload = provider_output("anthropic_api")
    payload["stop_reason"] = stop
    with pytest.raises(ConnectionError):
        run("anthropic_api", FakeTransport(response(payload)))


@pytest.mark.parametrize(
    "mutation", ["blocked", "truncated", "tool", "multiple", "empty"]
)
def test_gemini_requires_one_unblocked_complete_text_candidate(mutation):
    payload = provider_output("gemini_api")
    if mutation == "blocked":
        payload["promptFeedback"] = {"blockReason": "SAFETY"}
    elif mutation == "truncated":
        payload["candidates"][0]["finishReason"] = "MAX_TOKENS"
    elif mutation == "tool":
        payload["candidates"][0]["content"]["parts"] = [
            {"functionCall": {"name": "danger"}}
        ]
    elif mutation == "multiple":
        payload["candidates"] *= 2
    else:
        payload["candidates"] = []
    with pytest.raises(ConnectionError):
        run("gemini_api", FakeTransport(response(payload)))


@pytest.mark.parametrize(
    "events,tail",
    [
        ([{"type": "response.output_text.delta", "delta": "Partial"}], b""),
        (
            [
                {"type": "response.output_text.delta", "delta": "Partial"},
                {"type": "error", "message": SECRET},
            ],
            b"",
        ),
        ([{"type": "response.incomplete", "response": openai_output()}], b""),
        (
            [
                {"type": "response.completed", "response": openai_output()},
                {"type": "response.failed", "error": SECRET},
            ],
            b"",
        ),
        ([], b"data: [DONE]\n\n"),
        ([], b'data: {"type":"response.completed"}'),
    ],
)
def test_plan_never_accepts_partial_or_later_failed_stream(events, tail):
    with pytest.raises(ConnectionError):
        run("chatgpt_plan", FakeTransport(sse(*events, tail=tail)))


@pytest.mark.parametrize("kind", KINDS)
def test_cancel_before_send_avoids_any_request(kind):
    transport = FakeTransport()
    with pytest.raises(ConnectionError) as error:
        run(kind, transport, cancel=lambda: True)
    assert error.value.code == "cancelled"
    assert transport.requests == []


def test_cancel_mid_stream_closes_response_without_returning_partial_text():
    cancelled = False

    def chunks():
        nonlocal cancelled
        yield b'data: {"type":"response.output_text.delta","delta":"Partial"}\n\n'
        cancelled = True
        yield b"data: [DONE]\n\n"

    transport = FakeTransport(
        HTTPResponse(
            200, chunks=chunks(), headers={"content-type": "text/event-stream"}
        )
    )
    with pytest.raises(ConnectionError) as error:
        run("chatgpt_plan", transport, cancel=lambda: cancelled)
    assert error.value.code == "cancelled"
    assert transport.closed == 1


def test_deadline_is_for_whole_operation(monkeypatch):
    clock = 10.0
    monkeypatch.setattr("tsfm_lab.connections.providers.time.monotonic", lambda: clock)

    def chunks():
        nonlocal clock
        yield b'{"status":'
        clock = 12.0
        yield b'"completed","output":[]}'

    transport = FakeTransport(HTTPResponse(200, chunks=chunks()))
    with pytest.raises(ConnectionError) as error:
        run("openai_api", transport, limits=Limits(timeout_seconds=1))
    assert error.value.code == "timeout"
    assert transport.closed == 1


@pytest.mark.parametrize(
    "limits,text,code",
    [
        (Limits(max_input_bytes=1), "text", "bounds"),
        (Limits(max_text_bytes=1), "text", "bounds"),
        (Limits(max_response_bytes=1), "text", "bounds"),
    ],
)
def test_request_response_and_public_text_are_independently_bounded(limits, text, code):
    transport = FakeTransport(success_response("openai_api", text))
    with pytest.raises(ConnectionError) as error:
        run("openai_api", transport, limits=limits)
    assert error.value.code == code


@pytest.mark.parametrize(
    "identifier",
    [
        "../secret",
        "models/nested/name",
        "model?key=x",
        "model#x",
        "model\r\nx:y",
        "",
        "a" * 201,
    ],
)
@pytest.mark.parametrize("kind", KINDS)
def test_unsafe_model_path_is_rejected_before_transport(kind, identifier):
    transport = FakeTransport()
    with pytest.raises(ConnectionError) as error:
        generate(
            kind,
            credentials(kind),
            transport,
            model=identifier,
            instructions="",
            text="Explain.",
        )
    assert error.value.code == "model"
    assert transport.requests == []


def test_gemini_accepts_official_model_name_prefix_but_uses_single_safe_path_component():
    transport = FakeTransport(success_response("gemini_api"))
    generate(
        "gemini_api",
        credentials("gemini_api"),
        transport,
        model="models/chosen",
        instructions="",
        text="Explain.",
    )
    assert transport.requests[0].url.endswith("/models/chosen:generateContent")


@pytest.mark.parametrize(
    "pages,limits,code",
    [
        ([{"data": [{"id": "one"}, {"id": "two"}]}], Limits(max_models=1), "bounds"),
        ([{"data": [{"id": "one"}, {"id": "one"}]}], Limits(), "response"),
        ([{"data": [{"id": "../bad"}]}], Limits(), "model"),
        (
            [{"data": [{"id": "one", "display_name": "unsafe\x1b"}]}],
            Limits(),
            "response",
        ),
    ],
)
def test_untrusted_discovery_metadata_is_rejected(pages, limits, code):
    with pytest.raises(ConnectionError) as error:
        list_models(
            "openai_api",
            credentials("openai_api"),
            FakeTransport(*(response(page) for page in pages)),
            limits=limits,
        )
    assert error.value.code == code


def test_discovery_page_cap_and_repeated_cursor_are_rejected():
    page = {"models": [], "nextPageToken": "repeated"}
    transport = FakeTransport(response(page), response(page))
    with pytest.raises(ConnectionError) as error:
        list_models("gemini_api", credentials("gemini_api"), transport)
    assert error.value.code == "response"
    assert len(transport.requests) == 2
    with pytest.raises(ConnectionError) as error:
        list_models(
            "gemini_api",
            credentials("gemini_api"),
            FakeTransport(response(page)),
            limits=Limits(max_pages=1),
        )
    assert error.value.code == "bounds"


def test_raw_transport_exception_and_encoded_error_body_are_not_exposed():
    encoded = base64.b64encode(SECRET.encode()).decode()
    for reply in (RuntimeError(SECRET), response({"error": encoded}, 500)):
        with pytest.raises(ConnectionError) as error:
            run("openai_api", FakeTransport(reply))
        assert SECRET not in str(error.value)
        assert encoded not in str(error.value)
        assert error.value.__suppress_context__ or error.value.code == "provider"


@pytest.mark.parametrize(
    "value",
    [
        {"account": {"subject": SECRET}},
        {"workspace": "".join(f"%{c:02x}" for c in SECRET.encode())},
        {"label": "".join(f"\\u{ord(c):04x}" for c in SECRET)},
        {"label": "".join(f"&#{ord(c)};" for c in SECRET)},
        {"refresh_token": "unrelated"},
    ],
)
def test_public_metadata_secret_check_reads_only_sensitive_credential_fields(value):
    protected = credentials("chatgpt_plan").snapshot()
    protected.update({"refresh_token": "unrelated", "subject": "safe-subject"})
    with pytest.raises(ConnectionError) as error:
        reject_secret_reflection(value, Credentials(protected))
    assert error.value.code == "response"
    assert SECRET not in str(error.value)
    reject_secret_reflection(
        {"subject": "safe-subject", "label": "Bearer"}, Credentials(protected)
    )


@pytest.mark.parametrize(
    "raw",
    [
        b'data: {"type":"response.completed","type":"response.completed","response":{}}\n\n',
        b'data: {"type":"response.completed","response":{"number":NaN}}\n\n',
        b"event: error\n\n",
        b'event: response.completed\ndata: {"type":"response.failed"}\n\n',
    ],
)
def test_plan_uses_strict_event_json_and_rejects_ambiguous_or_empty_errors(raw):
    reply = HTTPResponse(200, raw, headers={"content-type": "text/event-stream"})
    with pytest.raises(ConnectionError):
        run("chatgpt_plan", FakeTransport(reply))


@pytest.mark.parametrize(
    "item",
    [
        {"type": "function_call", "name": "danger"},
        {"type": "message", "content": [{"type": "refusal", "refusal": "blocked"}]},
    ],
)
def test_plan_rejects_tools_or_refusals_even_if_later_terminal_response_omits_them(
    item,
):
    reply = sse(
        {"type": "response.output_item.done", "item": item},
        {"type": "response.completed", "response": openai_output()},
    )
    with pytest.raises(ConnectionError):
        run("chatgpt_plan", FakeTransport(reply))


def test_discovery_does_not_publish_reflected_secret_label():
    transport = FakeTransport(
        response(
            {
                "data": [
                    {
                        "id": "model",
                        "display_name": base64.b64encode(SECRET.encode()).decode(),
                    }
                ]
            }
        )
    )
    with pytest.raises(ConnectionError):
        list_models("openai_api", credentials("openai_api"), transport)


@pytest.mark.parametrize("output_tokens", [0, 4097, True, 1.5])
def test_invalid_generation_budget_never_sends_request(output_tokens):
    transport = FakeTransport()
    with pytest.raises(ConnectionError) as error:
        run("openai_api", transport, output_tokens=output_tokens)
    assert error.value.code == "invalid"
    assert transport.requests == []


@pytest.mark.parametrize("cursor", [SECRET, base64.b64encode(SECRET.encode()).decode()])
def test_reflected_credential_cursor_never_reaches_second_request_url(cursor):
    transport = FakeTransport(response({"models": [], "nextPageToken": cursor}))
    with pytest.raises(ConnectionError) as error:
        list_models("gemini_api", credentials("gemini_api"), transport)
    assert error.value.code == "response"
    assert len(transport.requests) == 1
    assert cursor not in transport.requests[0].url


@pytest.mark.parametrize("cursor", [False, 0, [], {}])
def test_malformed_falsy_cursor_is_not_treated_as_completed_discovery(cursor):
    transport = FakeTransport(response({"models": [], "nextPageToken": cursor}))
    with pytest.raises(ConnectionError) as error:
        list_models("gemini_api", credentials("gemini_api"), transport)
    assert error.value.code == "response"

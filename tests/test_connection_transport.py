"""Networking tests use in-memory HTTPX doubles and never provider endpoints."""

import asyncio
import json
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from tsfm_lab.connections.contracts import ConnectionError, Limits
from tsfm_lab.connections.transport import (
    HTTPRequest,
    HTTPResponse,
    HTTPXTransport,
    decode_json,
    read_json,
)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":1e999}',
        b"not-json",
        b'{"a":' + b"[" * 40 + b"0" + b"]" * 40 + b"}",
    ],
)
def test_strict_json_rejects_duplicates_nonfinite_and_deep_output(raw):
    with pytest.raises(ConnectionError):
        decode_json(raw)


@pytest.mark.parametrize(
    "url",
    [
        "http://api.openai.com/v1/models",
        "https://api.openai.com.evil/v1/models",
        "https://api.openai.com:443/v1/models",
        "https://api.openai.com/v1/modelsevil",
        "https://api.openai.com/v1/models?key=fake",
        "https://api.openai.com/v1/models#fake",
        "\thttps://api.openai.com/v1/models",
        "https://user@api.openai.com/v1/models",
        "https://generativelanguage.googleapis.com/v1beta/models/../secret:generateContent",
        "https://auth.openai.com/oauth/revoke?pageToken=fake",
        "https://auth.openai.com/oauth/../revoke",
        "https://auth.openai.com/oauth/%2e%2e/revoke",
        "https://auth.openai.com//oauth/revoke",
        "https://auth.openai.com:443/oauth/revoke",
    ],
)
def test_transport_refuses_untrusted_destinations(url):
    with pytest.raises(ConnectionError):
        HTTPRequest("GET", url)


@pytest.mark.parametrize(
    "path",
    [
        "/.well-known/openid-configuration",
        "/.well-known/jwks.json",
        "/api/accounts/oauth/token",
        "/api/accounts/oauth/revoke",
        "/oauth/revoke",
        "/future/discovered-endpoint",
    ],
)
def test_auth_discovery_can_supply_normalized_paths_on_only_the_pinned_issuer(path):
    assert HTTPRequest("POST", "https://auth.openai.com" + path).url.endswith(path)


def test_response_bounds_cancellation_error_body_and_cleanup():
    closed = []
    with pytest.raises(ConnectionError, match="local limits"):
        with HTTPResponse(
            200, b'{"a":"123456"}', close=lambda: closed.append(True)
        ) as response:
            read_json(response, limits=Limits(max_response_bytes=5))
    assert closed == [True]
    with pytest.raises(ConnectionError, match="cancelled"):
        read_json(HTTPResponse(200, b"{}"), limits=Limits(), cancel=lambda: True)
    with pytest.raises(ConnectionError) as error:
        read_json(
            HTTPResponse(401, b"fake-secret-should-never-be-parsed"), limits=Limits()
        )
    assert error.value.code == "auth"
    assert "fake-secret" not in str(error.value)


@pytest.fixture
def fake_httpx(monkeypatch):
    calls = {"clients": [], "transports": [], "requests": [], "sends": [], "closed": []}

    class Timeout(Exception):
        pass

    class Response:
        headers = {}
        status_code = 200

        async def aiter_bytes(self, **kwargs):
            calls.setdefault("chunk_sizes", []).append(kwargs)
            yield b'{"ok":true}'

        async def aclose(self):
            calls["closed"].append("response")

    class Client:
        def __init__(self, **kwargs):
            calls["clients"].append(kwargs)

        def build_request(self, *args, **kwargs):
            calls["requests"].append((args, kwargs))
            if kwargs.get("json") is not None:
                content = json.dumps(kwargs["json"]).encode()
            elif kwargs.get("data") is not None:
                content = urlencode(kwargs["data"]).encode()
            else:
                content = b""
            return SimpleNamespace(content=content)

        async def send(self, request, **kwargs):
            calls["sends"].append(kwargs)
            if calls.get("header_delay"):
                await asyncio.sleep(calls["header_delay"])
            return calls.get("response", Response())

        async def aclose(self):
            calls["closed"].append("client")

    def transport(**kwargs):
        calls["transports"].append(kwargs)
        return object()

    module = SimpleNamespace(
        AsyncClient=Client,
        AsyncHTTPTransport=transport,
        Timeout=lambda *a, **k: object(),
        TimeoutException=Timeout,
    )
    monkeypatch.setitem(sys.modules, "httpx", module)
    return calls, module, Response


def test_transport_disables_environment_redirects_retries_and_compression(
    fake_httpx, monkeypatch
):
    calls, _, _ = fake_httpx
    monkeypatch.setenv("HTTPS_PROXY", "https://fake-secret.invalid")
    monkeypatch.setenv("SSL_CERT_FILE", "fake-secret-file")
    request = HTTPRequest(
        "GET", "https://api.openai.com/v1/models", {"Authorization": "Bearer fake-key"}
    )
    assert "fake-key" not in repr(request)
    with HTTPXTransport().send(request, limits=Limits()) as response:
        assert read_json(response, limits=Limits()) == {"ok": True}
    assert len(calls["sends"]) == 1
    assert calls["transports"] == [{"retries": 0, "trust_env": False, "verify": True}]
    assert calls["clients"][0]["trust_env"] is False
    assert calls["clients"][0]["follow_redirects"] is False
    assert calls["clients"][0]["verify"] is True
    assert calls["requests"][0][1]["headers"]["Accept-Encoding"] == "identity"
    assert {"response", "client"} <= set(calls["closed"])
    assert calls["chunk_sizes"] == [{}]


def test_transport_redacts_network_failure_and_closes_on_interrupt(fake_httpx):
    calls, module, _ = fake_httpx

    async def failure(*args, **kwargs):
        raise RuntimeError("fake-key reflected in low-level error")

    module.AsyncClient.send = failure
    with pytest.raises(ConnectionError) as error:
        HTTPXTransport().send(
            HTTPRequest("GET", "https://api.openai.com/v1/models"), limits=Limits()
        )
    assert "fake-key" not in str(error.value)
    assert error.value.__suppress_context__
    assert calls["closed"] == ["client"]
    calls["closed"].clear()

    async def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    module.AsyncClient.send = interrupt
    with pytest.raises(KeyboardInterrupt):
        HTTPXTransport().send(
            HTTPRequest("GET", "https://api.openai.com/v1/models"), limits=Limits()
        )
    assert calls["closed"] == ["client"]


def test_optional_httpx_is_not_required_until_network_selected(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ConnectionError) as error:
        HTTPXTransport().send(
            HTTPRequest("GET", "https://api.openai.com/v1/models"), limits=Limits()
        )
    assert error.value.code == "dependency"


def test_total_timeout_covers_slow_headers_without_real_network(fake_httpx):
    calls, _, _ = fake_httpx
    calls["header_delay"] = 10
    start = time.monotonic()
    with pytest.raises(ConnectionError) as error:
        HTTPXTransport().send(
            HTTPRequest("GET", "https://api.openai.com/v1/models"),
            limits=Limits(timeout_seconds=0.04),
        )
    assert error.value.code == "timeout"
    assert time.monotonic() - start < 1
    assert len(calls["sends"]) == 1
    assert calls["closed"] == ["client"]


def test_total_timeout_covers_slow_drip_body_and_discards_partial_response(fake_httpx):
    calls, _, Response = fake_httpx

    class Drip(Response):
        async def aiter_bytes(self, **kwargs):
            for _ in range(1000):
                await asyncio.sleep(0.005)
                calls["drips"] = calls.get("drips", 0) + 1
                yield b" "

    calls["response"] = Drip()
    start = time.monotonic()
    with pytest.raises(ConnectionError) as error:
        HTTPXTransport().send(
            HTTPRequest("GET", "https://api.openai.com/v1/models"),
            limits=Limits(timeout_seconds=0.04),
        )
    assert error.value.code == "timeout"
    assert 0 < calls["drips"] < 1000
    assert time.monotonic() - start < 1
    assert {"response", "client"} <= set(calls["closed"])


def test_cancellation_interrupts_pending_headers_before_timeout(fake_httpx):
    calls, _, _ = fake_httpx
    calls["header_delay"] = 10
    start = time.monotonic()
    with pytest.raises(ConnectionError) as error:
        HTTPXTransport().send(
            HTTPRequest("GET", "https://api.openai.com/v1/models"),
            limits=Limits(timeout_seconds=2),
            cancel=lambda: time.monotonic() - start > 0.03,
        )
    assert error.value.code == "cancelled"
    assert time.monotonic() - start < 1
    assert len(calls["sends"]) == 1 and calls["closed"] == ["client"]


def test_response_is_bounded_before_return_and_resources_already_closed(fake_httpx):
    calls, _, _ = fake_httpx
    with pytest.raises(ConnectionError) as error:
        HTTPXTransport().send(
            HTTPRequest("GET", "https://api.openai.com/v1/models"),
            limits=Limits(max_response_bytes=3),
        )
    assert error.value.code == "bounds"
    assert {"response", "client"} <= set(calls["closed"])


def test_actual_form_encoding_is_bounded_before_send(fake_httpx):
    calls, _, _ = fake_httpx
    with pytest.raises(ConnectionError) as error:
        HTTPXTransport().send(
            HTTPRequest(
                "POST",
                "https://auth.openai.com/api/accounts/token",
                form_body={"code": "+" * 10},
            ),
            limits=Limits(max_input_bytes=30),
        )
    assert error.value.code == "bounds" and calls["sends"] == []
    assert calls["closed"] == ["client"]


def test_sync_transport_inside_running_loop_fails_without_secret_worker_or_request(
    fake_httpx,
):
    calls, _, _ = fake_httpx

    async def nested():
        with pytest.raises(ConnectionError) as error:
            HTTPXTransport().send(
                HTTPRequest("GET", "https://api.openai.com/v1/models"), limits=Limits()
            )
        assert error.value.code == "network"

    asyncio.run(nested())
    assert calls["clients"] == calls["sends"] == []


def test_installed_httpx_runtime_works_with_in_memory_transport_only(monkeypatch):
    httpx = pytest.importorskip("httpx")
    captured = []

    def handle(request):
        captured.append(request)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(handle)
    )
    with HTTPXTransport().send(
        HTTPRequest(
            "POST",
            "https://api.openai.com/v1/responses",
            {"Authorization": "Bearer fake-key"},
            json_body={"input": "fixed fake text"},
        ),
        limits=Limits(),
    ) as response:
        assert read_json(response, limits=Limits()) == {"ok": True}
    assert len(captured) == 1
    assert captured[0].headers["Authorization"] == "Bearer fake-key"
    assert captured[0].headers["Accept-Encoding"] == "identity"
    assert json.loads(captured[0].content) == {"input": "fixed fake text"}


def test_installed_httpx_slow_headers_obey_deadline_without_network(monkeypatch):
    httpx = pytest.importorskip("httpx")
    calls = []

    async def slow(request):
        calls.append(request)
        await asyncio.sleep(10)
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(slow)
    )
    start = time.monotonic()
    with pytest.raises(ConnectionError) as error:
        HTTPXTransport().send(
            HTTPRequest("GET", "https://api.openai.com/v1/models"),
            limits=Limits(timeout_seconds=0.04),
        )
    assert error.value.code == "timeout"
    assert len(calls) == 1
    assert time.monotonic() - start < 1

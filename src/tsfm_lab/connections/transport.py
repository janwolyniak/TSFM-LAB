"""Redacted HTTP boundaries; networking is lazy and always explicitly invoked."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic
from typing import Protocol
from urllib.parse import parse_qsl, urlsplit

from tsfm_lab.connections.contracts import CancelCheck, ConnectionError, Limits


@dataclass(repr=False)
class HTTPRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    json_body: dict | None = None
    form_body: dict[str, str] | None = None

    def __post_init__(self):
        if self.method not in {"GET", "POST"} or (
            self.json_body is not None and self.form_body is not None
        ):
            raise ConnectionError("invalid")
        validate_url(self.url)

    def __repr__(self):
        return "<HTTPRequest redacted>"


class HTTPResponse:
    def __init__(
        self, status: int, body: bytes = b"", *, chunks=None, headers=None, close=None
    ):
        self.status, self.headers = status, headers or {}
        self._chunks = chunks if chunks is not None else (body,)
        self._close = close or (lambda: None)
        self._closed = False

    def __repr__(self):
        return "<HTTPResponse redacted>"

    def iter_bytes(self):
        yield from self._chunks

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._close()
            except Exception:
                raise ConnectionError("network") from None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Transport(Protocol):
    def send(
        self, request: HTTPRequest, *, limits: Limits, cancel: CancelCheck = None
    ) -> HTTPResponse: ...


def validate_url(url):
    try:
        parts = urlsplit(url)
        prefixes = {
            "api.openai.com": ("/v1/models", "/v1/responses"),
            "auth.openai.com": ("/.well-known/", "/api/accounts/"),
            "api.anthropic.com": ("/v1/models", "/v1/messages"),
            "generativelanguage.googleapis.com": ("/v1beta/models",),
        }
        if (
            parts.scheme != "https"
            or parts.hostname not in prefixes
            or parts.netloc != parts.hostname
            or parts.fragment
            or parts.username
            or parts.password
            or not (
                (
                    parts.hostname == "auth.openai.com"
                    and not parts.query
                    and re.fullmatch(r"/[A-Za-z0-9._~/-]{0,199}", parts.path)
                    and "//" not in parts.path
                    and not any(
                        segment in {".", ".."} for segment in parts.path.split("/")
                    )
                )
                or (
                    parts.hostname == "generativelanguage.googleapis.com"
                    and re.fullmatch(
                        r"/v1beta/models(?:/[A-Za-z0-9][A-Za-z0-9._:-]{0,199}:generateContent)?",
                        parts.path,
                    )
                )
                or (
                    parts.hostname
                    not in {"auth.openai.com", "generativelanguage.googleapis.com"}
                    and parts.path in prefixes[parts.hostname]
                )
            )
            or any(
                key not in {"limit", "after_id", "pageSize", "pageToken"}
                for key, _ in parse_qsl(parts.query, keep_blank_values=True)
            )
            or "\\" in url
            or any(ord(char) < 33 or ord(char) == 127 for char in url)
        ):
            raise ValueError
    except (TypeError, ValueError, AttributeError):
        raise ConnectionError("invalid") from None


def _checked_json(raw):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def constant(value):
        raise ValueError

    value = json.loads(raw, object_pairs_hook=object_pairs, parse_constant=constant)
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 32 or count > 10000:
            raise ConnectionError("bounds")
        if isinstance(item, dict):
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            pending.extend((v, depth + 1) for v in item)
        elif isinstance(item, float) and not isfinite(item):
            raise ConnectionError("response")
    return value


def decode_json(raw):
    """Strict shared JSON parsing without returning provider text in failures."""
    try:
        if (
            not isinstance(raw, (bytes, str))
            or len(raw.encode() if isinstance(raw, str) else raw) > 1048576
        ):
            raise ConnectionError("bounds")
        return _checked_json(raw)
    except ConnectionError:
        raise
    except Exception:
        raise ConnectionError("response") from None


def check_status(status):
    if type(status) is not int:
        raise ConnectionError("response")
    if 200 <= status < 300:
        return
    code = {
        401: "auth",
        403: "permission",
        404: "model",
        408: "timeout",
        429: "limit",
    }.get(status, "provider")
    raise ConnectionError(code)


def read_json(response: HTTPResponse, *, limits: Limits, cancel: CancelCheck = None):
    check_status(response.status)
    raw = bytearray()
    deadline = monotonic() + limits.timeout_seconds
    try:
        for chunk in response.iter_bytes():
            if monotonic() >= deadline:
                raise ConnectionError("timeout")
            if cancel and cancel():
                raise ConnectionError("cancelled")
            if type(chunk) is not bytes:
                raise ConnectionError("response")
            raw.extend(chunk)
            if len(raw) > limits.max_response_bytes:
                raise ConnectionError("bounds")
        if cancel and cancel():
            raise ConnectionError("cancelled")
        if monotonic() >= deadline:
            raise ConnectionError("timeout")
        result = decode_json(bytes(raw))
        if type(result) is not dict:
            raise ConnectionError("response")
        return result
    except ConnectionError:
        raise
    except Exception:
        raise ConnectionError("response") from None


class HTTPXTransport:
    """One bounded async exchange behind the synchronous provider boundary.

    The entire response stays private and bounded until the exchange completes.
    DNS resolver calls can outlive cancellation inside the operating system.
    """

    def __repr__(self):
        return "<HTTPXTransport>"

    def send(self, request, *, limits, cancel=None):
        try:
            if not isinstance(request, HTTPRequest) or not isinstance(limits, Limits):
                raise ConnectionError("invalid")
            validate_url(request.url)
            if cancel and cancel():
                raise ConnectionError("cancelled")
            try:
                import httpx
            except ImportError:
                raise ConnectionError("dependency") from None
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                # Never move credentials into an implicit worker to nest loops.
                raise ConnectionError("network")
            return asyncio.run(self._bounded(request, limits, cancel, httpx))
        except ConnectionError as exc:
            raise ConnectionError(exc.code) from None
        except Exception as exc:
            code = "timeout" if isinstance(exc, TimeoutError) else "network"
            raise ConnectionError(code) from None

    async def _bounded(self, request, limits, cancel, httpx):
        current = asyncio.current_task()
        stopped = []

        async def cancellation_watch():
            try:
                while True:
                    await asyncio.sleep(0.025)
                    if cancel and cancel():
                        stopped.append("cancelled")
                        current.cancel()
                        return
            except ConnectionError as exc:
                stopped.append(exc.code)
                current.cancel()
            except Exception:
                stopped.append("network")
                current.cancel()

        watcher = asyncio.create_task(cancellation_watch()) if cancel else None
        try:
            async with asyncio.timeout(limits.timeout_seconds):
                return await self._exchange(request, limits, cancel, httpx)
        except asyncio.CancelledError:
            if stopped:
                raise ConnectionError(stopped[0]) from None
            raise
        except (TimeoutError, httpx.TimeoutException):
            raise ConnectionError("timeout") from None
        finally:
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)

    async def _exchange(self, request, limits, cancel, httpx):
        client = response = None
        deadline = monotonic() + limits.timeout_seconds
        try:
            body = (
                request.json_body
                if request.json_body is not None
                else request.form_body
            )
            if (
                body is not None
                and len(json.dumps(body, allow_nan=False).encode())
                > limits.max_input_bytes
            ):
                raise ConnectionError("bounds")
            timeout = httpx.Timeout(
                limits.timeout_seconds,
                connect=min(5.0, limits.timeout_seconds),
                read=min(5.0, limits.timeout_seconds),
                write=min(5.0, limits.timeout_seconds),
                pool=min(1.0, limits.timeout_seconds),
            )
            client = httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                verify=True,
                timeout=timeout,
                transport=httpx.AsyncHTTPTransport(
                    retries=0, trust_env=False, verify=True
                ),
            )
            outgoing = client.build_request(
                request.method,
                request.url,
                headers={**request.headers, "Accept-Encoding": "identity"},
                json=request.json_body,
                data=request.form_body,
            )
            if len(outgoing.content) > limits.max_input_bytes:
                raise ConnectionError("bounds")
            response = await client.send(outgoing, stream=True)
            if response.headers.get("content-encoding", "identity").casefold() not in {
                "",
                "identity",
            }:
                raise ConnectionError("response")
            chunks, total = [], 0
            async for chunk in response.aiter_bytes():
                if cancel and cancel():
                    raise ConnectionError("cancelled")
                if monotonic() >= deadline:
                    raise ConnectionError("timeout")
                if type(chunk) is not bytes:
                    raise ConnectionError("response")
                total += len(chunk)
                if total > limits.max_response_bytes:
                    raise ConnectionError("bounds")
                chunks.append(chunk)
            if cancel and cancel():
                raise ConnectionError("cancelled")
            return HTTPResponse(
                response.status_code,
                chunks=tuple(chunks),
                headers=dict(response.headers),
            )
        finally:
            interrupted = False
            for resource in (response, client):
                if resource is not None:
                    try:
                        await asyncio.wait_for(resource.aclose(), timeout=0.2)
                    except asyncio.CancelledError:
                        interrupted = True
                    except Exception:
                        pass
            if interrupted:
                raise asyncio.CancelledError

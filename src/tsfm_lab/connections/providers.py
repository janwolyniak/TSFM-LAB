"""Explicit provider protocols with no discovery, retries, or billing fallbacks."""

import base64
import html
import json
import re
import time
import unicodedata
from dataclasses import replace
from math import isfinite
from urllib.parse import quote, quote_plus, urlencode

from tsfm_lab.connections.contracts import KINDS, ConnectionError, Limits, ModelChoice
from tsfm_lab.connections.credentials import Credentials
from tsfm_lab.connections.streams import read_completed_response
from tsfm_lab.connections.transport import HTTPRequest, read_json

_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")
_DEFAULT_LIMITS = Limits()
_URLS = {
    "chatgpt_plan": "https://api.openai.com/v1/models",
    "openai_api": "https://api.openai.com/v1/models",
    "anthropic_api": "https://api.anthropic.com/v1/models",
    "gemini_api": "https://generativelanguage.googleapis.com/v1beta/models",
}


class _Operation:
    def __init__(self, limits, cancel):
        if not isinstance(limits, Limits) or (
            cancel is not None and not callable(cancel)
        ):
            raise ConnectionError("invalid")
        self.limits, self.cancel = limits, cancel
        self.deadline = time.monotonic() + limits.timeout_seconds

    def check(self):
        if self.cancel and self.cancel():
            raise ConnectionError("cancelled")
        if time.monotonic() >= self.deadline:
            raise ConnectionError("timeout")
        return False

    def remaining(self):
        self.check()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ConnectionError("timeout")
        return replace(self.limits, timeout_seconds=remaining)


def _headers(kind, credentials, workspace_id):
    if kind not in KINDS or not isinstance(credentials, Credentials):
        raise ConnectionError("invalid")
    if workspace_id is not None:
        if kind != "anthropic_api" or not _valid_id(workspace_id):
            raise ConnectionError("invalid")
    if kind == "chatgpt_plan":
        protected = credentials.snapshot()
        scopes, expiry = protected.get("scopes"), protected.get("access_expires_at")
        if protected.get("identity_pending") or protected.get("refresh_in_progress"):
            raise ConnectionError("auth")
        if type(scopes) not in (list, tuple) or any(type(s) is not str for s in scopes):
            raise ConnectionError("permission")
        if "chatgpt.tokens.use.direct" not in scopes:
            raise ConnectionError("permission")
        if (
            type(expiry) not in (int, float)
            or not isfinite(expiry)
            or expiry <= time.time()
        ):
            raise ConnectionError("auth")
    secret = credentials.reveal(KINDS[kind][2])
    headers = {"Accept": "application/json", "Authorization": f"Bearer {secret}"}
    if kind == "gemini_api":
        headers = {"Accept": "application/json", "x-goog-api-key": secret}
    elif kind == "anthropic_api":
        headers["anthropic-version"] = "2026-10-21"
        if workspace_id is not None:
            headers["anthropic-workspace-id"] = workspace_id
    return headers


def _valid_id(value):
    return type(value) is str and _MODEL_ID.fullmatch(value) is not None


def _model_id(value, kind):
    if kind == "gemini_api" and type(value) is str and value.startswith("models/"):
        value = value[len("models/") :]
    if not _valid_id(value):
        raise ConnectionError("model")
    return value


def _secret_variants(credentials):
    # A provider can echo authentication material in otherwise successful output.
    protected = credentials.snapshot()
    variants = set()
    for field in (
        "api_key",
        "access_token",
        "refresh_token",
        "id_token",
        "client_secret",
    ):
        secret = protected.get(field)
        if type(secret) is not str or not secret:
            continue
        raw = secret.encode()
        variants.update(
            {
                secret,
                quote(secret, safe=""),
                quote_plus(secret),
                html.escape(secret),
                json.dumps(secret)[1:-1],
                raw.hex(),
                raw.hex().upper(),
                "".join(f"%{c:02x}" for c in raw),
                "".join(f"%{c:02X}" for c in raw),
                "".join(f"&#{ord(c)};" for c in secret),
                "".join(f"&#x{ord(c):x};" for c in secret),
                "".join(f"\\u{ord(c):04x}" for c in secret),
            }
        )
        for encoder in (base64.b64encode, base64.urlsafe_b64encode):
            encoded = encoder(raw).decode()
            variants.update((encoded, encoded.rstrip("=")))
    return variants


def _reject_secret(value, variants):
    if any(variant in value for variant in variants):
        raise ConnectionError("response")


def reject_secret_reflection(value, credentials):
    """Reject reflected credentials before public text or JSON metadata is used."""
    try:
        if not isinstance(credentials, Credentials):
            raise ConnectionError("credentials")
        rendered = value if type(value) is str else json.dumps(value, allow_nan=False)
        if len(rendered.encode()) > 1048576:
            raise ConnectionError("bounds")
        variants = _secret_variants(credentials)
        _reject_secret(rendered, variants)
        pending = [value]
        while pending:
            item = pending.pop()
            if type(item) is str:
                _reject_secret(item, variants)
            elif type(item) is dict:
                pending.extend(item.keys())
                pending.extend(item.values())
            elif type(item) in (list, tuple):
                pending.extend(item)
    except ConnectionError:
        raise
    except Exception:
        raise ConnectionError("response") from None


def list_models(
    kind,
    credentials,
    transport,
    *,
    limits=_DEFAULT_LIMITS,
    cancel=None,
    workspace_id=None,
    endpoint=None,
):
    """Discover only; the caller must choose a model explicitly."""
    try:
        operation = _Operation(limits, cancel)
        from tsfm_lab.connections.endpoints import EXTRA_KINDS, endpoint_for

        endpoint_for(kind, endpoint)
        if kind in EXTRA_KINDS:
            if workspace_id is not None:
                raise ConnectionError("invalid")
            from tsfm_lab.connections.additional import list_models as additional_models

            return additional_models(kind, credentials, transport, operation, endpoint)
        headers = _headers(kind, credentials, workspace_id)
        variants = _secret_variants(credentials)
        models, cursors, identifiers = [], set(), set()
        cursor = None
        for _ in range(limits.max_pages):
            url = _URLS[kind]
            params = {}
            if kind == "anthropic_api":
                params = {"limit": min(100, limits.max_models)}
                if cursor is not None:
                    params["after_id"] = cursor
            elif kind == "gemini_api":
                params = {"pageSize": min(1000, limits.max_models)}
                if cursor is not None:
                    params["pageToken"] = cursor
            if params:
                url += "?" + urlencode(params)
            with transport.send(
                HTTPRequest("GET", url, headers),
                limits=operation.remaining(),
                cancel=operation.check,
            ) as response:
                page = read_json(response, limits=limits, cancel=operation.check)
            operation.check()
            choices, cursor = _model_page(kind, page, limits.max_models)
            for identifier, label in choices:
                operation.check()
                identifier = _model_id(identifier, kind)
                if identifier in identifiers:
                    raise ConnectionError("response")
                if (
                    type(label) is not str
                    or not label.strip()
                    or len(label.encode()) > 512
                    or any(unicodedata.category(c) in {"Cc", "Cf"} for c in label)
                ):
                    raise ConnectionError("response")
                _reject_secret(identifier + " " + label, variants)
                identifiers.add(identifier)
                models.append(ModelChoice(identifier, label))
                if len(models) > limits.max_models:
                    raise ConnectionError("bounds")
            if cursor is None:
                operation.check()
                return models
            if type(cursor) is not str or not cursor or len(cursor.encode()) > 2048:
                raise ConnectionError("response")
            _reject_secret(cursor, variants)
            if cursor in cursors:
                raise ConnectionError("response")
            cursors.add(cursor)
        raise ConnectionError("bounds")
    except ConnectionError:
        raise
    except Exception:
        raise ConnectionError("response") from None


def _model_page(kind, page, max_models):
    key = "models" if kind in {"chatgpt_plan", "gemini_api"} else "data"
    items = page.get(key)
    if type(items) is not list or any(type(item) is not dict for item in items):
        raise ConnectionError("response")
    if len(items) > max_models:
        raise ConnectionError("bounds")
    choices = []
    for item in items:
        if kind == "chatgpt_plan":
            if item.get("visibility") != "list":
                continue
            identifier = item.get("slug")
            label = item.get("display_name", identifier)
        elif kind == "gemini_api":
            methods = item.get("supportedGenerationMethods")
            if type(methods) is not list or any(type(m) is not str for m in methods):
                raise ConnectionError("response")
            if "generateContent" not in methods:
                continue
            identifier = item.get("name")
            label = item.get("displayName", identifier)
        else:
            identifier = item.get("id")
            label = item.get("display_name", identifier)
        choices.append((identifier, label))
    cursor = None
    if kind == "anthropic_api":
        if type(page.get("has_more")) is not bool:
            raise ConnectionError("response")
        if page["has_more"]:
            cursor = page.get("last_id")
            if cursor is None or not items:
                raise ConnectionError("response")
    elif kind == "gemini_api":
        cursor = page.get("nextPageToken")
        if cursor == "":
            cursor = None
    return choices, cursor


def generate(
    kind,
    credentials,
    transport,
    *,
    model,
    instructions,
    text,
    output_tokens=256,
    limits=_DEFAULT_LIMITS,
    cancel=None,
    workspace_id=None,
    endpoint=None,
):
    """Make exactly one bounded generation request, accepting only complete text."""
    try:
        operation = _Operation(limits, cancel)
        from tsfm_lab.connections.endpoints import EXTRA_KINDS, endpoint_for

        endpoint_for(kind, endpoint)
        if kind in EXTRA_KINDS and workspace_id is not None:
            raise ConnectionError("invalid")
        headers = (
            _headers(kind, credentials, workspace_id) if kind not in EXTRA_KINDS else {}
        )
        variants = _secret_variants(credentials)
        if kind not in EXTRA_KINDS:
            model = _model_id(model, kind)
        if (
            type(text) is not str
            or not text.strip()
            or type(instructions) is not str
            or type(output_tokens) is not int
            or not 1 <= output_tokens <= 4096
        ):
            raise ConnectionError("invalid")
        if len((text + instructions).encode()) > limits.max_input_bytes:
            raise ConnectionError("bounds")
        _reject_secret(model + instructions + text, variants)
        if kind in EXTRA_KINDS:
            from tsfm_lab.connections.additional import generate as additional_generate

            result = additional_generate(
                kind,
                credentials,
                transport,
                operation,
                endpoint,
                model,
                instructions,
                text,
                output_tokens,
            )
            if not result.strip() or any(
                unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in result
            ):
                raise ConnectionError("response")
            if len(result.encode()) > limits.max_text_bytes:
                raise ConnectionError("bounds")
            _reject_secret(result, variants)
            operation.check()
            return result
        url, body = _generation_request(kind, model, instructions, text, output_tokens)
        headers["Content-Type"] = "application/json"
        if kind == "chatgpt_plan":
            headers["Accept"] = "text/event-stream"
        with transport.send(
            HTTPRequest("POST", url, headers, json_body=body),
            limits=operation.remaining(),
            cancel=operation.check,
        ) as response:
            if kind == "chatgpt_plan":
                payload = read_completed_response(
                    response, limits=limits, cancel=operation.check
                )
            else:
                payload = read_json(response, limits=limits, cancel=operation.check)
        operation.check()
        result = _output_text(kind, payload)
        if not result.strip() or any(
            unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in result
        ):
            raise ConnectionError("response")
        if len(result.encode()) > limits.max_text_bytes:
            raise ConnectionError("bounds")
        _reject_secret(result, variants)
        operation.check()
        return result
    except ConnectionError:
        raise
    except Exception:
        raise ConnectionError("response") from None


def _generation_request(kind, model, instructions, text, output_tokens):
    if kind in {"openai_api", "chatgpt_plan"}:
        body = {
            "model": model,
            "input": text,
            "instructions": instructions,
            "store": False,
            "stream": kind == "chatgpt_plan",
        }
        if kind == "chatgpt_plan":
            body["input"] = [
                {"role": "user", "content": [{"type": "input_text", "text": text}]}
            ]
        else:
            body["max_output_tokens"] = output_tokens
        return "https://api.openai.com/v1/responses", body
    if kind == "anthropic_api":
        return "https://api.anthropic.com/v1/messages", {
            "model": model,
            "system": instructions,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": output_tokens,
            "stream": False,
        }
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "systemInstruction": {"parts": [{"text": instructions}]},
            "generationConfig": {
                "candidateCount": 1,
                "maxOutputTokens": output_tokens,
            },
        },
    )


def _output_text(kind, payload):
    if kind in {"openai_api", "chatgpt_plan"}:
        if (
            payload.get("status") != "completed"
            or payload.get("error") is not None
            or payload.get("incomplete_details") is not None
        ):
            raise ConnectionError("response")
        output = payload.get("output")
        if type(output) is not list:
            raise ConnectionError("response")
        parts = []
        for item in output:
            if type(item) is not dict:
                raise ConnectionError("response")
            if item.get("type") == "reasoning":
                continue
            if (
                item.get("type") != "message"
                or item.get("role") != "assistant"
                or item.get("status") != "completed"
            ):
                raise ConnectionError("response")
            parts.extend(_text_blocks(item.get("content"), "type", "output_text"))
        return "".join(parts)
    if kind == "anthropic_api":
        if (
            payload.get("type") != "message"
            or payload.get("role") != "assistant"
            or payload.get("stop_reason") != "end_turn"
            or payload.get("error") is not None
        ):
            raise ConnectionError("response")
        return "".join(
            _text_blocks(
                payload.get("content"),
                "type",
                "text",
                {"thinking", "redacted_thinking"},
            )
        )
    if payload.get("error") is not None or payload.get("promptFeedback", {}).get(
        "blockReason"
    ):
        raise ConnectionError("response")
    candidates = payload.get("candidates")
    if type(candidates) is not list or len(candidates) != 1:
        raise ConnectionError("response")
    candidate = candidates[0]
    if type(candidate) is not dict or candidate.get("finishReason") != "STOP":
        raise ConnectionError("response")
    ratings = candidate.get("safetyRatings", [])
    if type(ratings) is not list or any(
        type(rating) is not dict or rating.get("blocked") is True for rating in ratings
    ):
        raise ConnectionError("response")
    content = candidate.get("content")
    if type(content) is not dict or content.get("role") != "model":
        raise ConnectionError("response")
    parts = content.get("parts")
    if type(parts) is not list:
        raise ConnectionError("response")
    text = []
    for part in parts:
        if type(part) is not dict or set(part) - {
            "text",
            "thought",
            "thoughtSignature",
        }:
            raise ConnectionError("response")
        if part.get("thought") is True:
            continue
        if "thought" in part and type(part["thought"]) is not bool:
            raise ConnectionError("response")
        if type(part.get("text")) is not str:
            raise ConnectionError("response")
        text.append(part["text"])
    return "".join(text)


def _text_blocks(blocks, key, wanted, ignored=frozenset()):
    if type(blocks) is not list:
        raise ConnectionError("response")
    parts = []
    for block in blocks:
        if type(block) is not dict:
            raise ConnectionError("response")
        if block.get(key) in ignored:
            continue
        if block.get(key) != wanted or type(block.get("text")) is not str:
            raise ConnectionError("response")
        parts.append(block["text"])
    return parts

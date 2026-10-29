"""Conservative Chat Completions and installed local Ollama protocols."""

import re

from tsfm_lab.connections.contracts import ConnectionError, ModelChoice
from tsfm_lab.connections.endpoints import endpoint_for
from tsfm_lab.connections.transport import HTTPRequest, read_json


def model_id(value, kind):
    if (
        type(value) is not str
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or (kind == "ollama_local" and "cloud" in value.casefold())
    ):
        raise ConnectionError("model")
    return value


def _exchange(kind, credentials, transport, operation, endpoint, path, body=None):
    from tsfm_lab.connections.providers import reject_secret_reflection

    root = endpoint_for(kind, endpoint)
    headers = {"Accept": "application/json"}
    if kind != "ollama_local":
        headers["Authorization"] = f"Bearer {credentials.reveal('api_key')}"
    if body is not None:
        headers["Content-Type"] = "application/json"
    reject_secret_reflection(body or {}, credentials)
    with transport.send(
        HTTPRequest(
            "GET" if body is None else "POST",
            root + path,
            headers,
            json_body=body,
            endpoint=root,
        ),
        limits=operation.remaining(),
        cancel=operation.check,
    ) as response:
        result = read_json(
            response, limits=operation.remaining(), cancel=operation.check
        )
    operation.check()
    reject_secret_reflection(result, credentials)
    return result


def list_models(kind, credentials, transport, operation, endpoint):
    payload = _exchange(
        kind,
        credentials,
        transport,
        operation,
        endpoint,
        "/api/tags" if kind == "ollama_local" else "/models",
    )
    items = payload.get("models" if kind == "ollama_local" else "data")
    if type(items) is not list or payload.get("error") is not None:
        raise ConnectionError("response")
    if len(items) > operation.limits.max_models:
        raise ConnectionError("bounds")
    choices, seen = [], set()
    for item in items:
        if type(item) is not dict:
            raise ConnectionError("response")
        value = item.get("name" if kind == "ollama_local" else "id")
        if kind == "ollama_local" and (
            item.get("remote_host")
            or item.get("remote_model")
            or (type(value) is str and "cloud" in value.casefold())
        ):
            continue
        value = model_id(value, kind)
        if value in seen:
            raise ConnectionError("response")
        choices.append(ModelChoice(value, value))
        seen.add(value)
    # Unknown pagination is never silently truncated into a complete discovery.
    if payload.get("has_more") or payload.get("next_page") or payload.get("next"):
        raise ConnectionError("response")
    return choices


def generate(
    kind, credentials, transport, operation, endpoint, model, instructions, text, tokens
):
    model = model_id(model, kind)
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": text},
    ]
    if kind == "ollama_local":
        model = model if model.lower().endswith(":local") else model + ":local"
        # Cloud aliases are rejected even if their names disguise their route.
        shown = _exchange(
            kind,
            credentials,
            transport,
            operation,
            endpoint,
            "/api/show",
            {"model": model},
        )
        if (
            shown.get("remote_host")
            or shown.get("remote_model")
            or type(shown.get("model_info")) is not dict
            or not shown["model_info"]
            or type(shown.get("capabilities")) is not list
            or "completion" not in shown["capabilities"]
        ):
            raise ConnectionError("model")
        path = "/api/chat"
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": tokens},
        }
    else:
        path = "/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": tokens,
            "stream": False,
        }
    result = _exchange(kind, credentials, transport, operation, endpoint, path, body)
    if result.get("error") is not None:
        raise ConnectionError("response")
    if kind == "ollama_local":
        if (
            result.get("done") is not True
            or result.get("done_reason") != "stop"
            or result.get("remote_host")
            or result.get("remote_model")
        ):
            raise ConnectionError("response")
        message = result.get("message")
    else:
        choices = result.get("choices")
        if (
            type(choices) is not list
            or len(choices) != 1
            or type(choices[0]) is not dict
        ):
            raise ConnectionError("response")
        if choices[0].get("finish_reason") != "stop":
            raise ConnectionError("response")
        message = choices[0].get("message")
    if (
        type(message) is not dict
        or message.get("role") != "assistant"
        or type(message.get("content")) is not str
        or any(
            message.get(field) for field in ("tool_calls", "function_call", "refusal")
        )
    ):
        raise ConnectionError("response")
    return message["content"]

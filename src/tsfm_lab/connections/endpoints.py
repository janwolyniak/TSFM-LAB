"""Explicit destination policies for bounded additional assistant protocols."""

import ipaddress
import re
from urllib.parse import urlsplit

from tsfm_lab.connections.contracts import ConnectionError

EXTRA_KINDS = {"deepseek_api", "ollama_local", "compatible_api"}
DEFAULTS = {
    "deepseek_api": "https://api.deepseek.com",
    "ollama_local": "http://127.0.0.1:11434",
}


def endpoint_for(kind, endpoint=None):
    if kind not in EXTRA_KINDS:
        if endpoint is not None:
            raise ConnectionError("invalid")
        return None
    value = endpoint if endpoint is not None else DEFAULTS.get(kind)
    try:
        if type(value) is not str or len(value) > 512:
            raise ValueError
        parts = urlsplit(value)
        if (
            parts.username
            or parts.password
            or parts.query
            or parts.fragment
            or not parts.hostname
            or not parts.netloc.isascii()
            or any(ord(c) < 33 or ord(c) >= 127 for c in value)
            or "\\" in value
            or not re.fullmatch(r"(?:/[A-Za-z0-9_-]+)*/?", parts.path)
            or parts.port == 0
        ):
            raise ValueError
        local = parts.hostname in {"localhost", "127.0.0.1", "::1"}
        if kind == "ollama_local":
            if not local or parts.scheme != "http" or parts.path not in {"", "/"}:
                raise ValueError
        elif parts.scheme != "https":
            if parts.scheme != "http" or not local:
                raise ValueError
        if kind == "deepseek_api" and value.rstrip("/") != DEFAULTS[kind]:
            raise ValueError
        # Reject alternate loopback spellings, ambiguous IPs and URL delimiters.
        if parts.hostname != "localhost":
            try:
                address = ipaddress.ip_address(parts.hostname)
            except ValueError:
                if not re.fullmatch(
                    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*", parts.hostname
                ):
                    raise ValueError from None
            else:
                if address.is_loopback and not local:
                    raise ValueError
        return value.rstrip("/")
    except (ValueError, TypeError):
        raise ConnectionError("invalid") from None


def validate_endpoint_request(url, endpoint):
    """A request may reach only the reviewed API root and protocol paths."""
    local = endpoint_for("compatible_api", endpoint)
    suffix = url.removeprefix(local)
    if not url.startswith(local) or suffix not in {
        "/models",
        "/chat/completions",
        "/api/tags",
        "/api/show",
        "/api/chat",
    }:
        raise ConnectionError("invalid")
    if suffix.startswith("/api/"):
        endpoint_for("ollama_local", endpoint)


def destination(profile):
    kind = profile["kind"]
    return (
        endpoint_for(kind, profile.get("endpoint"))
        or {
            "chatgpt_plan": "https://api.openai.com/v1",
            "openai_api": "https://api.openai.com/v1",
            "anthropic_api": "https://api.anthropic.com/v1",
            "gemini_api": "https://generativelanguage.googleapis.com/v1beta",
        }[kind]
    )

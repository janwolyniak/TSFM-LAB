"""Small public contracts; credentials and provider details never enter errors."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

KINDS = {
    "chatgpt_plan": ("ChatGPT plan", "ChatGPT plan usage", "access_token"),
    "openai_api": ("OpenAI API key", "OpenAI API billing", "api_key"),
    "anthropic_api": ("Anthropic API key", "Anthropic API billing", "api_key"),
    "gemini_api": ("Google Gemini API key", "Gemini API billing", "api_key"),
    "deepseek_api": ("DeepSeek API key", "DeepSeek API billing", "api_key"),
    "ollama_local": ("Local Ollama", "Local server; no cloud key or model pull", None),
    "compatible_api": (
        "Compatible endpoint",
        "Selected endpoint billing; review destination",
        "api_key",
    ),
}

_MESSAGES = {
    "invalid": "The connection settings are invalid.",
    "credentials": "Credentials are missing or unavailable. Add or reconnect this connection.",
    "storage": "Protected credential storage is unavailable. Choose session-only storage explicitly.",
    "state": "Connection state changed. Review the selected connection and try again.",
    "auth": "Authentication failed. Replace the key or sign in again.",
    "permission": "This connection does not have the required permission.",
    "model": "The selected model is unavailable for this request. Choose a model explicitly.",
    "limit": "The provider usage limit was reached. No other billing route was selected.",
    "timeout": "The request timed out. No response was accepted.",
    "cancelled": "The request was cancelled. No response was accepted.",
    "network": "The provider could not be reached. No response was accepted.",
    "provider": "The provider request failed. No response was accepted.",
    "response": "The provider returned an incomplete or unsupported response.",
    "bounds": "The request or response exceeded the local limits.",
    "dependency": "This optional connection runtime is unavailable. See the connection setup guide.",
}


class ConnectionError(ValueError):
    def __init__(self, code: str):
        self.code = code if code in _MESSAGES else "provider"
        super().__init__(_MESSAGES[self.code])


@dataclass(frozen=True)
class Limits:
    timeout_seconds: float = 30.0
    max_input_bytes: int = 32768
    max_response_bytes: int = 262144
    max_text_bytes: int = 65536
    max_models: int = 1000
    max_pages: int = 10

    def __post_init__(self):
        if (
            type(self.timeout_seconds) not in (int, float)
            or not isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 120
        ):
            raise ConnectionError("invalid")
        for name in (
            "max_input_bytes",
            "max_response_bytes",
            "max_text_bytes",
            "max_models",
            "max_pages",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ConnectionError("invalid")
        if (
            self.max_input_bytes > 262144
            or self.max_response_bytes > 1048576
            or self.max_text_bytes > 262144
            or self.max_models > 1000
            or self.max_pages > 10
        ):
            raise ConnectionError("invalid")


@dataclass(frozen=True)
class ModelChoice:
    identifier: str
    label: str


CancelCheck = Callable[[], bool] | None

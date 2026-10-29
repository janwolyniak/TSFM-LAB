"""Bound user intent and untrusted annotations without granting execution authority."""

import json
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from tsfm_lab.connections.transport import decode_json

_PATH = re.compile(
    # A slash inside a unit, repository ID, or HTTPS URL is not a machine path.
    r"(?:^|[\s\"'`(=:\[,;])(?:~[/\\]|\.\.?[/\\]|[A-Za-z]:[/\\]|/(?!/)[^\s/])"
    r"|(?:^|[\s\"'`(=\[,;])(?://[^\s/]|\\\\[^\s\\])"
    r"|(?:^|[\s\"'`(=:\[,;])\\\\[^\s\\]"
    r"|\bfile:\s*[/\\]"
    r"|\b\S+\.(?:csv|parquet|pq|ya?ml|json|toml|env)\b",
    re.I,
)
_CREDENTIAL = re.compile(
    r"\b(?:sk-(?:proj-|ant-(?:api\d+-)?)?[A-Za-z0-9_-]{8,}"
    r"|AIza[A-Za-z0-9_-]{12,}|ya29\.[A-Za-z0-9_-]{12,}"
    r"|eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,})\b"
    r"|\bBearer\s+[A-Za-z0-9._~+/-]{8,}={0,2}"
    r"|\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|client[_ -]?secret|private[_ -]?key|authorization)\s*[:=]",
    re.I,
)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_FUTURE = re.compile(
    r"\b(?:tomorrow|next\s+(?:week|month|year)|future\s+(?:targets?|forecast(?:s|ing)?))\b",
    re.I,
)
_ADVICE = re.compile(
    r"\b(?:guaranteed|outperforms?|best\s+model|you\s+should\s+(?:buy|sell|short)|will\s+(?:profit|outperform))\b",
    re.I,
)


class AssistanceError(ValueError):
    def __init__(self, code="response"):
        messages = {
            "input": "Use a short plain-language goal without files, paths, credentials, code, or data tables. Choose files locally.",
            "future": "This goal asks for future forecasting. Select future forecasting explicitly and review a new preview; no historical experiment was substituted.",
            "mode": "The assistant proposal conflicts with the human-selected forecast mode. Choose the mode explicitly and review a fresh preview.",
            "response": "The assistant response could not be accepted. Nothing was executed or saved. Edit the goal or use guided setup.",
            "state": "The request changed or expired. Review a new preview before sending.",
            "conflict": "The clarification changed an earlier choice. Earlier choices were kept; use guided review to edit them explicitly.",
        }
        self.code = code if code in messages else "response"
        super().__init__(messages[self.code])


def safe_text(value, *, maximum=4096):
    try:
        valid = (
            type(value) is str
            and bool(value.strip())
            and len(value.encode("utf-8")) <= maximum
        )
    except UnicodeError:
        valid = False
    if not valid:
        raise AssistanceError("input")
    if any(unicodedata.category(char).startswith("C") for char in value):
        raise AssistanceError("input")
    return value.strip()


def user_goal(value):
    value = safe_text(value)
    if (
        _PATH.search(value)
        or _CREDENTIAL.search(value)
        or "```" in value
        or "{" in value
        or "[" in value
    ):
        raise AssistanceError("input")
    if re.search(r"\b\d+(?:\s*[,;]\s*[-+]?\d+){3,}", value):
        raise AssistanceError("input")
    return value


def validate_context(context):
    """Snapshot bounded evidence before preview or rendering any of its facts.

    Field projection and numerical validation belong to the evidence builders.
    This boundary also rejects private-looking content inside allowed metadata.
    It preserves every accepted scalar exactly rather than redacting or rounding.
    """
    from tsfm_lab.assistant.evidence import bounded_json

    try:
        snapshot = json.loads(bounded_json(context))
        if (
            set(snapshot)
            != {"schema_version", "kind", "data", "omitted", "glossary", "limitations"}
            or type(snapshot["schema_version"]) is not int
            or snapshot["schema_version"] != 1
            or snapshot["kind"]
            not in {
                "model",
                "model_comparison",
                "experiment",
                "finance",
                "future_forecast",
            }
            or type(snapshot["data"]) is not dict
            or not snapshot["data"]
            or type(snapshot["omitted"]) is not dict
            or any(
                type(value) not in {int, bool} or value < 0
                for value in snapshot["omitted"].values()
            )
            or type(snapshot["glossary"]) is not dict
            or any(type(value) is not str for value in snapshot["glossary"].values())
            or type(snapshot["limitations"]) is not list
            or len(snapshot["limitations"]) > 8
            or any(type(value) is not str for value in snapshot["limitations"])
        ):
            raise AssistanceError("input")
        pending = [(snapshot, 0)]
        while pending:
            value, depth = pending.pop()
            if depth > 16:
                raise AssistanceError("input")
            if type(value) is dict:
                pending.extend((key, depth + 1) for key in value)
                pending.extend((item, depth + 1) for item in value.values())
            elif type(value) is list:
                pending.extend((item, depth + 1) for item in value)
            elif type(value) is str and (
                len(value.encode("utf-8")) > 512
                or any(unicodedata.category(char).startswith("C") for char in value)
                or _PATH.search(value)
                or _CREDENTIAL.search(value)
                or "```" in value
            ):
                raise AssistanceError("input")
        return snapshot
    except Exception:
        raise AssistanceError("input") from None


def require_historical_goal(value):
    if _FUTURE.search(value):
        raise AssistanceError("future")


@dataclass(frozen=True)
class Annotation:
    text: str
    evidence_ids: tuple[str, ...]


def _numbers(value):
    if isinstance(value, bool) or value is None:
        return set()
    result = set()
    for number in _NUMBER.findall(str(value)):
        try:
            result.add(Decimal(number))
        except InvalidOperation:
            raise AssistanceError() from None
    return result


def decode_annotations(value, context):
    """Check structure/references/numbers; arbitrary prose is still an interpretation."""
    from tsfm_lab.assistant.evidence import fact_map

    try:
        payload = decode_json(value)
        if (
            type(payload) is not dict
            or set(payload) != {"schema_version", "annotations"}
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
        ):
            raise AssistanceError()
        items = payload["annotations"]
        if type(items) is not list or not 1 <= len(items) <= 6:
            raise AssistanceError()
        facts = fact_map(validate_context(context))
        result = []
        for item in items:
            if type(item) is not dict or set(item) != {"text", "evidence_ids"}:
                raise AssistanceError()
            text = safe_text(item["text"], maximum=1800)
            references = item["evidence_ids"]
            if (
                type(references) is not list
                or not 1 <= len(references) <= 16
                or any(type(ref) is not str or ref not in facts for ref in references)
                or len(set(references)) != len(references)
            ):
                raise AssistanceError()
            allowed = set().union(*(_numbers(facts[ref]) for ref in references))
            if (
                not _numbers(text) <= allowed
                or _PATH.search(text)
                or _CREDENTIAL.search(text)
                or _ADVICE.search(text)
                or "```" in text
            ):
                raise AssistanceError()
            result.append(Annotation(text, tuple(references)))
        return tuple(result)
    except Exception:
        raise AssistanceError() from None

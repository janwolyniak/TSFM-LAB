"""Bounded SSE parsing; only a final completed response is accepted."""

from tsfm_lab.connections.contracts import ConnectionError
from tsfm_lab.connections.transport import check_status, decode_json


def read_completed_response(response, *, limits, cancel):
    """Buffer privately so interrupted streams cannot expose partial answers."""
    check_status(response.status)
    content_type = next(
        (v for k, v in response.headers.items() if k.lower() == "content-type"), ""
    )
    if content_type.split(";", 1)[0].strip().lower() != "text/event-stream":
        raise ConnectionError("response")
    raw = bytearray()
    for chunk in response.iter_bytes():
        cancel()
        if type(chunk) is not bytes:
            raise ConnectionError("response")
        raw.extend(chunk)
        if len(raw) > limits.max_response_bytes:
            raise ConnectionError("bounds")
    cancel()
    try:
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeError:
        raise ConnectionError("response") from None
    frames = text.split("\n\n")
    if frames[-1].strip():
        raise ConnectionError("response")
    completed = None
    done = False
    for frame in frames[:-1]:
        cancel()
        event, data = _frame(frame)
        if event in {"error", "response.failed", "response.incomplete"}:
            raise ConnectionError("response")
        if not data:
            continue
        if data == "[DONE]":
            if completed is None or done:
                raise ConnectionError("response")
            done = True
            continue
        if completed is not None or done:
            raise ConnectionError("response")
        payload = decode_json(data.encode("utf-8"))
        if type(payload) is not dict:
            raise ConnectionError("response")
        kind = payload.get("type")
        if type(kind) is not str or (event and event != kind):
            raise ConnectionError("response")
        if kind == "response.completed":
            completed = payload.get("response")
            if type(completed) is not dict:
                raise ConnectionError("response")
        elif kind not in _PROGRESS_EVENTS:
            raise ConnectionError("response")
        elif kind in {"response.output_item.added", "response.output_item.done"}:
            item = payload.get("item")
            if type(item) is not dict or item.get("type") not in {
                "message",
                "reasoning",
            }:
                raise ConnectionError("response")
            if item["type"] == "message" and "content" in item:
                content = item["content"]
                if type(content) is not list or any(
                    type(part) is not dict or part.get("type") != "output_text"
                    for part in content
                ):
                    raise ConnectionError("response")
        elif kind in {"response.content_part.added", "response.content_part.done"}:
            part = payload.get("part")
            if type(part) is not dict or part.get("type") != "output_text":
                raise ConnectionError("response")
    if completed is None:
        raise ConnectionError("response")
    return completed


def _frame(frame):
    event = ""
    data = []
    for line in frame.split("\n"):
        if not line or line.startswith(":"):
            continue
        name, sep, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if name == "event":
            if event or not sep:
                raise ConnectionError("response")
            event = value
        elif name == "data":
            data.append(value)
        elif name not in {"id", "retry"}:
            raise ConnectionError("response")
    return event, "\n".join(data)


_PROGRESS_EVENTS = {
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.reasoning_summary_part.added",
    "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.delta",
    "response.reasoning_summary_text.done",
    "response.reasoning_text.delta",
    "response.reasoning_text.done",
}

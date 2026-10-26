"""Opaque in-memory credential bundles and an injectable protected-store boundary."""

import copy
import json
from typing import Protocol

from tsfm_lab.connections.contracts import ConnectionError


class Credentials:
    __slots__ = ("_values",)

    def __init__(self, values: dict):
        try:
            if type(values) is not dict or not values:
                raise ValueError
            raw = json.dumps(values, allow_nan=False)
            if len(raw.encode()) > 131072:
                raise ValueError
            self._values = json.loads(raw)
        except (TypeError, ValueError, RecursionError):
            raise ConnectionError("credentials") from None

    def __repr__(self):
        return "<Credentials protected>"

    __str__ = __repr__

    def reveal(self, name: str) -> str:
        value = self._values.get(name)
        if (
            type(value) is not str
            or not value
            or len(value) > 65536
            or any(ord(c) < 33 or ord(c) > 126 for c in value)
        ):
            raise ConnectionError("credentials")
        return value

    def snapshot(self) -> dict:
        """For protected store serialization only; never use in public metadata."""
        return copy.deepcopy(self._values)


class SecretStore(Protocol):
    def get(self, reference: str) -> Credentials | None: ...
    def put(self, reference: str, credentials: Credentials) -> None: ...
    def delete(self, reference: str) -> None: ...


class SessionStore:
    def __init__(self):
        self._secrets = {}

    def __repr__(self):
        return "<SessionStore protected>"

    def get(self, reference):
        value = self._secrets.get(reference)
        return Credentials(value.snapshot()) if value is not None else None

    def put(self, reference, credentials):
        if not isinstance(credentials, Credentials):
            raise ConnectionError("credentials")
        self._secrets[reference] = Credentials(credentials.snapshot())

    def delete(self, reference):
        self._secrets.pop(reference, None)

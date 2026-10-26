"""Explicit native-vault storage; no alternative plaintext backend is selected."""

import importlib
import json
import sys
from uuid import UUID

from tsfm_lab.connections.contracts import ConnectionError
from tsfm_lab.connections.credentials import Credentials
from tsfm_lab.connections.transport import decode_json

SERVICE = "TSFM-LAB.connections.v1"


class NativeVault:
    def __init__(self, *, backend=None):
        self._injected_backend = backend
        self._native_backend = None

    def __repr__(self):
        return "<NativeVault protected>"

    def _backend(self):
        if self._injected_backend is not None:
            return self._injected_backend
        if self._native_backend is None:
            names = {
                "darwin": ("keyring.backends.macOS", "Keyring"),
                "win32": ("keyring.backends.Windows", "WinVaultKeyring"),
                "linux": ("keyring.backends.SecretService", "Keyring"),
            }
            if sys.platform not in names:
                raise ConnectionError("storage")
            module, name = names[sys.platform]
            try:
                backend = getattr(importlib.import_module(module), name)()
                if type(backend).__module__ != module or type(backend).__name__ != name:
                    raise ConnectionError("storage")
                if backend.priority <= 0:
                    raise ConnectionError("storage")
                self._native_backend = backend
            except Exception:
                raise ConnectionError("storage") from None
        return self._native_backend

    @staticmethod
    def _reference(reference):
        try:
            if type(reference) is not str or str(UUID(reference)) != reference:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise ConnectionError("invalid") from None

    def get(self, reference):
        self._reference(reference)
        try:
            value = self._backend().get_password(SERVICE, reference)
            return Credentials(decode_json(value)) if value is not None else None
        except Exception:
            raise ConnectionError("storage") from None

    def put(self, reference, credentials):
        self._reference(reference)
        if not isinstance(credentials, Credentials):
            raise ConnectionError("credentials")
        try:
            self._backend().set_password(
                SERVICE, reference, json.dumps(credentials.snapshot(), allow_nan=False)
            )
        except Exception:
            raise ConnectionError("storage") from None

    def delete(self, reference):
        self._reference(reference)
        try:
            backend = self._backend()
            if backend.get_password(SERVICE, reference) is not None:
                backend.delete_password(SERVICE, reference)
        except Exception:
            raise ConnectionError("storage") from None

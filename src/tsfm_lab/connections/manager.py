"""Explicit connection/account/model selection, separate from numerical records."""

import copy
from functools import wraps
from threading import RLock
from time import time
from uuid import uuid4

from tsfm_lab.connections.contracts import KINDS, ConnectionError, Limits
from tsfm_lab.connections.credentials import Credentials, SessionStore
from tsfm_lab.connections.state import StateStore, account, identifier, text
from tsfm_lab.connections.transport import HTTPXTransport
from tsfm_lab.connections.vault import NativeVault

TEST_TEXT = "Reply with exactly: Connection confirmed."
TEST_INSTRUCTIONS = "This is a connection check. Return the requested short plain text. Do not call tools or include credentials."


def public_boundary(method):
    @wraps(method)
    def safe(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except ConnectionError as exc:
            raise ConnectionError(exc.code) from None
        except Exception:
            raise ConnectionError("state") from None

    return safe


class ConnectionManager:
    def __init__(
        self,
        state=None,
        *,
        session=None,
        vault=None,
        transport=None,
        oauth_factory=None,
        limits=None,
    ):
        self.state = state if state is not None else StateStore()
        self.session = session if session is not None else SessionStore()
        self.vault = vault if vault is not None else NativeVault()
        self.transport = transport if transport is not None else HTTPXTransport()
        self.limits = limits if limits is not None else Limits()
        self.oauth_factory = oauth_factory
        self._models, self._previews = {}, {}
        self._blocked = set()
        self._lock = RLock()

    def __repr__(self):
        return "<ConnectionManager protected>"

    def _store(self, mode):
        if mode not in {"session", "vault"}:
            raise ConnectionError("invalid")
        return self.session if mode == "session" else self.vault

    @staticmethod
    def _reference(state):
        used = {p["credential_ref"] for p in state["profiles"]}
        used.update(item["reference"] for item in state["retired_refs"])
        for _ in range(16):
            reference = str(uuid4())
            if reference not in used:
                return reference
        raise ConnectionError("state")

    @staticmethod
    def _public_safe(value, credentials):
        from tsfm_lab.connections.providers import reject_secret_reflection

        reject_secret_reflection(value, credentials)

    @staticmethod
    def _route(profile):
        route = copy.deepcopy(profile)
        route.pop("model")
        return route

    @staticmethod
    def _find(state, connection_id=None):
        connection_id = connection_id or state["selected"]
        for profile in state["profiles"]:
            if profile["id"] == connection_id:
                return profile
        raise ConnectionError("state")

    @staticmethod
    def _public(profile, selected):
        return {
            "id": profile["id"],
            "kind": profile["kind"],
            "label": profile["label"],
            "provider": KINDS[profile["kind"]][0],
            "usage": KINDS[profile["kind"]][1],
            "storage": profile["storage"],
            "model": profile["model"],
            "account": copy.deepcopy(profile["account"]),
            "workspace_id": profile["workspace_id"],
            "selected": profile["id"] == selected,
            "credentials_saved": profile["credential_ref"] is not None,
            "status": (
                "Saved; use explicit discovery/test to verify"
                if profile["credential_ref"] is not None
                else "Disconnected"
            ),
        }

    @public_boundary
    def list(self):
        state = self.state.read()
        return [
            self._public(profile, state["selected"]) for profile in state["profiles"]
        ]

    @public_boundary
    def status(self, connection_id=None):
        state = self.state.read()
        return self._public(self._find(state, connection_id), state["selected"])

    def _write(self, state):
        state["generation"] += 1
        self.state.write(state)
        self._previews.clear()

    @staticmethod
    def _new_profile(kind, label, storage, workspace_id):
        if kind not in KINDS or storage not in {"session", "vault"}:
            raise ConnectionError("invalid")
        text(label, maximum=80)
        if workspace_id is not None:
            text(workspace_id)
            if kind != "anthropic_api":
                raise ConnectionError("invalid")
        return {
            "id": str(uuid4()),
            "kind": kind,
            "label": label,
            "storage": storage,
            "credential_ref": None,
            "model": None,
            "account": None,
            "workspace_id": workspace_id,
            "registration": None,
        }

    @public_boundary
    def add_key(self, kind, label, value, *, storage, workspace_id=None):
        if kind == "chatgpt_plan":
            raise ConnectionError("invalid")
        credentials = Credentials({"api_key": value})
        credentials.reveal("api_key")
        profile = self._new_profile(kind, label, storage, workspace_id)
        self._public_safe(profile, credentials)
        with self._lock, self.state.locked() as state:
            reference = self._reference(state)
            try:
                self._store(storage).put(reference, credentials)
                profile["credential_ref"] = reference
                state["profiles"].append(profile)
                self._write(state)
            except BaseException as exc:
                try:
                    self._store(storage).delete(reference)
                except Exception:
                    pass
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise ConnectionError("storage") from None
        return self.status(profile["id"])

    @public_boundary
    def add_plan(self, label, *, storage):
        profile = self._new_profile("chatgpt_plan", label, storage, None)
        with self._lock, self.state.locked() as state:
            state["profiles"].append(profile)
            self._write(state)
        return self.status(profile["id"])

    @public_boundary
    def select(self, connection_id):
        identifier(connection_id)
        with self._lock, self.state.locked() as state:
            self._find(state, connection_id)
            state["selected"] = connection_id
            self._models.clear()
            self._write(state)
        return self.status(connection_id)

    def _retire(self, state, profile):
        if profile["credential_ref"] is not None:
            item = {
                "reference": profile["credential_ref"],
                "storage": profile["storage"],
            }
            if item not in state["retired_refs"]:
                state["retired_refs"].append(item)

    def _cleanup(self):
        with self.state.locked() as state:
            remaining = []
            for item in state["retired_refs"]:
                try:
                    self._store(item["storage"]).delete(item["reference"])
                except Exception:
                    remaining.append(item)
            if remaining != state["retired_refs"]:
                state["retired_refs"] = remaining
                self._write(state)
            return not remaining

    @public_boundary
    def replace_key(self, connection_id, value, *, workspace_id=None):
        credentials = Credentials({"api_key": value})
        credentials.reveal("api_key")
        with self._lock, self.state.locked() as state:
            profile = self._find(state, connection_id)
            if profile["kind"] == "chatgpt_plan":
                raise ConnectionError("invalid")
            if workspace_id is not None:
                text(workspace_id)
                if profile["kind"] != "anthropic_api":
                    raise ConnectionError("invalid")
            self._public_safe(
                {"label": profile["label"], "workspace_id": workspace_id}, credentials
            )
            reference = self._reference(state)
            store = self._store(profile["storage"])
            try:
                store.put(reference, credentials)
                self._retire(state, profile)
                profile.update(
                    credential_ref=reference, model=None, workspace_id=workspace_id
                )
                self._write(state)
            except BaseException as exc:
                try:
                    store.delete(reference)
                except Exception:
                    pass
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise ConnectionError("storage") from None
            self._models.pop(connection_id, None)
        cleaned = self._cleanup()
        return {**self.status(connection_id), "previous_credentials_removed": cleaned}

    def _oauth(self, state):
        if self.oauth_factory is not None:
            return self.oauth_factory(state["host_id"])
        from tsfm_lab.connections.oauth import OAuthClient

        return OAuthClient(self.transport, host_id=state["host_id"], limits=self.limits)

    @public_boundary
    def plan_setup(self, connection_id):
        state = self.state.read()
        profile = self._find(state, connection_id)
        if profile["kind"] != "chatgpt_plan":
            raise ConnectionError("invalid")
        credentials = (
            self._store(profile["storage"]).get(profile["credential_ref"])
            if profile["credential_ref"]
            else None
        )
        return (
            self._oauth(state),
            copy.deepcopy(profile),
            credentials,
            state["generation"],
        )

    @public_boundary
    def remember_registration(self, connection_id, registration, generation):
        if registration is None:
            return
        account(registration, registration=True)
        with self._lock, self.state.locked() as state:
            if state["generation"] != generation:
                raise ConnectionError("state")
            profile = self._find(state, connection_id)
            profile["registration"] = copy.deepcopy(registration)
            self._write(state)

    @public_boundary
    def finish_sign_in(self, connection_id, result, generation):
        account(result.account)
        self._bound(result.account, result.credentials)
        result.credentials.reveal("id_token")
        self._public_safe(result.account, result.credentials)
        if result.identity_pending:
            raise ConnectionError("auth")
        with self._lock, self.state.locked() as state:
            if state["generation"] != generation:
                raise ConnectionError("state")
            profile = self._find(state, connection_id)
            self._public_safe(profile["label"], result.credentials)
            reference = self._reference(state)
            if profile["kind"] != "chatgpt_plan":
                raise ConnectionError("invalid")
            old = profile["account"]
            if old is not None and any(
                old[key] != result.account[key]
                for key in ("issuer", "subject", "client_id")
            ):
                raise ConnectionError("auth")
            store = self._store(profile["storage"])
            try:
                store.put(reference, result.credentials)
                self._retire(state, profile)
                profile.update(
                    credential_ref=reference,
                    account=copy.deepcopy(result.account),
                    registration={
                        key: result.account[key] for key in ("issuer", "client_id")
                    },
                    model=None,
                )
                self._write(state)
            except BaseException as exc:
                try:
                    store.delete(reference)
                except Exception:
                    pass
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise ConnectionError("storage") from None
            self._models.pop(connection_id, None)
        self._cleanup()
        return self.status(connection_id)

    @staticmethod
    def _bound(saved, credentials):
        values = credentials.snapshot()
        if saved is None or any(
            values.get(key) != saved[key] for key in ("issuer", "subject", "client_id")
        ):
            raise ConnectionError("auth")

    def _ready(self, connection_id, cancel):
        with self._lock, self.state.locked() as state:
            profile = self._find(state, connection_id)
            if (
                profile["credential_ref"] is None
                or profile["credential_ref"] in self._blocked
            ):
                raise ConnectionError("credentials")
            store = self._store(profile["storage"])
            credentials = store.get(profile["credential_ref"])
            if not isinstance(credentials, Credentials):
                raise ConnectionError("credentials")
            if profile["kind"] == "chatgpt_plan":
                values = credentials.snapshot()
                saved = profile["account"]
                self._bound(saved, credentials)
                if values.get("refresh_in_progress") is True:
                    self._blocked.add(profile["credential_ref"])
                    self._retire(state, profile)
                    profile["credential_ref"] = None
                    self._write(state)
                    raise ConnectionError("auth")
                if (
                    type(values.get("scopes")) is not list
                    or "chatgpt.tokens.use.direct" not in values["scopes"]
                ):
                    raise ConnectionError("permission")
                expires = values.get("access_expires_at")
                if (
                    type(expires) not in (int, float)
                    or values.get("identity_pending")
                    or expires <= time() + 30
                ):
                    recovering = values.get("identity_pending") is True
                    try:
                        # Persist intent before spending a rotating token. A crash
                        # must require sign-in, never replay the old refresh token.
                        if not recovering:
                            store.put(
                                profile["credential_ref"],
                                Credentials({**values, "refresh_in_progress": True}),
                            )
                        refreshed = self._oauth(state).refresh(
                            saved, credentials, cancel=cancel
                        )
                        self._bound(saved, refreshed.credentials)
                        self._public_safe(profile, refreshed.credentials)
                        store.put(profile["credential_ref"], refreshed.credentials)
                        credentials = refreshed.credentials
                    except BaseException as exc:
                        # Pending identity recovery only verifies a saved token;
                        # a temporary failure has not spent a refresh token.
                        if recovering and not (
                            isinstance(exc, ConnectionError) and exc.code == "auth"
                        ):
                            raise
                        # A failed exchange can have rotated remotely. Never spend
                        # the old refresh token again after an uncertain result.
                        self._blocked.add(profile["credential_ref"])
                        self._retire(state, profile)
                        profile["credential_ref"] = None
                        self._write(state)
                        raise
                    if refreshed.identity_pending:
                        raise ConnectionError("auth")
            return copy.deepcopy(profile), credentials

    @public_boundary
    def models(self, connection_id=None, *, cancel=None):
        from tsfm_lab.connections.providers import list_models

        state = self.state.read()
        profile = self._find(state, connection_id)
        profile, credentials = self._ready(profile["id"], cancel)
        choices = list_models(
            profile["kind"],
            credentials,
            self.transport,
            limits=self.limits,
            cancel=cancel,
            workspace_id=profile["workspace_id"],
        )
        current = self.state.read()
        if self._route(self._find(current, profile["id"])) != self._route(profile):
            raise ConnectionError("state")
        self._models[profile["id"]] = (self._route(profile), tuple(choices))
        return choices

    @public_boundary
    def set_model(self, model, connection_id=None):
        text(model)
        with self._lock, self.state.locked() as state:
            profile = self._find(state, connection_id)
            cached = self._models.get(profile["id"])
            if (
                cached is None
                or cached[0] != self._route(profile)
                or model not in {item.identifier for item in cached[1]}
            ):
                raise ConnectionError("model")
            profile["model"] = model
            self._write(state)
        return self.status(profile["id"])

    @public_boundary
    def preview_test(self):
        return self._preview_message(
            TEST_TEXT,
            instructions=TEST_INSTRUCTIONS,
            output_tokens=32,
            data_included="Only this fixed connection-check text; no dataset, paths, saved forecasts or environment.",
            purpose="test",
            cancel=None,
        )

    @public_boundary
    def preview_message(
        self, text, *, instructions, output_tokens=1024, data_included, cancel=None
    ):
        """Approve exact assistant context locally, without refreshing or sending."""
        return self._preview_message(
            text,
            instructions=instructions,
            output_tokens=output_tokens,
            data_included=data_included,
            purpose="message",
            cancel=cancel,
        )

    @staticmethod
    def _check_cancel(cancel):
        if cancel is not None and not callable(cancel):
            raise ConnectionError("invalid")
        if cancel and cancel():
            raise ConnectionError("cancelled")

    def _preview_message(
        self, text, *, instructions, output_tokens, data_included, purpose, cancel
    ):
        self._check_cancel(cancel)
        if (
            type(text) is not str
            or not text.strip()
            or type(instructions) is not str
            or type(output_tokens) is not int
            or not 1 <= output_tokens <= 4096
            or type(data_included) is not str
            or not data_included.strip()
        ):
            raise ConnectionError("invalid")
        if (
            len((text + instructions).encode()) > self.limits.max_input_bytes
            or len(data_included.encode()) > 4096
        ):
            raise ConnectionError("bounds")
        with self._lock:
            return self._make_preview(
                text, instructions, output_tokens, data_included, purpose, cancel
            )

    def _make_preview(
        self, text, instructions, output_tokens, data_included, purpose, cancel
    ):
        state = self.state.read()
        profile = self._find(state)
        cached = self._models.get(profile["id"])
        if (
            profile["credential_ref"] is None
            or profile["credential_ref"] in self._blocked
        ):
            raise ConnectionError("credentials")
        if (
            cached is None
            or cached[0] != self._route(profile)
            or profile["model"] not in {item.identifier for item in cached[1]}
        ):
            raise ConnectionError("model")
        # Previewing must inspect secrets locally; _ready could refresh tokens.
        credentials = self._store(profile["storage"]).get(profile["credential_ref"])
        if not isinstance(credentials, Credentials):
            raise ConnectionError("credentials")
        if profile["kind"] == "chatgpt_plan":
            self._bound(profile["account"], credentials)
        token = str(uuid4())
        preview = {
            "id": token,
            "connection_id": profile["id"],
            "connection_label": profile["label"],
            "account": copy.deepcopy(profile["account"]),
            "workspace_id": profile["workspace_id"],
            "provider": KINDS[profile["kind"]][0],
            "usage": KINDS[profile["kind"]][1],
            "model": profile["model"],
            "text": text,
            "instructions": instructions,
            "output_tokens": (
                None if profile["kind"] == "chatgpt_plan" else output_tokens
            ),
            "local_limits": {
                "seconds": self.limits.timeout_seconds,
                "received_bytes": self.limits.max_response_bytes,
            },
            "data_included": data_included,
        }
        self._public_safe(preview, credentials)
        self._check_cancel(cancel)
        self._check_preview_state(self.state.read(), state["generation"], profile)
        self._previews.clear()
        self._previews[token] = (
            copy.deepcopy(preview),
            state["generation"],
            copy.deepcopy(profile),
            time(),
            purpose,
            output_tokens,
        )
        return preview

    @public_boundary
    def submit_test(self, preview, *, confirmed, cancel=None):
        return self._submit_message(
            preview, confirmed=confirmed, cancel=cancel, purpose="test"
        )

    @public_boundary
    def submit_message(self, preview, *, confirmed, cancel=None):
        """Spend one matching approval on at most one selected-route generation."""
        return self._submit_message(
            preview, confirmed=confirmed, cancel=cancel, purpose="message"
        )

    @staticmethod
    def _check_preview_state(state, generation, profile):
        if (
            state["generation"] != generation
            or state["selected"] != profile["id"]
            or ConnectionManager._find(state, profile["id"]) != profile
        ):
            raise ConnectionError("state")

    def _submit_message(self, preview, *, confirmed, cancel, purpose):
        from tsfm_lab.connections.providers import generate

        if type(preview) is not dict:
            raise ConnectionError("permission")
        with self._lock:
            pending = self._previews.pop(preview.get("id"), None)
        if confirmed is not True:
            raise ConnectionError("permission")
        if (
            pending is None
            or pending[0] != preview
            or not 0 <= time() - pending[3] <= 300
            or pending[4] != purpose
        ):
            raise ConnectionError("state")
        self._check_cancel(cancel)
        profile, approved = pending[2], pending[0]
        self._check_preview_state(self.state.read(), pending[1], profile)
        profile, credentials = self._ready(profile["id"], cancel)
        self._check_preview_state(self.state.read(), pending[1], pending[2])
        self._public_safe(approved, credentials)
        self._check_cancel(cancel)
        answer = generate(
            profile["kind"],
            credentials,
            self.transport,
            model=profile["model"],
            instructions=approved["instructions"],
            text=approved["text"],
            output_tokens=pending[5],
            limits=self.limits,
            cancel=cancel,
            workspace_id=profile["workspace_id"],
        )
        self._check_cancel(cancel)
        self._check_preview_state(self.state.read(), pending[1], pending[2])
        if type(answer) is not str:
            raise ConnectionError("response")
        self._public_safe(answer, credentials)
        self._check_cancel(cancel)
        return answer

    @public_boundary
    def disconnect(self, connection_id=None, *, cancel=None):
        with self._lock, self.state.locked() as state:
            profile = self._find(state, connection_id)
            remote = None
            if profile["kind"] == "chatgpt_plan" and profile["credential_ref"]:
                try:
                    credentials = self._store(profile["storage"]).get(
                        profile["credential_ref"]
                    )
                    remote = (
                        self._oauth(state).revoke(
                            profile["account"], credentials, cancel=cancel
                        )
                        if credentials is not None
                        else False
                    )
                except (Exception, KeyboardInterrupt):
                    remote = False
            self._retire(state, profile)
            profile.update(credential_ref=None, model=None)
            if state["selected"] == profile["id"]:
                state["selected"] = None
            self._write(state)
            self._models.pop(profile["id"], None)
        cleaned = self._cleanup()
        return {
            "id": profile["id"],
            "disconnected": True,
            "local_credentials_removed": cleaned,
            "remote_revocation_confirmed": remote,
        }

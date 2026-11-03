"""Private metadata and native-vault boundaries use disposable local fixtures."""

import json
import os
import stat
from uuid import uuid4

import pytest

from tsfm_lab.connections.contracts import ConnectionError
from tsfm_lab.connections.credentials import Credentials, SessionStore
from tsfm_lab.connections.state import StateStore
from tsfm_lab.connections.vault import SERVICE, NativeVault

SECRET = "fake-state-secret-ABC123"


def profile():
    return {
        "id": str(uuid4()),
        "kind": "openai_api",
        "label": "Local test",
        "storage": "session",
        "credential_ref": str(uuid4()),
        "model": None,
        "account": None,
        "workspace_id": None,
        "registration": None,
    }


@pytest.fixture
def state(tmp_path):
    store = StateStore(tmp_path / "private-connections")
    value = store.read()
    value["profiles"].append(profile())
    value["selected"] = value["profiles"][0]["id"]
    store.write(value)
    return store


def overwrite(store, raw):
    store.path.write_bytes(raw)
    store.path.chmod(0o600)


def test_atomic_state_round_trip_is_private_and_contains_only_metadata(state):
    value = state.read()
    assert value["profiles"][0]["label"] == "Local test"
    assert value["selected"] == value["profiles"][0]["id"]
    if os.name != "nt":
        assert stat.S_IMODE(state.directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(state.path.stat().st_mode) == 0o600
    value["generation"] += 1
    state.write(value)
    assert state.read() == value
    assert list(state.directory.glob(".connections-*")) == []


def test_unused_store_does_not_create_files_and_host_binding_survives_write(tmp_path):
    store = StateStore(tmp_path / "unused")
    first = store.read()
    assert not store.directory.exists()
    assert store.read()["host_id"] == first["host_id"]
    store.write(first)
    assert StateStore(store.directory).read()["host_id"] == first["host_id"]


@pytest.mark.parametrize(
    "raw",
    [
        b'{"private":"' + SECRET.encode() + b'"',
        b'{"schema_version":1,"schema_version":1}',
        b'{"generation":NaN}',
        b"[]",
        b'{"nested":' * 35 + b"null" + b"}" * 35,
        b"x" * 262145,
    ],
    ids=("malformed", "duplicate_keys", "nonfinite", "wrong_root", "deep", "oversized"),
)
def test_corrupt_or_oversized_state_returns_only_canned_error(state, raw):
    overwrite(state, raw)
    with pytest.raises(ConnectionError) as error:
        state.read()
    assert SECRET not in str(error.value)
    assert str(state.path) not in str(error.value)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_secret",
        "duplicate_profile",
        "duplicate_reference",
        "invalid_reference",
        "unknown_selected",
        "bad_host",
        "bool_generation",
        "account_on_api",
        "wrong_workspace",
        "unknown_schema",
        "retired_active",
        "invalid_issuer",
        "dynamic_client",
    ],
)
def test_malformed_metadata_cannot_rebind_connections_or_add_secrets(state, mutation):
    value = state.read()
    selected = value["profiles"][0]
    if mutation == "unknown_secret":
        selected["api_key"] = SECRET
    elif mutation == "duplicate_profile":
        value["profiles"].append(selected.copy())
    elif mutation == "duplicate_reference":
        other = profile()
        other["credential_ref"] = selected["credential_ref"]
        value["profiles"].append(other)
    elif mutation == "invalid_reference":
        selected["credential_ref"] = "../../another-profile"
    elif mutation == "unknown_selected":
        value["selected"] = str(uuid4())
    elif mutation == "bad_host":
        value["host_id"] = "urn:uuid:" + str(uuid4()).upper()
    elif mutation == "bool_generation":
        value["generation"] = True
    elif mutation == "account_on_api":
        selected["account"] = {
            "issuer": "https://auth.openai.com",
            "subject": "fake",
            "client_id": "fake-client",
        }
    elif mutation == "wrong_workspace":
        selected["workspace_id"] = "workspace-other-route"
    elif mutation == "unknown_schema":
        value["schema_version"] = 2
    elif mutation == "retired_active":
        value["retired_refs"] = [
            {"reference": selected["credential_ref"], "storage": "session"}
        ]
    else:
        selected["kind"] = "chatgpt_plan"
        selected["account"] = {
            "issuer": (
                "https://untrusted.invalid"
                if mutation == "invalid_issuer"
                else "https://auth.openai.com"
            ),
            "subject": "fake",
            "client_id": (
                "dynamic_agent_client"
                if mutation == "dynamic_client"
                else "fake-client"
            ),
        }
    overwrite(state, json.dumps(value).encode())
    with pytest.raises(ConnectionError) as error:
        state.read()
    assert SECRET not in str(error.value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission contract")
@pytest.mark.parametrize("target", ["directory", "file"])
def test_world_readable_state_is_refused(state, target):
    path = state.directory if target == "directory" else state.path
    path.chmod(0o755 if target == "directory" else 0o644)
    try:
        with pytest.raises(ConnectionError) as error:
            state.read()
        assert error.value.code == "storage"
    finally:
        path.chmod(0o700 if target == "directory" else 0o600)


@pytest.mark.parametrize("ancestor", [False, True])
def test_symlinked_directory_or_ancestor_is_refused(tmp_path, ancestor, symlinks):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    symlinks(link, real, target_is_directory=True)
    store = StateStore(link / "child" if ancestor else link)
    with pytest.raises(ConnectionError) as error:
        store.read()
    assert error.value.code == "storage"
    with pytest.raises(ConnectionError):
        store.write(StateStore(tmp_path / "fresh").read())
    assert list(real.iterdir()) == []


def test_symlinked_metadata_is_neither_read_nor_overwritten(state, tmp_path, symlinks):
    value = state.read()
    target = tmp_path / "unrelated"
    target.write_text(SECRET)
    state.path.unlink()
    symlinks(state.path, target)
    for operation in (state.read, lambda: state.write(value)):
        with pytest.raises(ConnectionError) as error:
            operation()
        assert error.value.code == "storage"
    assert target.read_text() == SECRET


def test_symlinked_lock_is_refused_without_touching_target(state, tmp_path, symlinks):
    target = tmp_path / "unrelated-lock"
    target.write_text(SECRET)
    symlinks(state.directory / ".lock", target)
    with pytest.raises(ConnectionError) as error:
        with state.locked():
            pytest.fail("A symlinked lock must not be accepted")
    assert error.value.code == "storage"
    assert target.read_text() == SECRET


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special-file fixture")
def test_special_metadata_file_is_refused_without_consuming_it(state):
    state.path.unlink()
    os.mkfifo(state.path, mode=0o600)
    # Holding a disposable local RDWR endpoint keeps even an erroneous blocking
    # reader from hanging this test; fstat still has to reject the FIFO.
    descriptor = os.open(state.path, os.O_RDWR | os.O_NONBLOCK)
    try:
        with pytest.raises(ConnectionError) as error:
            state.read()
        assert error.value.code == "storage"
    finally:
        os.close(descriptor)


def test_atomic_replace_failure_preserves_original_and_cleans_temporary_file(
    state, monkeypatch
):
    before = state.path.read_bytes()
    value = state.read()
    value["generation"] += 1

    def fail(*args):
        raise OSError(SECRET)

    monkeypatch.setattr("tsfm_lab.connections.state.os.replace", fail)
    with pytest.raises(ConnectionError) as error:
        state.write(value)
    assert error.value.code == "storage" and SECRET not in str(error.value)
    assert state.path.read_bytes() == before
    assert list(state.directory.glob(".connections-*")) == []


def test_private_lock_releases_after_failure(state):
    with pytest.raises(ConnectionError):
        with state.locked():
            raise RuntimeError("disposable fixture failure")
    with state.locked() as value:
        assert value["generation"] == state.read()["generation"]


class FakeNativeBackend:
    def __init__(self):
        self.values, self.calls = {}, []
        self.fail = False

    def get_password(self, service, reference):
        self.calls.append(("get", service, reference))
        if self.fail:
            raise RuntimeError(SECRET)
        return self.values.get((service, reference))

    def set_password(self, service, reference, value):
        self.calls.append(("put", service, reference))
        if self.fail:
            raise RuntimeError(SECRET)
        self.values[(service, reference)] = value

    def delete_password(self, service, reference):
        self.calls.append(("delete", service, reference))
        if self.fail:
            raise RuntimeError(SECRET)
        del self.values[(service, reference)]


def test_native_vault_round_trip_uses_only_explicit_injected_backend():
    backend = FakeNativeBackend()
    vault = NativeVault(backend=backend)
    reference = str(uuid4())
    credentials = Credentials({"api_key": SECRET})
    vault.put(reference, credentials)
    assert vault.get(reference).reveal("api_key") == SECRET
    assert SECRET not in repr(vault)
    assert all(call[1] == SERVICE for call in backend.calls)
    vault.delete(reference)
    vault.delete(reference)  # Already absent is a successful local deletion.
    assert vault.get(reference) is None


@pytest.mark.parametrize("operation", ["get", "put", "delete"])
def test_native_backend_failures_are_canned_without_plaintext_fallback(operation):
    backend = FakeNativeBackend()
    backend.fail = True
    vault = NativeVault(backend=backend)
    reference = str(uuid4())
    args = (
        (reference, Credentials({"api_key": SECRET}))
        if operation == "put"
        else (reference,)
    )
    with pytest.raises(ConnectionError) as error:
        getattr(vault, operation)(*args)
    assert error.value.code == "storage" and SECRET not in str(error.value)
    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    "raw", ["not-json-" + SECRET, '{"api_key":"one","api_key":"two"}', "[]"]
)
def test_corrupted_vault_entry_is_not_accepted_or_exposed(raw):
    backend = FakeNativeBackend()
    reference = str(uuid4())
    backend.values[(SERVICE, reference)] = raw
    with pytest.raises(ConnectionError) as error:
        NativeVault(backend=backend).get(reference)
    assert error.value.code == "storage" and SECRET not in str(error.value)


def test_missing_native_dependency_fails_closed_without_loading_other_backends(
    monkeypatch,
):
    calls = []

    def unavailable(name):
        calls.append(name)
        raise ImportError(SECRET)

    monkeypatch.setattr(
        "tsfm_lab.connections.vault.importlib.import_module", unavailable
    )
    with pytest.raises(ConnectionError) as error:
        NativeVault().get(str(uuid4()))
    assert error.value.code == "storage" and SECRET not in str(error.value)
    assert len(calls) <= 1
    assert all(
        name
        in {
            "keyring.backends.macOS",
            "keyring.backends.Windows",
            "keyring.backends.SecretService",
        }
        for name in calls
    )


@pytest.mark.parametrize("reference", ["../other", "", "not-a-uuid", SECRET])
def test_vault_rejects_untrusted_reference_before_backend_use(reference):
    backend = FakeNativeBackend()
    with pytest.raises(ConnectionError) as error:
        NativeVault(backend=backend).get(reference)
    assert error.value.code == "invalid" and backend.calls == []


def test_session_store_returns_isolated_protected_copies():
    store = SessionStore()
    original = Credentials({"api_key": SECRET, "scopes": ["fake-scope"]})
    store.put("fake-reference", original)
    snapshot = store.get("fake-reference").snapshot()
    snapshot["api_key"] = "changed"
    snapshot["scopes"].append("changed")
    assert store.get("fake-reference").snapshot() == original.snapshot()
    assert SECRET not in repr(store)

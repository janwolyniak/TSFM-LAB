"""Connection/account safety checks with local fake stores and providers only."""

import json
from types import SimpleNamespace

import pytest

from tsfm_lab.connections.contracts import ConnectionError, ModelChoice
from tsfm_lab.connections.credentials import Credentials, SessionStore
from tsfm_lab.connections.manager import (
    TEST_INSTRUCTIONS,
    TEST_TEXT,
    ConnectionManager,
)
from tsfm_lab.connections.state import StateStore

SECRET = "fake-manager-api-key-ABC123"
OTHER_SECRET = "fake-manager-api-key-DEF456"


class MemoryStore(SessionStore):
    def __init__(self):
        super().__init__()
        self.puts, self.deletes = [], []
        self.fail_put = self.fail_delete = False

    def put(self, reference, credentials):
        self.puts.append(reference)
        super().put(reference, credentials)
        if self.fail_put:
            raise RuntimeError(SECRET)

    def delete(self, reference):
        self.deletes.append(reference)
        if self.fail_delete:
            raise RuntimeError(SECRET)
        super().delete(reference)


class NoNetwork:
    def send(self, *args, **kwargs):
        raise AssertionError("This manager test must not use network transport")


class FakeOAuth:
    def __init__(self):
        self.refreshes, self.revocations = [], []
        self.refreshed = None
        self.revoked = True

    def refresh(self, account, credentials, *, cancel=None):
        self.refreshes.append((account.copy(), credentials.snapshot()))
        if isinstance(self.refreshed, BaseException):
            raise self.refreshed
        return self.refreshed

    def revoke(self, account, credentials, *, cancel=None):
        self.revocations.append((account.copy(), credentials.snapshot()))
        if isinstance(self.revoked, BaseException):
            raise self.revoked
        return self.revoked


@pytest.fixture
def setup(tmp_path, monkeypatch):
    state = StateStore(tmp_path / "connections")
    session, vault, oauth = MemoryStore(), MemoryStore(), FakeOAuth()
    manager = ConnectionManager(
        state,
        session=session,
        vault=vault,
        transport=NoNetwork(),
        oauth_factory=lambda host: oauth,
    )
    discovery, generation = [], []

    def list_models(kind, credentials, transport, **kwargs):
        discovery.append((kind, credentials.snapshot(), kwargs))
        return [ModelChoice("chosen-model", "Chosen model")]

    def generate(kind, credentials, transport, **kwargs):
        generation.append((kind, credentials.snapshot(), kwargs))
        return "Connection confirmed."

    monkeypatch.setattr("tsfm_lab.connections.providers.list_models", list_models)
    monkeypatch.setattr("tsfm_lab.connections.providers.generate", generate)
    return SimpleNamespace(
        manager=manager,
        state=state,
        session=session,
        vault=vault,
        oauth=oauth,
        discovery=discovery,
        generation=generation,
    )


def key_connection(setup, *, kind="openai_api", value=SECRET, storage="session"):
    return setup.manager.add_key(kind, "Test connection", value, storage=storage)["id"]


def prepared(setup, connection_id=None):
    connection_id = connection_id or key_connection(setup)
    setup.manager.select(connection_id)
    setup.manager.models()
    setup.manager.set_model("chosen-model")
    return connection_id


def plan_result(
    subject="subject-one", *, expiry=4_000_000_000, pending=False, suffix="one"
):
    account = {
        "issuer": "https://auth.openai.com",
        "client_id": "fake-public-client",
        "subject": subject,
        "email": f"{subject}@example.test",
    }
    credentials = Credentials(
        {
            **{key: account[key] for key in ("issuer", "client_id", "subject")},
            "access_token": f"fake-access-{suffix}-token",
            "refresh_token": f"fake-refresh-{suffix}-token",
            "id_token": f"fake-identity-{suffix}-token",
            "scopes": ["chatgpt.tokens.use.direct"],
            "access_expires_at": expiry,
            "identity_pending": pending,
        }
    )
    return SimpleNamespace(
        account=account, credentials=credentials, identity_pending=pending
    )


def plan_connection(setup, result=None, *, storage="session"):
    connection_id = setup.manager.add_plan("Test plan", storage=storage)["id"]
    setup.manager.finish_sign_in(
        connection_id, result or plan_result(), setup.state.read()["generation"]
    )
    return connection_id


def test_new_key_is_saved_without_discovery_selection_or_inference(setup):
    connection_id = key_connection(setup)
    public = setup.manager.status(connection_id)
    assert not public["selected"] and public["model"] is None
    assert public["usage"] == "OpenAI API billing"
    assert public["credentials_saved"] is True
    assert setup.discovery == setup.generation == []
    raw = setup.state.path.read_text()
    assert SECRET not in raw
    assert "api_key" not in raw
    assert SECRET not in repr(setup.manager)
    assert "credential_ref" not in json.dumps(public)


def test_session_restart_does_not_fall_back_to_vault_or_another_connection(setup):
    connection_id = prepared(setup)
    restarted = ConnectionManager(
        setup.state, session=MemoryStore(), vault=setup.vault, transport=NoNetwork()
    )
    assert restarted.status(connection_id)["credentials_saved"] is True
    with pytest.raises(ConnectionError) as error:
        restarted.models(connection_id)
    assert error.value.code == "credentials"
    assert setup.generation == []


@pytest.mark.parametrize("storage", ["session", "vault"])
def test_failed_credential_write_rolls_back_metadata_and_partial_store_write(
    setup, storage
):
    store = getattr(setup, storage)
    store.fail_put = True
    with pytest.raises(ConnectionError) as error:
        key_connection(setup, storage=storage)
    assert error.value.code == "storage"
    assert SECRET not in str(error.value)
    assert setup.manager.list() == []
    assert len(store.puts) == len(store.deletes) == 1
    assert store.get(store.puts[0]) is None
    assert getattr(setup, "vault" if storage == "session" else "session").puts == []


def test_failed_state_write_deletes_new_secret_and_preserves_old_connection(
    setup, monkeypatch
):
    connection_id = key_connection(setup)
    before = setup.state.read()
    old_ref = before["profiles"][0]["credential_ref"]

    def fail(_):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(setup.state, "write", fail)
    with pytest.raises(ConnectionError) as error:
        setup.manager.replace_key(connection_id, OTHER_SECRET)
    assert error.value.code == "storage"
    assert setup.state.read() == before
    assert setup.session.get(old_ref).reveal("api_key") == SECRET
    assert setup.session.get(setup.session.puts[-1]) is None


def test_key_replacement_invalidates_old_choices_and_preview(setup):
    connection_id = prepared(setup)
    preview = setup.manager.preview_test()
    old_ref = setup.state.read()["profiles"][0]["credential_ref"]
    replaced = setup.manager.replace_key(connection_id, OTHER_SECRET)
    assert replaced["model"] is None and replaced["previous_credentials_removed"]
    assert setup.session.get(old_ref) is None
    with pytest.raises(ConnectionError):
        setup.manager.submit_test(preview, confirmed=True)
    with pytest.raises(ConnectionError) as error:
        setup.manager.set_model("chosen-model")
    assert error.value.code == "model"
    assert setup.generation == []


def test_credential_cleanup_failure_disconnects_old_reference_without_fallback(setup):
    connection_id = prepared(setup)
    setup.session.fail_delete = True
    result = setup.manager.replace_key(connection_id, OTHER_SECRET)
    assert result["previous_credentials_removed"] is False
    state = setup.state.read()
    assert len(state["retired_refs"]) == 1
    assert (
        state["profiles"][0]["credential_ref"] != state["retired_refs"][0]["reference"]
    )
    assert SECRET not in setup.state.path.read_text()


@pytest.mark.parametrize("confirmed", [False, None, 1, "yes"])
def test_preview_requires_explicit_true_consent_and_cannot_be_replayed(
    setup, confirmed
):
    prepared(setup)
    preview = setup.manager.preview_test()
    with pytest.raises(ConnectionError) as error:
        setup.manager.submit_test(preview, confirmed=confirmed)
    assert error.value.code == "permission"
    with pytest.raises(ConnectionError):
        setup.manager.submit_test(preview, confirmed=True)
    assert setup.generation == []


def test_preview_lists_route_and_only_the_fixed_connection_check_is_sent(setup):
    prepared(setup)
    preview = setup.manager.preview_test()
    assert preview["provider"] == "OpenAI API key"
    assert preview["usage"] == "OpenAI API billing"
    assert preview["text"] == TEST_TEXT
    assert setup.generation == []
    assert setup.manager.submit_test(preview, confirmed=True) == "Connection confirmed."
    assert len(setup.generation) == 1
    kind, protected, sent = setup.generation[0]
    assert kind == "openai_api" and protected == {"api_key": SECRET}
    assert sent["text"] == TEST_TEXT and sent["instructions"] == TEST_INSTRUCTIONS
    assert sent["model"] == "chosen-model" and sent["output_tokens"] == 32
    with pytest.raises(ConnectionError):
        setup.manager.submit_test(preview, confirmed=True)
    assert len(setup.generation) == 1


def test_changed_preview_and_old_preview_are_rejected(setup):
    prepared(setup)
    first = setup.manager.preview_test()
    changed = setup.manager.preview_test()
    changed["text"] = "Send something else"
    for preview in (first, changed):
        with pytest.raises(ConnectionError):
            setup.manager.submit_test(preview, confirmed=True)
    assert setup.generation == []


def test_expired_preview_is_rejected_without_generation(setup, monkeypatch):
    prepared(setup)
    monkeypatch.setattr("tsfm_lab.connections.manager.time", lambda: 1000)
    preview = setup.manager.preview_test()
    monkeypatch.setattr("tsfm_lab.connections.manager.time", lambda: 1301)
    with pytest.raises(ConnectionError) as error:
        setup.manager.submit_test(preview, confirmed=True)
    assert error.value.code == "state" and setup.generation == []


@pytest.mark.parametrize(
    "field,value",
    [("kind", "anthropic_api"), ("storage", "vault"), ("label", "Changed label")],
)
def test_manual_metadata_change_cannot_reuse_preview_or_cached_model(
    setup, field, value
):
    prepared(setup)
    preview = setup.manager.preview_test()
    state = setup.state.read()
    state["profiles"][0][field] = value
    setup.state.write(state)  # Simulates an external writer retaining generation.
    with pytest.raises(ConnectionError) as error:
        setup.manager.submit_test(preview, confirmed=True)
    assert error.value.code == "state"
    with pytest.raises(ConnectionError):
        setup.manager.set_model("chosen-model")
    assert setup.generation == []


def test_provider_error_consumes_preview_without_retry_or_route_switch(
    setup, monkeypatch
):
    prepared(setup)
    calls = []

    def fail(*args, **kwargs):
        calls.append(args[0])
        raise ConnectionError("limit")

    monkeypatch.setattr("tsfm_lab.connections.providers.generate", fail)
    preview = setup.manager.preview_test()
    with pytest.raises(ConnectionError) as error:
        setup.manager.submit_test(preview, confirmed=True)
    assert error.value.code == "limit"
    with pytest.raises(ConnectionError):
        setup.manager.submit_test(preview, confirmed=True)
    assert calls == ["openai_api"]


def test_selection_change_during_inference_discards_answer(setup, monkeypatch):
    prepared(setup)
    other = key_connection(setup, kind="gemini_api", value=OTHER_SECRET)

    def changed(*args, **kwargs):
        setup.manager.select(other)
        return "Do not expose stale answer"

    monkeypatch.setattr("tsfm_lab.connections.providers.generate", changed)
    preview = setup.manager.preview_test()
    with pytest.raises(ConnectionError) as error:
        setup.manager.submit_test(preview, confirmed=True)
    assert error.value.code == "state"


def test_model_choices_are_bound_to_the_credential_and_account(setup):
    first = prepared(setup)
    second = key_connection(setup, value=OTHER_SECRET)
    setup.manager.select(second)
    with pytest.raises(ConnectionError) as error:
        setup.manager.set_model("chosen-model")
    assert error.value.code == "model"
    setup.manager.models()
    assert setup.discovery[-1][1]["api_key"] == OTHER_SECRET
    assert first != second


def test_plan_account_and_credentials_never_cross_profiles(setup):
    first = plan_connection(setup, plan_result("first"))
    generation = setup.state.read()["generation"]
    with pytest.raises(ConnectionError) as error:
        setup.manager.finish_sign_in(
            first, plan_result("second", suffix="two"), generation
        )
    assert error.value.code == "auth"
    second = plan_connection(setup, plan_result("second", suffix="two"))
    setup.manager.models(first)
    assert setup.discovery[-1][1]["subject"] == "first"
    setup.manager.models(second)
    assert setup.discovery[-1][1]["subject"] == "second"
    assert setup.manager.status(first)["account"]["subject"] == "first"


def test_expired_plan_refresh_rotates_protected_tokens_before_discovery(setup):
    connection_id = plan_connection(setup, plan_result(expiry=0))
    reference = setup.state.read()["profiles"][0]["credential_ref"]
    setup.oauth.refreshed = plan_result(suffix="rotated")
    setup.manager.models(connection_id)
    assert len(setup.oauth.refreshes) == 1
    assert setup.oauth.refreshes[0][1]["refresh_token"] == "fake-refresh-one-token"
    assert (
        setup.session.get(reference).reveal("refresh_token")
        == "fake-refresh-rotated-token"
    )
    assert setup.discovery[-1][1]["access_token"] == "fake-access-rotated-token"
    assert "fake-access" not in setup.state.path.read_text()


def test_uncertain_refresh_is_not_replayed_and_profile_becomes_unusable(setup):
    connection_id = plan_connection(setup, plan_result(expiry=0))
    setup.oauth.refreshed = RuntimeError("fake-refresh-one-token")
    with pytest.raises(ConnectionError) as error:
        setup.manager.models(connection_id)
    assert "fake-refresh-one-token" not in str(error.value)
    with pytest.raises(ConnectionError) as error:
        setup.manager.models(connection_id)
    assert error.value.code == "credentials"
    assert len(setup.oauth.refreshes) == 1
    assert setup.discovery == []
    assert setup.manager.status(connection_id)["credentials_saved"] is False


def test_pending_identity_keeps_rotated_tokens_protected_and_blocks_discovery(setup):
    connection_id = plan_connection(setup, plan_result(expiry=0))
    setup.oauth.refreshed = plan_result(suffix="rotated", pending=True)
    with pytest.raises(ConnectionError) as error:
        setup.manager.models(connection_id)
    assert error.value.code == "auth"
    reference = setup.state.read()["profiles"][0]["credential_ref"]
    values = setup.session.get(reference).snapshot()
    assert values["identity_pending"] is True
    assert values["refresh_token"] == "fake-refresh-rotated-token"
    assert setup.discovery == []
    assert "fake-refresh" not in setup.state.path.read_text()


def test_refresh_with_different_account_is_rejected_before_provider_use(setup):
    connection_id = plan_connection(setup, plan_result(expiry=0))
    setup.oauth.refreshed = plan_result("other-subject", suffix="rotated")
    with pytest.raises(ConnectionError) as error:
        setup.manager.models(connection_id)
    assert error.value.code == "auth" and setup.discovery == []
    assert setup.manager.status(connection_id)["credentials_saved"] is False


def test_refresh_rejects_secret_collision_with_existing_public_metadata(setup):
    connection_id = plan_connection(setup, plan_result(expiry=0))
    result = plan_result(suffix="rotated")
    values = result.credentials.snapshot()
    values["refresh_token"] = result.account["email"]
    result.credentials = Credentials(values)
    setup.oauth.refreshed = result
    with pytest.raises(ConnectionError) as error:
        setup.manager.models(connection_id)
    assert error.value.code == "response" and setup.discovery == []


@pytest.mark.parametrize("remote", [False, RuntimeError("fake-refresh-one-token")])
def test_failed_remote_revocation_still_disconnects_and_removes_local_secret(
    setup, remote
):
    connection_id = plan_connection(setup)
    setup.manager.select(connection_id)
    reference = setup.state.read()["profiles"][0]["credential_ref"]
    setup.oauth.revoked = remote
    result = setup.manager.disconnect()
    assert result["disconnected"] and result["local_credentials_removed"]
    assert result["remote_revocation_confirmed"] is False
    assert setup.session.get(reference) is None
    assert setup.manager.status(connection_id)["selected"] is False
    with pytest.raises(ConnectionError):
        setup.manager.models(connection_id)


def test_failed_local_deletion_is_reported_without_reactivating_credentials(setup):
    connection_id = key_connection(setup)
    setup.session.fail_delete = True
    result = setup.manager.disconnect(connection_id)
    assert result["disconnected"] is True
    assert result["local_credentials_removed"] is False
    assert setup.manager.status(connection_id)["credentials_saved"] is False
    with pytest.raises(ConnectionError):
        setup.manager.models(connection_id)


@pytest.mark.parametrize("label", [SECRET, "".join(f"\\u{ord(c):04x}" for c in SECRET)])
def test_secret_reflection_is_rejected_before_public_metadata_is_saved(setup, label):
    with pytest.raises(ConnectionError):
        setup.manager.add_key("openai_api", label, SECRET, storage="session")
    assert setup.manager.list() == []


@pytest.mark.parametrize("opened", [True, False])
def test_sign_in_bridge_binds_listener_first_reuses_bound_account_and_wipes_attempt(
    setup, opened
):
    from tsfm_lab.connections.sign_in import browser_sign_in

    connection_id = plan_connection(setup)
    events, captured = [], {}

    def cancel_check():
        return False

    class Attempt:
        authorization_url = "https://auth.openai.com/api/accounts/authorize?fake=public"
        registration = None

        def cancel(self):
            events.append("cancel-attempt")

    attempt = Attempt()

    class Client:
        def begin_sign_in(self, **kwargs):
            events.append("begin")
            captured.update(kwargs)
            return attempt

        def complete_callback(self, *args, **kwargs):
            events.append("complete")
            assert kwargs["cancel"] is cancel_check
            return plan_result()

    class Listener:
        redirect_uri = "http://127.0.0.1:12345/auth/callback"

        def __init__(self, **kwargs):
            pass

        def start(self):
            events.append("listen")

        def wait(self, *args, **kwargs):
            events.append("wait")
            assert kwargs["cancel"] is cancel_check
            return "fake-callback"

        def close(self):
            events.append("close-listener")

    def browser(url):
        events.append("browser")
        assert url == attempt.authorization_url
        return opened

    setup.manager.oauth_factory = lambda host: Client()
    if opened:
        result = browser_sign_in(
            setup.manager,
            connection_id,
            listener_factory=Listener,
            browser=browser,
            cancel=cancel_check,
        )
        assert result["id"] == connection_id
    else:
        with pytest.raises(ConnectionError) as error:
            browser_sign_in(
                setup.manager,
                connection_id,
                listener_factory=Listener,
                browser=browser,
                cancel=cancel_check,
            )
        assert error.value.code == "auth"
    assert events[:3] == ["listen", "begin", "browser"]
    assert events[-2:] == ["cancel-attempt", "close-listener"]
    assert captured["existing_registration"]["subject"] == "subject-one"
    assert isinstance(captured["existing_credentials"], Credentials)
    assert captured["cancel"] is cancel_check


def test_persisted_refresh_marker_blocks_old_refresh_token_on_restart(setup):
    connection_id = plan_connection(setup, plan_result(expiry=0), storage="vault")
    reference = setup.state.read()["profiles"][0]["credential_ref"]
    values = setup.vault.get(reference).snapshot()
    values["refresh_in_progress"] = True
    setup.vault.put(reference, Credentials(values))
    restarted = ConnectionManager(
        setup.state,
        session=MemoryStore(),
        vault=setup.vault,
        transport=NoNetwork(),
        oauth_factory=lambda host: setup.oauth,
    )
    with pytest.raises(ConnectionError):
        restarted.models(connection_id)
    assert setup.oauth.refreshes == setup.discovery == []
    assert restarted.status(connection_id)["credentials_saved"] is False


def test_remote_rotation_then_store_and_metadata_failure_cannot_replay_old_token(
    setup, monkeypatch
):
    connection_id = plan_connection(setup, plan_result(expiry=0), storage="vault")
    reference = setup.state.read()["profiles"][0]["credential_ref"]
    setup.oauth.refreshed = plan_result(suffix="rotated")
    saved_put, saved_write = setup.vault.put, setup.state.write

    def fail_rotated_write(ref, protected):
        if protected.snapshot().get("access_token") == "fake-access-rotated-token":
            raise RuntimeError("fake-refresh-rotated-token")
        saved_put(ref, protected)

    def fail_metadata(value):
        raise OSError("fake-refresh-one-token")

    monkeypatch.setattr(setup.vault, "put", fail_rotated_write)
    monkeypatch.setattr(setup.state, "write", fail_metadata)
    with pytest.raises(ConnectionError) as error:
        setup.manager.models(connection_id)
    assert "fake-refresh" not in str(error.value)
    assert len(setup.oauth.refreshes) == 1
    assert setup.vault.get(reference).snapshot().get("refresh_in_progress") is True
    monkeypatch.setattr(setup.state, "write", saved_write)
    restarted = ConnectionManager(
        setup.state,
        session=MemoryStore(),
        vault=setup.vault,
        transport=NoNetwork(),
        oauth_factory=lambda host: setup.oauth,
    )
    with pytest.raises(ConnectionError):
        restarted.models(connection_id)
    assert len(setup.oauth.refreshes) == 1 and setup.discovery == []


def test_interrupted_remote_revocation_still_disables_and_removes_local_credentials(
    setup,
):
    connection_id = plan_connection(setup)
    setup.manager.select(connection_id)
    reference = setup.state.read()["profiles"][0]["credential_ref"]
    setup.oauth.revoked = KeyboardInterrupt()
    try:
        setup.manager.disconnect(connection_id)
    except KeyboardInterrupt:
        pass
    assert setup.manager.status(connection_id)["credentials_saved"] is False
    assert setup.manager.status(connection_id)["selected"] is False
    assert setup.session.get(reference) is None
    with pytest.raises(ConnectionError):
        setup.manager.models(connection_id)
    setup.manager.disconnect(connection_id)
    assert len(setup.oauth.revocations) == 1


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionError("cancelled"),
        ConnectionError("network"),
        ConnectionError("timeout"),
        ConnectionError("provider"),
        ConnectionError("dependency"),
        ConnectionError("storage"),
        KeyboardInterrupt(),
        RuntimeError("fake provider failure"),
    ],
)
def test_pending_identity_recovery_failure_retains_rotated_bundle_without_refresh_respend(
    setup, failure
):
    connection_id = plan_connection(setup, plan_result(expiry=0), storage="vault")
    setup.oauth.refreshed = plan_result(suffix="rotated", pending=True)
    with pytest.raises(ConnectionError):
        setup.manager.models(connection_id)
    reference = setup.state.read()["profiles"][0]["credential_ref"]
    protected_before = setup.vault.get(reference).snapshot()
    assert protected_before["identity_pending"] is True
    assert not protected_before.get("refresh_in_progress")
    setup.oauth.refreshed = failure
    expected = (
        KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else ConnectionError
    )
    with pytest.raises(expected):
        setup.manager.models(connection_id)
    assert setup.manager.status(connection_id)["credentials_saved"] is True
    assert setup.state.read()["profiles"][0]["credential_ref"] == reference
    assert setup.vault.get(reference).snapshot() == protected_before
    assert setup.discovery == []
    setup.oauth.refreshed = plan_result(suffix="rotated")
    assert setup.manager.models(connection_id) == [
        ModelChoice("chosen-model", "Chosen model")
    ]
    assert setup.vault.get(reference).snapshot()["identity_pending"] is False
    assert setup.discovery[-1][1]["refresh_token"] == "fake-refresh-rotated-token"
    spent = [
        values["refresh_token"]
        for _, values in setup.oauth.refreshes
        if not values.get("identity_pending")
    ]
    assert spent == ["fake-refresh-one-token"]


@pytest.mark.parametrize("failure", ["auth_error", "wrong_account"])
def test_pending_identity_authentication_failure_still_invalidates_connection(
    setup, failure
):
    connection_id = plan_connection(setup, plan_result(expiry=0), storage="vault")
    setup.oauth.refreshed = plan_result(suffix="rotated", pending=True)
    with pytest.raises(ConnectionError):
        setup.manager.models(connection_id)
    setup.oauth.refreshed = (
        ConnectionError("auth")
        if failure == "auth_error"
        else plan_result("other-subject", suffix="rotated")
    )
    with pytest.raises(ConnectionError) as error:
        setup.manager.models(connection_id)
    assert error.value.code == "auth"
    assert setup.manager.status(connection_id)["credentials_saved"] is False
    assert setup.discovery == []


def test_pre_cancelled_sign_in_touches_no_manager_listener_or_browser():
    from tsfm_lab.connections.sign_in import browser_sign_in

    touched = []

    def forbidden(*args, **kwargs):
        touched.append("side effect")
        raise AssertionError("Pre-cancel must precede any sign-in work")

    manager = SimpleNamespace(plan_setup=forbidden)
    with pytest.raises(ConnectionError) as error:
        browser_sign_in(
            manager,
            "unused",
            listener_factory=forbidden,
            browser=forbidden,
            cancel=lambda: True,
        )
    assert error.value.code == "cancelled"
    assert touched == []


def test_failing_pre_cancel_callback_is_canned_without_sign_in_side_effects():
    from tsfm_lab.connections.sign_in import browser_sign_in

    touched = []

    def failed_cancel():
        raise RuntimeError(SECRET)

    def forbidden(*args, **kwargs):
        touched.append("side effect")
        raise AssertionError("Cancellation failure must stop sign-in work")

    manager = SimpleNamespace(plan_setup=forbidden)
    with pytest.raises(ConnectionError) as error:
        browser_sign_in(
            manager,
            "unused",
            listener_factory=forbidden,
            browser=forbidden,
            cancel=failed_cancel,
        )
    assert SECRET not in str(error.value)
    assert touched == []

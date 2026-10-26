"""Synthetic authorization only: no provider, browser, socket, or credentials."""

import base64
import hashlib
import io
import json
import socket
from email.message import Message
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from tsfm_lab.connections import oauth_callback
from tsfm_lab.connections.contracts import ConnectionError
from tsfm_lab.connections.credentials import Credentials
from tsfm_lab.connections.identity import ISSUER, IdentityUnavailable, IdentityVerifier
from tsfm_lab.connections.oauth import RESOURCE, OAuthClient
from tsfm_lab.connections.oauth_callback import AuthListener
from tsfm_lab.connections.transport import HTTPResponse

NOW = 1800000000
CLIENT = "client_fixture"
SUBJECT = "subject_fixture"
JWKS = ISSUER + "/.well-known/jwks.json"
DISCOVERY = {
    "issuer": ISSUER,
    "authorization_endpoint": ISSUER + "/api/accounts/authorize",
    "token_endpoint": ISSUER + "/api/accounts/oauth/token",
    "jwks_uri": JWKS,
    "revocation_endpoint": ISSUER + "/api/accounts/oauth/revoke",
}
ACCOUNT = {"issuer": ISSUER, "client_id": CLIENT, "subject": SUBJECT}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refused(*_args, **_kwargs):
        raise AssertionError("Network access is forbidden in synthetic OAuth tests")

    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket, "create_connection", refused)


class FakeHTTP:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def send(self, request, **_kwargs):
        self.requests.append(request)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, HTTPResponse):
            return result
        return HTTPResponse(200, json.dumps(result).encode())


class FakeVerifier:
    def __init__(self, *errors):
        self.errors = list(errors)
        self.calls = []

    def verify(self, token, **kwargs):
        self.calls.append((token, kwargs))
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        return {"sub": SUBJECT, "email": "fake@example.invalid"}


def tokens(**changes):
    result = {
        "access_token": "fake_access",
        "refresh_token": "fake_refresh",
        "id_token": "fake_identity",
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": "openid offline_access chatgpt.tokens.use.direct",
    }
    result.update(changes)
    return result


def credentials(**changes):
    values = {
        **ACCOUNT,
        **tokens(),
        "scopes": ["openid", "offline_access", "chatgpt.tokens.use.direct"],
        "access_expires_at": NOW + 3600,
        "identity_pending": False,
    }
    values.pop("scope")
    values.pop("expires_in")
    values.update(changes)
    return Credentials(values)


def callback(attempt, **changes):
    query = parse_qs(urlsplit(attempt.authorization_url).query)
    fields = {"state": query["state"][0], "code": "fake_code", "client_id": CLIENT}
    fields.update(changes)
    return attempt.redirect_uri + "?" + urlencode(fields)


def client(http, verifier=None):
    return OAuthClient(
        http,
        verifier=verifier or FakeVerifier(),
        clock=lambda: NOW,
        host_id="urn:uuid:3af3e11e-0d50-4ca4-9641-f0e0e9ceaa85",
    )


def test_dynamic_registration_pkce_and_protected_representations():
    http = FakeHTTP(DISCOVERY, tokens())
    verifier = FakeVerifier()
    auth = client(http, verifier)
    pending = auth.begin_sign_in(redirect_uri="http://127.0.0.1:51999/auth/callback")
    query = parse_qs(urlsplit(pending.authorization_url).query)
    assert query["client_id"] == ["dynamic_agent_client"]
    assert query["agent_name_hint"] == ["TSFM-LAB"]
    assert query["resource"] == [RESOURCE]
    assert query["code_challenge_method"] == ["S256"]
    assert "id_token_hint" not in query and "login_hint" not in query
    assert pending.registration is None
    result = auth.complete_callback(pending, callback(pending))
    form = http.requests[-1].form_body
    challenge = (
        base64.urlsafe_b64encode(
            hashlib.sha256(form["code_verifier"].encode()).digest()
        )
        .decode()
        .rstrip("=")
    )
    assert query["code_challenge"] == [challenge]
    assert form["redirect_uri"] == pending.redirect_uri
    assert result.account["subject"] == SUBJECT
    assert pending.registration == {"issuer": ISSUER, "client_id": CLIENT}
    assert verifier.calls[0][1]["nonce"] == query["nonce"][0]
    assert not result.identity_pending
    for value in (pending, result, result.credentials, http.requests[-1]):
        assert "fake_" not in repr(value)
        assert query["state"][0] not in repr(value)
    with pytest.raises(ConnectionError):
        auth.complete_callback(pending, callback(pending))
    assert len(http.requests) == 2


def test_returning_registration_and_expected_account_are_bound():
    http = FakeHTTP(DISCOVERY, tokens())
    verifier = FakeVerifier()
    auth = client(http, verifier)
    pending = auth.begin_sign_in(ACCOUNT, credentials())
    query = parse_qs(urlsplit(pending.authorization_url).query)
    assert query["client_id"] == [CLIENT] and "agent_name_hint" not in query
    fields = parse_qs(urlsplit(callback(pending)).query)
    fields.pop("client_id")
    auth.complete_callback(
        pending,
        pending.redirect_uri + "?" + urlencode({k: v[0] for k, v in fields.items()}),
    )
    assert verifier.calls[0][1]["subject"] == SUBJECT
    with pytest.raises(ConnectionError):
        auth.begin_sign_in(ACCOUNT, credentials(subject="other"))


def test_wrong_state_does_not_consume_attempt_but_errors_do():
    http = FakeHTTP(DISCOVERY)
    auth = client(http)
    pending = auth.begin_sign_in()
    with pytest.raises(ConnectionError):
        auth.complete_callback(pending, callback(pending, state="wrong"))
    with pytest.raises(ConnectionError):
        auth.complete_callback(pending, callback(pending, error="access_denied"))
    with pytest.raises(ConnectionError):
        auth.complete_callback(pending, callback(pending))
    assert len(http.requests) == 1


@pytest.mark.parametrize(
    "suffix", ["&state=duplicate", "&code=duplicate", "&client_id=duplicate"]
)
def test_duplicate_callback_fields_are_rejected(suffix):
    auth = client(FakeHTTP(DISCOVERY))
    pending = auth.begin_sign_in()
    with pytest.raises(ConnectionError):
        auth.complete_callback(pending, callback(pending) + suffix)


def test_issued_registration_survives_exchange_failure():
    http = FakeHTTP(DISCOVERY, HTTPResponse(400, b'{"secret":"do-not-expose"}'))
    auth = client(http)
    pending = auth.begin_sign_in()
    with pytest.raises(ConnectionError) as failure:
        auth.complete_callback(pending, callback(pending))
    assert failure.value.code == "auth"
    assert "do-not-expose" not in str(failure.value)
    assert pending.registration == {"issuer": ISSUER, "client_id": CLIENT}


@pytest.mark.parametrize(
    "uri",
    [
        "http://localhost:1455/auth/callback",
        "https://127.0.0.1:1455/auth/callback",
        "http://127.0.0.1:1455/wrong",
        "http://127.0.0.1:1455/auth/callback?state=x",
    ],
)
def test_redirect_uris_are_loopback_only(uri):
    with pytest.raises(ConnectionError):
        client(FakeHTTP(DISCOVERY)).begin_sign_in(redirect_uri=uri)


@pytest.mark.parametrize(
    "field",
    ["authorization_endpoint", "token_endpoint", "jwks_uri", "revocation_endpoint"],
)
def test_discovery_cannot_redirect_credentials_to_another_authority(field):
    bad = {**DISCOVERY, field: "https://example.invalid/collect"}
    with pytest.raises(ConnectionError):
        client(FakeHTTP(bad)).begin_sign_in()


@pytest.mark.parametrize(
    "changes",
    [
        {"token_type": "mac"},
        {"expires_in": True},
        {"expires_in": 0},
        {"access_token": "bad\nvalue"},
        {"scope": None},
        {"refresh_token": ""},
    ],
)
def test_malformed_token_contracts_fail_closed(changes):
    auth = client(FakeHTTP(DISCOVERY, tokens(**changes)))
    pending = auth.begin_sign_in()
    with pytest.raises(ConnectionError):
        auth.complete_callback(pending, callback(pending))


def test_identity_only_scope_is_retained_without_inference_grant():
    auth = client(FakeHTTP(DISCOVERY, tokens(scope="openid profile email")))
    pending = auth.begin_sign_in()
    result = auth.complete_callback(pending, callback(pending))
    assert result.credentials.snapshot()["scopes"] == ["openid", "profile", "email"]


def test_refresh_rotates_tokens_and_preserves_omitted_scopes():
    fresh = tokens(access_token="new_access", refresh_token="new_refresh")
    fresh.pop("scope")
    auth = client(FakeHTTP(DISCOVERY, fresh))
    result = auth.refresh(ACCOUNT, credentials())
    assert result.credentials.reveal("refresh_token") == "new_refresh"
    assert result.credentials.snapshot()["scopes"] == credentials().snapshot()["scopes"]


@pytest.mark.parametrize(
    "error",
    [
        IdentityUnavailable(),
        ConnectionError("cancelled"),
        ConnectionError("dependency"),
    ],
)
def test_pending_rotation_is_protected_and_resumed_without_refresh_replay(error):
    http = FakeHTTP(DISCOVERY, tokens(refresh_token="rotated_refresh"))
    verifier = FakeVerifier(error, None)
    auth = client(http, verifier)
    pending = auth.refresh(ACCOUNT, credentials())
    assert pending.identity_pending
    assert pending.credentials.reveal("refresh_token") == "rotated_refresh"
    recovered = auth.refresh(ACCOUNT, pending.credentials)
    assert not recovered.identity_pending
    assert len(http.requests) == 2
    assert verifier.calls[-1][1]["now"] == NOW


def test_invalid_fresh_identity_is_terminal_and_not_retried():
    http = FakeHTTP(DISCOVERY, tokens(refresh_token="rotated_refresh"))
    auth = client(http, FakeVerifier(ConnectionError("auth")))
    with pytest.raises(ConnectionError) as failure:
        auth.refresh(ACCOUNT, credentials())
    assert failure.value.code == "auth" and len(http.requests) == 2


def test_refresh_without_new_identity_keeps_existing_verified_account():
    fresh = tokens(refresh_token="rotated_refresh")
    fresh.pop("id_token")
    verifier = FakeVerifier()
    result = client(FakeHTTP(DISCOVERY, fresh), verifier).refresh(
        ACCOUNT, credentials()
    )
    assert result.account == ACCOUNT and not verifier.calls


@pytest.mark.parametrize("status,expected", [(200, True), (500, False)])
def test_revocation_is_explicit_and_does_not_parse_secret_bodies(status, expected):
    http = FakeHTTP(DISCOVERY, HTTPResponse(status, b"secret"))
    assert client(http).revoke(ACCOUNT, credentials()) is expected
    assert http.requests[-1].form_body["token_type_hint"] == "refresh_token"


@pytest.fixture(scope="module")
def signing_key():
    jwt = pytest.importorskip("jwt")
    rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    public.update(kid="synthetic-key", alg="RS256", use="sig")
    return jwt, key, public


def signed(signing_key, changes=None, headers=None):
    jwt, key, _ = signing_key
    claims = {
        "iss": ISSUER,
        "aud": CLIENT,
        "sub": SUBJECT,
        "iat": NOW - 10,
        "exp": NOW + 3600,
        "nonce": "synthetic-nonce",
    }
    claims.update(changes or {})
    return jwt.encode(
        claims,
        key,
        algorithm="RS256",
        headers={"kid": "synthetic-key", **(headers or {})},
    )


def verify(signing_key, token, **kwargs):
    transport = FakeHTTP({"keys": [signing_key[2]]})
    return IdentityVerifier(transport, clock=lambda: NOW).verify(
        token,
        client_id=CLIENT,
        jwks_uri=JWKS,
        nonce="synthetic-nonce",
        subject=SUBJECT,
        **kwargs,
    )


def test_synthetic_rsa_token_is_verified(signing_key):
    assert verify(signing_key, signed(signing_key))["sub"] == SUBJECT


@pytest.mark.parametrize(
    "changes",
    [
        {"iss": "https://example.invalid"},
        {"aud": "other-client"},
        {"sub": "other-subject"},
        {"nonce": "wrong"},
        {"exp": NOW - 6},
        {"iat": NOW + 6},
        {"iat": True},
        {"exp": True},
        {"azp": "other-client"},
        {"aud": [CLIENT, "other"]},
        {"nbf": NOW + 6},
        {"sub": ""},
    ],
)
def test_signed_invalid_identity_claims_fail_closed(signing_key, changes):
    with pytest.raises(ConnectionError) as failure:
        verify(signing_key, signed(signing_key, changes))
    assert failure.value.code == "auth"


@pytest.mark.parametrize(
    "header",
    [
        {"jku": "https://example.invalid/keys"},
        {"x5u": "https://example.invalid/cert"},
        {"kid": "unknown"},
    ],
)
def test_untrusted_key_hints_never_determine_a_network_destination(signing_key, header):
    with pytest.raises(ConnectionError) as failure:
        verify(signing_key, signed(signing_key, headers=header))
    assert failure.value.code == "auth"


def test_signature_and_algorithm_are_verified(signing_key):
    jwt, _, _ = signing_key
    forged = jwt.encode(
        {"sub": SUBJECT},
        "synthetic-secret",
        algorithm="HS256",
        headers={"kid": "synthetic-key"},
    )
    with pytest.raises(ConnectionError):
        verify(signing_key, forged)
    token = signed(signing_key)
    head, body, signature = token.split(".")
    signature = ("A" if signature[0] != "A" else "B") + signature[1:]
    with pytest.raises(ConnectionError):
        verify(signing_key, ".".join((head, body, signature)))


def test_jwks_outage_is_distinct_from_bad_identity(signing_key):
    verifier = IdentityVerifier(FakeHTTP(ConnectionError("network")), clock=lambda: NOW)
    with pytest.raises(IdentityUnavailable):
        verifier.verify(signed(signing_key), client_id=CLIENT, jwks_uri=JWKS)


def test_jwks_cache_refetches_unknown_rotation_key(signing_key):
    jwt, key, public = signing_key
    second = {**public, "kid": "rotated-key"}
    http = FakeHTTP({"keys": [public]}, {"keys": [second]})
    verifier = IdentityVerifier(http, clock=lambda: NOW)
    verifier.verify(signed(signing_key), client_id=CLIENT, jwks_uri=JWKS)
    rotated = jwt.encode(
        {"iss": ISSUER, "aud": CLIENT, "sub": SUBJECT, "iat": NOW, "exp": NOW + 60},
        key,
        algorithm="RS256",
        headers={"kid": "rotated-key"},
    )
    verifier.verify(rotated, client_id=CLIENT, jwks_uri=JWKS)
    assert len(http.requests) == 2


class FakeServer:
    def __init__(self, address, handler):
        assert address == ("127.0.0.1", 0)
        self.server_address = ("127.0.0.1", 51234)
        self.handler = handler
        self.paths = []
        self.statuses = []
        self.closed = False

    def handle_request(self):
        handler = self.handler.__new__(self.handler)
        handler.path = self.paths.pop(0)
        handler.headers = Message()
        handler.headers["Host"] = "127.0.0.1:51234"
        handler.send_response = self.statuses.append
        handler.send_header = lambda *_args: None
        handler.end_headers = lambda: None
        handler.wfile = io.BytesIO()
        handler.do_GET()

    def server_close(self):
        self.closed = True


def test_loopback_listener_uses_fake_socket_and_ignores_unrelated_state(monkeypatch):
    monkeypatch.setattr(oauth_callback, "HTTPServer", FakeServer)
    auth = client(FakeHTTP(DISCOVERY))
    with AuthListener() as listener:
        pending = auth.begin_sign_in(redirect_uri=listener.redirect_uri)
        server = listener._server
        server.paths = [
            "/auth/callback?state=wrong&code=wrong",
            urlsplit(callback(pending)).path + "?" + urlsplit(callback(pending)).query,
        ]
        assert listener.wait(pending) == callback(pending)
        assert server.statuses == [400, 200]
    assert server.closed


def test_listener_cancellation_consumes_attempt_without_socket_io(monkeypatch):
    monkeypatch.setattr(oauth_callback, "HTTPServer", FakeServer)
    auth = client(FakeHTTP(DISCOVERY))
    with AuthListener() as listener:
        pending = auth.begin_sign_in(redirect_uri=listener.redirect_uri)
        with pytest.raises(ConnectionError) as failure:
            listener.wait(pending, cancel=lambda: True)
        assert failure.value.code == "cancelled"
        with pytest.raises(ConnectionError):
            auth.complete_callback(pending, callback(pending))


@pytest.mark.parametrize("value", [None, "", 12, False])
def test_refresh_requires_a_fresh_access_token(value):
    fresh = tokens(refresh_token="rotated_refresh")
    if value is None:
        fresh.pop("access_token")
    else:
        fresh["access_token"] = value
    http = FakeHTTP(DISCOVERY, fresh)
    with pytest.raises(ConnectionError) as failure:
        client(http).refresh(ACCOUNT, credentials())
    assert failure.value.code == "response"


@pytest.mark.parametrize("mode", ["timeout", "cancelled"])
def test_slow_local_headers_are_bounded_without_real_sockets(monkeypatch, mode):
    import threading
    import time

    entered, released = threading.Event(), threading.Event()

    class SlowSocket:
        def shutdown(self, _how):
            released.set()

        def close(self):
            released.set()

    class SlowServer(FakeServer):
        def handle_request(self):
            with self.socket_lock:
                self.active_socket = SlowSocket()
            entered.set()
            released.wait(2)

    monkeypatch.setattr(oauth_callback, "HTTPServer", SlowServer)
    auth = client(FakeHTTP(DISCOVERY))
    with AuthListener(timeout_seconds=0.02) as listener:
        pending = auth.begin_sign_in(redirect_uri=listener.redirect_uri)
        started = time.monotonic()
        with pytest.raises(ConnectionError) as failure:
            listener.wait(
                pending,
                cancel=(lambda: entered.is_set()) if mode == "cancelled" else None,
            )
        assert failure.value.code == mode
        assert released.is_set()
        assert time.monotonic() - started < 0.8


def test_actual_identity_verifier_completes_synthetic_authorization(signing_key):
    http = FakeHTTP(DISCOVERY)
    auth = OAuthClient(http, clock=lambda: NOW)
    pending = auth.begin_sign_in()
    nonce = parse_qs(urlsplit(pending.authorization_url).query)["nonce"][0]
    http.responses.extend(
        [
            tokens(id_token=signed(signing_key, {"nonce": nonce})),
            {"keys": [signing_key[2]]},
        ]
    )
    result = auth.complete_callback(pending, callback(pending))
    assert result.account == ACCOUNT


def test_cancellation_before_exchange_still_retains_issued_registration():
    http = FakeHTTP(DISCOVERY)
    auth = client(http)
    pending = auth.begin_sign_in()
    with pytest.raises(ConnectionError) as failure:
        auth.complete_callback(pending, callback(pending), cancel=lambda: True)
    assert failure.value.code == "cancelled"
    assert pending.registration == {"issuer": ISSUER, "client_id": CLIENT}
    assert len(http.requests) == 1

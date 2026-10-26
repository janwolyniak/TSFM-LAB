"""Public-client ChatGPT authorization, with isolated one-use attempts."""

import base64
import hashlib
import math
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlsplit

from tsfm_lab.connections.contracts import ConnectionError, Limits
from tsfm_lab.connections.credentials import Credentials
from tsfm_lab.connections.identity import (
    ISSUER,
    IdentityUnavailable,
    IdentityVerifier,
    safe_text,
    trusted_endpoint,
)
from tsfm_lab.connections.transport import HTTPRequest, read_json

RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
_CLIENT = re.compile(r"[A-Za-z0-9_-]{1,200}\Z")


def _client_id(value):
    if (
        type(value) is not str
        or not _CLIENT.fullmatch(value)
        or value == "dynamic_agent_client"
    ):
        raise ConnectionError("auth")
    return value


def _redirect(value):
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
            or not 1 <= parsed.port <= 65535
            or parsed.path != "/auth/callback"
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError
        return value
    except (TypeError, ValueError):
        raise ConnectionError("invalid") from None


def _account(value, *, registration_only=False):
    if type(value) is not dict or value.get("issuer") != ISSUER:
        raise ConnectionError("auth")
    result = {"issuer": ISSUER, "client_id": _client_id(value.get("client_id"))}
    if "subject" in value:
        if not safe_text(value["subject"], 512):
            raise ConnectionError("auth")
        result["subject"] = value["subject"]
    elif not registration_only:
        raise ConnectionError("auth")
    if "email" in value and safe_text(value["email"], 320):
        result["email"] = value["email"]
    return result


@dataclass(frozen=True, repr=False)
class AuthResult:
    account: dict
    credentials: Credentials

    @property
    def identity_pending(self):
        return self.credentials.snapshot().get("identity_pending") is True

    def __repr__(self):
        return "<AuthResult protected>"


class PendingSignIn:
    def __init__(
        self,
        *,
        url,
        redirect,
        state,
        nonce,
        verifier,
        account,
        endpoints,
        deadline,
        owner,
    ):
        self.authorization_url = url
        self.redirect_uri = redirect
        self._state, self._nonce, self._verifier = state, nonce, verifier
        self._account, self._endpoints = account, endpoints
        self._deadline, self._owner = deadline, owner
        self._registration = account.copy() if account else None
        self._used = False
        self._lock = threading.Lock()

    def __repr__(self):
        return "<PendingSignIn protected>"

    @property
    def registration(self):
        if self._registration is None:
            return None
        return {key: self._registration[key] for key in ("issuer", "client_id")}

    def cancel(self):
        with self._lock:
            self._used = True
            self._verifier = ""

    def matches_state(self, state):
        return type(state) is str and secrets.compare_digest(state, self._state)

    def callback(self, callback_url):
        try:
            if type(callback_url) is not str or len(callback_url) > 16384:
                raise ValueError
            parsed = urlsplit(callback_url)
            expected = urlsplit(self.redirect_uri)
            if (parsed.scheme, parsed.netloc, parsed.path) != (
                expected.scheme,
                expected.netloc,
                expected.path,
            ) or parsed.fragment:
                raise ValueError
            fields = parse_qs(
                parsed.query,
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=20,
            )
            if any(len(value) != 1 for value in fields.values()):
                raise ValueError
            if not self.matches_state(fields.get("state", [None])[0]):
                raise ValueError
            return {key: value[0] for key, value in fields.items()}
        except (TypeError, ValueError):
            raise ConnectionError("auth") from None


class OAuthClient:
    def __init__(
        self,
        transport,
        *,
        verifier=None,
        limits=None,
        clock=time.time,
        host_id=None,
    ):
        self.transport, self.limits, self.clock = transport, limits or Limits(), clock
        self.verifier = verifier or IdentityVerifier(
            transport, limits=self.limits, clock=clock
        )
        self.host_id = host_id or f"urn:uuid:{uuid.uuid4()}"
        try:
            host_uuid = uuid.UUID(self.host_id.removeprefix("urn:uuid:"))
            if self.host_id != f"urn:uuid:{host_uuid}" or host_uuid.version != 4:
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            raise ConnectionError("invalid") from None
        self._discovered = None
        self._owner = object()

    def _discovery(self, cancel=None):
        if self._discovered is None:
            with self.transport.send(
                HTTPRequest("GET", ISSUER + "/.well-known/openid-configuration"),
                limits=self.limits,
                cancel=cancel,
            ) as response:
                payload = read_json(response, limits=self.limits, cancel=cancel)
            if payload.get("issuer") != ISSUER:
                raise ConnectionError("auth")
            endpoints = {
                key: trusted_endpoint(payload.get(key))
                for key in ("authorization_endpoint", "token_endpoint", "jwks_uri")
            }
            if payload.get("revocation_endpoint"):
                endpoints["revocation_endpoint"] = trusted_endpoint(
                    payload["revocation_endpoint"]
                )
            self._discovered = endpoints
        return self._discovered.copy()

    def begin_sign_in(
        self,
        existing_registration=None,
        existing_credentials=None,
        *,
        redirect_uri="http://127.0.0.1:1455/auth/callback",
        cancel=None,
    ):
        if cancel and cancel():
            raise ConnectionError("cancelled")
        account = (
            _account(existing_registration, registration_only=True)
            if existing_registration is not None
            else None
        )
        if existing_credentials is not None:
            if not account or "subject" not in account:
                raise ConnectionError("auth")
            self._bound(account, existing_credentials)
        endpoints = self._discovery(cancel)
        redirect_uri = _redirect(redirect_uri)
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        query = {
            "response_type": "code",
            "client_id": account["client_id"] if account else "dynamic_agent_client",
            "redirect_uri": redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "ext_agent_host_id": self.host_id,
        }
        if account is None:
            query["agent_name_hint"] = "TSFM-LAB"
        return PendingSignIn(
            url=endpoints["authorization_endpoint"] + "?" + urlencode(query),
            redirect=redirect_uri,
            state=state,
            nonce=nonce,
            verifier=verifier,
            account=account,
            endpoints=endpoints,
            deadline=self.clock() + 600,
            owner=self._owner,
        )

    def complete_callback(self, attempt, callback_url, cancel=None):
        if not isinstance(attempt, PendingSignIn) or attempt._owner is not self._owner:
            raise ConnectionError("auth")
        fields = attempt.callback(callback_url)
        with attempt._lock:
            if attempt._used or self.clock() > attempt._deadline:
                raise ConnectionError("auth")
            attempt._used = True
            verifier, attempt._verifier = attempt._verifier, ""
        if "error" in fields:
            raise ConnectionError("auth")
        code = fields.get("code")
        if not safe_text(code, 8192):
            raise ConnectionError("auth")
        previous = attempt._account
        client_id = _client_id(
            fields.get("client_id", previous["client_id"] if previous else None)
        )
        if previous and client_id != previous["client_id"]:
            raise ConnectionError("auth")
        attempt._registration = {"issuer": ISSUER, "client_id": client_id}
        if cancel and cancel():
            raise ConnectionError("cancelled")
        payload = self._post(
            attempt._endpoints["token_endpoint"],
            {
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": attempt.redirect_uri,
                "resource": RESOURCE,
            },
            cancel,
        )
        received = self.clock()
        values = self._tokens(payload, client_id=client_id, received=received)
        claims = self.verifier.verify(
            values.get("id_token"),
            client_id=client_id,
            jwks_uri=attempt._endpoints["jwks_uri"],
            nonce=attempt._nonce,
            subject=previous.get("subject") if previous else None,
            now=received,
            cancel=cancel,
        )
        account = self._identity(claims, client_id)
        values["subject"] = account["subject"]
        return AuthResult(account, Credentials(values))

    def _post(self, endpoint, fields, cancel):
        with self.transport.send(
            HTTPRequest(
                "POST",
                endpoint,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                form_body=fields,
            ),
            limits=self.limits,
            cancel=cancel,
        ) as response:
            if response.status == 400:
                # Error bodies can contain credentials; expose only a canned failure.
                raise ConnectionError("auth")
            return read_json(response, limits=self.limits, cancel=cancel)

    def _tokens(self, payload, *, client_id, received, previous=None):
        values = previous.copy() if previous else {}
        if not payload.get("access_token"):
            raise ConnectionError("response")
        scope = payload.get(
            "scope", " ".join(values.get("scopes", [])) if previous else None
        )
        if type(scope) is not str or len(scope) > 8192:
            raise ConnectionError("response")
        scopes = scope.split()
        if any(not safe_text(item, 256) for item in scopes) or len(scopes) > 100:
            raise ConnectionError("response")
        token_type, expires = payload.get("token_type"), payload.get("expires_in")
        if (
            type(token_type) is not str
            or token_type.lower() != "bearer"
            or type(expires) not in (int, float)
            or not math.isfinite(expires)
            or not 0 < expires <= 31536000
        ):
            raise ConnectionError("response")
        for field in ("access_token", "refresh_token", "id_token"):
            if field in payload:
                value = payload[field]
                if (
                    type(value) is not str
                    or not 0 < len(value) <= 65536
                    or any(ord(char) < 33 or ord(char) > 126 for char in value)
                ):
                    raise ConnectionError("response")
                values[field] = value
        if not values.get("access_token") or (
            "offline_access" in scopes and not values.get("refresh_token")
        ):
            raise ConnectionError("response")
        values.update(
            issuer=ISSUER,
            client_id=client_id,
            scopes=list(dict.fromkeys(scopes)),
            token_type="Bearer",
            access_expires_at=received + expires,
            identity_pending=False,
        )
        return values

    @staticmethod
    def _identity(claims, client_id):
        if type(claims) is not dict or not safe_text(claims.get("sub"), 512):
            raise ConnectionError("auth")
        account = {"issuer": ISSUER, "client_id": client_id, "subject": claims["sub"]}
        if safe_text(claims.get("email"), 320):
            account["email"] = claims["email"]
        return account

    @staticmethod
    def _bound(account, credentials):
        account = _account(account)
        if not isinstance(credentials, Credentials):
            raise ConnectionError("credentials")
        values = credentials.snapshot()
        if any(
            values.get(key) != account[key]
            for key in ("issuer", "client_id", "subject")
        ):
            raise ConnectionError("auth")
        return account, values

    def refresh(self, account, credentials, cancel=None):
        account, old = self._bound(account, credentials)
        endpoints = self._discovery(cancel)
        if old.get("identity_pending") is True:
            values, received = old, old.get("identity_received_at")
            if type(received) not in (int, float) or not math.isfinite(received):
                raise ConnectionError("auth")
        else:
            payload = self._post(
                endpoints["token_endpoint"],
                {
                    "grant_type": "refresh_token",
                    "client_id": account["client_id"],
                    "refresh_token": credentials.reveal("refresh_token"),
                    "resource": RESOURCE,
                },
                cancel,
            )
            received = self.clock()
            values = self._tokens(
                payload, client_id=account["client_id"], received=received, previous=old
            )
            if "id_token" not in payload:
                return AuthResult(account, Credentials(values))
        try:
            claims = self.verifier.verify(
                values.get("id_token"),
                client_id=account["client_id"],
                jwks_uri=endpoints["jwks_uri"],
                subject=account["subject"],
                now=received,
                cancel=cancel,
            )
        except ConnectionError as error:
            if not isinstance(error, IdentityUnavailable) and error.code not in (
                "cancelled",
                "dependency",
            ):
                raise
            values.update(identity_pending=True, identity_received_at=received)
            return AuthResult(account, Credentials(values))
        values["identity_pending"] = False
        values.pop("identity_received_at", None)
        return AuthResult(
            self._identity(claims, account["client_id"]), Credentials(values)
        )

    def revoke(self, account, credentials, cancel=None):
        account, _ = self._bound(account, credentials)
        endpoint = self._discovery(cancel).get("revocation_endpoint")
        if not endpoint:
            return False
        try:
            with self.transport.send(
                HTTPRequest(
                    "POST",
                    endpoint,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    form_body={
                        "token": credentials.reveal("refresh_token"),
                        "token_type_hint": "refresh_token",
                        "client_id": account["client_id"],
                    },
                ),
                limits=self.limits,
                cancel=cancel,
            ) as response:
                return response.status == 200
        except ConnectionError as error:
            if error.code == "cancelled":
                raise
            return False

"""Verify ChatGPT identities using a pinned issuer and trusted public keys."""

import math
import secrets
import time
from urllib.parse import urlsplit

from tsfm_lab.connections.contracts import ConnectionError, Limits
from tsfm_lab.connections.transport import HTTPRequest, read_json

ISSUER = "https://auth.openai.com"


def trusted_endpoint(value):
    try:
        parsed = urlsplit(value)
        valid = (
            type(value) is str
            and parsed.scheme == "https"
            and parsed.hostname == "auth.openai.com"
            and parsed.port in (None, 443)
            and not parsed.username
            and not parsed.password
            and not parsed.fragment
            and not parsed.query
            and parsed.path.startswith("/")
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ConnectionError("auth")
    return value


class IdentityUnavailable(ConnectionError):
    """Fresh rotated credentials must be saved but cannot yet be used."""

    def __init__(self):
        super().__init__("provider")


class IdentityVerifier:
    def __init__(self, transport, *, limits=None, clock=time.time):
        self.transport, self.limits, self.clock = transport, limits or Limits(), clock
        self._keys = None
        self._keys_at = 0

    def _key(self, kid, jwks_uri, jwt, cancel):
        jwks_uri = trusted_endpoint(jwks_uri)
        now = self.clock()
        keys = self._keys if now - self._keys_at < 300 else None
        if keys is None or not any(key.get("kid") == kid for key in keys):
            try:
                with self.transport.send(
                    HTTPRequest("GET", trusted_endpoint(jwks_uri)),
                    limits=self.limits,
                    cancel=cancel,
                ) as response:
                    payload = read_json(response, limits=self.limits, cancel=cancel)
                keys = payload.get("keys")
                if (
                    type(keys) is not list
                    or not 0 < len(keys) <= 100
                    or any(type(key) is not dict for key in keys)
                ):
                    raise ValueError
                self._keys, self._keys_at = keys, now
            except ConnectionError as error:
                if error.code == "cancelled":
                    raise
                raise IdentityUnavailable() from None
            except Exception:
                raise IdentityUnavailable() from None
        matches = [key for key in keys if key.get("kid") == kid]
        if len(matches) != 1:
            raise ConnectionError("auth")
        key = matches[0]
        if (
            key.get("kty") != "RSA"
            or key.get("alg", "RS256") != "RS256"
            or key.get("use", "sig") != "sig"
            or "d" in key
            or ("key_ops" in key and key["key_ops"] != ["verify"])
        ):
            raise ConnectionError("auth")
        try:
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
            if public_key.key_size < 2048:
                raise ValueError
            return public_key
        except Exception:
            raise ConnectionError("auth") from None

    def verify(
        self,
        token,
        *,
        client_id,
        jwks_uri,
        nonce=None,
        subject=None,
        now=None,
        cancel=None,
    ):
        if cancel and cancel():
            raise ConnectionError("cancelled")
        try:
            import jwt
        except ImportError:
            raise ConnectionError("dependency") from None
        try:
            if type(token) is not str or not 0 < len(token) <= 65536:
                raise ValueError
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if (
                header.get("alg") != "RS256"
                or type(kid) is not str
                or not 0 < len(kid) <= 200
                or any(field in header for field in ("jku", "x5u", "jwk", "crit"))
            ):
                raise ValueError
        except Exception:
            raise ConnectionError("auth") from None
        key = self._key(kid, jwks_uri, jwt, cancel)
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=client_id,
                issuer=ISSUER,
                options={
                    "require": ["iss", "aud", "sub", "exp", "iat"],
                    "verify_exp": False,
                    "verify_iat": False,
                    "verify_nbf": False,
                },
            )
            _validate_claims(
                claims,
                client_id=client_id,
                nonce=nonce,
                subject=subject,
                now=self.clock() if now is None else now,
            )
            return claims
        except Exception:
            raise ConnectionError("auth") from None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate_claims(claims, *, client_id, nonce, subject, now):
    expires, issued = claims["exp"], claims["iat"]
    if (
        not _number(expires)
        or not _number(issued)
        or expires <= now - 5
        or issued > now + 5
        or expires <= issued
        or not safe_text(claims["sub"], 512)
    ):
        raise ValueError
    if "nbf" in claims and (not _number(claims["nbf"]) or claims["nbf"] > now + 5):
        raise ValueError
    audience = claims["aud"]
    if type(audience) not in (str, list) or (
        type(audience) is list
        and (not audience or any(type(item) is not str for item in audience))
    ):
        raise ValueError
    if ("azp" in claims and claims["azp"] != client_id) or (
        type(audience) is list and len(audience) > 1 and claims.get("azp") != client_id
    ):
        raise ValueError
    if nonce is not None and (
        type(claims.get("nonce")) is not str
        or not secrets.compare_digest(claims["nonce"], nonce)
    ):
        raise ValueError
    if subject is not None and claims["sub"] != subject:
        raise ValueError


def safe_text(value, maximum):
    return (
        type(value) is str
        and 0 < len(value) <= maximum
        and value == value.strip()
        and not any(ord(char) < 32 or ord(char) == 127 for char in value)
    )

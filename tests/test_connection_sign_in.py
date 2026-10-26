"""Browser sequencing is entirely injected; no actual browser or login opens."""

from types import SimpleNamespace

import pytest

from tsfm_lab.connections.contracts import ConnectionError
from tsfm_lab.connections.sign_in import browser_sign_in


def setup(*, failure=None, stale=False):
    events = []
    registration = {
        "issuer": "https://auth.openai.com",
        "client_id": "fake-issued-client",
    }
    attempt = SimpleNamespace(
        authorization_url="https://auth.openai.com/api/accounts/authorize?fake=1",
        registration=None,
    )

    class Client:
        def begin_sign_in(self, **kwargs):
            events.append("begin")
            assert "start" in events
            assert kwargs["redirect_uri"] == "http://127.0.0.1:54321/auth/callback"
            assert kwargs["existing_registration"] is None
            return attempt

        def complete_callback(self, pending, callback, **kwargs):
            events.append("complete")
            assert callback == "fake-callback" and pending is attempt
            attempt.registration = registration
            if failure is not None:
                raise failure
            return "protected-result"

    class Manager:
        def plan_setup(self, identifier):
            return Client(), {"registration": None}, None, 3

        def finish_sign_in(self, identifier, result, generation):
            events.append("finish")
            assert result == "protected-result" and generation == 3
            if stale:
                raise ConnectionError("state")
            return "public-status"

        def remember_registration(self, identifier, saved, generation):
            assert saved == registration and generation == 3
            events.append("remember")
            if stale:
                raise ConnectionError("state")

    class Listener:
        def __init__(self, *, timeout_seconds):
            assert timeout_seconds == 120

        def start(self):
            events.append("start")

        @property
        def redirect_uri(self):
            assert events[0] == "start"
            return "http://127.0.0.1:54321/auth/callback"

        def wait(self, pending, **kwargs):
            events.append("callback")
            return "fake-callback"

        def close(self):
            events.append("close")

    def browser(url):
        events.append("browser")
        assert events == ["start", "begin", "browser"]
        assert url == attempt.authorization_url
        return True

    return Manager(), Listener, browser, events


def test_listener_precedes_browser_and_closes_after_success():
    manager, listener, browser, events = setup()
    assert (
        browser_sign_in(
            manager, "connection", listener_factory=listener, browser=browser
        )
        == "public-status"
    )
    assert events == [
        "start",
        "begin",
        "browser",
        "callback",
        "complete",
        "finish",
        "close",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionError("network"),
        RuntimeError("fake-sensitive-error"),
        KeyboardInterrupt(),
    ],
)
def test_failed_exchange_retains_issued_registration_but_no_raw_errors(failure):
    manager, listener, browser, events = setup(failure=failure)
    with pytest.raises(
        KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else ConnectionError
    ) as error:
        browser_sign_in(
            manager, "connection", listener_factory=listener, browser=browser
        )
    assert "fake-sensitive-error" not in str(error.value)
    assert events[-2:] == ["remember", "close"]
    assert "finish" not in events


def test_stale_sign_in_cannot_overwrite_newer_connection_state():
    manager, listener, browser, events = setup(stale=True)
    with pytest.raises(ConnectionError) as error:
        browser_sign_in(
            manager, "connection", listener_factory=listener, browser=browser
        )
    assert error.value.code == "state"
    assert events[-1] == "close"


def test_browser_failure_closes_listener_without_exchange():
    manager, listener, _, events = setup()
    with pytest.raises(ConnectionError):
        browser_sign_in(
            manager, "connection", listener_factory=listener, browser=lambda _: False
        )
    assert events == ["start", "begin", "close"]

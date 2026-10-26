"""Bind the loopback listener before opening official authorization."""

import webbrowser

from tsfm_lab.connections.contracts import ConnectionError


def browser_sign_in(
    manager, connection_id, *, listener_factory=None, browser=None, cancel=None
):
    listener, attempt = None, None
    try:
        if cancel and cancel():
            raise ConnectionError("cancelled")
        if listener_factory is None:
            from tsfm_lab.connections.oauth_callback import AuthListener

            listener_factory = AuthListener
        client, profile, credentials, generation = manager.plan_setup(connection_id)
        listener = listener_factory(timeout_seconds=120.0)
        listener.start()
        attempt = client.begin_sign_in(
            existing_registration=profile.get("account") or profile["registration"],
            existing_credentials=credentials,
            redirect_uri=listener.redirect_uri,
            cancel=cancel,
        )
        if cancel and cancel():
            raise ConnectionError("cancelled")
        opener = browser if browser is not None else webbrowser.open
        if not opener(attempt.authorization_url):
            raise ConnectionError("auth")
        callback = listener.wait(attempt, cancel=cancel)
        result = client.complete_callback(attempt, callback, cancel=cancel)
        return manager.finish_sign_in(connection_id, result, generation)
    except BaseException as exc:
        if attempt is not None and attempt.registration is not None:
            try:
                manager.remember_registration(
                    connection_id, attempt.registration, generation
                )
            except Exception:
                pass
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, ConnectionError):
            raise ConnectionError(exc.code) from None
        raise ConnectionError("auth") from None
    finally:
        if attempt is not None:
            try:
                attempt.cancel()
            except Exception:
                pass
        if listener is not None:
            try:
                listener.close()
            except Exception:
                pass

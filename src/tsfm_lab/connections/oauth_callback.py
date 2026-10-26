"""A bounded loopback-only listener; creating the object performs no I/O."""

import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

from tsfm_lab.connections.contracts import ConnectionError


class AuthListener:
    def __init__(self, *, timeout_seconds=120.0):
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 600
        ):
            raise ConnectionError("invalid")
        self.timeout_seconds = timeout_seconds
        self._server = None
        self._attempt = None
        self._callback = None

    def __repr__(self):
        return "<AuthListener loopback>"

    @property
    def redirect_uri(self):
        if self._server is None:
            raise ConnectionError("state")
        return f"http://127.0.0.1:{self._server.server_address[1]}/auth/callback"

    def start(self):
        if self._server is not None:
            raise ConnectionError("state")
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                accepted = False
                try:
                    expected_host = urlsplit(owner.redirect_uri).netloc
                    path = urlsplit(self.path)
                    if (
                        len(self.path) > 16384
                        or path.scheme
                        or path.netloc
                        or path.fragment
                        or path.path != "/auth/callback"
                        or self.headers.get_all("Host", []) != [expected_host]
                        or owner._attempt is None
                        or owner._callback is not None
                    ):
                        raise ValueError
                    callback = owner.redirect_uri + "?" + path.query
                    owner._attempt.callback(callback)
                    owner._callback = callback
                    accepted = True
                except (ValueError, ConnectionError):
                    pass
                self.send_response(200 if accepted else 400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; frame-ancestors 'none'",
                )
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(
                    b"Return to TSFM-LAB."
                    if accepted
                    else b"Sign-in callback rejected."
                )
                self.close_connection = True

            def log_message(self, *_args):
                pass

        class Server(HTTPServer):
            def __init__(self, *args):
                self.active_socket = None
                self.closing = False
                self.socket_lock = threading.Lock()
                super().__init__(*args)

            def close_active(self):
                with self.socket_lock:
                    self.closing = True
                    client = self.active_socket
                    self.active_socket = None
                if client is not None:
                    try:
                        client.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    client.close()

            def get_request(self):
                client, address = super().get_request()
                client.settimeout(0.5)
                with self.socket_lock:
                    if self.closing:
                        client.close()
                        raise OSError
                    self.active_socket = client
                return client, address

            def handle_error(self, *_args):
                pass

        try:
            self._server = Server(("127.0.0.1", 0), Handler)
            self._server.timeout = 0.1
        except OSError:
            raise ConnectionError("network") from None
        return self

    def wait(self, attempt, cancel=None):
        if self._server is None or attempt.redirect_uri != self.redirect_uri:
            raise ConnectionError("state")
        self._attempt = attempt
        deadline = time.monotonic() + self.timeout_seconds
        while self._callback is None:
            if cancel and cancel():
                attempt.cancel()
                raise ConnectionError("cancelled")
            if time.monotonic() >= deadline:
                attempt.cancel()
                raise ConnectionError("timeout")
            # The foreground deadline also bounds slow local header transmission.
            server = self._server
            worker = threading.Thread(target=server.handle_request, daemon=True)
            worker.start()
            while worker.is_alive():
                if cancel and cancel():
                    attempt.cancel()
                    server.close_active()
                    worker.join(0.5)
                    raise ConnectionError("cancelled")
                if time.monotonic() >= deadline:
                    attempt.cancel()
                    server.close_active()
                    worker.join(0.5)
                    raise ConnectionError("timeout")
                worker.join(0.05)
        return self._callback

    def close(self):
        if self._server is not None:
            self._server.close_active()
            self._server.server_close()
            self._server = None
        self._attempt, self._callback = None, None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_args):
        self.close()

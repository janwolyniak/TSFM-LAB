"""Human-operated connection menu; secret input has its own hidden boundary."""

from tsfm_lab.connections.contracts import KINDS, ConnectionError
from tsfm_lab.connections.manager import ConnectionManager
from tsfm_lab.repl.text import literal


class ConnectionMenu:
    def __init__(self, editor, emit, *, manager=None, sign_in=None):
        self.editor = editor
        self.emit = lambda value: emit(literal(value))
        self.manager = manager if manager is not None else ConnectionManager()
        self.sign_in = sign_in

    def _choice(self, label, choices, default=""):
        value = self.editor.ask(label, default, choices)
        if value not in choices:
            raise ConnectionError("invalid")
        return value

    def _profiles(self):
        items = self.manager.list()
        if not items:
            self.emit(
                "No connections. Add one to begin; local forecasting works without one."
            )
        for number, item in enumerate(items, 1):
            active = " · active" if item["selected"] else ""
            self.emit(f"{number}. {item['label']} · {item['provider']}{active}")
            self.emit(
                f"   {item['usage']} · {item['storage']} · model {item['model'] or 'not selected'}"
            )
            self.emit(f"   {item['status']} · credentials hidden")
            if item["account"]:
                self.emit(
                    f"   Account {item['account'].get('email', item['account']['subject'])}"
                )
            if item["workspace_id"]:
                self.emit(f"   Anthropic workspace {item['workspace_id']}")
        return items

    def _pick(self):
        items = self._profiles()
        if not items:
            raise ConnectionError("state")
        answer = self._choice(
            "Connection number", tuple(str(i) for i in range(1, len(items) + 1))
        )
        return items[int(answer) - 1]

    def _secret(self):
        hidden = getattr(self.editor, "ask_secret", None)
        if hidden is None:
            raise ConnectionError("credentials")
        return hidden("API key (hidden; never enter a key in the command prompt)")

    def _workspace(self, kind):
        if kind != "anthropic_api":
            return None
        return (
            self.editor.ask(
                "Anthropic workspace ID (optional; leave empty if not required)"
            )
            or None
        )

    def _add(self):
        for kind, (name, usage, _) in KINDS.items():
            self.emit(f"{kind}: {name} · {usage}")
        kind = self._choice("Connection type", tuple(KINDS))
        label = self.editor.ask("Nonsecret connection name")
        self.emit(
            "Session keeps credentials only until this terminal closes. Vault uses the native OS credential store; failure never falls back to a file."
        )
        storage = self._choice("Credential storage", ("session", "vault"), "session")
        if kind == "chatgpt_plan":
            self.manager.add_plan(label, storage=storage)
            self.emit(
                "Connection added. Select it, then choose sign-in to open official authorization."
            )
        else:
            workspace = self._workspace(kind)
            self.manager.add_key(
                kind, label, self._secret(), storage=storage, workspace_id=workspace
            )
            self.emit(
                "Connection added. Select it, then discover and choose an assistant model."
            )

    def _replace(self):
        item = self._pick()
        if item["kind"] == "chatgpt_plan":
            self.emit(
                "Use sign-in to renew this account. Add a separate connection for another account."
            )
            return
        workspace = self._workspace(item["kind"])
        result = self.manager.replace_key(
            item["id"], self._secret(), workspace_id=workspace
        )
        self.emit("Key replaced. Discover and choose a model again.")
        if not result["previous_credentials_removed"]:
            self.emit(
                "Old credentials are disconnected but could not be removed from the vault. Retry disconnect after unlocking the vault."
            )

    def _models(self):
        active = self.manager.status()
        self.emit(
            f"Discovering models · {active['provider']} · {active['usage']}. No inference request."
        )
        choices = self.manager.models()
        for item in choices:
            self.emit(f"{item.identifier}: {item.label}")
        if not choices:
            self.emit("No usable models returned. No model was selected.")
            return
        model = self._choice(
            "Assistant model (explicit selection)",
            tuple(item.identifier for item in choices),
        )
        self.manager.set_model(model)

    def _test(self):
        preview = self.manager.preview_test()
        self.emit(
            f"Connection {preview['connection_label']} · {preview['provider']} · {preview['usage']}"
        )
        self.emit(f"Assistant model {preview['model']}")
        if preview["account"]:
            self.emit(
                f"Account {preview['account'].get('email', preview['account']['subject'])}"
            )
        if preview["workspace_id"]:
            self.emit(f"Anthropic workspace {preview['workspace_id']}")
        self.emit("Exact outgoing connection-check context:")
        self.emit(f"Instructions: {preview['instructions']}")
        self.emit(f"Message: {preview['text']}")
        self.emit(preview["data_included"])
        limits = preview["local_limits"]
        allowance = (
            f"{preview['output_tokens']} tokens"
            if preview["output_tokens"]
            else "provider controlled; local byte limit applies"
        )
        self.emit(
            f"Request timeout {limits['seconds']:g}s · received limit {limits['received_bytes']:,} bytes · output allowance {allowance}"
        )
        self.emit(
            "This sends one inference request through the shown usage method. Local limits do not guarantee a monetary cap. Ctrl+c cancels; no automatic retries."
        )
        confirmed = (
            self._choice("Send this connection check?", ("no", "yes"), "no") == "yes"
        )
        if not confirmed:
            try:
                self.manager.submit_test(preview, confirmed=False)
            except ConnectionError as exc:
                if exc.code != "permission":
                    raise
            self.emit("Connection check cancelled. Nothing sent.")
            return
        answer = self.manager.submit_test(preview, confirmed=True)
        self.emit("Connection-check reply (untrusted provider text):")
        self.emit(answer)

    def _disconnect(self):
        item = self._pick()
        if self._choice("Disconnect this connection?", ("no", "yes"), "no") != "yes":
            return
        result = self.manager.disconnect(item["id"])
        self.emit("Connection disconnected. No other connection was selected.")
        if not result["local_credentials_removed"]:
            self.emit(
                "Local credential removal is incomplete. Unlock the vault and retry disconnect."
            )
        if result["remote_revocation_confirmed"] is False:
            self.emit(
                "Remote sign-out was not confirmed. Local access is disabled; review sessions at the provider if needed."
            )

    def _sign_in(self):
        item = self.manager.status()
        if item["kind"] != "chatgpt_plan":
            raise ConnectionError("invalid")
        self.emit(
            "Official ChatGPT sign-in · eligible plan usage. No API-key fallback."
        )
        if self._choice("Open the browser to sign in?", ("no", "yes"), "no") != "yes":
            return
        if self.sign_in is None:
            from tsfm_lab.connections.sign_in import browser_sign_in

            action = browser_sign_in
        else:
            action = self.sign_in
        action(self.manager, item["id"])
        self.emit(
            "Signed in. Discover and choose a model before the explicit connection check."
        )

    def run(self):
        self.emit(
            "Assistant connections · Setup and connection checks. Language setup and explanations are available in the main tsfm terminal."
        )
        actions = {
            "status": self._profiles,
            "add": self._add,
            "select": lambda: self.manager.select(self._pick()["id"]),
            "replace": self._replace,
            "sign-in": self._sign_in,
            "models": self._models,
            "test": self._test,
            "disconnect": self._disconnect,
        }
        while True:
            try:
                action = self._choice(
                    "Connections: status / add / select / replace / sign-in / models / test / disconnect / back",
                    (*actions, "back"),
                    "back",
                )
                if action == "back":
                    return
                actions[action]()
            except ConnectionError as exc:
                self.emit(f"Connection error: {exc}")
            except (EOFError, KeyboardInterrupt):
                self.emit("Connection action cancelled. Returning to the terminal.")
                return
            except Exception:
                self.emit(str(ConnectionError("state")))

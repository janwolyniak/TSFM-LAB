"""Optional language assistance, with local review and explicit outgoing consent."""

import json
from pathlib import Path

from tsfm_lab.assistant.drafts import DraftResolution, resolve_draft
from tsfm_lab.assistant.evidence import fact_map
from tsfm_lab.assistant.protocol import AssistanceError, validate_context
from tsfm_lab.connections.contracts import ConnectionError
from tsfm_lab.repl.text import literal


class AssistantFlow:
    def __init__(self, editor, emit, service, *, connections):
        self.editor, self.service, self.connections = editor, service, connections
        self.emit = lambda value: emit(literal(value))

    def _choice(self, label, choices, default):
        while True:
            value = self.editor.ask(label, default, choices).strip().lower()
            if value in {"/cancel", "/back"}:
                return "cancel"
            if value in choices:
                return value
            self.emit(f"Choose {', '.join(choices)}.")

    def _consent(self, preview):
        self.emit(
            f"Connection {preview['connection_label']} · {preview['provider']} · {preview['usage']}"
        )
        self.emit(f"Destination {preview.get('destination', preview['provider'])}")
        self.emit(f"Assistant model {preview['model']}")
        if preview["account"]:
            self.emit(
                f"Account {preview['account'].get('email',preview['account']['subject'])}"
            )
        if preview["workspace_id"]:
            self.emit(f"Anthropic workspace {preview['workspace_id']}")
        self.emit(f"Included: {preview['data_included']}")
        self.emit("Exact outgoing instructions:")
        self.emit(preview["instructions"])
        self.emit("Exact outgoing message:")
        self.emit(preview["text"])
        limits = preview["local_limits"]
        allowance = (
            f"{preview['output_tokens']} tokens"
            if preview["output_tokens"]
            else "provider controlled; local byte limit applies"
        )
        self.emit(
            f"Timeout {limits['seconds']:g}s · received limit {limits['received_bytes']:,} bytes · output allowance {allowance}"
        )
        self.emit(
            "One request through the shown usage method. Local limits do not guarantee a monetary cap. Ctrl+c cancels; no automatic retry."
        )
        try:
            accepted = self._choice("Send this message?", ("no", "yes"), "no") == "yes"
        except BaseException:
            self.service.discard(preview)
            raise
        if not accepted:
            self.service.discard(preview)
            self.emit("Message cancelled. Nothing sent.")
        return accepted

    def _dataset(self, default):
        while True:
            value = self.editor.ask(
                "Choose a local dataset (its path stays on this machine)",
                str(default) if default else "",
            )
            if value in {"/cancel", "/back"}:
                return None
            if value == "/connections":
                self.connections()
                continue
            path = Path(value).expanduser()
            if path.suffix.lower() in {".csv", ".parquet", ".pq"} and path.is_file():
                return path.resolve()
            self.emit("Choose an existing local CSV or Parquet file, or /cancel.")

    def _forecast_mode(self, default="historical_evaluation"):
        selected = self._choice(
            "Forecast mode",
            ("historical evaluation", "future forecasting"),
            (
                "future forecasting"
                if default == "future_forecast"
                else "historical evaluation"
            ),
        )
        if selected == "cancel":
            return None
        return (
            "future_forecast"
            if selected == "future forecasting"
            else "historical_evaluation"
        )

    def describe(self, review, *, default_dataset=None):
        self.emit(
            "Choose files and forecast mode locally; leave paths, credentials and data tables out of the message."
        )
        try:
            dataset = self._dataset(default_dataset)
            if dataset is None:
                return None
            mode = self._forecast_mode()
            if mode is None:
                return None
            goal = self.editor.ask(
                "What would you like to forecast?"
                if mode == "future_forecast"
                else "What would you like to evaluate?"
            )
        except (EOFError, KeyboardInterrupt):
            self.emit("Assisted setup cancelled. Nothing was executed or saved.")
            return None
        draft, confirm_mode = None, None

        def guided():
            return review(
                DraftResolution(
                    "needs_input",
                    {"dataset_path": str(dataset), "mode": mode},
                    ("dataset_path", "mode"),
                    (),
                    (),
                    (),
                )
            )

        def change_mode():
            nonlocal mode, confirm_mode
            selected = self._forecast_mode(mode)
            if selected is None:
                return False
            mode, confirm_mode = selected, selected
            return True

        while True:
            if goal in {"/cancel", "/back"}:
                return None
            try:
                preview = self.service.preview_draft(
                    goal,
                    draft=draft,
                    mode=mode,
                    **({"confirm_mode": confirm_mode} if confirm_mode else {}),
                )
                if not self._consent(preview):
                    return None
                draft = self.service.submit_draft(preview, confirmed=True)
                resolved = resolve_draft(
                    draft, dataset_refs={"selected_dataset": dataset}, mode=mode
                )
                self.emit(
                    f"Local draft status: {resolved.state.replace('_',' ')}. Nothing has run or been saved."
                )
                for key, value in draft.fields.items():
                    self.emit(f"  {key.replace('_',' ')}: {value or 'not supplied'}")
                if resolved.missing_fields:
                    self.emit(
                        "Still needed: "
                        + ", ".join(
                            item.replace("_", " ") for item in resolved.missing_fields
                        )
                    )
                for error in resolved.errors:
                    self.emit(f"Local validation: {error}")
                for assumption in resolved.assumptions:
                    self.emit(f"Assistant assumption to review: {assumption}")
                eligible = draft.intent == mode and resolved.state != "unsupported"
                choices = (
                    ("review", "clarify", "guided", "mode", "cancel")
                    if eligible
                    else ("clarify", "guided", "mode", "cancel")
                )
                choice = self._choice(
                    "Review every setting, clarify with another message, use fresh guided setup, or cancel?",
                    choices,
                    "cancel",
                )
                if choice == "review":
                    return review(resolved)
                if choice == "guided":
                    return guided()
                if choice == "cancel":
                    return None
                if choice == "mode" and not change_mode():
                    return None
                self.emit(
                    "Clarification can fill missing choices. Mode changes require a fresh preview; use guided review to edit an existing choice."
                )
                goal = self.editor.ask(
                    "Clarify your goal (a new preview and confirmation follow)"
                )
            except (ConnectionError, AssistanceError) as exc:
                self.emit(f"Assistant: {exc}")
                choices = (
                    ("connections", "retry", "review", "guided", "mode", "cancel")
                    if draft is not None and draft.intent == mode
                    else ("connections", "retry", "guided", "mode", "cancel")
                )
                choice = self._choice(
                    "Use connections, retry, review the current draft, use guided setup, or cancel?",
                    choices,
                    "cancel",
                )
                if choice == "guided":
                    return guided()
                if choice == "review":
                    return review(
                        resolve_draft(
                            draft, dataset_refs={"selected_dataset": dataset}, mode=mode
                        )
                    )
                if choice == "cancel":
                    return None
                if choice == "connections":
                    self.connections()
                if choice == "mode" and not change_mode():
                    return None
                goal = self.editor.ask("Describe or clarify the selected forecast mode")
            except (EOFError, KeyboardInterrupt):
                self.emit("Assisted setup cancelled. Nothing was executed or saved.")
                return None

    def _facts(self, context, references=None):
        facts = fact_map(context)
        selected = list(references or facts)
        if references:
            selected = list(
                dict.fromkeys(
                    [
                        *selected,
                        *(
                            key
                            for key in facts
                            if key.endswith(
                                ("/units", "/evaluation_start", "/evaluation_end")
                            )
                        ),
                    ]
                )
            )
        self.emit("Checked local facts:")
        for identifier in selected:
            label = (
                identifier.removeprefix("/data/")
                .removeprefix("/glossary/")
                .replace("/", " · ")
                .replace("_", " ")
            )
            self.emit(
                f"  {label}: {json.dumps(facts[identifier],ensure_ascii=False,allow_nan=False)}"
            )

    def explain(self, context, *, default_question):
        try:
            context = validate_context(context)
            self._facts(context)
            self.emit(
                "The checked values above remain authoritative. Assistant wording is an optional interpretation, not an independent fact check."
            )
            question = self.editor.ask(
                "What would you like explained?", default_question
            )
            if question in {"/cancel", "/back"}:
                return
            preview = self.service.preview_explanation(question, context)
            if not self._consent(preview):
                return
            annotations = self.service.submit_explanation(preview, confirmed=True)
            self._facts(
                context, [ref for item in annotations for ref in item.evidence_ids]
            )
            self.emit(
                "Assistant interpretation (untrusted wording; check against the local facts):"
            )
            for item in annotations:
                self.emit(item.text)
        except (ConnectionError, AssistanceError) as exc:
            self.emit(f"Assistant: {exc}")
            self.emit(
                "No explanation was accepted. Use /connections or submit a new explicit request; /models and /show remain available for local inspection."
            )
        except (EOFError, KeyboardInterrupt):
            self.emit("Explanation cancelled. Nothing was changed.")

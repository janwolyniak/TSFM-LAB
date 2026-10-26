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

    def describe(self, review, *, default_dataset=None):
        self.emit(
            "Describe a historical evaluation. Choose files locally; leave paths, credentials and data tables out of the message."
        )
        dataset = self._dataset(default_dataset)
        if dataset is None:
            return None
        goal = self.editor.ask("What would you like to evaluate?")
        draft = None
        confirm_historical = False

        def guided():
            return review(
                DraftResolution(
                    "needs_input",
                    {"dataset_path": str(dataset)},
                    ("dataset_path",),
                    (),
                    (),
                    (),
                )
            )

        while True:
            if goal in {"/cancel", "/back"}:
                return None
            try:
                options = {"confirm_historical": True} if confirm_historical else {}
                preview = self.service.preview_draft(goal, draft=draft, **options)
                if not self._consent(preview):
                    return None
                draft = self.service.submit_draft(preview, confirmed=True)
                resolved = resolve_draft(
                    draft, dataset_refs={"selected_dataset": dataset}
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
                choices = (
                    ("review", "clarify", "guided", "cancel")
                    if draft.intent == "historical_evaluation"
                    and resolved.state != "unsupported"
                    else ("clarify", "guided", "cancel")
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
                if draft.intent != "historical_evaluation":
                    confirm_historical = (
                        self._choice(
                            "Explicitly switch to historical evaluation with observed targets?",
                            ("no", "yes"),
                            "no",
                        )
                        == "yes"
                    )
                    if not confirm_historical:
                        return None
                self.emit(
                    "Clarification can fill missing choices. Use guided review to edit an existing choice."
                )
                goal = self.editor.ask(
                    "Clarify your goal (a new preview and confirmation follow)"
                )
            except (ConnectionError, AssistanceError) as exc:
                self.emit(f"Assistant: {exc}")
                choice = self._choice(
                    "Use connections, retry, review the current draft, use guided setup, or cancel?",
                    (
                        ("connections", "retry", "review", "guided", "cancel")
                        if draft is not None and draft.intent == "historical_evaluation"
                        else ("connections", "retry", "guided", "cancel")
                    ),
                    "cancel",
                )
                if choice == "guided":
                    return guided()
                if choice == "review":
                    return review(
                        resolve_draft(draft, dataset_refs={"selected_dataset": dataset})
                    )
                if choice == "cancel":
                    return None
                if choice == "connections":
                    self.connections()
                goal = self.editor.ask("Describe or clarify the historical evaluation")
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

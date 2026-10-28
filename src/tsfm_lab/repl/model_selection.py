"""Friendly curated model selection without runtime imports or checkpoint access."""

import re

from tsfm_lab.config import ConfigError
from tsfm_lab.models.catalog import catalog_entries
from tsfm_lab.models.settings import model_settings

_PROMPTS = {
    "size": "Model size",
    "training_profile": "Training profile",
    "context_horizon": "Context/horizon profile",
}
_SELECTION_FIELDS = ("source", "repository", "revision", "checkpoint_path")


def _sha(value):
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ConfigError("Enter an exact 40-character lowercase checkpoint SHA")


def _required(value):
    if not value.strip():
        raise ConfigError("Enter a local checkpoint directory, or /cancel")


def _cache(value):
    if "://" in value or any(
        ord(char) < 32 or 127 <= ord(char) <= 159 for char in value
    ):
        raise ConfigError("Choose a local cache directory, or - for the default")


def _identity(variants, fields, profile):
    matches = [
        item for item in variants if item["repository"] == fields.get("repository")
    ]
    exact = next(
        (
            item
            for item in matches
            if item["reference_revision"] == fields.get("revision")
        ),
        None,
    )
    if profile is not None:
        selected = next((item for item in matches if item["variant"] == profile), None)
        if selected is not None and (exact is None or exact == selected):
            return selected["variant"]
        return None
    if exact is not None:
        return exact["variant"]
    return matches[0]["variant"] if len(matches) == 1 else None


class _Selection:
    def __init__(
        self,
        ask,
        emit,
        fields,
        checkpoint_variant,
        allow_preparation=False,
        offline=False,
    ):
        self.ask, self.emit = ask, emit
        self.fields = dict(fields)
        self.fields.pop("checkpoint_variant", None)
        self.profile = checkpoint_variant
        self.allow_preparation, self.offline = allow_preparation, offline
        self.entries = {item["adapter_id"]: item for item in catalog_entries()}
        self.labels = {key: model_settings(key) for key in self.entries}
        self.adapter = self.fields.get("adapter", "last_value")
        self.family = (
            self.labels[self.adapter].family_label
            if self.adapter in self.labels
            else self.adapter
        )
        self.saved = {}
        self.selected = None

    def question(self, label, default="", choices=(), validate=None):
        # Imported only when called, so the guide can import this helper lazily.
        from tsfm_lab.repl.wizard import _question

        return _question(self.ask, self.emit, label, default, choices, validate)

    def switch(self, adapter):
        if adapter == self.adapter:
            return
        self.saved[self.adapter] = (
            {key: self.fields.get(key, "") for key in _SELECTION_FIELDS},
            self.profile,
        )
        self.adapter = adapter
        values, self.profile = self.saved.get(
            adapter,
            (
                {
                    "source": "huggingface",
                    "repository": "",
                    "revision": "",
                    "checkpoint_path": "",
                },
                None,
            ),
        )
        self.fields.update(values)
        self.fields["adapter"] = adapter
        self.selected = None

    def choose_family(self):
        self.family = self.question(
            "Model family",
            self.family,
            tuple(dict.fromkeys(item.family_label for item in self.labels.values())),
        )

    def choose_generation(self):
        candidates = {
            key: item
            for key, item in self.labels.items()
            if item.family_label == self.family
        }
        if self.family == "Baselines":
            names = {item.variant_label: key for key, item in candidates.items()}
            default = (
                self.labels[self.adapter].variant_label
                if self.adapter in candidates
                else next(iter(names))
            )
            self.switch(names[self.question("Baseline", default, tuple(names))])
            self.fields.update(
                source="builtin", repository="", revision="", checkpoint_path=""
            )
            self.profile = None
            return
        names = {item.generation_label: key for key, item in candidates.items()}
        default = (
            self.labels[self.adapter].generation_label
            if self.adapter in candidates
            else next(iter(names))
        )
        if len(names) > 1:
            generation = self.question("Model generation", default, tuple(names))
        else:
            generation = next(iter(names))
            self.emit(f"Generation: {generation}")
        self.switch(names[generation])

    def choose_source(self):
        if self.allow_preparation:
            self.emit(
                "Automatic preparation checks metadata now; missing weights download only after confirmation. Cached mode stays offline."
            )
        else:
            self.emit(
                "Use an existing cached or local checkpoint; setup does not download models."
            )
        default = (
            "local"
            if self.fields.get("source") == "local"
            else (
                "automatic" if self.allow_preparation and not self.offline else "cached"
            )
        )
        source = self.question(
            "Checkpoint source",
            default,
            (
                ("automatic", "cached", "local")
                if self.allow_preparation
                else ("cached", "local")
            ),
        )
        if self.fields.get("source") != "huggingface" and source in {
            "automatic",
            "cached",
        }:
            self.fields["revision"] = ""
        self.fields["source"] = "local" if source == "local" else "huggingface"
        self.offline = source != "automatic"

    def choose_checkpoint(self):
        if self.fields["source"] == "local":
            self.selected = None
            self.fields["checkpoint_path"] = self.question(
                "Local checkpoint directory",
                self.fields.get("checkpoint_path", ""),
                validate=_required,
            )
            self.fields.update(repository="", revision="")
            self.profile = None
            return
        variants = self.entries[self.adapter]["variants"]
        names = {item["display_name"]: item for item in variants}
        identity = _identity(variants, self.fields, self.profile)
        current = next(
            (
                item
                for item in variants
                if item["variant"] == identity
                and item["repository"] == self.fields.get("repository")
            ),
            None,
        )
        default = current["display_name"] if current else variants[0]["display_name"]
        if self.profile is not None and current is None:
            default = self.profile
            self.emit(
                "The suggested checkpoint profile needs correction; choose a supported variant."
            )
        chosen = self.question(
            _PROMPTS[self.labels[self.adapter].variant_kind], default, tuple(names)
        )
        self.selected = names[chosen]
        if (
            identity != self.selected["variant"]
            or self.fields.get("repository") != self.selected["repository"]
            or not self.fields.get("revision")
        ):
            self.fields["revision"] = self.selected["reference_revision"]
        self.fields.update(repository=self.selected["repository"], checkpoint_path="")
        self.profile = self.selected["variant"]
        self.emit(
            f"Selected: {self.family} · {self.labels[self.adapter].generation_label} · {chosen}"
        )

    def advanced(self):
        answer = self.question("Advanced checkpoint options?", "no", ("yes", "no"))
        if answer == "yes":
            self.emit(f"Repository: {self.fields['repository']}")
            self.emit(f"Revision: {self.fields['revision']}")
            return True
        try:
            self.validate_revision(self.fields["revision"])
        except ValueError:
            self.emit("The suggested revision needs correction before continuing.")
            self.fields["revision"] = self.question(
                "Exact checkpoint SHA",
                self.fields["revision"],
                validate=self.validate_revision,
            )
        return False

    def validate_revision(self, value):
        _sha(value)
        model_settings(
            self.adapter,
            repository=self.fields["repository"],
            variant=self.profile,
            revision=value,
        )

    def finish(self):
        if self.allow_preparation:
            self.fields["preparation_offline"] = "true" if self.offline else "false"
        if self.selected is not None:
            if self.fields["revision"] == self.selected["reference_revision"]:
                self.emit(self.selected["verification_label"])
            else:
                self.emit(
                    "Custom checkpoint revision; individual inference verification is not established."
                )
            for reference in self.entries[self.adapter]["variants"]:
                if (
                    reference != self.selected
                    and reference["project_inference_status"] == "recorded"
                ):
                    self.emit(
                        f"Adapter reference: {self.family} · {self.labels[self.adapter].generation_label} · "
                        f"{reference['display_name']} · {reference['verification_label']}"
                    )
            self.fields["checkpoint_variant"] = self.profile
        return self.fields

    def run(self):
        from tsfm_lab.repl.wizard import _Back

        phase = 0
        while True:
            try:
                if phase == 0:
                    self.choose_family()
                elif phase == 1:
                    self.choose_generation()
                    if self.family == "Baselines":
                        return self.finish()
                elif phase == 2:
                    self.choose_source()
                elif phase == 3:
                    self.choose_checkpoint()
                    if self.fields["source"] == "local":
                        return self.finish()
                elif phase == 4:
                    if not self.advanced():
                        return self.finish()
                elif phase == 5:
                    self.fields["revision"] = self.question(
                        "Exact checkpoint SHA",
                        self.fields["revision"],
                        validate=self.validate_revision,
                    )
                else:
                    cache = self.question(
                        "Cache directory (optional; - uses the default)",
                        self.fields.get("cache_dir", ""),
                        validate=_cache,
                    )
                    self.fields["cache_dir"] = "" if cache == "-" else cache
                    return self.finish()
                phase += 1
            except _Back:
                if phase == 0:
                    raise
                phase -= 1
                if (
                    phase == 1
                    and sum(
                        item.family_label == self.family
                        for item in self.labels.values()
                    )
                    == 1
                ):
                    phase = 0


def select_model(
    ask,
    emit,
    fields: dict[str, str],
    *,
    checkpoint_variant: str | None = None,
    allow_preparation: bool = False,
    offline: bool = False,
) -> dict[str, str]:
    """Return reviewed selection fields, preserving caller input on cancellation."""
    return _Selection(
        ask, emit, fields, checkpoint_variant, allow_preparation, offline
    ).run()

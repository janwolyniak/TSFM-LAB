"""Friendly model selection preserves exact identities and never prepares models."""

import builtins
from collections import defaultdict, deque

import pytest

from tsfm_lab.models.catalog import CURATED_REPOSITORIES, catalog_entries
from tsfm_lab.models.settings import model_settings
from tsfm_lab.repl.model_selection import select_model
from tsfm_lab.repl.wizard import _Back, _Cancel
from tsfm_lab.tui.forms import default_fields


class Questions:
    def __init__(self, **answers):
        self.answers = defaultdict(deque)
        for label, values in answers.items():
            self.answers[label].extend(values if isinstance(values, list) else [values])
        self.calls = []

    def __call__(self, label, default="", choices=()):
        self.calls.append((label, default, choices))
        answer = self.answers[label].popleft() if self.answers[label] else ""
        if isinstance(answer, BaseException):
            raise answer
        return answer


def variants():
    return [
        (entry["adapter_id"], variant)
        for entry in catalog_entries()
        for variant in entry["variants"]
    ]


def fields(adapter, variant=None, **changes):
    result = default_fields(adapter)
    if variant is not None:
        result.update(
            repository=variant["repository"], revision=variant["reference_revision"]
        )
    return {**result, **changes}


@pytest.mark.parametrize("adapter,variant", variants())
def test_every_curated_variant_has_friendly_choices_and_exact_identity(
    adapter, variant
):
    settings = model_settings(adapter)
    prompt = {
        "size": "Model size",
        "training_profile": "Training profile",
        "context_horizon": "Context/horizon profile",
    }[settings.variant_kind]
    questions = Questions(**{prompt: variant["display_name"]})
    emitted = []
    source = fields(adapter)
    result = select_model(questions, emitted.append, source)
    assert result["adapter"] == adapter
    assert result["repository"] == variant["repository"]
    assert result["revision"] == variant["reference_revision"]
    assert result["checkpoint_variant"] == variant["variant"]
    assert result["source"] == "huggingface"
    assert source == fields(adapter)
    assert variant["verification_label"] in emitted
    assert not any(
        choice in CURATED_REPOSITORIES
        for _, _, choices in questions.calls
        for choice in choices
    )
    assert "Checkpoint repository" not in [item[0] for item in questions.calls]
    assert "Exact checkpoint SHA" not in [item[0] for item in questions.calls]
    assert not any(variant["repository"] in line for line in emitted)


@pytest.mark.parametrize(
    "name,adapter", [("Last value", "last_value"), ("Seasonal naive", "seasonal_naive")]
)
def test_baselines_remain_selectable_and_clear_stale_checkpoint_choices(name, adapter):
    questions = Questions(**{"Model family": "Baselines", "Baseline": name})
    result = select_model(
        questions, lambda _: None, fields("chronos_bolt"), checkpoint_variant="tiny"
    )
    assert result["adapter"] == adapter
    assert result["source"] == "builtin"
    assert result["revision"] == result["repository"] == ""
    assert "checkpoint_variant" not in result
    assert [item[0] for item in questions.calls] == ["Model family", "Baseline"]


def test_chronos_generation_change_resets_revision_and_stale_variant():
    original = fields("chronos_bolt", revision="a" * 40)
    questions = Questions(**{"Model generation": "2", "Model size": "Standard"})
    result = select_model(
        questions, lambda _: None, original, checkpoint_variant="tiny"
    )
    selected = next(
        item
        for adapter, item in variants()
        if adapter == "chronos_2" and item["variant"] == "standard"
    )
    assert result["adapter"] == "chronos_2"
    assert result["revision"] == selected["reference_revision"]
    assert result["checkpoint_variant"] == "standard"
    assert original["adapter"] == "chronos_bolt"


def test_custom_revision_survives_same_identity_without_inherited_verification():
    variant = next(item for adapter, item in variants() if adapter == "chronos_bolt")
    original = fields("chronos_bolt", variant, revision="a" * 40)
    emitted = []
    result = select_model(
        Questions(), emitted.append, original, checkpoint_variant="tiny"
    )
    assert result["revision"] == "a" * 40
    assert any("Custom checkpoint revision" in line for line in emitted)
    assert variant["verification_label"] not in emitted


def test_changing_profile_in_same_repository_resets_custom_revision():
    variants_ttm = [item for adapter, item in variants() if adapter == "ttm_r2"]
    original = fields("ttm_r2", variants_ttm[0], revision="a" * 40)
    result = select_model(
        Questions(**{"Context/horizon profile": variants_ttm[1]["display_name"]}),
        lambda _: None,
        original,
        checkpoint_variant=variants_ttm[0]["variant"],
    )
    assert result["revision"] == variants_ttm[1]["reference_revision"]
    assert result["repository"] == variants_ttm[0]["repository"]


def test_invalid_seed_profile_is_visible_until_explicitly_corrected():
    questions = Questions(
        **{"Context/horizon profile": ["", "R2 1024 context / 96 horizon"]}
    )
    emitted = []
    result = select_model(
        questions, emitted.append, fields("ttm_r2"), checkpoint_variant="invented"
    )
    calls = [item for item in questions.calls if item[0] == "Context/horizon profile"]
    assert calls[0][1] == "invented"
    assert len(calls) == 2
    assert any("needs correction" in line for line in emitted)
    assert result["checkpoint_variant"] == "1024-96-r2"


def test_unknown_seeded_adapter_is_correctable_through_family_choice():
    questions = Questions(**{"Model family": ["", "Baselines"]})
    emitted = []
    result = select_model(
        questions,
        emitted.append,
        {**fields("last_value"), "adapter": "invented_adapter"},
    )
    assert questions.calls[0][1] == "invented_adapter"
    assert result["adapter"] == "last_value"
    assert any("Choose Baselines" in line for line in emitted)


def test_advanced_view_exposes_exact_identity_and_validates_sha_and_cache():
    questions = Questions(
        **{
            "Advanced checkpoint options?": "yes",
            "Exact checkpoint SHA": ["main", "b" * 40],
            "Cache directory (optional; - uses the default)": [
                "https://bad.example/cache",
                "-",
            ],
        }
    )
    emitted = []
    result = select_model(questions, emitted.append, fields("chronos_bolt"))
    assert result["revision"] == "b" * 40
    assert result["cache_dir"] == ""
    assert any(line.startswith("Repository: amazon/") for line in emitted)
    assert any("40-character" in line for line in emitted)
    assert any("local cache directory" in line for line in emitted)


@pytest.mark.parametrize("correction", ["correct", "back", "custom"])
def test_advanced_sha_cannot_silently_change_known_native_profile(correction):
    profiles = [item for adapter, item in variants() if adapter == "ttm_r2"]
    first, other = profiles[:2]
    answer = {
        "correct": first["reference_revision"],
        "back": "/back",
        "custom": "c" * 40,
    }[correction]
    questions = Questions(
        **{
            "Advanced checkpoint options?": ["yes", "no"],
            "Exact checkpoint SHA": [other["reference_revision"], answer],
        }
    )
    emitted = []
    original = fields("ttm_r2", first)
    result = select_model(
        questions, emitted.append, original, checkpoint_variant=first["variant"]
    )
    assert any("disagrees with the selected profile" in line for line in emitted)
    assert result["checkpoint_variant"] == first["variant"]
    expected = "c" * 40 if correction == "custom" else first["reference_revision"]
    assert result["revision"] == expected
    if correction == "back":
        assert (
            sum(
                label == "Advanced checkpoint options?"
                for label, _, _ in questions.calls
            )
            == 2
        )
    if correction == "custom":
        assert any("Custom checkpoint revision" in line for line in emitted)
    assert original["revision"] == first["reference_revision"]


@pytest.mark.parametrize(
    "answer,exception",
    [
        ("/cancel", _Cancel),
        ("/back", _Back),
        (KeyboardInterrupt(), KeyboardInterrupt),
        (EOFError(), EOFError),
    ],
)
def test_cancel_and_back_at_family_leave_caller_fields_unchanged(answer, exception):
    original = fields("chronos_bolt")
    before = dict(original)
    with pytest.raises(exception):
        select_model(Questions(**{"Model family": answer}), lambda _: None, original)
    assert original == before


def test_back_from_single_generation_source_revisits_family():
    questions = Questions(
        **{
            "Model family": ["TimesFM", "Baselines"],
            "Checkpoint source": "/back",
            "Baseline": "Last value",
        }
    )
    result = select_model(questions, lambda _: None, fields("timesfm_2p5"))
    assert result["adapter"] == "last_value"
    assert sum(label == "Model family" for label, _, _ in questions.calls) == 2


def test_back_from_variant_revisits_source_and_can_choose_local_without_stale_profile():
    questions = Questions(
        **{
            "Advanced checkpoint options?": "/back",
            "Model size": ["Small", "/back"],
            "Checkpoint source": ["cached", "local"],
            "Local checkpoint directory": "an uninspected local path",
        }
    )
    result = select_model(questions, lambda _: None, fields("chronos_bolt"))
    assert result["source"] == "local"
    assert result["checkpoint_path"] == "an uninspected local path"
    assert result["revision"] == result["repository"] == ""
    assert "checkpoint_variant" not in result


def test_selection_imports_no_optional_runtime_or_hub(monkeypatch):
    original_import = builtins.__import__
    forbidden = {
        "torch",
        "chronos",
        "timesfm",
        "tirex",
        "tsfm_public",
        "transformers",
        "huggingface_hub",
    }

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in forbidden
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = select_model(
        Questions(**{"Model family": "IBM Granite TTM"}),
        lambda _: None,
        fields("last_value"),
    )
    assert result["adapter"] == "ttm_r2"

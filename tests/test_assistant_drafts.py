"""Provider suggestions remain bounded, path-free and locally resolved."""

import json

import pytest

from tsfm_lab.assistant.drafts import (
    ALLOWED_FIELDS,
    DraftError,
    decode_draft,
    draft_contract,
    resolve_draft,
)


def payload(**updates):
    result = {
        "schema_version": 1,
        "state": "ready",
        "intent": "historical_evaluation",
        "dataset_ref": "selected_dataset",
        "fields": {
            "usage_profile": "generic",
            "adapter": "last_value",
            "timestamp_column": "timestamp",
            "target_column": "demand",
            "frequency": "1h",
            "units": "synthetic_units",
            "context_length": "8",
            "horizon": "4",
            "stride": "4",
            "mase_period": "4",
            "evaluation_start": "2026-01-01T16:00:00Z",
            "evaluation_end": "2026-01-02T08:00:00Z",
            "point_statistic": "median",
        },
        "explicit_fields": [],
        "assumptions": [],
        "missing_fields": [],
    }
    result.update(updates)
    return result


def resolve(value, tmp_path):
    return resolve_draft(
        decode_draft(json.dumps(value)),
        dataset_refs={"selected_dataset": tmp_path / "seasonal.csv"},
    )


def test_complete_draft_uses_local_actuals_and_preserves_all_proposals(
    config_factory, tmp_path, monkeypatch
):
    config_factory()
    monkeypatch.setattr(
        "tsfm_lab.tui.forms.build_model",
        lambda *_args, **_kwargs: pytest.fail("Resolution must not initialize a model"),
    )
    result = resolve(payload(state="unsupported"), tmp_path)
    assert result.state == "ready"
    assert result.errors == result.missing_fields == ()
    assert result.seed_fields["dataset_path"] == str(tmp_path / "seasonal.csv")
    assert set(payload()["fields"]) <= set(result.explicit_fields)
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "field",
    [
        "dataset_path",
        "checkpoint_path",
        "revision",
        "source",
        "cache_dir",
        "output",
        "runtime",
        "code",
        "tool_calls",
    ],
)
def test_generated_paths_and_authority_fields_are_rejected(field):
    value = payload()
    value["fields"][field] = "/invented/location"
    with pytest.raises(DraftError, match="unknown or forbidden"):
        decode_draft(json.dumps(value))


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": True},
        {"state": "execute"},
        {"intent": "trading"},
        {"fields": {"horizon": 4}},
        {"fields": {"name": "unsafe\x1b[2J"}},
        {"fields": {"name": "unsafe\x9b2J"}},
        {"fields": {"name": "unsafe\ud800"}},
        {"explicit_fields": ["horizon", "horizon"]},
        {"explicit_fields": ["name"]},
        {"missing_fields": ["secret"]},
        {"assumptions": ["a"] * 41},
        {"dataset_ref": "a" * 101},
    ],
)
def test_invalid_structure_and_unbounded_text_reject(update):
    with pytest.raises(DraftError):
        decode_draft(json.dumps(payload(**update)))


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version":1,"schema_version":1}',
        "```json\n{}\n```",
        "{} trailing text",
        '{"schema_version":NaN}',
        " " * 32769,
    ],
    ids=("duplicate_keys", "markdown_fence", "trailing_text", "nonfinite", "oversized"),
)
def test_malformed_duplicate_or_oversized_json_rejects(text):
    with pytest.raises(DraftError):
        decode_draft(text)


def test_ready_and_claimed_missing_fields_do_not_establish_local_readiness(tmp_path):
    value = payload()
    for name in ("units", "evaluation_start", "evaluation_end"):
        value["fields"].pop(name)
    result = resolve(value, tmp_path)
    assert result.state == "needs_input"
    assert set(result.missing_fields) == {"units", "evaluation_start", "evaluation_end"}


def test_future_and_unclear_intent_never_become_ready(tmp_path):
    future = resolve(payload(intent="future_forecast"), tmp_path)
    assert future.state == "unsupported"
    assert "observed actuals" in future.errors[0]
    unclear = resolve(payload(intent="unclear"), tmp_path)
    assert unclear.state == "needs_input" and "intent" in unclear.missing_fields


def test_unknown_opaque_reference_cannot_become_a_path(tmp_path):
    result = resolve(payload(dataset_ref="/invented/data.csv"), tmp_path)
    assert "dataset_path" not in result.seed_fields
    assert "dataset_ref" in result.missing_fields
    assert result.errors


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"adapter": "invented"}, "Unsupported model"),
        (
            {"adapter": "tirex_2", "repository": "invented/model"},
            "curated repository",
        ),
        (
            {"adapter": "tirex_2", "repository": "NX-AI/TiRex-2", "horizon": "500"},
            "horizon is outside",
        ),
        (
            {
                "adapter": "ttm_r2",
                "repository": "ibm-granite/granite-timeseries-ttm-r2",
                "checkpoint_variant": "1024-96-r2",
                "context_length": "512",
                "point_statistic": "native_point",
            },
            "context_length is outside",
        ),
        (
            {
                "adapter": "ttm_r2",
                "repository": "ibm-granite/granite-timeseries-ttm-r2",
                "checkpoint_variant": "invented",
            },
            "curated checkpoint profile",
        ),
    ],
)
def test_unsupported_choices_survive_for_explicit_correction(tmp_path, updates, match):
    value = payload()
    value["fields"].update(updates)
    result = resolve(value, tmp_path)
    assert result.state == "needs_input"
    assert match in "\n".join(result.errors)
    assert updates.items() <= result.seed_fields.items()


def test_ttm_repository_requires_variant_and_derives_sha_locally(tmp_path):
    value = payload()
    value["fields"].update(
        adapter="ttm_r2", repository="ibm-granite/granite-timeseries-ttm-r2"
    )
    ambiguous = resolve(value, tmp_path)
    assert "checkpoint_variant" in ambiguous.missing_fields
    assert "revision" not in ambiguous.seed_fields
    value["fields"].update(
        checkpoint_variant="1024-96-r2",
        context_length="1024",
        point_statistic="native_point",
    )
    selected = resolve(value, tmp_path)
    assert (
        selected.seed_fields["revision"] == "1a382618f3c5b63836281ad632e54724e55e85a5"
    )
    assert selected.seed_fields["source"] == "huggingface"


def test_historical_dates_beyond_actuals_cannot_be_ready(config_factory, tmp_path):
    config_factory()
    value = payload()
    value["fields"]["evaluation_end"] = "2030-01-01T00:00:00Z"
    result = resolve(value, tmp_path)
    assert result.state == "needs_input"
    assert "dataset grid" in result.errors[0]


def test_contract_is_the_decoder_allowlist_and_returns_fresh_objects():
    first = draft_contract()
    assert set(first["properties"]["fields"]["properties"]) == ALLOWED_FIELDS
    first["properties"].clear()
    assert draft_contract()["properties"]

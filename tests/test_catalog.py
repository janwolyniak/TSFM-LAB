"""Model discovery is curated, stable, and independent of optional runtimes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tsfm_lab.models.catalog import (
    CHECKPOINT_REPOSITORIES,
    CURATED_REPOSITORIES,
    catalog_entries,
    catalog_info,
)


def test_catalog_metadata_is_json_compatible_and_has_cpu_only_verified_bounds():
    entries = catalog_entries()
    assert [entry["adapter_id"] for entry in entries] == [
        "last_value",
        "seasonal_naive",
        "chronos_bolt",
        "timesfm_2p5",
        "chronos_2",
    ]
    assert json.loads(json.dumps(entries, allow_nan=False)) == entries
    bolt = catalog_info("chronos_bolt")
    assert bolt["capabilities"]["devices"] == ["cpu"]
    assert bolt["dtypes"] == ["float32"]
    assert bolt["point_statistic"] == "median"
    assert bolt["capabilities"]["max_context"] == 2048
    assert bolt["capabilities"]["max_horizon"] == 64
    assert bolt["repositories"] == list(CHECKPOINT_REPOSITORIES["chronos_bolt"])
    timesfm = catalog_info("timesfm_2p5")
    assert timesfm["family"] == "timesfm"
    assert timesfm["capabilities"]["devices"] == ["cpu"]
    assert timesfm["capabilities"]["max_context"] == 16256
    assert timesfm["capabilities"]["max_horizon"] == 128
    assert timesfm["dtypes"] == ["float32"]
    assert timesfm["point_statistic"] == "median"
    assert timesfm["repositories"] == ["google/timesfm-2.5-200m-pytorch"]
    chronos2 = catalog_info("chronos_2")
    assert chronos2["family"] == "chronos"
    assert chronos2["capabilities"]["max_context"] == 8192
    assert chronos2["capabilities"]["max_horizon"] == 1024
    assert chronos2["point_statistic"] == "median"
    assert chronos2["repositories"] == [
        "autogluon/chronos-2-small",
        "amazon/chronos-2",
    ]


@pytest.mark.parametrize(
    "identifier",
    ["last_value", "seasonal_naive", *CHECKPOINT_REPOSITORIES, *CURATED_REPOSITORIES],
)
def test_catalog_supports_adapter_and_curated_repository_lookup(identifier):
    entry = catalog_info(identifier)
    if identifier in CURATED_REPOSITORIES:
        assert identifier in CHECKPOINT_REPOSITORIES[entry["adapter_id"]]
        assert entry["repository"] == identifier
    else:
        assert entry["adapter_id"] == identifier


def test_catalog_returns_independent_metadata_objects():
    altered = catalog_info("chronos_bolt")
    altered["capabilities"]["devices"].append("cuda")
    altered["repositories"].append("unknown/model")
    assert catalog_info("chronos_bolt")["capabilities"]["devices"] == ["cpu"]
    assert "unknown/model" not in catalog_info("chronos_bolt")["repositories"]


@pytest.mark.parametrize(
    "identifier", ["amazon/chronos-3", "unknown/model", "", "amazon/chronos-bolt-large"]
)
def test_catalog_rejects_unknown_models(identifier):
    with pytest.raises(ValueError, match="Unsupported model"):
        catalog_info(identifier)


def test_catalog_and_checkpoint_metadata_import_without_optional_runtimes():
    source = Path(__file__).resolve().parents[1] / "src"
    script = (
        "import sys; import tsfm_lab.models.catalog; import tsfm_lab.models.checkpoints; "
        "assert not ({'torch', 'chronos', 'timesfm', 'huggingface_hub', "
        "'safetensors', 'transformers'} & set(sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

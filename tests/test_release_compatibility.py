"""Frozen pre-profile bytes must remain readable by new code without migration."""

import json
import runpy
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "release_legacy_v1"


def test_pre_profile_records_keep_identity_scores_comparison_and_finance():
    result = runpy.run_path(str(FIXTURE / "verify.py"))["verify"]()
    assert result["runs"] == 2
    assert result["unchanged_artifacts"] == 17
    assert result["comparison_policy"] == "controlled_point_v1"


@pytest.mark.parametrize("filename", ["last_value.yaml", "comparison.json"])
def test_historical_probe_refuses_changed_fixture_bytes(tmp_path, filename):
    import shutil

    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURE, copied)
    path = copied / filename
    path.write_bytes(path.read_bytes() + b" ")
    identity = json.loads((copied / "identity.json").read_text())
    assert filename in identity["files"]
    verify = runpy.run_path(str(FIXTURE / "verify.py"))["verify"]
    with pytest.raises(AssertionError):
        verify(copied)

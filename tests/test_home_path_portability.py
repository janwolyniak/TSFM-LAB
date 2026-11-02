"""A Windows named-user expansion must not invent a writable home."""

import os
from pathlib import Path

import pytest

from tsfm_lab.local_paths import expand_local_user


def _windows_expansion(monkeypatch, tmp_path):
    def expand(path):
        text = str(path)
        if text.startswith("~unknown"):
            return tmp_path / "unknown" / text[len("~unknown") :].lstrip("/\\")
        if text.startswith("~known"):
            return tmp_path / "known" / text[len("~known") :].lstrip("/\\")
        if text.startswith("~"):
            return tmp_path / "self" / text[1:].lstrip("/\\")
        return path

    monkeypatch.setattr(Path, "expanduser", expand)


def test_unknown_windows_named_home_is_rejected_without_creating_it(
    monkeypatch, tmp_path
):
    _windows_expansion(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="named user's home"):
        expand_local_user("~unknown/results", windows=True)
    assert not (tmp_path / "unknown").exists()


def test_existing_windows_named_home_allows_a_new_output_directory(
    monkeypatch, tmp_path
):
    _windows_expansion(monkeypatch, tmp_path)
    (tmp_path / "known").mkdir()
    result = expand_local_user("~known/results", windows=True)
    assert result == tmp_path / "known/results"
    assert not result.exists()


def test_current_windows_home_expansion_does_not_require_output_to_exist(
    monkeypatch, tmp_path
):
    _windows_expansion(monkeypatch, tmp_path)
    assert expand_local_user("~/results", windows=True) == tmp_path / "self/results"


@pytest.mark.skipif(os.name == "nt", reason="POSIX native home lookup")
def test_posix_unknown_named_home_keeps_native_rejection():
    with pytest.raises((RuntimeError, ValueError)):
        expand_local_user("~tsfmlab-no-such-user-928374/results", windows=False)

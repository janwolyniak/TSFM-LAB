"""Export durability uses writable handles and retains atomic no-overwrite IO."""

import hashlib
import os

import pytest

from tsfm_lab.application import run_from_config
from tsfm_lab.exports import export_predictions


@pytest.mark.parametrize("format", ["csv", "json", "parquet"])
def test_export_syncs_writable_handle_before_publication(
    config_factory, tmp_path, monkeypatch, format
):
    if format == "parquet":
        pytest.importorskip("pyarrow")
    run = run_from_config(config_factory())
    target = tmp_path / "exports" / f"forecasts.{format}"
    original_fsync = os.fsync
    synced = []

    def fsync(descriptor):
        # This fails for a read-only descriptor even on POSIX, without modifying
        # the file. Windows rejects those descriptors when committing to disk.
        assert os.write(descriptor, b"") == 0
        assert not target.exists()
        original_fsync(descriptor)
        synced.append(True)

    monkeypatch.setattr("tsfm_lab.exports.os.fsync", fsync)
    exported = export_predictions(run["run_id"], tmp_path / "workspace", target)
    assert synced == [True]
    assert exported["rows"] == 16
    assert exported["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not list(target.parent.glob(".tsfm-export-*"))


@pytest.mark.parametrize("format", ["csv", "json", "parquet"])
def test_concurrent_existing_destination_is_preserved(
    config_factory, tmp_path, monkeypatch, format
):
    if format == "parquet":
        pytest.importorskip("pyarrow")
    run = run_from_config(config_factory())
    target = tmp_path / "exports" / f"forecasts.{format}"
    original_link = os.link
    other_output = b"another process published this file"

    def publish(source, destination):
        target.write_bytes(other_output)
        original_link(source, destination)

    monkeypatch.setattr("tsfm_lab.exports.os.link", publish)
    with pytest.raises(FileExistsError):
        export_predictions(run["run_id"], tmp_path / "workspace", target)
    assert target.read_bytes() == other_output
    assert not list(target.parent.glob(".tsfm-export-*"))


def test_durability_failure_does_not_publish_or_leave_temporary_files(
    config_factory, tmp_path, monkeypatch
):
    run = run_from_config(config_factory())
    target = tmp_path / "exports" / "forecasts.json"

    def fsync(descriptor):
        raise OSError("injected durability failure")

    monkeypatch.setattr("tsfm_lab.exports.os.fsync", fsync)
    with pytest.raises(OSError, match="injected durability failure"):
        export_predictions(run["run_id"], tmp_path / "workspace", target)
    assert not target.exists()
    assert not list(target.parent.glob(".tsfm-export-*"))

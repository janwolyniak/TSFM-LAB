"""Onboarding diagnostics stay offline and distinguish metadata compatibility."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tsfm_lab import runtime_diagnostics as diagnostics
from tsfm_lab.models.registry import ADAPTER_PROFILES

VERSIONS = {
    "chronos-forecasting": "2.2.0",
    "torch": "2.8.0+cpu",
    "transformers": "4.56.0",
    "huggingface-hub": "0.35.3",
    "safetensors": "0.6.2",
    "accelerate": "1.12.0",
    "timesfm": "3.0.2",
    "tirex-2": "0.4.0",
    "numpy": "2.1.3",
    "flashrnn": "1.0.8",
    "mlstm-kernels": "2.0.6",
    "ninja": "1.13.2",
    "narwhals": "2.13.0",
    "granite-tsfm": "0.3.4",
    "tokenizers": "0.22.0",
    "scikit-learn": "1.7.2",
    "filelock": "3.20.3",
}


def result_for(adapter):
    return next(
        entry
        for entry in diagnostics.runtime_diagnostics()
        if entry["adapter_id"] == adapter
    )


def test_supported_catalog_versions_are_metadata_checks_only(monkeypatch):
    calls = []

    def version(name):
        calls.append(name)
        return VERSIONS[name]

    monkeypatch.setattr(diagnostics, "version", version)
    entries = diagnostics.runtime_diagnostics()
    assert len(entries) == 7
    assert all(entry["status"] == "available" for entry in entries)
    assert len(calls) == len(set(calls))
    assert {entry["adapter_id"] for entry in entries[2:]} == {
        profile.adapter_id for profile in ADAPTER_PROFILES
    }
    assert all(
        "platform support unverified" in entry["detail"] for entry in entries[2:]
    )


def test_missing_optionals_leave_baselines_available(monkeypatch):
    def absent(name):
        raise diagnostics.PackageNotFoundError(name)

    monkeypatch.setattr(diagnostics, "version", absent)
    entries = diagnostics.runtime_diagnostics()
    assert all(entry["status"] == "available" for entry in entries[:2])
    assert all(entry["status"] == "missing" for entry in entries[2:])
    assert "torch>=2.8,<2.9" in entries[2]["detail"]


@pytest.mark.parametrize(
    "adapter,package,version",
    [
        ("chronos_bolt", "chronos-forecasting", "2.3.0"),
        ("chronos_2", "accelerate", "1.13.0"),
        ("timesfm_2p5", "timesfm", "3.0.1"),
        ("tirex_2", "numpy", "2.1.2"),
        ("tirex_2", "numpy", "3.0.0"),
        ("tirex_2", "numpy", "2.1.3rc1"),
        ("ttm_r2", "transformers", "4.56.1"),
        ("ttm_r2", "torch", "2.8.0rc1"),
        ("ttm_r2", "torch", "broken"),
    ],
)
def test_unqualified_runtime_identifies_package_and_requirement(
    monkeypatch, adapter, package, version
):
    monkeypatch.setattr(
        diagnostics, "version", {**VERSIONS, package: version}.__getitem__
    )
    entry = result_for(adapter)
    assert entry["status"] == "incompatible"
    assert f"{package}: found {version}, requires" in entry["detail"]


def test_missing_and_incompatible_packages_are_both_reported(monkeypatch):
    def version(name):
        if name == "timesfm":
            raise diagnostics.PackageNotFoundError(name)
        return "2.9.0" if name == "torch" else VERSIONS[name]

    monkeypatch.setattr(diagnostics, "version", version)
    entry = result_for("timesfm_2p5")
    assert entry["status"] == "missing"
    assert "timesfm==3.0.2" in entry["detail"]
    assert "torch: found 2.9.0" in entry["detail"]


def test_unreadable_metadata_has_actionable_diagnostic_without_raw_error(monkeypatch):
    def version(name):
        if name == "timesfm":
            raise OSError("private filesystem detail")
        return VERSIONS[name]

    monkeypatch.setattr(diagnostics, "version", version)
    entry = result_for("timesfm_2p5")
    assert entry["status"] == "incompatible"
    assert "timesfm: cannot read package metadata" in entry["detail"]
    assert "private filesystem detail" not in entry["detail"]


def test_diagnostics_do_not_import_optional_runtimes_or_access_network():
    code = """
import importlib.abc
import socket
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {
            'torch', 'chronos', 'transformers', 'timesfm', 'tirex2',
            'tsfm_public', 'huggingface_hub', 'safetensors', 'packaging'
        }:
            raise AssertionError('Unexpected runtime import: ' + fullname)
def forbidden(*args, **kwargs):
    raise AssertionError('Unexpected network request')
sys.meta_path.insert(0, Block())
socket.create_connection = socket.socket = forbidden
from tsfm_lab.runtime_diagnostics import runtime_diagnostics
assert len(runtime_diagnostics()) == 7
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

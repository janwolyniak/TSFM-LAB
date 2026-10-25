"""Parquet input tests, optional when the file-format dependency is absent."""

from __future__ import annotations

import builtins
import hashlib
import importlib
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.contracts import EvaluationSpec
from tsfm_lab.data import canonical_fingerprint, load_csv
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.models.baselines import LastValue
from tsfm_lab.parquet import load_parquet


@pytest.fixture
def arrow():
    return pytest.importorskip("pyarrow")


@pytest.fixture
def parquet(arrow):
    return importlib.import_module("pyarrow.parquet")


def table(arrow, *, timestamps=None, values=None):
    return arrow.table(
        {
            "timestamp": (
                pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC")
                if timestamps is None
                else timestamps
            ),
            "target": list(range(1, 9)) if values is None else values,
        }
    )


def write(tmp_path, data, parquet, name="data.parquet"):
    path = tmp_path / name
    parquet.write_table(data, path)
    return path


def load(path, **kwargs):
    return load_parquet(
        path,
        timestamp_column="timestamp",
        target_column="target",
        frequency=kwargs.pop("frequency", "1h"),
        **kwargs,
    )


def test_roundtrip_matches_csv_forecasts_and_metrics(tmp_path, arrow, parquet):
    path = write(tmp_path, table(arrow), parquet)
    parquet_data = load(path, units="kWh", series_id="meter-1")
    csv_path = tmp_path / "data.csv"
    pd.DataFrame(
        {"timestamp": parquet_data.timestamps, "target": parquet_data.values}
    ).to_csv(csv_path, index=False)
    csv_data = load_csv(
        csv_path,
        timestamp_column="timestamp",
        target_column="target",
        frequency="1h",
        units="kWh",
        series_id="meter-1",
    )
    np.testing.assert_array_equal(parquet_data.values, csv_data.values)
    assert parquet_data.timestamps.equals(csv_data.timestamps)
    spec = EvaluationSpec(
        context_length=3,
        horizon=2,
        evaluation_start=parquet_data.timestamps[4],
        evaluation_end=parquet_data.timestamps[-1] + pd.Timedelta(hours=1),
    )
    first = run_experiment(parquet_data, LastValue(), spec)
    second = run_experiment(csv_data, LastValue(), spec)
    assert first.scores == second.scores
    assert first.scores.overall["mae"].value == 1.5
    assert (
        parquet_data.provenance["source_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert parquet_data.provenance["parser_policy"] == "parquet_strict_v1"


def test_typed_nanoseconds_are_preserved(tmp_path, arrow, parquet):
    timestamps = pd.date_range("2026-01-01T00:00:00.000000001Z", periods=3, freq="1ns")
    path = write(
        tmp_path, table(arrow, timestamps=timestamps, values=[1.0, 2.0, 3.0]), parquet
    )
    dataset = load(path, frequency="1ns")
    assert dataset.timestamps.equals(timestamps)
    assert dataset.timestamps[0].nanosecond == 1


def test_source_bytes_are_read_once(tmp_path, arrow, parquet, monkeypatch):
    path = write(tmp_path, table(arrow), parquet)
    original = Path.read_bytes
    reads = []

    def read_bytes(self):
        reads.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    load(path)
    assert reads == [path]


def test_naive_datetimes_require_explicit_timezone(tmp_path, arrow, parquet):
    timestamps = pd.date_range("2026-01-01T01:00", periods=8, freq="1h")
    path = write(tmp_path, table(arrow, timestamps=timestamps), parquet)
    with pytest.raises(ValueError, match="require a timezone"):
        load(path)
    dataset = load(path, timezone="Europe/Warsaw")
    assert dataset.timestamps[0] == pd.Timestamp("2026-01-01T00:00Z")


@pytest.mark.parametrize("timestamp", ["2026-10-25T02:30", "2026-03-29T02:30"])
def test_dst_ambiguity_and_nonexistent_times_fail(tmp_path, arrow, parquet, timestamp):
    data = table(arrow, timestamps=[pd.Timestamp(timestamp)], values=[1])
    path = write(tmp_path, data, parquet)
    with pytest.raises(ValueError, match="Cannot localize"):
        load(path, timezone="Europe/Warsaw")


def test_iso_strings_accept_explicit_dst_offsets(tmp_path, arrow, parquet):
    data = table(
        arrow,
        timestamps=["2026-10-25T02:00:00+02:00", "2026-10-25T02:00:00+01:00"],
        values=[1, 2],
    )
    dataset = load(write(tmp_path, data, parquet))
    assert dataset.timestamps[1] - dataset.timestamps[0] == pd.Timedelta(hours=1)


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2026-01-01T00:00Z", "2026-01-01T00:00Z"],
        ["2026-01-01T01:00Z", "2026-01-01T00:00Z"],
        ["2026-01-01T00:00Z", "2026-01-01T02:00Z"],
        ["2026-01-01T00:00Z", "2026-01-01T01:00"],
        ["2026-01-01T00:00Z", "now"],
        ["2026-01-01T00:00Z", None],
    ],
)
def test_invalid_string_timestamps_fail(tmp_path, arrow, parquet, timestamps):
    data = table(arrow, timestamps=timestamps, values=[1, 2])
    with pytest.raises(ValueError):
        load(write(tmp_path, data, parquet))


@pytest.mark.parametrize(
    "values",
    [
        [True] * 8,
        ["1"] * 8,
        [Decimal("1.0")] * 8,
        [[1]] * 8,
        [None] * 8,
        [1, 2, 3, 4, 5, 6, 7, None],
        [1.0, 2, 3, 4, 5, 6, 7, float("nan")],
        [1.0, 2, 3, 4, 5, 6, 7, float("inf")],
    ],
)
def test_invalid_target_schemas_or_values_fail(tmp_path, arrow, parquet, values):
    with pytest.raises(ValueError, match="target column"):
        load(write(tmp_path, table(arrow, values=values), parquet))


def test_duplicate_fields_are_rejected(tmp_path, arrow, parquet):
    data = arrow.Table.from_arrays(
        [arrow.array([1]), arrow.array([2])], names=["timestamp", "timestamp"]
    )
    with pytest.raises(ValueError, match="duplicate field names"):
        load(write(tmp_path, data, parquet))


def test_empty_and_missing_column_files_are_rejected(tmp_path, arrow, parquet):
    empty = arrow.table(
        {"timestamp": arrow.array([], type=arrow.timestamp("ns")), "target": []}
    )
    with pytest.raises(ValueError, match="at least one observation"):
        load(write(tmp_path, empty, parquet))
    with pytest.raises(ValueError, match="missing selected column"):
        load(write(tmp_path, arrow.table({"other": [1]}), parquet))


def test_invalid_timestamp_type_is_rejected(tmp_path, arrow, parquet):
    with pytest.raises(ValueError, match="datetimes or ISO strings"):
        load(write(tmp_path, table(arrow, timestamps=list(range(8))), parquet))


def test_fingerprint_tracks_content_and_semantics_not_path(tmp_path, arrow, parquet):
    first_path = write(tmp_path, table(arrow), parquet)
    second_path = tmp_path / "copy.parquet"
    second_path.write_bytes(first_path.read_bytes())
    first = load(first_path)
    second = load(second_path, frequency="60min")
    assert canonical_fingerprint(first) == canonical_fingerprint(second)
    assert canonical_fingerprint(first) != canonical_fingerprint(
        load(first_path, units="W")
    )
    assert canonical_fingerprint(first) != canonical_fingerprint(
        load(first_path, series_id="other")
    )
    modified = table(arrow, values=list(range(2, 10)))
    parquet.write_table(modified, second_path)
    assert canonical_fingerprint(first) != canonical_fingerprint(load(second_path))


def test_missing_optional_dependency_has_actionable_error(tmp_path, monkeypatch):
    original = importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == "pyarrow":
            raise ModuleNotFoundError("No module named pyarrow")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_module)
    with pytest.raises(ImportError, match=r"tsfm-lab\[parquet\]"):
        load(tmp_path / "not-read.parquet")


def test_module_import_does_not_require_pyarrow(monkeypatch):
    original = builtins.__import__

    def import_without_arrow(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise AssertionError("Parquet module imported an optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_arrow)
    importlib.reload(importlib.import_module("tsfm_lab.parquet"))


@pytest.mark.parametrize("content", [b"", b"not a parquet file", b"PAR1brokenPAR1"])
def test_bad_file_fails_clearly(tmp_path, arrow, content):
    path = tmp_path / "bad.parquet"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="valid Parquet file"):
        load(path)


@pytest.mark.parametrize("frequency", ["0h", "-1h", "ME", None])
def test_invalid_frequency_fails_before_file_read(tmp_path, frequency):
    with pytest.raises(ValueError, match="positive fixed duration"):
        load(tmp_path / "not-read.parquet", frequency=frequency)

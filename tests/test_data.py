"""CSV boundaries and content identity tests with disposable local fixtures."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.data import canonical_fingerprint, load_csv


def write_csv(tmp_path, content, name="data.csv"):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def load(path, **kwargs):
    return load_csv(
        path,
        timestamp_column="timestamp",
        target_column="target",
        frequency=kwargs.pop("frequency", "1h"),
        **kwargs,
    )


def test_csv_maps_columns_normalizes_utc_and_preserves_units(tmp_path):
    path = write_csv(
        tmp_path,
        "unused,target,timestamp\na,1.5,2026-01-01T01:00:00+01:00\n"
        "b,-2,2026-01-01T02:00:00+01:00\n",
    )
    dataset = load(path, units="kWh", series_id="meter-1")
    np.testing.assert_array_equal(dataset.values, [1.5, -2])
    assert dataset.values.dtype == np.dtype("float64")
    assert dataset.timestamps.equals(
        pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    )
    assert dataset.units == "kWh"
    assert dataset.series_id == "meter-1"
    assert dataset.target_name == "target"
    assert (
        dataset.provenance["source_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert dataset.provenance["canonical_sha256"] == canonical_fingerprint(dataset)
    assert not dataset.values.flags.writeable


def test_naive_timestamps_need_declared_timezone(tmp_path):
    path = write_csv(tmp_path, "timestamp,target\n2026-01-01 01:00,1\n")
    with pytest.raises(ValueError, match="require a timezone"):
        load(path)
    dataset = load(path, timezone="Europe/Warsaw")
    assert dataset.timestamps[0] == pd.Timestamp("2026-01-01T00:00Z")


def test_mixed_iso_date_and_time_precision_is_supported(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,target\n2026-01-01,1\n2026-01-01T01:00:00.000000000,2\n",
    )
    dataset = load(path, timezone="UTC")
    assert dataset.timestamps.equals(
        pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    )


@pytest.mark.parametrize(
    "timestamp",
    ["2026-10-25T02:30:00", "2026-03-29T02:30:00"],
)
def test_dst_ambiguity_and_nonexistent_local_times_fail(tmp_path, timestamp):
    path = write_csv(tmp_path, f"timestamp,target\n{timestamp},1\n")
    with pytest.raises(ValueError, match="Cannot localize"):
        load(path, timezone="Europe/Warsaw")


def test_explicit_offsets_across_dst_are_regular_in_utc(tmp_path):
    path = write_csv(
        tmp_path,
        "timestamp,target\n2026-10-25T02:00:00+02:00,1\n"
        "2026-10-25T02:00:00+01:00,2\n",
    )
    dataset = load(path)
    assert dataset.timestamps[1] - dataset.timestamps[0] == pd.Timedelta(hours=1)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("timestamp,target,target\n2026-01-01Z,1,2\n", "duplicate column"),
        ("timestamp,\n2026-01-01,1\n", "nonempty"),
        ("timestamp,other\n2026-01-01,1\n", "missing selected column"),
        ("timestamp,target\n", "at least one observation"),
        ("", "UTF-8 header"),
        ("timestamp,target\n2026-01-01,1,extra\n", "fields"),
        ("timestamp,target\n2026-01-01\n", "fields"),
        ("timestamp,target\n,1\n", "Invalid timestamp"),
        ("timestamp,target\nnot-a-date,1\n", "Invalid timestamp"),
        ("timestamp,target\nnow,1\n", "Invalid timestamp"),
        ("timestamp,target\ntoday,1\n", "Invalid timestamp"),
        ("timestamp,target\n01/02/2026,1\n", "Invalid timestamp"),
        ("timestamp,target\nNaT,1\n", "Invalid timestamp"),
        ("timestamp,target\n2026-01-01T00:00Z,word\n", "CSV row 2"),
        ("timestamp,target\n2026-01-01T00:00Z,\n", "finite numeric"),
        ("timestamp,target\n2026-01-01T00:00Z,nan\n", "finite numeric"),
        ("timestamp,target\n2026-01-01T00:00Z,inf\n", "finite numeric"),
        ("timestamp,target\n2026-01-01T00:00Z,-inf\n", "finite numeric"),
        (
            "timestamp,target\n2026-01-01T00:00Z,1\n2026-01-01T01:00,2\n",
            "mixes timezone-aware and naive",
        ),
    ],
)
def test_invalid_csv_fails_with_context(tmp_path, content, message):
    with pytest.raises(ValueError, match=message):
        load(write_csv(tmp_path, content))


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2026-01-01T00:00Z", "2026-01-01T00:00Z"],
        ["2026-01-01T01:00Z", "2026-01-01T00:00Z"],
        ["2026-01-01T00:00Z", "2026-01-01T02:00Z"],
        ["2026-01-01T00:00Z", "2026-01-01T00:30Z"],
    ],
)
def test_duplicate_unordered_or_irregular_grid_fails(tmp_path, timestamps):
    text = "timestamp,target\n" + "".join(
        f"{t},{i}\n" for i, t in enumerate(timestamps)
    )
    with pytest.raises(ValueError, match="column|Column"):
        load(write_csv(tmp_path, text))


@pytest.mark.parametrize("frequency", ["0h", "-1h", "ME", "invalid-frequency", None, 1])
def test_invalid_frequency_is_rejected(tmp_path, frequency):
    path = write_csv(tmp_path, "timestamp,target\n2026-01-01T00:00Z,1\n")
    with pytest.raises(ValueError, match="positive fixed duration"):
        load(path, frequency=frequency)


def test_fingerprint_ignores_paths_and_csv_number_format(tmp_path):
    first = write_csv(
        tmp_path,
        "timestamp,target\n2026-01-01T00:00Z,1\n2026-01-01T01:00Z,2\n",
    )
    second = write_csv(
        tmp_path,
        "timestamp,target\n2026-01-01T00:00:00+00:00,1.0\n"
        "2026-01-01T01:00:00+00:00,2.000\n",
        name="copy.csv",
    )
    original, equivalent = load(first), load(second, frequency="60min")
    assert (
        original.provenance["source_sha256"] != equivalent.provenance["source_sha256"]
    )
    assert canonical_fingerprint(original) == canonical_fingerprint(equivalent)
    assert canonical_fingerprint(original) != canonical_fingerprint(
        load(first, units="MW")
    )
    assert canonical_fingerprint(original) != canonical_fingerprint(
        load(first, series_id="different")
    )
    second.write_text(second.read_text().replace("2.000", "3.000"))
    assert canonical_fingerprint(original) != canonical_fingerprint(load(second))


def test_utf8_bom_and_quoted_headers_are_supported(tmp_path):
    path = write_csv(tmp_path, '\ufeff"timestamp","target"\n2026-01-01T00:00Z,1\n')
    assert load(path).values.tolist() == [1.0]


def test_non_utf8_input_is_rejected(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_bytes(b"timestamp,target\n\xff,1\n")
    with pytest.raises(ValueError, match="UTF-8"):
        load(path)


def test_same_timestamp_and_target_column_is_rejected(tmp_path):
    path = write_csv(tmp_path, "timestamp,target\n2026-01-01T00:00Z,1\n")
    with pytest.raises(ValueError, match="must differ"):
        load_csv(
            path, timestamp_column="target", target_column="target", frequency="1h"
        )

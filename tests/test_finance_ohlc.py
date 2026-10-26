"""Independent OHLC ingestion, chronology, and source-alignment checks."""

import csv
import hashlib
import io
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.contracts import Dataset
from tsfm_lab.finance.contracts import FinanceError
from tsfm_lab.finance.ohlc import (
    OHLCData,
    load_ohlc_csv,
    ohlc_fingerprint,
    validate_price_alignment,
)

_COLUMNS = ["open_timestamp", "close_timestamp", "open", "high", "low", "close"]


def bars(**changes):
    closes = pd.date_range("2026-01-01T01:00Z", periods=3, freq="h")
    fields = {
        "open_timestamps": closes - pd.Timedelta(minutes=59),
        "close_timestamps": closes,
        "opens": [100, 110, 90],
        "highs": [106, 115, 101],
        "lows": [98, 102, 88],
        "closes": [105, 104, 99],
        "frequency": "1h",
        "units": "USD",
        "series_id": "asset",
    }
    return OHLCData(**(fields | changes))


def csv_file(tmp_path, *, name="ohlc.csv", transform=lambda rows: rows):
    data = bars()
    values = zip(
        data.open_timestamps,
        data.close_timestamps,
        data.opens,
        data.highs,
        data.lows,
        data.closes,
        strict=True,
    )
    rows = [[str(value) for value in row] for row in values]
    output = io.StringIO(newline="")
    csv.writer(output).writerows(transform([_COLUMNS, *rows]))
    path = tmp_path / name
    path.write_text(output.getvalue())
    return path


def prices(data, **changes):
    fields = {
        "timestamps": data.close_timestamps,
        "values": data.closes,
        "frequency": data.frequency,
        "units": data.units,
        "series_id": data.series_id,
        "target_name": "source_close_price",
    }
    return Dataset(**(fields | changes))


def test_ohlc_snapshots_are_frozen_readonly_and_do_not_alias_inputs():
    original = bars()
    opens = original.opens.copy()
    times = original.open_timestamps.copy(deep=True)
    provenance = {"note": {"label": "before"}}
    snapshot = replace(
        original, opens=opens, open_timestamps=times, provenance=provenance
    )
    opens[0] = 1
    times.values[0] += np.timedelta64(1, "s")
    provenance["note"]["label"] = "after"
    assert snapshot.opens[0] == 100
    assert snapshot.open_timestamps[0] == original.open_timestamps[0]
    assert snapshot.provenance["note"]["label"] == "before"
    for name in ("opens", "highs", "lows", "closes"):
        array = getattr(snapshot, name)
        assert array.dtype == np.dtype("float64") and not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 1
    with pytest.raises(FrozenInstanceError):
        snapshot.frequency = "2h"


@pytest.mark.parametrize("field", ["opens", "highs", "lows", "closes"])
@pytest.mark.parametrize(
    "value",
    [
        [True, 110, 90],
        ["100", "110", "90"],
        [100, np.nan, 90],
        [100, np.inf, 90],
        [100, 0, 90],
        [100, -1, 90],
        [100 + 1j, 110, 90],
        [[100, 110, 90]],
        [],
        [None, 110, 90],
    ],
)
def test_prices_reject_coercion_nonfinite_nonpositive_and_wrong_shapes(field, value):
    with pytest.raises(FinanceError):
        bars(**{field: value})


def test_masked_prices_are_not_silently_unwrapped():
    with pytest.raises(FinanceError, match="masked"):
        bars(opens=np.ma.array([100, 110, 90], mask=[True, False, False]))


@pytest.mark.parametrize(
    "changes",
    [
        {"opens": [100]},
        {"highs": [104, 115, 101]},
        {"lows": [101, 102, 88]},
        {"units": ""},
        {"series_id": None},
        {"frequency": "ME"},
        {"provenance": []},
    ],
)
def test_price_bounds_lengths_and_metadata_are_strict(changes):
    with pytest.raises(FinanceError):
        bars(**changes)


def test_high_bound_and_price_alignment_use_exact_float64_values():
    with pytest.raises(FinanceError, match="highs"):
        bars(highs=[np.nextafter(105.0, 0), 115, 101])
    data = bars()
    changed = data.closes.copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    with pytest.raises(FinanceError, match="Price values"):
        validate_price_alignment(prices(data, values=changed), data)


@pytest.mark.parametrize("field", ["open_timestamps", "close_timestamps"])
@pytest.mark.parametrize(
    "kind", ["naive", "duplicate", "descending", "missing", "empty", "short"]
)
def test_timestamps_require_aware_ordered_unique_complete_indices(field, kind):
    values = getattr(bars(), field)
    replacements = {
        "naive": values.tz_localize(None),
        "duplicate": values.take([0, 0, 2]),
        "descending": values[::-1],
        "missing": pd.DatetimeIndex([values[0], pd.NaT, values[2]]),
        "empty": values[:0],
        "short": values[:2],
    }
    with pytest.raises(FinanceError):
        bars(**{field: replacements[kind]})


def test_bar_order_regular_closes_and_nonoverlap_are_independent_requirements():
    original = bars()
    with pytest.raises(FinanceError, match="strictly before"):
        bars(open_timestamps=original.close_timestamps)
    with pytest.raises(FinanceError, match="overlap"):
        bars(open_timestamps=original.close_timestamps - pd.Timedelta(minutes=61))
    with pytest.raises(FinanceError, match="regular frequency"):
        bars(
            close_timestamps=original.close_timestamps
            + pd.to_timedelta([0, 1, 0], unit="ns")
        )
    touching = bars(open_timestamps=original.close_timestamps - pd.Timedelta(hours=1))
    assert touching.open_timestamps[1] == touching.close_timestamps[0]


def test_nanosecond_timing_is_preserved_in_data_and_identity():
    closes = pd.date_range("2026-01-01", periods=3, freq="ns", tz="UTC")
    data = bars(
        close_timestamps=closes,
        open_timestamps=closes - pd.Timedelta(1, "ns"),
        frequency="1ns",
    )
    assert data.close_timestamps[1] - data.open_timestamps[1] == pd.Timedelta(1, "ns")
    shifted = replace(
        data,
        open_timestamps=data.open_timestamps + pd.Timedelta(1, "ns"),
        close_timestamps=data.close_timestamps + pd.Timedelta(1, "ns"),
    )
    assert ohlc_fingerprint(data) != ohlc_fingerprint(shifted)


def test_csv_maps_fixed_columns_and_records_byte_and_canonical_identities(tmp_path):
    path = csv_file(tmp_path)
    data = load_ohlc_csv(path, frequency="1h", units="USD", series_id="asset")
    np.testing.assert_array_equal(data.closes, [105, 104, 99])
    assert str(data.open_timestamps.tz) == str(data.close_timestamps.tz) == "UTC"
    assert (
        data.provenance["source_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert data.provenance["source_path"] == str(path.resolve())
    assert data.provenance["open_source_timezone"] == "embedded"
    assert data.provenance["close_source_timezone"] == "embedded"
    assert data.provenance["canonical_sha256"] == ohlc_fingerprint(data)
    validate_price_alignment(prices(data), data)


def test_equivalent_bytes_paths_headers_and_frequency_labels_keep_canonical_identity(
    tmp_path,
):
    first = csv_file(tmp_path, name="first.csv")

    def reordered(rows):
        return [[row[index] for index in [5, 4, 3, 2, 1, 0]] for row in rows]

    second = csv_file(tmp_path, name="second.csv", transform=reordered)
    a = load_ohlc_csv(first, frequency="1h")
    b = load_ohlc_csv(second, frequency="60min")
    assert a.provenance["source_sha256"] != b.provenance["source_sha256"]
    assert ohlc_fingerprint(a) == ohlc_fingerprint(b)
    assert ohlc_fingerprint(
        replace(a, provenance=a.provenance | {"parser_policy": "other"})
    ) != ohlc_fingerprint(a)
    assert ohlc_fingerprint(replace(a, units="other")) != ohlc_fingerprint(a)


@pytest.mark.parametrize("timestamp", ["2026-10-25T02:30:00", "2026-03-29T02:30:00"])
def test_csv_rejects_ambiguous_or_nonexistent_naive_dst_times(tmp_path, timestamp):
    path = tmp_path / "dst.csv"
    closing = timestamp[:10] + "T04:00:00"
    path.write_text(",".join(_COLUMNS) + f"\n{timestamp},{closing},100,110,90,105\n")
    with pytest.raises(FinanceError, match="Cannot localize"):
        load_ohlc_csv(path, frequency="1h", timezone="Europe/Warsaw")


def test_naive_csv_requires_timezone_and_normalizes_it_explicitly(tmp_path):
    path = csv_file(
        tmp_path,
        transform=lambda rows: [
            [value.replace("+00:00", "") for value in row] for row in rows
        ],
    )
    with pytest.raises(FinanceError, match="require a timezone"):
        load_ohlc_csv(path, frequency="1h")
    local = load_ohlc_csv(path, frequency="1h", timezone="Europe/Warsaw")
    assert local.close_timestamps[0] == pd.Timestamp("2026-01-01T00:00Z")
    assert local.provenance["open_source_timezone"] == "Europe/Warsaw"


@pytest.mark.parametrize(
    "transform,message",
    [
        (lambda rows: [rows[0] + ["close"], *rows[1:]], "duplicate column"),
        (lambda rows: [rows[0], rows[1][:-1]], "fields"),
        (lambda rows: [rows[0][:-1], *[row[:-1] for row in rows[1:]]], "exactly"),
        (
            lambda rows: [rows[0] + ["volume"], *[row + ["1"] for row in rows[1:]]],
            "exactly",
        ),
        (lambda rows: [rows[0], ["today", *rows[1][1:]]], "Invalid timestamp"),
        (lambda rows: [rows[0], [*rows[1][:2], "nan", *rows[1][3:]]], "finite numeric"),
        (
            lambda rows: [rows[0], [*rows[1][:2], "True", *rows[1][3:]]],
            "finite numeric",
        ),
    ],
)
def test_malformed_csv_fails_without_repair(tmp_path, transform, message):
    path = csv_file(tmp_path, transform=transform)
    with pytest.raises(FinanceError, match=message):
        load_ohlc_csv(path, frequency="1h")


def test_alignment_requires_entire_index_values_and_declared_identity():
    data = bars()
    validate_price_alignment(prices(data, frequency="60min"), data)
    validate_price_alignment(
        prices(data, timestamps=data.close_timestamps.as_unit("us")), data
    )
    for changes in [
        {"series_id": "other"},
        {"units": "other"},
        {"timestamps": data.close_timestamps + pd.Timedelta(1, "ns")},
        {"timestamps": data.close_timestamps[:2], "values": data.closes[:2]},
        {
            "timestamps": data.close_timestamps[0]
            + pd.timedelta_range(0, periods=3, freq="2h"),
            "frequency": "2h",
        },
    ]:
        with pytest.raises(FinanceError):
            validate_price_alignment(prices(data, **changes), data)


def test_canonical_identity_normalizes_zones_but_covers_each_ohlc_value():
    original = bars()
    equivalent = replace(
        original,
        open_timestamps=original.open_timestamps.tz_convert("Europe/Warsaw"),
        close_timestamps=original.close_timestamps.tz_convert("Europe/Warsaw"),
    )
    assert ohlc_fingerprint(original) == ohlc_fingerprint(equivalent)
    for name in ("opens", "highs", "lows", "closes"):
        changed = getattr(original, name).copy()
        changed[0] = np.nextafter(changed[0], np.inf)
        assert ohlc_fingerprint(
            replace(original, **{name: changed})
        ) != ohlc_fingerprint(original)

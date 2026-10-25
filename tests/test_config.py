import copy
import json
from pathlib import Path

import pytest
import yaml

from tsfm_lab.config import ConfigError, load_config


@pytest.fixture
def valid_config():
    return {
        "schema_version": 1,
        "model": {"source": "builtin", "adapter_id": "last_value"},
        "dataset": {
            "path": "../datasets/input.csv",
            "timestamp_column": "timestamp",
            "target_column": "demand",
            "frequency": "1h",
        },
        "evaluation": {
            "context_length": 4,
            "horizon": 2,
            "evaluation_start": "2026-01-02T00:00:00Z",
            "evaluation_end": "2026-01-03T00:00:00Z",
        },
    }


def write_config(tmp_path, data, *, name="experiment.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_resolved_paths_defaults_and_exact_snapshot(tmp_path, valid_config):
    path = write_config(tmp_path, valid_config)
    config = load_config(path)
    assert config.name == "experiment"
    assert config.source_path == path
    assert config.source_yaml == path.read_text()
    assert config.dataset == {
        **valid_config["dataset"],
        "path": str((tmp_path.parent / "datasets/input.csv").resolve()),
        "format": "csv",
        "units": "unspecified",
        "series_id": "series_0",
    }
    assert "timezone" not in config.dataset
    assert config.runtime == {
        "device": "cpu",
        "dtype": "float64",
        "batch_size": 1,
        "seed": 0,
    }
    assert config.output_directory == tmp_path / "results"
    assert not config.output_directory.exists()
    assert not Path(config.dataset["path"]).exists()
    resolved = config.resolved_dict()
    assert resolved["evaluation"]["evaluation_start"] == "2026-01-02T00:00:00+00:00"
    assert resolved["evaluation"]["metrics"] == ["mae", "rmse", "mase"]
    assert json.loads(json.dumps(resolved, allow_nan=False)) == resolved
    resolved["dataset"]["path"] = "changed.csv"
    assert config.dataset["path"] != "changed.csv"


def test_relative_paths_use_config_directory_after_cwd_change(
    tmp_path, valid_config, monkeypatch
):
    directory = tmp_path / "configs"
    directory.mkdir()
    valid_config["output"] = {"directory": "../runs"}
    path = write_config(directory, valid_config)
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(path)
    assert config.output_directory == tmp_path / "runs"
    assert config.dataset["path"] == str(tmp_path / "datasets/input.csv")


def test_home_paths_expand_without_creating_directories(tmp_path, valid_config):
    valid_config["dataset"]["path"] = "~/tsfm-config-test.csv"
    valid_config["output"] = {"directory": "~/tsfm-config-test-results"}
    config = load_config(write_config(tmp_path, valid_config))
    assert config.dataset["path"] == str(Path.home() / "tsfm-config-test.csv")
    assert config.output_directory == Path.home() / "tsfm-config-test-results"


def test_seasonal_configuration_keeps_two_distinct_periods(tmp_path, valid_config):
    valid_config["name"] = "seasonal"
    valid_config["model"] = {
        "source": "builtin",
        "adapter_id": "seasonal_naive",
        "options": {"seasonal_period": 4},
    }
    valid_config["evaluation"]["seasonal_period"] = 2
    valid_config["dataset"].update(timezone="UTC", units="MW", series_id="grid")
    valid_config["runtime"] = {"seed": 42}
    config = load_config(write_config(tmp_path, valid_config))
    assert config.model["options"]["seasonal_period"] == 4
    assert config.evaluation.seasonal_period == 2
    assert config.runtime["seed"] == 42


@pytest.mark.parametrize(
    "suffix,format_name",
    [(".csv", "csv"), (".parquet", "parquet"), (".pq", "parquet"), (".CSV", "csv")],
)
def test_file_format_inference(tmp_path, valid_config, suffix, format_name):
    valid_config["dataset"]["path"] = "series" + suffix
    config = load_config(write_config(tmp_path, valid_config))
    assert config.dataset["format"] == format_name


def test_unquoted_iso_timestamps_remain_strings_without_global_loader_changes(
    tmp_path, valid_config
):
    path = write_config(tmp_path, valid_config)
    source = path.read_text().replace("'2026-01-02T00:00:00Z'", "2026-01-02T00:00:00Z")
    path.write_text(source)
    assert (
        load_config(path).evaluation.evaluation_start.isoformat()
        == "2026-01-02T00:00:00+00:00"
    )
    assert not isinstance(yaml.safe_load("date: 2026-01-02")["date"], str)


@pytest.mark.parametrize(
    "source",
    [
        "schema_version: 1\nschema_version: 1\n",
        "model:\n  options:\n    seasonal_period: 2\n    seasonal_period: 3\n",
        "dataset:\n  path: one.csv\n  path: two.csv\n",
        "evaluation:\n  metrics:\n    - {mae: 1, mae: 2}\n",
    ],
)
def test_duplicate_keys_rejected_at_every_depth(tmp_path, source):
    path = tmp_path / "invalid.yaml"
    path.write_text(source)
    with pytest.raises(ConfigError, match="Duplicate"):
        load_config(path)


@pytest.mark.parametrize(
    "source",
    [
        "root: &loop [*loop]",
        "root: &first {value: 1}\nsecond: *first\n",
        "!!python/name:os.system ''",
        "!unsupported tag",
        "{? [one, two]: value}",
        "{1: value}",
        "[unterminated",
        "---\n{}\n---\n{}",
        "null",
        "[]",
        "",
    ],
)
def test_unsafe_or_malformed_yaml_rejected(tmp_path, source):
    path = tmp_path / "invalid.yaml"
    path.write_text(source)
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize(
    "section,field,value",
    [
        (None, "schema_version", True),
        (None, "schema_version", 1.0),
        (None, "schema_version", 2),
        (None, "schema_version", "1"),
        (None, "name", " "),
        (None, "unexpected", 1),
        ("model", "source", "huggingface"),
        ("model", "source", True),
        ("model", "adapter_id", "unknown"),
        ("model", "family", "baseline"),
        ("model", "adapter_version", "spoofed"),
        ("model", "resolved_revision", "spoofed"),
        ("model", "options", {"unused": 1}),
        ("model", "options", []),
        ("dataset", "target_column", "timestamp"),
        ("dataset", "frequency", "ME"),
        ("dataset", "frequency", "0h"),
        ("dataset", "frequency", 1),
        ("dataset", "timezone", None),
        ("dataset", "format", "parquet"),
        ("dataset", "path", "https://example.com/input.csv"),
        ("dataset", "path", "file:///tmp/input.csv"),
        ("dataset", "path", "//server/input.csv"),
        ("dataset", "path", "input.txt"),
        ("dataset", "target_column", 3),
        ("dataset", "units", ""),
        ("dataset", "unknown", 3),
        ("evaluation", "context_length", True),
        ("evaluation", "horizon", 1.0),
        ("evaluation", "stride", "1"),
        ("evaluation", "seasonal_period", 0),
        ("evaluation", "metrics", "mae"),
        ("evaluation", "metrics", ["mae", "mae"]),
        ("evaluation", "metrics", []),
        ("evaluation", "metrics", [True]),
        ("evaluation", "metrics", ["unknown"]),
        ("evaluation", "evaluation_start", 42),
        ("evaluation", "evaluation_start", "2026-01-02"),
        ("evaluation", "evaluation_end", "2026-01-01T00:00:00Z"),
        ("evaluation", "point_statistic", "mean"),
        ("evaluation", "aggregation", "other"),
        ("evaluation", "unknown", 3),
        ("runtime", "device", "cuda"),
        ("runtime", "dtype", "float32"),
        ("runtime", "batch_size", 2),
        ("runtime", "batch_size", True),
        ("runtime", "seed", -1),
        ("runtime", "seed", 1.0),
        ("runtime", "seed", True),
        ("runtime", "unknown", 3),
        ("output", "directory", "https://example.com/runs"),
        ("output", "directory", ""),
        ("output", "format", "json"),
    ],
)
def test_invalid_field_values_rejected(tmp_path, valid_config, section, field, value):
    target = valid_config if section is None else valid_config.setdefault(section, {})
    target[field] = value
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize(
    "section", ["schema_version", "dataset", "model", "evaluation"]
)
def test_required_root_fields(tmp_path, valid_config, section):
    del valid_config[section]
    with pytest.raises(ConfigError, match="Missing"):
        load_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize(
    "section", ["dataset", "model", "evaluation", "runtime", "output"]
)
def test_sections_must_be_mappings(tmp_path, valid_config, section):
    valid_config[section] = []
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize(
    "options",
    [{}, {"seasonal_period": True}, {"seasonal_period": 0}, {"seasonal_period": 5}],
)
def test_seasonal_baseline_requires_positive_period_and_sufficient_context(
    tmp_path, valid_config, options
):
    valid_config["model"] = {
        "source": "builtin",
        "adapter_id": "seasonal_naive",
        "options": options,
    }
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, valid_config))


def test_large_and_non_utf8_sources_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_bytes(b"#" * (1024 * 1024 + 1))
    with pytest.raises(ConfigError, match="1 MiB"):
        load_config(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_source_has_config_error(tmp_path):
    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_config(tmp_path / "missing.yaml")


def test_config_input_is_not_mutated_by_validation(tmp_path, valid_config):
    original = copy.deepcopy(valid_config)
    load_config(write_config(tmp_path, valid_config))
    assert valid_config == original


@pytest.mark.parametrize("frequency", ["100000000000000000000000h", "9999999999D"])
def test_oversized_frequency_has_a_config_error(tmp_path, valid_config, frequency):
    valid_config["dataset"]["frequency"] = frequency
    with pytest.raises(ConfigError, match="positive fixed duration"):
        load_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize(
    "section,field", [("dataset", "path"), ("output", "directory")]
)
def test_unknown_home_user_in_config_paths_has_a_config_error(
    tmp_path, valid_config, section, field
):
    valid_config.setdefault(section, {})[
        field
    ] = "~tsfmlab-no-such-user-928374/input.csv"
    with pytest.raises(ConfigError, match=f"Invalid {section}.{field}"):
        load_config(write_config(tmp_path, valid_config))
    assert not (tmp_path / "results").exists()


def test_unknown_home_user_in_source_path_has_a_config_error():
    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_config("~tsfmlab-no-such-user-928374/config.yaml")

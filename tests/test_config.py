import builtins
import copy
import json
from pathlib import Path

import pytest
import yaml

from tsfm_lab.config import ConfigError, config_from_mapping, load_config


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
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("utf-8"))
    return path


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_resolved_paths_defaults_and_exact_snapshot(
    tmp_path, valid_config, line_ending
):
    path = write_config(tmp_path, valid_config)
    path.write_bytes(path.read_bytes().replace(b"\n", line_ending))
    config = load_config(path)
    assert config.name == "experiment"
    assert config.source_path == path
    assert config.source_yaml == path.read_bytes().decode("utf-8")
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


def bolt_model(source="huggingface", **changes):
    model = {"source": source, "adapter_id": "chronos_bolt"}
    if source == "huggingface":
        model.update(repository="amazon/chronos-bolt-tiny", revision="a" * 40)
    else:
        model["path"] = "../models/checkpoint"
    return model | changes


@pytest.mark.parametrize(
    "repository", ["amazon/chronos-bolt-tiny", "amazon/chronos-bolt-small"]
)
def test_curated_hub_model_defaults_to_exact_offline_cpu(
    tmp_path, valid_config, repository
):
    valid_config["model"] = bolt_model(repository=repository)
    config = load_config(write_config(tmp_path, valid_config))
    assert config.model == {
        **valid_config["model"],
        "offline": True,
        "options": {},
    }
    assert config.runtime == {
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 1,
        "seed": 0,
    }
    assert config.resolved_dict()["model"] == config.model
    assert (
        json.loads(json.dumps(config.resolved_dict(), allow_nan=False))
        == config.resolved_dict()
    )


def test_hub_cache_directory_resolves_against_configuration(
    tmp_path, valid_config, monkeypatch
):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    valid_config["model"] = bolt_model(
        cache_dir="../model-cache", offline=False, revision="main"
    )
    path = write_config(config_dir, valid_config)
    monkeypatch.chdir(tmp_path.parent)
    config = load_config(path)
    assert config.model["cache_dir"] == str(tmp_path / "model-cache")
    assert config.model["revision"] == "main"
    assert config.model["offline"] is False
    assert not (tmp_path / "model-cache").exists()


def test_local_checkpoint_configuration_needs_no_existing_directory(
    tmp_path, valid_config
):
    valid_config["model"] = bolt_model("local")
    config = load_config(write_config(tmp_path, valid_config))
    assert config.model == {
        "source": "local",
        "adapter_id": "chronos_bolt",
        "path": str(tmp_path.parent / "models/checkpoint"),
        "options": {},
    }
    assert config.runtime["dtype"] == "float32"
    assert not Path(config.model["path"]).exists()
    assert not config.output_directory.exists()


@pytest.mark.parametrize(
    "revision", ["main", "v1", "a" * 39, "a" * 41, "A" * 40, "g" * 40, " " + "a" * 40]
)
def test_offline_checkpoint_requires_exact_lowercase_sha(
    tmp_path, valid_config, revision
):
    valid_config["model"] = bolt_model(revision=revision)
    with pytest.raises(ConfigError, match="exact lowercase 40-character SHA"):
        load_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize("revision", ["main", "release/v1", "a" * 40])
def test_online_metadata_selection_can_preserve_requested_revision(
    tmp_path, valid_config, revision
):
    valid_config["model"] = bolt_model(offline=False, revision=revision)
    config = load_config(write_config(tmp_path, valid_config))
    assert config.model["revision"] == revision
    assert config.model["offline"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"repository": "amazon/chronos-2"},
        {"repository": "other/model"},
        {"repository": None},
        {"repository": []},
        {"revision": ""},
        {"revision": 123},
        {"revision": None},
        {"offline": "true"},
        {"offline": 1},
        {"offline": None},
        {"offline": []},
        {"cache_dir": "https://example.com/cache"},
        {"cache_dir": None},
        {"cache_dir": []},
        {"path": "./local-checkpoint"},
        {"adapter_id": "chronos_2"},
        {"adapter_id": "last_value"},
    ],
)
def test_invalid_hub_configuration_is_rejected(tmp_path, valid_config, changes):
    valid_config["model"] = bolt_model(**changes)
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize("field", ["repository", "revision"])
def test_hub_configuration_requires_checkpoint_selection(tmp_path, valid_config, field):
    valid_config["model"] = bolt_model()
    del valid_config["model"][field]
    with pytest.raises(ConfigError, match="Missing model fields"):
        load_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize(
    "changes",
    [
        {"repository": "amazon/chronos-bolt-tiny"},
        {"revision": "a" * 40},
        {"offline": True},
        {"cache_dir": "./cache"},
        {"path": "https://example.com/model"},
        {"path": None},
        {"path": []},
        {"adapter_id": "seasonal_naive"},
        {"adapter_id": "chronos_3"},
    ],
)
def test_invalid_local_configuration_is_rejected(tmp_path, valid_config, changes):
    valid_config["model"] = bolt_model("local", **changes)
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, valid_config))


def test_local_configuration_requires_explicit_path(tmp_path, valid_config):
    valid_config["model"] = {"source": "local", "adapter_id": "chronos_bolt"}
    with pytest.raises(ConfigError, match="Missing model fields"):
        load_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize("source", ["local", "huggingface"])
@pytest.mark.parametrize(
    "changes",
    [
        {"family": "chronos"},
        {"adapter_version": "spoofed"},
        {"resolved_revision": "a" * 40},
        {"trust_remote_code": True},
        {"remote_code": False},
        {"options": {"quantile_levels": [0.5]}},
        {"options": {"trust_remote_code": True}},
        {"options": None},
        {"options": []},
    ],
)
def test_checkpoint_configs_reject_derived_fields_and_unsupported_options(
    tmp_path, valid_config, source, changes
):
    valid_config["model"] = bolt_model(source, **changes)
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository", "amazon/chronos-bolt-tiny"),
        ("revision", "a" * 40),
        ("path", "model"),
        ("offline", True),
        ("cache_dir", "cache"),
    ],
)
def test_builtin_models_keep_their_original_source_schema(
    tmp_path, valid_config, field, value
):
    valid_config["model"][field] = value
    with pytest.raises(ConfigError, match="Unknown model fields"):
        load_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize("source", ["unknown", "", True, None, []])
def test_unknown_or_malformed_model_source_is_rejected(tmp_path, valid_config, source):
    valid_config["model"]["source"] = source
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, valid_config))


@pytest.mark.parametrize(
    "runtime",
    [
        {"device": "cuda"},
        {"device": "mps"},
        {"device": "auto"},
        {"dtype": "float64"},
        {"dtype": "bfloat16"},
        {"batch_size": 2},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_checkpoint_runtime_rejects_unverified_choices(tmp_path, valid_config, runtime):
    valid_config["model"] = bolt_model()
    valid_config["runtime"] = runtime
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, valid_config))


def test_checkpoint_runtime_preserves_explicit_supported_seed(tmp_path, valid_config):
    valid_config["model"] = bolt_model()
    valid_config["runtime"] = {
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 1,
        "seed": 42,
    }
    config = load_config(write_config(tmp_path, valid_config))
    assert config.runtime == valid_config["runtime"]


@pytest.mark.parametrize("limits", [{"context_length": 2049}, {"horizon": 65}])
def test_checkpoint_capability_preflight_rejects_extension(
    tmp_path, valid_config, limits
):
    valid_config["model"] = bolt_model()
    valid_config["evaluation"].update(limits)
    with pytest.raises(ConfigError, match="context_length <= 2048 and horizon <= 64"):
        load_config(write_config(tmp_path, valid_config))


def test_checkpoint_capability_preflight_accepts_native_limits(tmp_path, valid_config):
    valid_config["model"] = bolt_model()
    valid_config["evaluation"].update(context_length=2048, horizon=64)
    config = load_config(write_config(tmp_path, valid_config))
    assert config.evaluation.context_length == 2048
    assert config.evaluation.horizon == 64


@pytest.mark.parametrize("source", ["local", "huggingface"])
def test_checkpoint_validation_imports_no_runtime_and_opens_only_yaml(
    tmp_path, valid_config, source, monkeypatch
):
    valid_config["model"] = bolt_model(source)
    path = write_config(tmp_path, valid_config)
    original_open = Path.open
    original_import = builtins.__import__
    opened = []

    def only_yaml(self, *args, **kwargs):
        assert self == path
        opened.append(self)
        return original_open(self, *args, **kwargs)

    def no_runtime(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "chronos", "huggingface_hub"}:
            raise AssertionError("Configuration must not import a model runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(Path, "open", only_yaml)
    monkeypatch.setattr(builtins, "__import__", no_runtime)
    config = load_config(path)
    assert opened == [path]
    assert not config.output_directory.exists()


def test_mapping_configuration_matches_the_yaml_boundary_without_writing(
    tmp_path, valid_config, monkeypatch
):
    valid_config["name"] = "Selected in terminal"
    path = write_config(tmp_path, valid_config)
    expected = load_config(path)

    def no_open(*args, **kwargs):
        raise AssertionError("Mapping validation must not read or write files")

    monkeypatch.setattr(Path, "open", no_open)
    result = config_from_mapping(valid_config, base_directory=tmp_path)
    assert result.resolved_dict() == expected.resolved_dict()
    assert result.source_path == tmp_path / "interactive.yaml"
    assert yaml.safe_load(result.source_yaml) == valid_config


@pytest.mark.parametrize(
    "mapping", [None, [], {"schema_version": True}, {"extra": object()}]
)
def test_mapping_configuration_rejects_invalid_or_unserializable_values(
    tmp_path, mapping
):
    with pytest.raises(ConfigError):
        config_from_mapping(mapping, base_directory=tmp_path)


def test_mapping_configuration_snapshot_is_detached(tmp_path, valid_config):
    original = copy.deepcopy(valid_config)
    config = config_from_mapping(valid_config, base_directory=tmp_path)
    valid_config["dataset"]["path"] = "changed.csv"
    assert yaml.safe_load(config.source_yaml) == original
    assert config.dataset["path"] != "changed.csv"


def test_mapping_configuration_rejects_oversized_snapshot(tmp_path, valid_config):
    valid_config["name"] = "x" * (1024 * 1024)
    with pytest.raises(ConfigError, match="1 MiB"):
        config_from_mapping(valid_config, base_directory=tmp_path)

"""Native command spelling and detached cross-platform records need no runtimes."""

import copy
import hashlib
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
import yaml
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completion
from prompt_toolkit.document import Document
from test_checkpoints import REPOSITORY, SHA, write_checkpoint
from test_future_store import make_record
from test_tirex2 import write_checkpoint as write_tirex_checkpoint
from test_tirex2_workflow import fake_adapter as shared_tirex

from tsfm_lab.config import ConfigError
from tsfm_lab.exports import export_predictions
from tsfm_lab.future_application import run_future_config
from tsfm_lab.future_config import future_config_from_mapping
from tsfm_lab.future_store import FutureStore
from tsfm_lab.future_validation import validate_configuration
from tsfm_lab.guided import review_guided
from tsfm_lab.local_paths import active_path_text, recorded_path
from tsfm_lab.models import checkpoints
from tsfm_lab.repl.arguments import quote_argument, split_arguments
from tsfm_lab.repl.prompt import CommandCompleter

fake_adapter = shared_tirex


@pytest.mark.parametrize(
    "text",
    [r"C:\Users\Jan\Łódź\a.csv", "C:/My files/a.csv", "relative.csv", r"sub\a.csv"],
)
def test_windows_local_paths_are_distinct_from_url_schemes(text):
    assert active_path_text(text, windows=True) == text


@pytest.mark.parametrize(
    "text",
    [
        r"C:relative.csv",
        r"\rooted.csv",
        "/rooted.csv",
        "https://host/a.csv",
        "file:///a.csv",
        r"\\server\share\a.csv",
        "//server/share/a.csv",
        r"\\?\C:\a.csv",
        "bad\npath",
    ],
)
def test_windows_ambiguous_network_and_url_paths_are_refused(text):
    with pytest.raises(ValueError):
        active_path_text(text, windows=True)


@pytest.mark.parametrize(
    "tail",
    [
        "series:stream.csv",
        "NUL.csv",
        "com1.csv",
        "LPT².txt",
        "CONIN$",
        "data.\\a.csv",
        "data \\a.csv",
        "wild*.csv",
    ],
)
def test_windows_streams_devices_and_ambiguous_components_fail_for_active_and_recorded_paths(
    tail,
):
    text = "C:\\data\\" + tail
    for action in (
        lambda: active_path_text(text, windows=True),
        lambda: recorded_path(text),
    ):
        with pytest.raises(ValueError):
            action()


def test_foreign_drive_is_not_a_local_execution_path_on_posix():
    with pytest.raises(ValueError):
        active_path_text(r"C:\input.csv", windows=False)
    assert active_path_text("/tmp/input.csv", windows=False) == "/tmp/input.csv"


@pytest.mark.parametrize(
    "line,expected",
    [
        (r"/run C:\Users\Jan\config.yaml", ["/run", r"C:\Users\Jan\config.yaml"]),
        ('/workspace "C:\\My files\\Łódź\\"', ["/workspace", "C:\\My files\\Łódź\\"]),
        (
            r"/export latest 'C:\My files\out.csv'",
            ["/export", "latest", r"C:\My files\out.csv"],
        ),
    ],
)
def test_windows_slash_commands_preserve_backslashes_and_quoted_spaces(line, expected):
    assert split_arguments(line, windows=True) == expected
    path = expected[-1]
    assert split_arguments(quote_argument(path, windows=True), windows=True) == [path]


def test_windows_completion_and_dispatch_share_the_same_quoting(monkeypatch):
    from types import SimpleNamespace

    import tsfm_lab.repl.arguments as arguments

    monkeypatch.setattr(arguments, "os", SimpleNamespace(name="nt"))
    completer = CommandCompleter()

    def paths(document, event):
        assert document.text == r"C:\My files\par"
        yield Completion("t.yaml")

    monkeypatch.setattr(completer.paths, "get_completions", paths)
    text = '/run "C:\\My files\\par'
    buffer = Buffer(document=Document(text))
    buffer.apply_completion(
        next(completer.get_completions(Document(text), CompleteEvent()))
    )
    assert split_arguments(buffer.text) == ["/run", r"C:\My files\part.yaml"]


@pytest.mark.parametrize(
    "text,kind",
    [
        (r"C:\Users\Jan\Łódź\weights", PureWindowsPath),
        ("/home/jan/Łódź/weights", PurePosixPath),
    ],
)
def test_recorded_paths_are_lexical_and_preserve_the_original_spelling(
    text, kind, monkeypatch
):
    monkeypatch.setattr(Path, "stat", lambda *a, **k: pytest.fail("filesystem access"))
    path = recorded_path(text)
    assert type(path) is kind and str(path) == text


@pytest.mark.parametrize(
    "text",
    [
        "relative",
        r"C:relative",
        r"C:\a\..\b",
        "/a/../b",
        "/a/./b",
        "/a//b",
        "/a/",
        "https://host/a",
        r"\\server\share",
        "C:\\bad\x00name",
        r"C:\a:stream",
        "C:/not-canonical",
    ],
)
def test_recorded_paths_require_canonical_absolute_local_spelling(text):
    with pytest.raises(ValueError):
        recorded_path(text)


@pytest.mark.parametrize("source", ["local", "huggingface"])
@pytest.mark.parametrize("root", ["/home/jan/cache", r"C:\Users\Jan\cache"])
def test_foreign_checkpoint_records_validate_without_probing_paths(
    tmp_path, monkeypatch, source, root
):
    record = checkpoints.inspect_local_checkpoint(
        write_checkpoint(tmp_path / "weights")
    ).record()
    pure = PureWindowsPath(root) if root.startswith("C:") else PurePosixPath(root)
    if source == "huggingface":
        pure = pure / ("models--" + REPOSITORY.replace("/", "--")) / "snapshots" / SHA
        record.update(
            source=source,
            repository=REPOSITORY,
            requested_revision="main",
            resolved_revision=SHA,
        )
    record["path"] = str(pure)
    before = copy.deepcopy(record)
    monkeypatch.setattr(Path, "stat", lambda *a, **k: pytest.fail("filesystem access"))
    assert checkpoints.validate_checkpoint_record(record) == before


@pytest.mark.parametrize(
    "path", ["/tmp//old/./weights", r"C:\old\\weights", r"\\server\share\weights"]
)
def test_legacy_checkpoint_location_spelling_and_fingerprint_remain_unchanged(
    tmp_path, path, monkeypatch
):
    record = checkpoints.inspect_local_checkpoint(
        write_checkpoint(tmp_path / "weights")
    ).record()
    record["path"] = path
    before = copy.deepcopy(record)
    monkeypatch.setattr(Path, "stat", lambda *a, **k: pytest.fail("filesystem access"))
    assert checkpoints.validate_checkpoint_record(record) == before


@pytest.mark.parametrize("windows", [False, True])
def test_foreign_future_record_reload_and_export_preserve_all_saved_bytes(
    tmp_path, windows
):
    made = make_record(tmp_path)
    result = copy.deepcopy(made.result)
    pure = (
        PureWindowsPath(r"C:\Users\Jan\Łódź")
        if windows
        else PurePosixPath("/home/jan/Łódź")
    )
    result["configuration"]["dataset"]["path"] = str(pure / "source.csv")
    result["configuration"]["output"]["directory"] = str(pure / "results")
    result["provenance"]["dataset"]["source_path"] = str(pure / "source.csv")
    text = yaml.safe_dump(result["configuration"], allow_unicode=True)
    result["provenance"]["input_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    store = FutureStore(tmp_path / "detached")
    run_id = store.create(result["configuration"], text)
    result["run_id"] = run_id
    store.complete(run_id, result, made.predictions, made.observations)
    directory = store.root / run_id
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    loaded, frame, _ = store.load_future_result(run_id)
    assert loaded == result and len(frame) == 3
    assert (
        export_predictions(run_id, store.workspace, tmp_path / "export.json")["rows"]
        == 3
    )
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    detached = validate_configuration(result["configuration"])
    assert detached.resolved_dict() == result["configuration"] and detached._record_only
    for action in (run_future_config, review_guided):
        with pytest.raises(ConfigError, match="read-only"):
            action(detached)


@pytest.mark.parametrize(
    "field,value",
    [
        ("forecast", {"horizon": 0}),
        ("runtime", {"batch_size": 2}),
        ("dataset", {"extra": "ignored?"}),
        ("forecast", {"origin": "latest"}),
        ("output", {"directory": r"C:\a\..\b"}),
    ],
)
def test_detached_future_path_validation_retains_strict_numerical_and_field_checks(
    tmp_path, field, value
):
    result = make_record(tmp_path).result
    result["configuration"][field].update(value)
    with pytest.raises(ValueError):
        validate_configuration(result["configuration"])


@pytest.mark.parametrize("source", ["local", "huggingface"])
@pytest.mark.parametrize("observed_candidate", [False, True])
def test_pretrained_future_foreign_checkpoint_paths_and_optional_windows_provenance_reload_offline(
    tmp_path, fake_adapter, source, observed_candidate
):
    made = make_record(tmp_path)
    checkpoint = write_tirex_checkpoint(tmp_path / "tirex-weights")
    raw = made.config.resolved_dict()
    raw["model"] = {
        "source": "local",
        "adapter_id": "tirex_2",
        "path": str(checkpoint.path),
    }
    raw["runtime"]["dtype"] = "float32"
    completed = run_future_config(
        future_config_from_mapping(raw, base_directory=tmp_path)
    )
    result, predictions, observations = FutureStore(
        made.config.output_directory
    ).load_future_result(completed["run_id"])
    foreign = PureWindowsPath(r"C:\Users\Jan\Łódź")
    result["configuration"]["dataset"]["path"] = str(foreign / "input.csv")
    result["configuration"]["output"]["directory"] = str(foreign / "results")
    result["provenance"]["dataset"]["source_path"] = str(foreign / "input.csv")
    if observed_candidate:
        result["provenance"]["environment"]["triton-windows"] = "3.4.0.post21"
    record = result["provenance"]["model"]["options"]["checkpoint"]
    fingerprint = record["fingerprint"]
    if source == "local":
        record["path"] = str(foreign / "weights")
        result["configuration"]["model"]["path"] = record["path"]
    else:
        repository, revision = "NX-AI/TiRex-2", "a" * 40
        record.update(
            source=source,
            repository=repository,
            requested_revision="main",
            resolved_revision=revision,
            path=str(foreign / "models--NX-AI--TiRex-2" / "snapshots" / revision),
        )
        result["configuration"]["model"] = {
            "source": source,
            "adapter_id": "tirex_2",
            "options": {},
            "repository": repository,
            "revision": revision,
            "offline": True,
            "cache_dir": str(foreign),
        }
        result["provenance"]["model"]["source"] = source
    assert checkpoints.validate_checkpoint_record(record)["fingerprint"] == fingerprint
    text = yaml.safe_dump(result["configuration"], allow_unicode=True)
    result["provenance"]["input_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    store = FutureStore(tmp_path / "foreign-pretrained")
    identifier = store.create(result["configuration"], text)
    result["run_id"] = identifier
    store.complete(identifier, result, predictions, observations)
    directory = store.root / identifier
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    program = """
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','tirex2','triton','safetensors','huggingface_hub','flashrnn','mlstm_kernels'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, Block())
from tsfm_lab.future_store import FutureStore
from tsfm_lab.exports import export_predictions
result, frame, _ = FutureStore(sys.argv[2]).load_future_result(sys.argv[3])
assert len(frame) == 3
assert export_predictions(sys.argv[3], sys.argv[2], sys.argv[4])['rows'] == 3
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            program,
            str(Path(__file__).resolve().parents[1] / "src"),
            str(store.workspace),
            identifier,
            str(tmp_path / "foreign-export.csv"),
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.skipif(os.name != "nt", reason="Requires native Windows drive resolution")
def test_native_windows_config_roundtrip_uses_local_drive_paths(config_factory):
    from tsfm_lab.config import load_config

    config = load_config(config_factory())
    assert PureWindowsPath(config.dataset["path"]).is_absolute()
    assert config.output_directory.is_absolute()

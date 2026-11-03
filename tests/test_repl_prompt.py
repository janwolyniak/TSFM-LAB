from pathlib import Path

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from tsfm_lab.repl.arguments import split_arguments
from tsfm_lab.repl.prompt import CommandCompleter


def complete(text, ids=()):
    return list(
        CommandCompleter(lambda: list(ids)).get_completions(
            Document(text), CompleteEvent()
        )
    )


def test_command_completion_contains_help_and_filters_prefix():
    assert "/new" in {item.text for item in complete("/")}
    assert [item.text for item in complete("/hist")] == ["/history"]


@pytest.mark.parametrize("command", ["/new", "/demo"])
def test_timesfm_adapter_completion_comes_from_the_model_catalog(command):
    assert [item.text for item in complete(f"{command} times")] == ["timesfm_2p5"]


def test_id_completion_does_not_read_experiment_files():
    assert [item.text for item in complete("/show abc", ["abcdef", "12345"])] == [
        "abcdef"
    ]
    assert [item.text for item in complete("/compare latest 12", ["12345"])] == [
        "12345"
    ]


@pytest.mark.parametrize("command", ["/run", "/workspace", "/export latest"])
def test_path_completion_completes_only_argument_and_preserves_spaces(
    tmp_path, command
):
    (tmp_path / "experiment file.yaml").touch()
    text = f"{command} {tmp_path}/exper"
    completion = complete(text)[0]
    buffer = Buffer(document=Document(text))
    buffer.apply_completion(completion)
    result = buffer.text
    words = split_arguments(result)
    assert words[:-1] == split_arguments(command)
    assert Path(words[-1]).samefile(tmp_path / "experiment file.yaml")


def test_quoted_path_completion_handles_unclosed_quote(tmp_path):
    (tmp_path / "experiment file.yaml").touch()
    text = f'/run "{tmp_path}/exper'
    completion = complete(text)[0]
    buffer = Buffer(document=Document(text))
    buffer.apply_completion(completion)
    result = buffer.text
    words = split_arguments(result)
    assert words[:-1] == ["/run"]
    assert Path(words[-1]).samefile(tmp_path / "experiment file.yaml")


@pytest.mark.parametrize("quote,closing", [('"', '"'), ("'", "")])
def test_completion_replaces_quote_context_safely(tmp_path, quote, closing):
    target = tmp_path / "example's file.yaml"
    target.touch()
    text = f"/run {quote}{tmp_path}/exam{closing}"
    completion = complete(text)[0]
    buffer = Buffer(document=Document(text))
    buffer.apply_completion(completion)
    words = split_arguments(buffer.text)
    assert words[:-1] == ["/run"]
    assert Path(words[-1]).samefile(target)

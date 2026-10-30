"""Guided downloads require explicit terminal consent; scripts stay offline."""

import io
from importlib.resources import files

import pytest
from rich.console import Console
from test_model_setup import Questions

from tsfm_lab.config import load_config
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.guided import GuidedRunPlan
from tsfm_lab.models.preparation import CheckpointPlan
from tsfm_lab.repl import wizard
from tsfm_lab.repl.model_selection import select_model
from tsfm_lab.repl.progress import execute
from tsfm_lab.repl.session import TerminalSession
from tsfm_lab.tui.forms import default_fields


def build(tmp_path, questions, emitted):
    return wizard.build_experiment(
        questions,
        emitted.append,
        workspace=tmp_path / "results",
        demo_path=files("tsfm_lab.resources").joinpath("seasonal.csv"),
        adapter="chronos_bolt",
        guided_preparation=True,
    )


@pytest.mark.parametrize("choice,offline", [("automatic", False), ("cached", True)])
def test_source_mode_is_transient_and_never_changes_configuration_schema(
    choice, offline
):
    selected = select_model(
        Questions({"Checkpoint source": choice}),
        lambda _: None,
        default_fields("chronos_bolt"),
        allow_preparation=True,
    )
    assert selected.pop("preparation_offline") == str(offline).lower()
    assert selected["source"] == "huggingface"
    assert selected["revision"]
    assert "offline" not in selected


@pytest.mark.parametrize("choice", ["automatic", "cached"])
@pytest.mark.parametrize("answer", ["yes", "no"])
def test_download_review_requires_specific_confirmation_and_makes_no_run(
    tmp_path, monkeypatch, choice, answer
):
    reviews = []
    cached = choice == "cached"

    def review(config, **kwargs):
        reviews.append(kwargs)
        checkpoint = CheckpointPlan(
            config.model["repository"],
            config.model["revision"],
            config.model["revision"],
            tmp_path / "cache",
            cached,
            0 if cached else None,
            (),
            b"{}",
            b"{}",
        )
        return GuidedRunPlan(config, checkpoint, "a" * 64)

    monkeypatch.setattr("tsfm_lab.guided.review_guided", review)
    monkeypatch.setattr(
        "tsfm_lab.models.preparation.prepare_checkpoint",
        lambda *_a, **_kw: pytest.fail("Review cannot download weights"),
    )
    confirmation = (
        "Run experiment?"
        if cached
        else "Download missing checkpoint files and run experiment?"
    )
    questions = Questions({"Checkpoint source": choice, confirmation: answer})
    emitted = []
    result = build(tmp_path, questions, emitted)
    assert isinstance(result, GuidedRunPlan) if answer == "yes" else result is None
    assert reviews == [{"offline": cached}]
    assert (confirmation, "no", ("yes", "no", "back")) in questions.calls
    if not cached:
        assert "Required transfer: size unknown" in emitted
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "results").exists()


def test_missing_runtime_blocks_before_any_hub_work(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tsfm_lab.guided.runtime_diagnostics",
        lambda: [
            {
                "adapter_id": "chronos_bolt",
                "label": "Chronos-Bolt",
                "status": "missing",
                "detail": "Install the optional runtime explicitly.",
            }
        ],
    )
    monkeypatch.setattr(
        "tsfm_lab.guided.plan_checkpoint",
        lambda *_a, **_kw: pytest.fail("Runtime must be checked before Hub access"),
    )
    emitted = []
    assert build(tmp_path, Questions(), emitted) is None
    assert any("runtime is unavailable" in text for text in emitted)
    assert not (tmp_path / "results").exists()


def test_script_run_never_uses_the_guided_download_service(config_factory, tmp_path):
    calls = []

    def ordinary(config, console):
        calls.append(config)
        return {"type": "cancelled"}

    app = TerminalSession(
        tmp_path / "results",
        runner=ordinary,
        guided_runner=lambda *_: pytest.fail("Saved YAML cannot authorize a download"),
        console=Console(file=io.StringIO()),
    )
    app.dispatch(f'/run "{config_factory()}"')
    assert len(calls) == 1


@pytest.mark.parametrize("status", ["pending", "failed", "cancelled"])
def test_show_unfinished_records_does_not_request_completed_outputs(
    config_factory, tmp_path, status
):
    config = load_config(config_factory())
    store = ExperimentStore(config.output_directory)
    run_id = store.create(config.name, config.resolved_dict(), config.source_yaml)
    if status != "pending":
        store.set_status(run_id, status)
    output = io.StringIO()
    app = TerminalSession(config.output_directory, console=Console(file=output))
    app.dispatch(f"/show {run_id}")
    assert "Error:" not in output.getvalue()
    assert "Only completed experiments" not in output.getvalue()
    assert status in output.getvalue().lower()


def test_live_progress_retains_measured_bytes_window_counts_and_elapsed_time():
    class Worker:
        def __init__(self, config):
            self.batches = iter(
                [
                    [
                        {"type": "stage", "stage": "downloading"},
                        {
                            "type": "transfer",
                            "completed_bytes": 12,
                            "total_bytes": None,
                        },
                    ],
                    [{"type": "stage", "stage": "verifying"}],
                    [
                        {"type": "stage", "stage": "running"},
                        {"type": "progress", "completed": 4, "total": 4},
                    ],
                    [{"type": "stage", "stage": "saving"}],
                    [{"type": "completed"}],
                ]
            )

        def start(self):
            pass

        def poll(self):
            return next(self.batches)

        def close(self):
            return []

    output = io.StringIO()
    result = execute(
        None,
        Console(file=output, width=160),
        worker_factory=Worker,
        pause=lambda _: None,
    )
    assert result == {"type": "completed"}
    text = output.getvalue()
    assert "12 bytes measured" in text
    assert "4/4 windows" in text
    assert "Elapsed" in text
    assert "%" not in text

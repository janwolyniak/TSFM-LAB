# Persistent terminal prompt: implementation and verification

Date: 25 October 2026. Branch: `feat/terminal-prompt`, based on the unmerged
`feat/interactive-terminal` branch at `49fa20e`. Prerequisite milestones must merge
first; no publication or merge is part of this local change.

## Agreed direction

The owner found the full-screen button/wizard interface too similar to a separate
application. The requested experience is a terminal with TSFM capabilities inside
it, drawing interaction inspiration from Codex, Claude Code, and OpenClaw. In
clarification, the owner selected a persistent prompt with guided commands and
local deterministic actions, with no AI service.

The replacement uses native terminal scrollback, a small welcome block, editable
input, slash-command suggestions, Tab completion, in-memory command history,
inline questions, and ordinary Rich output. It does not enter an alternate
screen or launch the previous button screens. `/demo` is a short first path;
`/new` guides a full experiment. Model discovery, history, verified results,
matching-task comparison, and export stay in the same transcript.

`prompt_toolkit` 3.0.50 and `wcwidth` 0.2.13 were already available. Their
requirements are declared/recorded; no new third-party package installation or
upgrade was needed. `[cli]` now supplies the prompt editor. `[tui]` remains
compatible and the original Textual source/tests remain for regression coverage.
The numerical core remains independent of both terminal implementations.

## Verification

| Check | Observed result |
| --- | --- |
| Full routine suite | **886 passed, 1 skipped** in 45.14 seconds in the existing disposable environment on macOS/Python 3.13.6. The skipped test is the opt-in Chronos real-checkpoint smoke test. |
| Formatting/linting | Black and Ruff passed for source, tests, and examples. Diff whitespace and local Markdown links checked. |
| Actual terminal input | Six controlling-pseudo-terminal cases at **80×24 and 120×36** cover completion, Up-arrow command recall, setup interruption, a real baseline run, idle Ctrl+D, and Ctrl+C during real inference. Exit status is 0. |
| Native scrollback | The terminal stream contains no alternate-screen entry sequences (`?1049h`, `?1047h`, `?47h`); command output stays above subsequent prompts. |
| Cancellation | The foreground process group receives real Ctrl+C. Workers ignore terminal SIGINT and stop through the cancellation event; the main monitor handles SIGINT between IPC reads, restores its previous handler after cleanup, waits for worker termination, and retains the cancelled record before accepting another command. Real-spawn and race tests remain passing. |
| Inline questions | Nineteen focused tests cover default acceptance, confirmation default no, /back, /cancel, Ctrl+C/EOF, invalid fields/bounds, model capabilities, local/cached checkpoint metadata, and preservation of explicit choices. Baseline setup imports no model runtime or Textual and creates no experiment until confirmed. |
| Records/output | Twenty-eight focused tests cover missing/ambiguous IDs, latest verified completed, changed workspaces, corrupt/incomplete records, undefined metrics, strict comparison, export/overwrite refusal, literal markup/control characters, and timestamp precision. Record commands do not run inference. |
| Command routing | Existing shell/JSON behavior remains covered. New tests verify prompt launch without Textual/model libraries and explain a missing prompt-editor dependency. Syntax/usage errors return to the prompt. Commands do not invoke arbitrary shell execution or an AI model. |
| Numerical equivalence | A scripted command session's saved baseline predictions and all metrics exactly equal the existing YAML/shell workflow on the original seasonal fixture. |
| Installed wheel | Built with no dependency resolution; installed only the local project wheel into the existing disposable environment with `--no-deps --no-index`. A real installed `tsfm` session outside the checkout completed both baselines, compared them, exported 16 cases to a path with spaces, and exited normally. Module imports came from site-packages. |
| Output layout | Recorded model/result reports were rendered to SVG/PNG at 80 and 120 columns and visually inspected. Live editor behavior was tested through real terminal streams. These report renders are not native Terminal-app screenshots; native GUI capture was unavailable. Long timestamp cells fold without dropping precision. |

The installed wheel was `tsfm_lab-0.1.0-py3-none-any.whl`, 91,982 bytes, SHA-256
`7e6bb0b38184e7998561dfc9465ecfcc6773280a3666e58cf4f9b6012a365e23`.
Independent reviews found and repaired path/quote completion, stale question
completers, stale run-ID suggestions after workspace changes, SIGINT handling
at process/IPC boundaries, and terminal control characters in stored text.
The full suite passed after these repairs; no actionable review findings remain.

## Reproduce and limits

See the [current terminal guide](tui.md) for setup and commands. The standard
[checks](../README.md#checks) run the prompt tests; install the existing `[tui]`
extra as well to include the original Textual regression suite. Routine tests
need neither network access nor weights/GPU access. The POSIX terminal checks
skip on Windows.

Remote CI, Windows, other Python/platform combinations, and a fully clean
dependency installation were not executed here. No new checkpoint download,
GPU compute, model training, or scientific performance comparison was performed.
Source runs during implementation recorded dirty Git provenance; installed-wheel
runs identify the package rather than claiming a clean source revision.

The pre-existing untracked `PLAN.md` remains unchanged. Temporary environments,
PTY captures, rendered reports, and generated records stay outside Git. No push,
pull request, or merge was performed.

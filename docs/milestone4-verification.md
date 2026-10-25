# Interactive terminal verification record

This records the original Textual milestone. The owner subsequently requested a
persistent terminal prompt; see the [rework record](terminal-rework.md) for the
current interaction and its verification.

Date: 25 October 2026. Scope: Milestone 4 following the owner's continuation
request. Local branch: `feat/interactive-terminal`, based on the unmerged
`feat/chronos-bolt-integration` branch. Merge the prerequisite milestones first.

## Implemented behavior

The optional Textual application provides a four-step model/data/evaluation/review
wizard, offline model catalog, progress, cancellation, saved results, history,
compatible comparisons, and CSV/JSON/optional Parquet export. The installed
package includes the original synthetic CSV and terminal stylesheet. `tsfm`
without arguments opens the UI only when both streams are terminals; explicit
shell commands preserve their noninteractive behavior.

The application uses the same validated configurations and numerical functions
as the shell. A spawned process owns inference; a one-way pipe carries bounded
progress events. Cancellation distinguishes a request from a reaped worker,
allows cooperative cleanup, then terminates after three seconds with a kill
fallback after another 0.25 seconds. The controller reconciles persisted status
off the UI thread, preserving completed records when completion wins a race.
Worker output goes to a temporary log; failure screens show a bounded tail.

## Environment and authorization

Local verification used Python 3.13.6 on macOS. The owner explicitly authorized
installation into a disposable environment. Only Textual 8.2.8, mdit-py-plugins
0.6.1, and linkify-it-py 2.2.0 were added there. The environment reused existing
numerical, terminal, development, and model libraries through system site
packages; shared libraries were not upgraded. This was not a fully clean
installation of the complete dependency stack.

## Checks and observed results

| Check | Outcome |
| --- | --- |
| Full routine suite | **814 passed, 1 skipped** in 37.45 seconds. The skipped test is the opt-in Chronos smoke test. No routine test downloads models. |
| Black, Ruff, diff whitespace | Passed for source, tests, and examples. |
| Keyboard workflow | At 80×24, Tab/Enter navigate home and all four wizard steps, select the packaged demo, execute a real spawn worker, display results, and return home. The saved 16-case prediction table and metrics exactly equal the shell workflow. |
| Form/configuration validation | Strict in-memory configuration parsing, asynchronous data inspection, explicit-bound preservation, model/season constraints, dataset-only demo selection, and local checkpoint/capability checks passed. Review reads metadata without loading Torch/Chronos or downloading weights. |
| UI lifecycle | Headless tests cover progress, failures, cancellation pending until the worker stops, retained diagnostics, Ctrl+Q cleanup, model/history navigation, and startup failure. A fresh-process regression covers Python resource-tracker startup when terminal capture streams report file descriptor -1. |
| Process lifecycle | Real-spawn tests cover cooperative and forced cancellation, a SIGTERM-resistant worker with kill fallback, crashes, unknown UUIDs, completion/cancel races, nonblocking manifest reconciliation, and native/Python log isolation. |
| Saved views | History, comparison, undefined MASE reasons, corrupt/incomplete records, exports without inference/overwrite, and responsive navigation during slow reads passed. Preview timestamps retain seconds, nanoseconds, and year boundaries. |
| Rendered terminal QA | SVG screenshots rendered to PNG and visually inspected at 80×24 and 120×36. Checked home, wizard steps, progress, requested cancellation, stopped state, results, history, comparison, model selection, and export controls. Navigation stays above the footer; long forms/tables scroll. Progress/cancellation screenshots use controlled events; the keyboard execution test uses a real process. |
| Wheel and resources | Built without dependency resolution; installed the local wheel with `--no-deps --no-index` into the disposable environment. Demo CSV and stylesheet are present. The installed module loaded from site-packages outside the checkout. |
| Installed baseline | Real installed UI/worker workflow produced four successful windows, 16 cases, and MAE 2.5 on the hourly synthetic demand task, with context 8/horizon 4/stride 4/MASE lag 4. |
| Installed Chronos | The installed UI/worker completed the same 16-case task using the previously authorized cached Tiny revision on CPU, with Hub/Transformers offline settings. No further download occurred. This checks integration, not forecasting superiority. |
| Actual terminal entry point | In an 80×24 pseudo-terminal, installed `tsfm` without arguments rendered home; `q` exited with status 0. Redirected entry-point and missing-extra behavior are covered separately. |

Independent reviews found and repaired demo-selection resets, premature checkpoint
readiness, timestamp truncation, and a command-palette/modal interference path.
Combined verification also repaired guarded version exits and initial spawn under
Textual's captured streams. Focused regressions and the full suite passed after
the repairs. No outstanding actionable review findings remain.

The wheel was `tsfm_lab-0.1.0-py3-none-any.whl` (77,835 bytes), SHA-256
`c0c18d2ba93fb57153741e692f4d172835126999bcd09fa1dfeaa0e10383b7f8`.
The installed CPU check reused
`amazon/chronos-bolt-tiny@a0e552de83495b5c28c14c71c374f3e33280b340` from the
[Milestone 3 download](milestone3-verification.md). Its synthetic scores are
recorded only as integration evidence; no model advantage claim follows.

## Reproduce and limits

After installing the approved optional dependencies as described in
[the terminal guide](tui.md), run:

```sh
python -m black --check src tests examples
python -m ruff check src tests examples
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
python -m pip wheel --no-deps --no-build-isolation --no-index --wheel-dir /tmp/tsfm-lab-wheels .
tsfm ui --workspace results
```

The configured Linux/macOS Python 3.11–3.13 CI matrix now includes the UI extra,
but remote CI and those other runtime combinations have not been executed here.
No Windows, GPU/MPS, Small checkpoint, training, or scientific performance claim
is covered. The installed real-model check used the pinned local cache already
available from Milestone 3. Source-checkout runs during implementation record a
dirty Git state; installed-wheel runs record package identity instead of claiming
a clean source revision.

The pre-existing untracked `PLAN.md` is unchanged. The isolated environment,
screenshots, checkpoint cache, and generated experiment records are temporary
verification artifacts outside Git. No branch was pushed, pull request opened,
or merge performed.

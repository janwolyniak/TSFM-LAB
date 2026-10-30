# v0.3.0 S4 future forecasting verification

Date: 29 October 2026. Scope: the [approved S4 contract](future-forecast-contract.md)
and [release plan](release-v0.3.0-plan.md). Branch
`codex/v0.3.0-future-forecast` depends on contract commit `b574c4d`, finance repair
`4a2344d`, S3 `30d6ac8`, S2 `d2eeab9`, and S1 `f35e069`; integrate prerequisites
first. No publication or merge is included.

## Implemented behavior

Separate future configuration and numerical services consume only the final L
observations and generate exactly H regular future timestamps after the final
observed origin. Operational and native bounds, numerical preflight, detached
input/output snapshots, actual point labels, and original units are enforced.
Request identity binds consumed context and verified numerical model/runtime
identities, independently of source prefix, paths and category-only metadata.

Future manifests carry schema 3/kind from creation. Completion requires eight
checked artifacts, including consumed observations, deterministic SVG/provenance,
and explicit unavailable accuracy. Dedicated validators reconstruct context/grid/
request identity and reject inconsistent artifacts even after checksum updates.
Historical configurations, scores, fingerprints, renderers, comparisons and
finance reports retain their existing contracts. Future comparison/finance are
explicitly refused.

The shared managed process and guided checkpoint preparation dispatch explicit
configuration types. Reviewed data/raw bytes, origin, known checkpoint identities
and prepared fingerprints remain bound through late completion and IPC recovery.
Future progress skips evaluation. Both setup forms and the persistent prompt
support future mode; assistants follow human-selected mode and send metadata-only
explanation evidence. Scriptable future Python APIs and CLI dispatch preserve
historical-only Python entry points and no-download defaults.

## Verification evidence

Environment: macOS ARM64, Python 3.13, existing local dependencies. Routine checks
use small fixtures, fake providers/runtimes and real offline baseline workers.

- Full suite: **3,007 passed, 15 skipped** in 78.82 seconds using
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q --tb=short`.
  Configured Black/Ruff checks pass across `src tests examples scripts`; Black
  checked 210 files. Whitespace/diff checks pass.
- Documentation checks pass for **159 local links/anchors across 12 changed
  documents**, balanced code fences, parsed YAML/JSON examples and the runnable
  future configuration. All **104 Python files** in the built wheel match the
  implementation source bytes.
- Configuration/core tests cover strict mixed-mode fields, N=L and N=L=H=1,
  explicit/UTC-equivalent final-row origins, overflow/resource guards, detached
  timestamp mutation, actual output labels, unsupported numerics and no scoring.
  Both baselines and all five actual pretrained adapter classes run with existing
  fake runtimes at native limits, including all three TTM contexts.
- Saved-record tests cover strict initial/failure/cancel/completed schemas,
  eight-artifact atomicity, rehashed metadata/CSV/chart/provenance tampering,
  float64/negative-zero round trips, fixed-grid identities and weight-free reload.
  Historical fixtures, fingerprints, finance accounting and guided-v2 reports pass.
- Managed checks cover real subprocess completion, failures, cancellation, lost
  IPC, late valid completion, wrong-configuration refusal, explicit-origin
  substitution and changed local checkpoint bytes. Remote recovery checks bind
  known file identities plus the post-preparation fingerprint, preserving the
  distinction between raw configuration bytes and canonical parsed metadata.
- Manual/assisted tests verify equivalent choices, authoritative human mode and
  fresh consent across changes using all four existing fake provider routes.
  Metadata-only explanations retain the existing outgoing-value/path omissions.
- Final review closed the Textual confirmation gap: the screen keeps the reviewed
  offline plan through managed execution. Service regressions reject source drift
  both during review and after confirmation. Historical exports also retain their
  original NumPy/pandas-only import boundary; an isolated subprocess blocks YAML,
  terminal, assistant transport and model-runtime imports while exporting a legacy
  fixture successfully.
- Offline source/wheel build passes: **343 source files, 119 wheel files**,
  including every future module and bundled example. Core dependencies remain
  NumPy/pandas; the wheel is built from the source archive. This is a development
  snapshot with uncommitted changes, not a published release or final S9 identity.
- A freshly extracted wheel passes an **80x24 terminal** guided future-baseline
  walkthrough using the existing disposable CLI environment without installation:
  mode after data preview, exact two-row export, schema-3 eight-file completion,
  schema-2 completed preparation without evaluation, exact output paths, and
  restart/reload without inference-framework imports or source-checkout imports.
- The future baseline SVG was rendered with an existing offline image library
  and visually inspected. Original units, UTC axis, last-observation origin,
  separated context/future lines, actual point label and unavailable-accuracy
  wording agree with the checked data. Full-context/sampling disclosure is legible.

## Qualification limits

No dependency installation, actual model weight download/pretrained inference,
live provider request or vault login was performed. All five pretrained classes
are exercised with existing fake runtimes; this establishes software contracts,
not new real-model qualification. Optional Textual execution is skipped in the
general environment because Textual is absent. Form services and prompt flows are
executed; screen edits receive static inspection. Native Windows/Linux and the
Python-version matrix remain release qualification work. Existing S3 real-model
qualification remains open.

Recorded source fingerprints describe the input check. Detached reload cannot
authenticate unsaved source rows or historical availability. Unknown upstream
weight hashes remain observed-byte identities, as documented in
[guided preparation](guided-preparation.md). Later actual import/scoring remains
deferred. Package version and published v0.2.0 assets remain unchanged.

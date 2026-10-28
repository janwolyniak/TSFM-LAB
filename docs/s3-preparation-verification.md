# v0.3.0 S3 preparation verification

Date: 28 October 2026. Scope: S3 implementation from the
[release plan](release-v0.3.0-plan.md), following [S2](s2-model-setup-verification.md).
Branch `codex/v0.3.0-preparation-outputs` depends on S2 commit `d2eeab9` and S1
commit `f35e069`; integrate in that order. No publication or merge is included.

## Implemented boundary

- Shared metadata review validates data/settings/runtime first, resolves and
  freezes an exact safe checkpoint/file plan, and transfers no weights.
- Guided manual/assisted setup offers automatic, offline cached, or local sources,
  then requires explicit download/run confirmation. Script behavior is unchanged.
- Managed execution checks real runtime imports before weight transfer, prepares
  reviewed files, rechecks data and checkpoint identity, runs the existing
  numerical contract, and publishes checked outputs atomically.
- Durable preparation records cover pre-inference failures/cancellation and link
  numerical records. Worker cleanup and late-completion reconciliation reuse the
  original managed-process boundary. Transfer persistence/IPC is throttled while
  preserving current counters and terminal truth.
- Guided schema-2 manifests require chart/provenance plus the original five files.
  Legacy schema-1 records and numerical/result/configuration identities remain
  unchanged. Completion/reload expose checked paths without inference.

## Verification work

Environment: macOS ARM64, Python 3.13.6, existing local dependencies. Checks use
small fixtures, fake Hub/runtime transports, and actual offline baseline workers.
No runtime installation, real weight download, paid request, or live provider
login is authorized or performed by this implementation task.

- Full suite: **2,799 passed, 14 skipped** in 63.78 seconds using
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`.
- Configured Black/Ruff checks pass across `src tests examples scripts`; Black
  checked 195 files. `git diff --check` passes.
- Offline fake-Hub/runtime checks cover empty-cache review without weights,
  specifically confirmed preparation, exact revision/file consent, integrity
  receipts, cached replay, unknown totals, failures, and cancellation. Real
  baseline workers cover process startup, forced stop, crashes, late completion,
  and lost-IPC reconciliation.
- Future-actual perturbation tests retain historical-only inputs, including the
  existing TTM normalization tests and a new guided-run regression. Actuals and
  scores can change while the earlier-origin predictions remain unchanged.
- Output tests cover strict old/new manifests, atomic failure/cancellation,
  checksum plus deterministic artifact validation, safe XML, UTC/subsecond ticks,
  original negative values, unavailable metrics, sampling disclosure, credential
  field omission, and reload without source datasets or optional runtimes.
- All 164 local links across ten changed documents pass. The final baseline SVG
  was rendered with an existing offline renderer and visually inspected: labels,
  dates/times, original units, separate windows, evaluation bounds, metric values,
  and scope/sampling captions are legible and consistent with the saved record.
- Offline source/wheel build passes required module/resource, license, entry-point,
  metadata, and optional-import boundaries: 324 source files and 111 wheel files;
  core dependencies remain NumPy/pandas. The wheel is built from the source archive
  and its Python files match the implementation.
- A fresh extracted-wheel 80x24 terminal passes guided setup, original-unit data
  preview, offline managed baseline execution, seven-artifact schema-2 publication,
  durable completed preparation history, checked chart/provenance paths, 16-row
  JSON export, Chronos-2 selection/cancellation before checkpoint access, and
  saved-result reload after restart without inference-framework imports. It uses
  an existing isolated CLI environment without installing dependencies.

Independent reviews found and resolved three additional regressions: newly
missing reviewed-present files cannot expand download authorization; recovered
failed/cancelled run IDs remain visible after IPC loss; `/show` handles unfinished
records without requesting completed-only paths. Cache/data TOCTOU, progress
throttling, output atomicity, and these recovery cases have retained tests.

## Qualification limits

The first S3 acceptance criterion remains open for representative real pretrained
download/runtime/inference qualification. Fake transports demonstrate the empty
cache, confirmed download, verified replay, failure/cancel, and output workflow;
they do not establish live Hub/runtime compatibility. The installed general
environment's Hub version is not silently changed to match the pinned runtime.

Copied caches lacking prior receipts/upstream digests support observed-byte
identity and same-plan revalidation, rather than a newly established upstream
authenticity claim. Chart display sampling is disclosed; full CSV data remains.
Only evaluated observations are charted because saved results omit context history.

S2 mode-switching acceptance remains tied to S4. S4-S9 remain pending; native
Windows/Linux/Python-matrix and new real provider/model qualification have not
been performed. Package version/public v0.2.0 assets remain unchanged. See
[the guide](guided-preparation.md) and [contribution rules](../CONTRIBUTING.md).

# v0.3.0 S1 onboarding verification

Date: 27 October 2026. Scope: the first implementation stage of the
[agreed v0.3.0 plan](release-v0.3.0-plan.md). Base: v0.2.0 on `main`, commit
`47bb07c`. Implementation branch: `codex/v0.3.0-onboarding`.

## Implemented boundary

- Interactive first-launch offer, persistent skip, transactional completion,
  back/cancel navigation, and later `/setup` or `/settings` editing.
- Strict versioned nonsecret preferences with atomic replacement. Workspace,
  results/cache paths, category, and export format survive restarts; no existing
  results or caches move. Filesystem probes validate reviewed choices, and the
  effective output path is checked again before interactive execution.
- Guided and assisted setup use defaults without replacing explicit selections.
  Explicit UI workspace, saved YAML, export extensions, and explicit formats keep
  precedence. Ordinary script commands do not read preferences or prompt.
- Optional assistant setup delegates to the existing hidden-input connection
  menu. No provider request is introduced by onboarding; credential storage and
  redaction retain the v0.2.0 boundary.
- Runtime diagnostics reuse the catalog's requirement definitions and inspect
  distribution metadata without model imports, installation, network access, or
  model allocation. The bundled offline baseline retains its original numerical
  configuration and explicit execution confirmation.

See [the user guide](onboarding.md) for paths, environment variables, and exact
override semantics. Cancelling a never-completed setup leaves the offer pending;
skipping persists dismissal even when unused work/cache paths are unavailable.

## Verification evidence

Local environment: macOS 27.0.1 ARM64, Python 3.13.6, existing `.venv`.
Routine checks use small disposable fixtures and fake connections. No dependency
installation, model download, paid request, live provider login, or cloud compute
is part of this stage.

- Full offline suite: **2,617 passed, 14 skipped** in 59.25 seconds using
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`.
- Configured formatting/lint: Black checked 181 files and Ruff passed across
  `src tests examples scripts`.
- Focused onboarding, persistence, runtime, terminal, wizard, and distribution
  regressions passed, including real pseudo-terminal sessions at 80x24 and
  120x36, native scrollback, completion, Ctrl+C, and worker cancellation.
- Documentation: 160 local links checked across the nine changed guides/plans;
  `git diff --check` passed. The v0.2.0 plan and package-version files have no
  changes.
- Offline source/wheel build passed the repository's archive, module/resource,
  metadata, entry-point, license, and optional-dependency checks. The wheel was
  built from the source archive. New preference/runtime/onboarding modules and
  their documentation/tests are required by the distribution checker.
- A fresh 80x24 terminal loaded the extracted wheel with existing isolated CLI
  dependencies: onboarding saved a workspace with spaces and Polish characters,
  the managed baseline completed offline with MAE 2.5, suffixless export produced
  16 JSON rows, and a restarted session loaded preferences and saved results
  without repeating onboarding. No cache directory or assistant connection was
  created. This is an extracted-wheel smoke check, not a new dependency install.
- Independent read-only reviews found and resolved two regressions: Skip probing
  unused unwritable cache paths, and execution starting before output path
  validation. Both have retained regression tests.

The build checks use development artifacts at version 0.2.0. They do not replace
published assets or establish v0.3.0 release qualification. S1 acceptance is local;
new clean-install and cross-platform qualification remains part of S9.

## Remaining work and limits

S2-S9 remain open. This stage does not qualify future forecasting, automatic
checkpoint preparation, new assistant providers, new forecasting families,
public documentation deployment, or native Windows support. Windows path and
settings branches have automated tests where applicable; native Windows and the
Linux/macOS Python 3.11-3.13 matrix have not been run for this implementation.
Runtime diagnostics certify metadata compatibility only. Actual optional runtime
imports/loading, real provider requests, and OS-vault behavior retain their
existing qualification limits.

Package version and published v0.2.0 artifacts remain unchanged. Publication,
pull requests, merging, package releases, and native qualification remain separate
steps under the [contribution workflow](../CONTRIBUTING.md).

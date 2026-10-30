# S8 portability implementation and verification

Status, 29 October 2026: **implementation complete; native acceptance pending**.
This record separates local macOS execution, simulated platform cases, live
dependency metadata research and checks that require a native Windows host.
It does not waive the [release plan](release-v0.3.0-plan.md)'s acceptance gates.

## Implementation

- Native drive paths are distinguished from URLs and drive-relative paths.
  Shared Windows component validation rejects alternate streams, device names,
  trailing dots/spaces and invalid filename characters before execution.
- Slash-command dispatch/completion share Windows quoting with literal
  backslashes. Windows redirected command output uses UTF-8.
- Checkpoint provenance uses pure lexical paths for cross-platform reload,
  retaining original strings/fingerprints and previously accepted absolute legacy
  spelling. Actual checkpoint resolution/loading retains native path, symlink,
  file-integrity and revision checks.
- Future metadata uses the same strict parser with a private lexical resolver.
  Its original resolved configuration must compare exactly. The nonserialized
  record-only marker blocks planning, preparation, synchronous and managed runs
  before path/model work. No artifact schema or numerical fingerprint changed.
- MSVCRT locks one byte without a prior write; bounded contention retries and
  cleanup are tested. Microsoft documents
  [locking beyond EOF](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/locking?view=msvc-170),
  so an empty lock file needs no initialization write that could conflict with
  another process's byte lock.
- Verification selects native environment executables. The CI helper builds,
  creates a clean environment and invokes isolated verification with `--wheel`
  for the exact artifact. CI covers Linux/macOS/Windows, Python 3.11–3.13.
- TiRex's Windows candidate constraint/check is execution-only. The global
  catalog's required provenance does not vary with the reader's OS. An optional
  platform-independent `triton-windows` observation can be read on any host and
  is not required in older records.

## Local evidence

Host: `macOS-27.0.1-arm64-arm-64bit-Mach-O`, ARM64, Python 3.13.6, Clang 16.0.0.
Existing environment: NumPy 2.1.3, pandas 2.3.3, pytest 9.0.2, Black 26.1.0,
Ruff 0.15.4, prompt-toolkit 3.0.50, Rich 14.3.2, Typer 0.24.1 and PyYAML 6.0.2.
No dependency installation, weight transfer, real model inference, native vault
operation or provider request was performed for S8.

Executed checks:

- Integrated suite: **3,274 passed, 17 skipped, 2 failed**. Both failures were
  pre-existing S6 TOTO catalog expectations omitting the new adapter. Those two
  expectations were repaired within the current task; their rerun passed.
- Final Windows/path/worker/verifier/diagnostic selection: **133 passed,
  2 skipped**. The skips require actual Windows drive resolution and console
  control events; neither ran on this macOS host.
- Catalog repair checks plus public-documentation regression tests: **34 passed**.
- Independent primary checks: **70 passed, 2 native skips** on each existing
  Python 3.11.13 and 3.12.11 environment, plus **120 passed, 1 native skip** for
  package-build contracts, Windows/path boundaries and public documentation.
  The Windows Triton candidate is declared conditionally in both the optional
  installation extra and constraints; no installation was performed.
- Configured Black/Ruff checks and `git diff --check`: passed. Local documentation
  inspection checked 75 link destinations across seven maintained guides/records.

Commands used the existing `.venv/bin/python`, with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for pytest. The final targeted selection was
`tests/test_portable_paths.py tests/test_windows_boundaries.py
tests/test_windows_console.py tests/test_repl_progress.py tests/test_execution.py
tests/test_guided_execution.py tests/test_distribution_verifier.py
tests/test_runtime_diagnostics.py`. The two repaired tests were rerun together
with `tests/test_public_docs.py`; no redundant full-suite rerun was required.
The CI helper's installation commands were inspected/mocked, not executed; the
fresh installed candidate and exact source/wheel qualification belong to S9.

The focused checks exercise Windows path spelling/quoting,
simulated MSVCRT contention/timeout, forced-stop escalation, read-only execution
boundaries, conditional candidate metadata and CI command construction. They
also reload/export simulated Windows-produced pretrained future records with
foreign local or Hub snapshot paths and optional Triton provenance in a fresh
process that blocks model/kernel/Hub imports. Saved bytes and numerical checkpoint
fingerprints remain unchanged. Legacy checkpoint location spellings are retained.

Windows-only drive-resolution and private-console event cases are skipped on
this macOS host. The existing POSIX terminal tests still execute locally. A fake
process verifies escalation beyond termination; real forced-stop/reaping tests
remain active across platforms, although Windows uses `TerminateProcess` rather
than catchable POSIX `SIGTERM`.

## Verification limits and release gate

No Windows 11, Windows Terminal, PowerShell, Windows Server CI image or native
Windows Python/runtime version was executed here. Their exact versions are
therefore unavailable, not inferred from configured targets. CI remains blocked
by account billing; new matrix entries are configuration, not passed evidence.
Linux/macOS matrix jobs likewise have not qualified this new candidate.

All pretrained Windows installations, native imports, representative CPU inference,
cached replay and distinct profile behaviors remain pending. TOTO's agreed
Python 3.12–3.13 exception does not qualify its runtime. Browser authentication,
Windows Credential Manager persistence and terminal keyboard/scrollback behavior
require actual native checks. No Windows support exclusion or CI waiver was
silently substituted for these acceptance items.

See the [platform matrix, candidate PowerShell commands and native checklist](windows.md).

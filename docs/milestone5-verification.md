# Release-readiness verification record

Date: 25 October 2026. Branch: `build/release-readiness`, based on the unmerged
persistent-terminal branch at `5dca17d`. The owner accepted the terminal and
authorized the next implementation milestone, then separately authorized a clean
temporary core/CLI/Parquet dependency installation. Publication remains outside
that authorization.

## Scope and environment

This milestone prepares source and wheel distributions, installed-package checks,
installation/release/adapter documentation, and configured CI verification. The
interactive workflow, evaluation definitions, supported catalog, and model loading
policy are unchanged. An educational direct-Python adapter is an extension
example, not a newly supported catalog model.

Local execution uses Python 3.13.6, macOS 27.0.1 on arm64, setuptools 80.9.0, and
the repository's development constraints. The owner-authorized dependency download
fetched 18 pinned runtime wheels totaling 53,470,242 bytes. No model checkpoint,
Torch, Chronos, Textual, or GPU runtime was downloaded for the clean environment.

## Verified locally

The clean environment was created without system site packages. Dependencies were
installed from the owner-authorized local wheel directory, with package-index
access disabled during installation and runtime checks.

| Check | Outcome |
| --- | --- |
| Full routine suite | **904 passed, 1 skipped** in 41.96 seconds. The skipped check is the opt-in real Chronos smoke test. |
| Black/Ruff/whitespace | Passed for source, tests, examples, and release scripts. |
| Source/wheel build | A source archive containing the documentation, examples, constraints, test fixtures, and verification tools passed content inspection. pip built the wheel from that archive with no dependency resolution or build isolation. |
| Package metadata | Project name/version, Python requirement, Apache-2.0 license/file, generated `tsfm` entry point, bundled CSV, retained stylesheet, and numerical-core-only dependency boundary passed inspection. |
| Archive guard regressions | Absolute/traversal/backslash paths, including directory members, are rejected. Core dependencies cannot be concealed behind platform markers. Existing output directories are preserved. |
| Core-only installation | The installed numerical package ran both Python examples from the unpacked source archive, outside the checkout, using isolated Python. NumPy/pandas were present; CLI, Parquet, model, and Textual modules were absent. Baseline scores matched the documented arithmetic. |
| Clean CLI/Parquet installation | `pip check` passed. The installed verifier passed all eight phases: command routing, offline catalog, bundled-data arithmetic, both baseline runs, history/reload/comparison, all exports and reloads, overwrite/invalid-config refusal, and missing optional-runtime boundaries. |
| Isolation | The imported module and generated launcher came from the clean environment. Python isolated mode was enabled, system site packages were disabled, and Torch/Chronos/Transformers/Hub/Textual were absent. Source/editable imports and a different launcher's PATH are refused. |
| Installed interactive terminal | An 80×24 controlling pseudo-terminal launched the clean wheel's `tsfm`, confirmed `/demo`, completed four windows/16 cases with MAE 2.5, displayed results, and exited with status 0. No alternate screen was entered. |
| Adapter guide/example | The educational LinearDrift adapter passed five tests covering hand arithmetic, analytical trend/alignment, future-value isolation, pre-load rejection, and standalone execution without optional runtimes. It remains outside the catalog/YAML schema. |
| Documentation/CI | Installation, release, adapter, and methodology links were checked. Shell snippets and all 11 configured CI shell steps passed syntax checks. The CI matrix now includes source-to-wheel rebuilding and a separate clean installed-wheel check. |

Independent review found and repaired archive path/dependency-audit gaps. The
installed positive path also corrected an overly strict verifier assumption about
Rich's help headings; this was a test expectation, not a product-routing defect.
Focused regressions and the final full suite passed. No actionable review findings
remain in the reviewed changes.

## Remaining release gates

Linux/macOS Python 3.11–3.13 CI is configured, with a wheel rebuilt from the source
archive and installed into a separate clean environment. The remote matrix has
not been run here. Windows and other Python/platform combinations remain unverified.
The prior pinned Tiny CPU integration remains historical evidence; this step
does not make a new model-advantage claim.

Local implementation and installation checks are complete. The public-release
milestone remains gated on the unrun platform matrix, reviewed integration of the
prerequisite branches, and explicit owner approval to publish.

Final artifact paths, sizes, SHA-256 identities, and source Git state are emitted by
`scripts/build_distribution.py` and retained with the installed-check JSON outside
the source archive. The source archive includes this document, so storing that
same archive's own hash here would change the artifact being identified.
The final build is repeated from the local commit and its installed checks rerun
against that artifact. An untracked local brief can make the reported Git state
dirty even when task-owned tracked changes are committed; its content is excluded.

The pre-existing untracked `PLAN.md` remains unchanged and excluded from the
distribution. No push, pull request, merge, tag, package upload, or public release
has been performed.

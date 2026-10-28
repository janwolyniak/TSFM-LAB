# v0.3.0 S2 model setup verification

Date: 28 October 2026. Scope: S2 of the
[agreed release plan](release-v0.3.0-plan.md), extending the locally verified
[S1 onboarding](s1-onboarding-verification.md). Branch:
`codex/v0.3.0-model-setup`, based on S1 commit `f35e069`. Integration order is S1,
then S2; no branch has been merged or published as part of this implementation.

## Implemented contract

- Friendly family/generation/variant choices cover all five categories, both
  baselines, and all 13 curated pretrained variants. Model size, training profile,
  and native context/horizon profile remain distinct labels.
- One immutable metadata-only settings description supplies limits, defaults,
  task support, frequency policy, and fixed preprocessing explanations. Manual
  setup, assistant validation, YAML, and Python construction share the contract.
- Context/horizon options precede dataset selection. Strict original-unit data
  preview, measurement declarations, cadence/history checks, historical evaluation
  bounds, final preparation, and explicit confirmation remain required.
- Back navigation preserves deliberate compatible edits while invalidating stale
  checkpoint/data identities and regenerating implicit date bounds. Conflicting
  explicit values remain available for correction. Invalid datasets can be retried.
- Advanced revision selection rejects a known SHA belonging to a different TTM
  native profile; unknown custom revisions remain unverified and require actual
  checkpoint validation. Representative inference evidence remains separate from
  the selected variant's verification label.

## Evidence

Local environment: macOS ARM64, Python 3.13.6, existing dependencies. Tests use tiny
local fixtures and fake model metadata.

- Full suite: **2,702 passed, 14 skipped** in 61.65 seconds with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`.
- Configured checks: Black checked 186 files; Ruff passed across
  `src tests examples scripts`. `git diff --check` passed.
- Model settings, friendly selection, integrated entry-point equivalence, invalid
  data/options, prior terminal navigation, and native-checkpoint regressions pass.
  Invalid seeded stride/MASE values can be corrected at their evaluation questions
  after data inspection. The same valid configuration resolves through manual,
  assisted, YAML, and Python paths for all seven adapters.
- Documentation: 154 local links checked across ten changed documents; the
  metadata-only Python example in the setup guide executed successfully.
- Offline source/wheel build passed the repository's archive, required-module,
  resource, metadata, entry-point, license, and optional-dependency checks. The
  wheel was built from the source archive; its Python files match current source.
- Fresh extracted-wheel 80x24 terminal smoke passed using an existing isolated
  CLI environment: setup skip, friendly baseline selection, context/horizon before
  data, original-unit preview, managed offline historical run, 16-row JSON export,
  Chronos-2 generation/size selection with cancellation, and saved-result reload
  after restart. No model cache was created or inference framework imported.

No new dependency installation, model download, paid API, or live provider login
was used. Development package checks do not replace v0.3.0 release qualification.

Read-only review identified and repaired stale parsed-dataset limits blocking
baseline-to-TTM navigation, a false universal TiRex differencing description, and
contradictory advanced TTM profile/revision selections. Regression tests retain
these cases. Manual/assisted/YAML/Python resolved configurations are compared for
all seven adapters, and preexisting records/finance contracts remain under the
full regression suite.

## Limits and next stage

Four S2 acceptance criteria are locally verified. The combined choice-change
criterion remains open for its future-mode portion: historical category, family,
variant, and dataset changes are covered, while future-mode selection and
mode-change revalidation remain S4 integration work. Future intents continue to
be refused explicitly. Checkpoint preparation/download coordination remains S3. No new
pretrained inference evidence, external provider requests, dependency installation,
Windows-native qualification, or public documentation deployment is claimed.
Runtime and catalog verification limits from v0.2.0 remain in force.

The package version remains 0.2.0 during development. S3-S9 remain pending. See
[the setup guide](model-setup.md) for supported controls and
[CONTRIBUTING.md](../CONTRIBUTING.md) for publication and merge boundaries.

# Development roadmap and acceptance criteria

The first delivery is a small offline numerical experiment. The first product
release is the complete terminal workflow with one pretrained model adapter.
Milestones build on verified behavior; generating module stubs does not satisfy
their acceptance criteria.

## Current state and review decisions

The repository now contains the numerical Python package, scriptable CLI,
YAML configuration, local records, Parquet support, tests, synthetic examples,
and configured checks/CI, plus an interactive terminal. Chronos-Bolt Tiny/Small
and TimesFM 2.5 have passed pinned local CPU integrations on synthetic data.
Chronos-2 Small adds the bounded P2 CPU integration; standard is compatibility
checked without individual inference.
TiRex-2 adds four training profiles through P3, with a verified representative
CPU workflow and shared-Hub regression checks recorded separately.
The pre-existing local `PLAN.md` remains the product brief;
these tracked documents make the actionable proposal usable without relying on
that untracked file.

The owner agreed to the following initial direction on 25 October 2026:

1. Start with a single regular univariate series, explicit timezone/frequency,
   no covariates or imputation, and last-value plus seasonal-naive baselines.
2. Adopt the [core contracts](contracts.md) and
   [evaluation semantics](evaluation.md), including complete horizons, fixed
   pre-evaluation MASE scaling, and equal weighting of forecast cases.
3. Use Python, NumPy, pandas, typed dataclasses, setuptools, pytest, Ruff, and
   Black for the first implementation. Introduce terminal/model dependencies
   later. Dependency installation remains a separate authorization.
4. Use Typer/Rich for shell commands and Textual for the interactive application,
   sharing application functions and eventually one inference worker process.
5. Start persistence with run directories and manifests, deferring SQLite and
   automatic forecast cache lookup.
6. Integrate Chronos-Bolt Tiny on CPU first, followed by an optional larger Bolt
   checkpoint; keep Chronos-2 as a separate adapter and finance outside the MVP.

The contribution workflow requires discussion of architecture, interfaces,
dependencies, and evaluation methodology before implementation. The agreed
proposal establishes the implementation direction. The owner's continuation
requests authorized Milestones 2 through 5; the owner separately authorized the
pinned Tiny download, brief CPU test, isolated UI dependencies, and a clean
core/CLI/Parquet installation check.
Continuation does not by itself authorize
dependency installation, model downloads, publication, merging, or cloud compute.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for the maintained permission and Git
workflow rules.

After inspecting the first terminal implementation, the owner revised its
interaction direction on 25 October 2026: use a persistent prompt with guided
commands, native scrollback, and local deterministic actions, with no AI service.
The active interface now uses prompt_toolkit and Rich. The forecasting contracts
and managed worker remain shared with the shell. See [the rework](terminal-rework.md).

The owner selected TimesFM for the second family and authorized a bounded Small
download/CPU test and an isolated TimesFM runtime installation. The broader
benchmarking compatibility study is authorized in a separate temporary runtime;
the first finance timing, position, and cost rules are now agreed below.
These decisions do not authorize publishing a package release.

On 26 October 2026, the owner selected the next feature-release scope:
additional forecasting families and compatible checkpoint sizes, usage profiles
in the terminal wizard, and optional natural-language setup/explanations through
four explicit LLM connections. The detailed
[next release plan](next-release-plan.md) defines the implementation order and
acceptance criteria. Larger compatible variants need not receive separate real
inference runs; checkpoint compatibility checks and accurate verification status
remain required. Large benchmark comparisons are postponed.

P1's catalog, variant selection, Base metadata, and historical verification
snapshots are implemented and verified locally; see the
[P1 verification record](p1-catalog-verification.md). P2 adds Chronos-2 Small and
standard; see the [P2 verification record](p2-chronos2-verification.md).
P3 implements TiRex-2 with the owner-approved shared Hub 0.35.3 update and NXAI
runtime notices; qualification is recorded in the
[P3 verification record](p3-tirex2-verification.md). The four choices are training
profiles with distinct context/horizon bounds, not model sizes. TTM still needs
the runtime and output-semantic decisions in the [catalog guide](model-catalog.md).

The agreed [v0.3.0 plan](release-v0.3.0-plan.md) covers onboarding, model-specific
setup, integrated checkpoint preparation and outputs, future forecasting,
additional assistant connections and bounded model expansion, public documentation
and GEO, and native Windows 11 x64 support. S1 onboarding is locally verified;
S2 adds [model-specific setup](model-setup.md). See the
[S1 record](s1-onboarding-verification.md) and [S2 record](s2-model-setup-verification.md).
S3 adds [guided preparation and outputs](guided-preparation.md); its
[verification record](s3-preparation-verification.md) retains the pending real
pretrained check. S4-S8 implementation and offline S9 qualification are now
complete; the owner requested review, merge and release publication on 30 October
2026. See the [0.3.0 verification record](v0.3.0-verification.md) for review fixes,
local checks and still-unqualified model/platform/provider combinations.
The v0.2.0 plan and release remain historical and unchanged.

## Milestone 0 Architecture review

Deliver the bounded MVP, stack, contracts, methodology, checkpoint plan, repository
structure, and acceptance criteria. Review the following before closing it:

- [x] Architecture critique and proposed simplifications are documented.
- [x] Dataset, model, forecast, and experiment contracts are specified.
- [x] Temporal and metric definitions include boundary and undefined cases.
- [x] First adapter candidates and official sources are identified.
- [x] Implementation milestones have observable acceptance criteria.
- [x] Owner agrees to the implementation scope and proposed choices.

## Milestone 1 Numerical evaluation core

Implemented and verified locally, with subsequent clean-install and remote
Linux/macOS Python 3.11–3.13 checks. Original milestone records remain historical;
current candidates must pass their own configured checks.

Implement only the Python numerical workflow. No UI, remote models, database,
or domain-specific metrics are needed to pass this milestone.

| Task | Deliverable and required verification |
| --- | --- |
| M1.1 Package and checks | Add `pyproject.toml`, a `src` package, declared core/dev dependencies, and pytest/Ruff/Black configuration. Build and inspect a wheel; document exact setup/check commands. Add lightweight Linux/macOS CI for the supported Python matrix. |
| M1.2 Contracts | Implement validated dataset, evaluation, request, output, model metadata, and result types. Reject invalid integer lengths, unsupported fields, and impossible evaluation bounds. |
| M1.3 CSV import | Explicit column/timezone/frequency mapping, finite numeric values, strict grid validation, and versioned fingerprints. Test duplicates, unordered rows, gaps, timezone ambiguity, and invalid values. |
| M1.4 Windows | Implement fixed-context rolling origins with separate detached requests and evaluator-only truth. Test exact timestamp bounds, window counts, and incomplete horizons. |
| M1.5 Baselines | Last-value and seasonal-naive adapters with the same protocol. Test multiple leads, season wrap, and insufficient context. |
| M1.6 Metrics | Implement pooled and per-lead MAE/RMSE/MASE, explicit scale metadata, and undefined statuses. Check against independent hand calculations. |
| M1.7 Runner | Validate before inference, load once, score aligned outputs, and always release model resources. Test malformed adapters, failed runs, and future-value perturbations. |
| M1.8 Demonstration | Bundle a small original synthetic seasonal CSV with generation formula/provenance and a runnable Python example. Repeated runs reproduce predictions and method metadata. |

Acceptance: a clean local environment can execute the documented example offline
and inspect aligned predictions and scores. Tests, formatting, linting, and wheel
checks pass. CI requires no model downloads, credentials, or GPUs. The package
core imports without any terminal or model runtime. A hand-worked fixture verifies
that reported metrics mean what the documentation says.

## Milestone 2 Shell workflow and experiment records

Implemented and verified locally; see [CLI usage](cli.md) and the
[verification record](milestone2-verification.md). The integrated pre-TimesFM
workflow subsequently passed remote CI and clean installation checks.

Add Typer/Rich, safe YAML configuration, Parquet ingestion/export, UUID run
directories, configuration snapshots, JSON manifests/metrics, prediction export,
history, and compatibility checks for comparisons. Start with synchronous runs.

Acceptance: a user can run a baseline experiment, inspect it from a separate
process, compare two compatible runs, and export/reload predictions without
inference. Relative paths resolve against the config file. Invalid configs fail
before work begins. JSON is parseable without terminal decoration, and automated
commands never prompt. Atomic-write and failure-injection tests show that partial
runs do not appear completed. Package installation exposes the `tsfm` command.

## Milestone 3 First pretrained adapter

Implemented and verified with the pinned Tiny checkpoint on CPU. Routine tests,
cached replay, local copy, installed CLI, and the marked real smoke test passed.
See [the checkpoint guide](chronos.md) and
[verification record](milestone3-verification.md). Small was subsequently verified
on CPU; see the [expanded record](milestone6-verification.md). Other devices remain
unverified. Routine integration checks subsequently passed remote CI.

Add a small explicit model catalog, optional Chronos dependencies, metadata-first
validation, Hub revision resolution, explicit model pulls, cache reuse, offline
loading, and the Chronos-Bolt adapter. Verify version-specific output shapes,
median point semantics, safe loading, context/horizon limits, and CPU behavior.

The current tiny checkpoint configuration specifies context length 2048 and
native prediction length 64. The first adapter should reject longer requests
until horizon extension is specified and tested, and reject excess context
rather than relying on silent truncation.
[Official checkpoint configuration](https://huggingface.co/amazon/chronos-bolt-tiny/blob/main/config.json).
Recheck these limits at the resolved checkpoint revision during implementation.

Acceptance: after an explicitly authorized download, a user evaluates the exact
resolved checkpoint on local synthetic data. A second run uses cached files and
works offline. Mocked Hub/adapter tests cover moving revisions, missing files,
unsupported architectures, and missing optional dependencies. A separately
tested local checkpoint path loads without network access; its file manifest
changes identity when a file changes. A separately
marked real CPU smoke test records checkpoint SHA, dependency versions, device,
shapes, and finite aligned outputs. This proves integration, not model advantage.

## Milestone 4 Interactive terminal

Implemented and verified locally. See the [keyboard guide](tui.md) and
[original verification record](milestone4-verification.md), followed by the
[persistent-prompt revision](terminal-rework.md). The integrated workflow
subsequently passed the Linux/macOS Python 3.11–3.13 matrix; Windows remains
unverified.

Add model/dataset selection, a validated experiment wizard, progress and errors,
results, history, comparison, and cancellation. Use the same configuration objects
and application functions as shell commands. Introduce one managed inference
process and persisted stage transitions when needed for responsiveness.

Acceptance: a user completes the documented synthetic baseline workflow using
only keyboard commands and inline questions. Commands and results stay in native
terminal scrollback with no alternate-screen takeover. Invalid selections block submission with actionable
errors. Headless interaction tests cover navigation, execution, failure, and
cancellation. Inspect rendered terminal screens at common sizes, including
80 by 24. Both terminal modes produce equivalent numerical results; a cancelled
or crashed process never becomes a completed experiment.

## Milestone 5 First public release

Local release-candidate preparation is implemented and verified. See the
[installation guide](installation.md), [release workflow](releasing.md),
[adapter extension guide](adapters.md), and
[verification record](milestone5-verification.md).

- [x] Source archive includes reproducible examples, documentation, and fixtures.
- [x] Wheel is built from the source archive and its metadata/resources checked.
- [x] Clean core/CLI/Parquet installation passes locally on Python 3.13 macOS.
- [x] Installation, methodology, model/license, and adapter guidance are available.
- [x] Clean installed-wheel verification is configured in the CI matrix.
- [x] Original core/CLI/Bolt/terminal/release-tooling branches are integrated.
- [x] Publication is authorized for v0.2.0 with hosted CI skipped.
- [x] Final release metadata selects v0.2.0; PyPI publication remains separate.

The Linux/macOS Python 3.11–3.13 matrix remains configured in
[GitHub Actions](https://github.com/janwolyniak/TSFM-LAB/actions/workflows/checks.yml).
The owner explicitly waived new hosted CI for v0.2.0 publication; current release
evidence and unverified platform limits are recorded in the
[v0.2.0 verification record](v0.2.0-verification.md).

Complete installation and methodology documentation, a baseline demo and pinned
Chronos example, optional dependency instructions, model/license metadata, CI,
and adapter extension guidance. Keep downloads out of routine checks and results
claims within the recorded evidence.

Acceptance: an external user can install a built package, run the baseline demo
offline, explicitly pull a checkpoint, run a saved config, and reload/export its
results. Verify the clean-install path on the advertised platform/Python matrix.
Publish only within owner authorization and the contribution workflow.

## Milestone 6 Second model family and expanded CPU verification

The owner selected TimesFM, following the original product brief. The bounded
adapter is `timesfm_2p5`: the explicit 2.5 Torch model through runtime 3.0.2,
CPU/float32, context up to 16,256, horizon up to 128, median point forecasts only.
The [model guide](timesfm.md) and [verification record](milestone6-verification.md)
state exact identities and evidence.

- [x] Add per-family catalog/config/checkpoint selection without changing old
  Bolt record fingerprints or enabling arbitrary repositories.
- [x] Implement lazy optional loading, native-bound refusal, true median output,
  recorded forecast settings, and cleanup after failures.
- [x] Integrate shell/guided selection and preserve no-runtime saved-result
  reload, controlled comparison, and export.
- [x] Verify real cached TimesFM CPU inference/replay and the approved Small
  workflow, without a real-data forecasting advantage claim.
- [x] Complete routine, package, and installed-wheel checks for the preparation tree.
- [x] Review/integrate this milestone and verify its current remote CI:
  [PR #2](https://github.com/janwolyniak/TSFM-LAB/pull/2), merged `b8d5647`,
  [six-job post-merge run](https://github.com/janwolyniak/TSFM-LAB/actions/runs/40240609625).

Acceptance: a baseline and TimesFM use identical forecast cases and metric
definitions; saved records remain usable without either model's weights/runtime;
wrong-family identities, unsupported limits, unsafe files, and unverified runtime
versions fail explicitly. Routine CI needs no weights. Larger device/model claims
require separate verification. The subsequent P2 integration adds Chronos-2 as
a distinct adapter without changing the TimesFM contract.

## Milestone 7 Explicit OHLC finance reports

The owner agreed to the [finance contract](finance-extension.md) before coding:
positive close-price targets, lead 1 and stride 1; signal from forecasted close
relative to origin close; next-bar open entry and that bar's close exit;
unit long/flat/short exposure with full liquidation every bar. Capital, threshold,
fees, slippage, per-held-bar short financing, and signal latency are required
inputs. Explicit open/close timestamps must permit entry strictly after the
origin close plus latency. Continuous touching bars are ineligible even at zero
latency.

The separate extension consumes validated saved forecasts and exact original
prices without inference. It preserves core forecast metrics, adds close-return
and three-way direction diagnostics, and reconstructs strategy/cash/costed
buy-and-hold equity, return, period-end drawdown, turnover, and sample volatility.
Checked self-contained JSON reports support independent reload and CSV export.
See the [verification record](milestone7-verification.md) for current evidence
and publication status. Live orders, margin simulation, irregular trading
calendars, and selection/parameter tuning remain outside this contract.

## Benchmarking compatibility study

The bounded [fev compatibility study](benchmarking-extension.md) was completed.
Its isolated prototype records where contracts agree or differ; it does not add
a mandatory dependency or replace the numerical scorer. Production integration
and larger benchmark comparisons are deferred from the next feature release.

## Later work

Forecast accuracy and trading performance remain different evaluations. A second
domain must remain possible without importing finance into the numerical core.
The bounded finance simulator does not establish a profitable trading rule or
real-data generalization. Further execution policies require separate agreement.

SQLite indexing, resumability, forecast caches, environment isolation, advanced
terminal charts, and performance telemetry require a demonstrated need.
No such convenience feature is added merely to populate this list.
Training, fine-tuning, web infrastructure, and distributed compute are outside
this roadmap.

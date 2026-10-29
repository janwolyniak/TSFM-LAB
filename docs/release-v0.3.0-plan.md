# v0.3.0 release plan

Status: **S1-S5 implemented and locally verified through 29 October 2026;
S3 real pretrained qualification remains pending**. S4 closes S2 mode-switching
acceptance; S6-S9 remain open. Additional connection evidence is retained in the
[S5 verification record](s5-connections-verification.md).
Scope was agreed with the owner on 26 October 2026. Acceptance items remain open
until verified; see the [S1 verification record](s1-onboarding-verification.md)
and [S2 verification record](s2-model-setup-verification.md), followed by the
[S3 verification record](s3-preparation-verification.md).
The [S4 contract](future-forecast-contract.md) completed source and independent
review and was approved for implementation on 29 October 2026. See the
[S4 verification record](s4-future-verification.md) for executed evidence and limits.

Release goal: a new user can go from first launch to a usable forecast file,
appropriate results report, and reproducible saved experiment without leaving
the terminal.

## Relationship to v0.2.0

The [v0.2.0 plan](next-release-plan.md) remains unchanged. Its five forecasting
families, five usage profiles, four assistant connections, natural-language
drafting, and grounded explanations are the foundation for this release.
The v0.2.0 implementation is now integrated and released; its
[verification limits](v0.2.0-verification.md) still apply.

v0.3.0 integrates onboarding, model-specific setup, approved automatic checkpoint
preparation, clearer outputs, and future forecasting. It also adds assistant
connections, a bounded expansion of forecasting models, public project
documentation for AI discovery and visibility (GEO), and native Windows support
alongside Linux and macOS. Windows support is a release target, not a claim about
the current implementation.

This document records scope and implementation gates. The owner authorized
starting implementation on 27 October 2026. This does not change the package
version or authorize installations,
downloads, paid requests, publication, or changes to account settings. Follow
the [contribution workflow](../CONTRIBUTING.md) for those actions.

## Intended user journey

First launch offers workspace, results, and model-cache directory selection;
runtime checks; optional assistant connection setup; saved preferences; and an
optional bundled demonstration. Users can skip assistant setup and forecast
without API keys. `/setup` revisits onboarding; `/settings` edits preferences.
These commands are implemented in the S1 development checkout; they are absent
from the published v0.2.0 package.

The `/new` flow is:

1. Choose Generic, Finance, Energy, Temperature, or Weather.
2. Choose a forecasting family and its generation where applicable.
3. Choose a supported checkpoint variant: size, training profile, or native
   context/horizon profile, according to that family.
4. Configure applicable model options, including context, horizon, and supported
   preprocessing. Distinguish editable, required/fixed, and unsupported settings.
5. Link a local dataset and review its preview, timestamp/target columns,
   timezone, frequency, units, and profile-specific measurement declarations.
6. Choose historical evaluation or future forecasting, then resolve the
   evaluation bounds or forecast origin. Revalidate model settings against data.
7. Review the final configuration, checkpoint identity, cache status, missing
   requirements, required downloads, and output location.
8. Confirm the displayed download and run, follow live stages, and cancel if
   needed. Reuse already verified cached weights.
9. Receive a forecast file, chart, reproducibility information, and the
   appropriate evaluation summary. Optionally request an assistant explanation.

Manual and natural-language setup resolve through the same workflow. The
forecasting model and assistant model remain separate selections. Back navigation
preserves compatible user choices and requires review of any invalidated choice.
Keep the persistent prompt, keyboard operation, and native terminal scrollback.

## Contracts and boundaries

- Extend the existing [application boundaries](architecture.md), managed worker,
  checkpoint cache, [catalog](model-catalog.md), and [usage profiles](usage-profiles.md).
  Keep preparation, run, and export coordination in shared application services;
  the terminal renders their progress. No generic environment manager is implied.
- Model controls must describe the selected variant's real capabilities.
  TTM's exact context and required scaling are examples of fixed requirements,
  not universal toggles. A generation change can require a distinct adapter.
- Curate supported Hugging Face checkpoints. Hub discoverability alone does not
  establish adapter compatibility. Preserve safe serialization, exact revisions,
  strict loading, truthful verification labels, and disabled remote code.
- Check runtime availability and compatibility before a large weight download.
  Runtime installation remains explicit and separate from downloading weights.
  Private/gated Hub authentication is not added implicitly.
- Validate data before expensive work. Fit preprocessing only on permitted
  history, invert transformations to original units, and record all numerical
  choices in identity/provenance and controlled-comparison eligibility.
  No hidden imputation, resampling, clipping, or unit conversion is introduced.
- Keep the single regular series and UTC/fixed-frequency contract. Historical
  evaluation retains the [existing metric definitions](evaluation.md).
  Future inference receives only observations available at its declared origin.
- Preserve old configurations, fingerprints, forecast semantics, saved records,
  comparisons, and finance reports. Future forecasts require a separate,
  explicitly versioned record contract; do not weaken historical validators.
- Reuse v0.2.0's credential vault, outgoing-context controls, request bounds,
  and validated assistant drafts. Assistants do not gain authority to start
  downloads/runs, execute code, or overwrite numerical results.
- Retain the [finance contract](finance-extension.md), including task eligibility
  and explicit execution assumptions. This release adds no trading policy.

## Implementation stages and acceptance criteria

Each stage needs a focused implementation and retained verification evidence.
Use small offline fixtures and fake transports for routine checks. Real runtime,
checkpoint, and connection checks follow the contribution permission boundaries.
Prior evidence is reusable only when the relevant implementation and identities
are unchanged.

| Stage | Deliverable | Dependency |
| --- | --- | --- |
| S1 | First-launch onboarding and persistent preferences | v0.2.0 settings/connection contracts available before integration |
| S2 | Model-specific `/new` workflow | S1 and the existing catalog/profile contracts |
| S3 | Integrated preparation, live tracking, and results handoff | S2 |
| S4 | Future forecasting alongside historical evaluation | S2/S3 and the approved future-record contract |
| S5 | DeepSeek, local Ollama, and compatible endpoint connections | v0.2.0 connection boundary; S1/S2 for terminal integration |
| S6 | More existing-family variants and one additional family | Compatibility decisions first; S2/S3 for integration |
| S7 | Public documentation and GEO preparation | Can proceed alongside implementation; final examples depend on verified behavior |
| S8 | Native Windows compatibility and verification | Portability audit can start early; S1-S6 for integrated workflow checks |
| S9 | Release qualification | S1-S8 accepted |

### S1: First-launch onboarding

Implement directory selection, runtime diagnostics, optional assistant setup,
saved defaults, and an optional bundled demonstration. Keep nonsecret preferences
separate from credentials and experiment configuration. Directory changes apply
to subsequent work; they do not silently move existing results or caches.

Acceptance:

- [x] A fresh interactive installation offers onboarding once and permits skip,
  cancellation, and later completion without leaving broken preferences.
- [x] Workspace, results/cache locations, category, and export preferences survive
  restart; unwritable paths produce actionable errors before running work.
- [x] `/setup` and `/settings` revisit choices. Explicit command/configuration
  options override defaults; scripts never enter interactive onboarding.
- [x] Connection setup reuses the v0.2.0 credential boundary. No key is required
  for manual forecasting and none enters preferences, run artifacts, or logs.
- [x] Runtime checks identify missing/incompatible optional packages without
  installing them. The bundled baseline demonstration runs offline.

### S2: Model-specific experiment setup

Implement the ordered `/new` journey above. Display friendly family, generation,
and variant names; expose repository/revision details through an advanced view.
Use one validated settings description per adapter/variant across manual setup,
assistant drafting, YAML, and Python entry points.

Acceptance:

- [x] All five categories, both baselines, and supported pretrained choices remain
  selectable. No training profile is presented as a model size.
- [x] Editable options show valid bounds/defaults; fixed preprocessing and native
  requirements are explained; unsupported settings cannot become executable.
- [x] Dataset preview and measurement questions preserve declared units. Invalid
  timestamps, insufficient history, unsupported cadence, or impossible horizons
  block execution before downloads or model allocation.
- [x] Changing category, family, variant, dataset, or mode revalidates dependent
  choices. Back/cancel paths preserve compatible edits without stale settings.
- [x] Equivalent manual and assisted choices resolve to the same validated
  configuration and numerical identity. Existing YAML remains readable.

Historical choice changes are verified in the S2 record. S4 verifies future-mode
switching, fresh review/consent and preservation of compatible choices.

### S3: Automatic preparation, progress, and outputs

Before final confirmation, validate the configuration, resolve the exact revision,
inspect the cache, and show the download identity and size when known. After
confirmation, coordinate missing-file download, weight verification, data
preparation, model loading, forecasting, applicable evaluation, and atomic result
publication. Download and load the reviewed revision rather than resolving a
moving branch again.
The guided action explicitly authorizes its displayed download and run. Preserve
the standalone pull command and the existing scriptable no-download default.

Acceptance:

- [ ] A supported model with an installed runtime and empty cache completes the
  guided path after download/run confirmation. Declining makes no weight download
  or inference request; missing runtimes are reported before weight transfer.
- [x] A second run reuses verified cached files. Offline cached execution works
  without metadata resolution; absent files fail clearly without network access.
- [x] The terminal shows cache/download/verification/preparation/loading/forecast/
  evaluation/saving stages, elapsed time, bytes when measurable, and completed
  forecast windows. Unknown durations are not shown as invented percentages.
- [x] Cancellation and injected failures leave truthful stage/status records,
  terminate managed work, and never publish an incomplete run as completed.
  Retry is explicit; cache reuse rechecks file integrity.
- [x] Preprocessing uses historical input only, reverses to original units, and
  records its policy. Future-value perturbation checks prove isolation.
- [x] Completion surfaces exact paths to the forecast CSV, machine-readable
  results/provenance, concise evaluation summary, and saved chart. Existing
  JSON/Parquet export remains available where its optional support is installed.
- [x] Charts show observed/predicted values with timestamps, units, and clear
  evaluation boundaries; missing metrics include reasons. Saved outputs reload
  without model weights, assistant packages, or rerunning inference.

Checked S3 items are verified with local fixtures, fake transports, and real
offline baseline workers. The first item remains open for an authorized real
pretrained download/runtime/inference check; see the S3 verification record.

### S4: Future forecasting

Add an explicit future mode for one regular series, forecasting beyond its last
observed timestamp. Specify and review its request/output/record contract before
coding: origin, historical cutoff, future timestamps, variant/settings identity,
declared point semantics, artifact validation, and metric-availability status.

Acceptance:

- [x] Both modes are available in guided and assisted setup and have equivalent
  scriptable configuration. Historical runs retain their prior score semantics.
- [x] Future mode produces exactly the requested supported horizon on the declared
  timestamp grid, using only observations available at the origin. Unsupported
  horizons or required unavailable inputs fail before inference.
- [x] Future records/CSV/chart distinguish observations from predictions and retain
  original units, checkpoint/runtime provenance, and preprocessing policy.
- [x] Accuracy is explicitly unavailable because future actuals are absent;
  no zeros, invented actuals, historical scores, or financial returns substitute
  for unavailable future evaluation.
- [x] Historical comparison and finance validators refuse ineligible future
  records. Old artifacts remain readable without migration or rewriting.

Importing actual observations and scoring saved future forecasts later is
deferred; it requires a separately agreed matching and revision policy.
Checked items use offline fixtures, fake pretrained/provider transports and real
baseline workers/terminal execution. Optional Textual screen execution, real
pretrained/provider and platform qualification limits remain recorded in S4/S9.

### S5: Additional assistant connections

Add DeepSeek API-key support, local Ollama, and a configurable OpenAI-compatible
endpoint. Retain the four v0.2.0 routes and their distinct authentication and
usage semantics. Compatibility must be demonstrated per supported endpoint;
sharing a request format does not guarantee identical capabilities.

Acceptance:

- [x] Users can add/select/replace/disconnect each route and select an applicable
  assistant model. Show destination and local/cloud usage before a request.
- [x] Provider/endpoint capabilities govern request fields, streaming completion,
  model discovery, and structured draft validation. Invalid or incomplete model
  output cannot bypass the ordinary configuration and execution boundary.
- [x] Endpoint switches never forward another connection's credentials or silently
  fall back to paid services. Errors, logs, settings, and exports redact secrets.
- [x] Fake-provider checks cover drafting/explanations, malformed/truncated output,
  unsupported fields, authentication/quota failures, timeout, and cancellation.
- [x] The local Ollama path uses an already installed local server/model, makes
  no implicit cloud fallback or model pull, and needs no cloud key. Manual use
  remains available without any server or assistant connection.

### S6: Forecasting model expansion

Existing-family implementation is retained in the
[TTM expansion verification record](s6-ttm-verification.md). The owner approved
TOTO 2 on Python 3.12-3.13 on 29 October 2026; the application keeps Python 3.11
support. Real inference is deferred at the owner's request: use offline tests
without installing optional runtimes or downloading weights.

Prioritize additional useful variants of existing families, particularly TTM
R2.1 daily/weekly profiles and additional supported native horizons. Add **one**
new family, choosing between Moirai and TOTO after a bounded compatibility study.
The chosen generation, checkpoints, runtime pins, scaling/output semantics,
resource needs, and licensing remain decision gates before adapter work.
If neither candidate fits, request a scope decision rather than silently
substituting another family or introducing an environment manager.

Acceptance:

- [ ] A retained decision record selects exact existing-family additions and one
  new family, with official model/config/runtime/license references and explicit
  supported frequency/context/horizon and output semantics.
- [ ] Runtime compatibility is settled before implementation. Dependency changes
  include existing-family regression checks and optional-import isolation.
- [ ] Each new integration/profile behavior has contract tests and representative
  bounded real inference evidence; materially different architectures/behaviors
  are not treated as untested larger sizes of an unchanged adapter.
- [ ] Compatible larger variants retain separate compatibility and real-inference
  status under the existing catalog policy; no inherited accuracy, resource,
  or maintainer-verification claim is fabricated.
- [ ] New choices work through the wizard, configuration, safe loading, progress,
  cancellation, original-unit outputs, and saved-result reload/export. Applicable
  future-mode support is verified explicitly; unsupported combinations are shown.

TimesFM 3.0 is not a committed addition. Its downloaded-weight terms differ from
2.5; Moirai generations also require checkpoint-specific license review. Preserve
the distinction between project-code, runtime, and model-weight licenses.

### S7: Public documentation and GEO

Prepare public, indexable documentation that accurately explains what TSFM-LAB
does, how to install/use it, and which model workflows are supported. Public
static documentation is in scope; the forecasting application remains local.
Choose the documentation host/public URL before deployment. Preparing this
material does not make a private repository or dataset public.

Acceptance:

- [ ] Documentation includes a clear overview, installation/first-run quickstart,
  model/variant capability matrix, examples for all five categories, historical
  versus future semantics, limitations, release notes, and citation information.
- [ ] Examples match the qualified release; stable URLs, internal links, metadata,
  and any structured data agree with visible content. Core information is text
  accessible without running the forecasting app or an assistant.
- [ ] A publication-ready static artifact has checked links, crawl/index settings,
  and no credentials, private paths, datasets, or unapproved experiment outputs.
- [ ] The verification record distinguishes locally checked documentation from
  authorized publication and subsequent public reachability/indexability checks.
  Search indexing and AI citations are not guaranteed acceptance outcomes.

An optional `llms.txt` may aid navigation but is not a release requirement or a
substitute for crawlable, useful documentation. Public checks follow publication
authorization under the existing contribution workflow.

### S8: Native Windows compatibility

Target native Windows 11 x64 with Python 3.11-3.13, alongside the existing Linux
and macOS targets. Users must be able to install and run the application from
PowerShell in Windows Terminal without WSL or a Unix shell. Keep CPU-first
verification; Windows ARM and GPU qualification are separate future work.

Start with a portability audit of packaging, optional model runtimes, filesystem
operations, terminal input/rendering, worker startup/cancellation, credential
storage, and browser-based authentication. Record a per-platform support matrix
before declaring optional forecasting families or assistant routes supported.
An unavailable native dependency is a blocker requiring an explicit support/scope
decision, rather than an automatic WSL fallback or silent removal of a family.

Acceptance:

- [ ] A clean native Windows installation completes onboarding, the offline
  baseline demonstration, historical evaluation, future forecasting, saved-result
  reload, comparison where eligible, and CSV/JSON/optional Parquet export.
- [ ] `/new`, back/cancel navigation, completion, native scrollback, Unicode text,
  and live progress work in PowerShell/Windows Terminal. Redirected commands stay
  noninteractive; Ctrl+C and worker failures leave no orphan inference process or
  falsely completed run. Native terminal checks supplement automated coverage.
- [ ] Workspace/cache/output settings, relative and absolute paths, drive letters,
  spaces and non-ASCII characters, file handles/locking, and atomic publication
  pass native checks. Existing results are never overwritten or migrated silently.
- [ ] Optional runtime installation is audited for every curated family. Each
  advertised Windows integration has a compatible pinned runtime and recorded
  representative CPU inference/cache-replay evidence; larger variants retain
  their truthful individual verification status. Unresolved platform exclusions
  require an owner decision before release and explicit catalog/documentation labels.
- [ ] Available assistant routes pass native connection-management checks,
  including protected credential persistence or explicit session-only fallback,
  browser sign-in where applicable, cancellation, and secret redaction. Routine
  checks use fake providers; real connection checks remain separately authorized.
- [ ] CI includes native Windows jobs for Python 3.11-3.13, covering configured
  lint/format checks, applicable tests, source/wheel builds, and a clean installed
  baseline workflow. Replace POSIX-only shell assumptions; platform-specific test
  exclusions need documented reasons and equivalent Windows coverage.
- [ ] Installation, terminal, model, and release guides include verified PowerShell
  commands, platform-specific prerequisites, and the support matrix. Record exact
  Windows/Python/terminal/runtime versions and retained verification evidence;
  Linux/macOS checks continue to pass.

### S9: Release qualification

- [ ] A fresh installed application completes onboarding, a manual baseline,
  historical evaluation, and future forecasting with clearly located outputs.
- [ ] An authorized representative pretrained run covers missing-weight download,
  cached repeat, live stages, cancellation/failure recovery, and artifact reload.
- [ ] Offline fixtures cover every category, supported option boundary, new
  connection, and mode; old configuration/result/finance regressions pass.
- [ ] Targeted and integrated tests, configured Black/Ruff checks, source/wheel
  contents, installed workflows, and the Linux/macOS/native-Windows Python
  3.11-3.13 CI matrix pass for the exact release candidate. Record source/artifact
  identities, platform support decisions, and verification limits.
- [ ] Guides, examples, licenses/notices, and GEO artifacts match verified behavior;
  all S1-S8 decision gates and acceptance items have supporting evidence.
- [ ] The final release follows the [release workflow](releasing.md). Versioning,
  package/repository publication, and public documentation deployment occur only
  within their authorized scope and are recorded separately from implementation.

## Deferred work

Keep large comparative benchmark campaigns, accuracy rankings, training and
fine-tuning, multi-series panels, covariates, live feeds, irregular calendars,
new probability-scoring contracts, automatic model optimization, autonomous
coding/shell agents, hosted application infrastructure, and distributed compute
outside this release. Do not add automatic dependency installation, arbitrary
Hub execution, private/gated Hub authentication, general run resumability, or
GPU/MPS certification incidentally. Existing finance policies remain unchanged.

## Primary references for implementation decisions

Recheck upstream contracts and exact revisions when implementing; these links
support planning and do not establish project integration or forecast advantage.

- [DeepSeek API](https://api-docs.deepseek.com/) and
  [Ollama API compatibility](https://docs.ollama.com/api/openai-compatibility).
- [IBM TTM model profiles](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2).
- [Moirai / Uni2TS](https://github.com/SalesforceAIResearch/uni2ts),
  [Moirai 2.0 model card](https://huggingface.co/Salesforce/moirai-2.0-R-small), and
  [TOTO](https://github.com/datadog/toto).
- [TimesFM generations and license notice](https://github.com/google-research/timesfm#license-notice-for-pretrained-weights).
- [Google guidance on AI search features](https://developers.google.com/search/docs/appearance/ai-features).

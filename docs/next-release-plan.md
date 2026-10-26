# Next feature release plan

Working target: **v0.2.0**, following the prepared v0.1.0 release.
Scope agreed with the owner on 26 October 2026. P1 and P2 are implemented and verified
locally. P3 TiRex-2, P4 TTM, and P5 usage profiles are implemented and verified locally;
P6–P8 remain
future work. The existing package version and publication
status remain unchanged.

P1 behavior is described in the [catalog guide](model-catalog.md), with checks
recorded in the [P1 verification record](p1-catalog-verification.md).
Chronos-2 behavior and checks are documented in its [guide](chronos2.md) and
[P2 verification record](p2-chronos2-verification.md).
P3's [guide](tirex2.md) and [verification record](p3-tirex2-verification.md)
document its four training profiles and completed checks. The owner
approved a shared Hub 0.35.3 pin, existing-family CPU regression checks, isolated
runtime setup, and explicit NXAI runtime notices; no per-family environment
manager is introduced.
Individual release criteria below stay open until the corresponding implementation is
verified.

The release will let a user start an experiment in the terminal, choose its
usage profile and forecasting model, and complete setup either through guided
questions or by describing the goal in natural language. Optional language-model
connections will also explain experiments, results, and model differences.
Large benchmark comparisons are postponed.

## Terminal experience

The active prompt-based terminal remains the main interface. Its existing
scrollback, explicit commands, managed inference process, and cancellation
behavior stay in place. The shell and guided terminal use the same validated
configuration and application services.

The proposed `/new` sequence is:

1. Choose guided setup or describe the goal in natural language.
2. Choose or confirm the usage profile.
3. Select the local dataset and confirm columns, frequency, timezone, and units.
4. Choose or confirm the forecasting family and checkpoint variant.
5. Set evaluation bounds, context, horizon, stride, and applicable profile options.
6. Review all resolved choices, missing requirements, and checkpoint status.
7. Confirm execution through the existing run confirmation.
8. Inspect forecast results and any explicitly requested domain report; optionally
   ask for an explanation.

Natural-language setup pre-fills this same workflow. Every choice remains
editable, and manual setup works without any LLM connection. Forecasting model
selection and assistant model selection are separate controls.
The existing last-value and seasonal-naive baselines remain selectable alongside
the five pretrained families.

## Forecasting families and checkpoint choices

| Family | Release choices and adapter work | Representative real inference check |
| --- | --- | --- |
| Chronos-Bolt | Preserve Tiny and Small; add Base to the curated catalog using the existing compatible adapter. | Reuse the verified Tiny/Small evidence for the unchanged adapter; rerun affected checks if its implementation changes. Base needs no separate inference run solely for catalog inclusion. |
| TimesFM | Preserve the existing TimesFM 2.5 200M adapter and checkpoint. Add further sizes only when they actually share its supported architecture and runtime contract. | Reuse current evidence for the unchanged adapter. |
| Chronos-2 | Add a separate adapter; offer Small and the standard checkpoint. Begin with the existing univariate evaluation workflow. | Use the smaller supported checkpoint for the bounded CPU check. |
| TiRex-2 | Add the separate CPU adapter and four official training profiles: default, FEV benchmark, GiftEval zero-shot, and GiftEval pretrain. Context/horizon bounds are profile specific. | Use the default 82.5M checkpoint as the representative check; other profiles remain individually untested unless evidence says otherwise. These are not size tiers. |
| IBM Granite TTM | Add an adapter with explicit context, horizon, frequency, and preprocessing profiles for curated variants. | Use one small representative variant; add compatible variants after the checks below. |

Official references: [Bolt Base](https://huggingface.co/amazon/chronos-bolt-base),
[Chronos-2](https://huggingface.co/amazon/chronos-2),
[Chronos-2 Small](https://huggingface.co/autogluon/chronos-2-small),
[TiRex-2](https://huggingface.co/NX-AI/TiRex-2), and
[TTM](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2).

### Verification policy for larger variants

A compatible larger checkpoint can become selectable without a separate real
inference run. Catalog and loader checks must still establish its architecture,
runtime interface, configuration, native limits, safe serialization, license,
and exact revision. Parameter count alone does not establish compatibility.

Keep two distinct facts visible in model information, setup, and saved records:

- The adapter's representative verification record identifies the checkpoint,
  revision, runtime, device, dtype, and checks actually executed.
- Each selected checkpoint records its own compatibility checks and whether
  real inference was run for that checkpoint. Larger variants can display
  **Metadata/API compatibility checked; individual inference not run**.

Do not copy a representative checkpoint's inference result onto another size.
Existing verified checkpoints retain their evidence. Metadata must describe each
variant's actual context, horizon, runtime requirements, download size when
known, and cache availability. Weight-file size is not a runtime RAM measurement.
Normal execution still checks the selected files' identities and loads weights
strictly on first use; representative testing does not bypass those safeguards.

TimesFM 2.0 and 3.0 are separate generations, rather than size choices covered by
the current 2.5 integration. They are outside this release's committed adapter
scope; the [official TimesFM repository](https://github.com/google-research/timesfm)
also documents different licensing for the downloaded 3.0 weights.

### Adapter contracts

Resolve and pin each new optional runtime before adapter implementation. Inspect
compatibility with the supported Python/PyTorch environment without silently
upgrading shared dependencies. If runtimes need separate environments, settle
that execution design before introducing an environment manager.

New adapters must preserve causal context-only input, native capability checks,
safe cache/local loading, resource cleanup, exact checkpoint identity, and
no-runtime saved-result reload/export. Initial integrations are univariate and
CPU-first. Multivariate and covariate features advertised upstream are not
implicitly added to the product.

P4 completed the following contract step before its adapter was coded, under
the [approved TTM contract](ttm-contract.md):

- Select and pin release profile, training loss, context, horizon, and frequency
  variants explicitly; prevent automatic substitution, padding, rolling horizon
  extension, or random initialization.
- Specify causal standardization and inverse transformation to original units.
  Future observations must not affect an earlier forecast or its scaler.
- Add a backward-compatible native-point task if required by the selected
  checkpoint. Record its actual output semantics and training-loss metadata;
  never relabel an MSE-trained point output as a median or a verified mean.
  Existing median tasks and records retain their meaning, and controlled
  comparisons must respect the chosen task.

See the [TTM usage contract](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2#usage-guidelines)
and [official prediction implementation](https://github.com/ibm-granite/granite-tsfm/blob/main/tsfm_public/models/tinytimemixer/modeling_tinytimemixer.py).

## Usage profiles

Profiles provide terminology, examples, setup questions, validation, and result
presentation. They do not change the numerical definitions of existing scores
or silently transform data.

| Profile | Guided workflow and reporting |
| --- | --- |
| Generic | Preserve the current domain-independent numeric-series workflow. |
| Finance | Identify the target and units. Offer the existing OHLC report only for an explicitly declared positive close-price target, with separate execution data and all timing/cost inputs. Forecast-only use remains available. |
| Energy | Ask whether the target represents power, energy, or another measurement; preserve its declared units and interval. Offer suitable seasonal examples and errors in those units. |
| Temperature | Preserve Celsius/Kelvin/other declared units and legitimate negative readings. Provide temperature-specific wording and seasonal examples. |
| Weather | Evaluate one station-level measured variable, such as wind speed, rainfall, or humidity, with explicit units and measurement semantics. Temperature remains its own convenient preset. |

All profiles initially use one regular series under the current UTC/fixed-frequency
contract. No hidden resampling, imputation, clipping, unit conversion, or inferred
trading rules is introduced. Percentage-error metrics and new risk formulas are
not added incidentally. Existing finance timing and accounting stay governed by
the [finance contract](finance-extension.md).

Persist the selected profile and its resolved options as validated workflow
metadata. Profile labels alone must not change dataset fingerprints, forecast
scores, or comparison eligibility. Existing configurations and saved records
without a profile display as Generic without rewriting their files.
Any actual change to the target, preprocessing, task, or evaluation policy must
remain explicit in its numerical identity.

This release evaluates historical forecasts with observed targets. A request
such as "forecast next week" must clarify the intended historical evaluation or
explain that future targets without actuals need a separate prospective-forecast
feature. Weather grids, live feeds, multi-series panels, and future covariates
are outside this release.

## Optional natural language assistance

The assistant has two jobs:

1. Translate a plain-language goal into an editable experiment draft, asking for
   missing or ambiguous information through the normal setup flow.
2. Explain supported models, configuration choices, validated experiments,
   scores, and domain reports in plain language.

It does not generate code, invoke a shell, select trading assumptions, download
weights, start runs, or modify authoritative experiment results.

### Four connection choices

| Connection shown to the user | Authentication and usage |
| --- | --- |
| ChatGPT plan | Official Sign in with ChatGPT for eligible Plus/Pro accounts and permitted local/open-source use. Respect granted inference permission, account limits, and current route restrictions. |
| OpenAI API key | The user's separately selected API credentials and API billing. |
| Anthropic API key | The user's Claude Console credentials and API billing or eligible linked API credits. No Claude subscription OAuth is included. |
| Google Gemini API key | The user's Gemini Developer API credentials, project quotas, and billing. No consumer Google subscription login is included. |

Each connection has its own credential and request handling. Show the active
connection, assistant model, and usage method. Switching requires the user's
choice; a plan limit or API error never silently selects another paid route.
List models applicable to the selected connection and handle unavailable models
explicitly.

A connection manager is reachable from the terminal and new-experiment setup.
It offers add, select, replace, and disconnect actions with masked status.
ChatGPT sign-in opens the official browser authorization flow; API-key setup
uses a separate hidden prompt and can save the key in the credential vault.
Never ask for credentials inside an ordinary assistant conversation. A real
inference connection test is an explicit user action with its usage method shown.

The [OpenAI sign-in guide](https://developers.openai.com/siwc/quickstart)
documents eligible plan usage. Its [preview request contract](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
differs from ordinary API-key requests; do not assume universal request fields
or token-limit parameters. Use the documented [registration flow](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
and [session/credential handling](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions).
For the current ChatGPT-plan HTTP route, tests must require `store: false` and
`stream: true`, omit unsupported fields such as `max_output_tokens`, and accept
output only after `response.completed`. A valid-looking draft followed by a
stream failure or quota error remains unaccepted.
Anthropic [Max/Team API credits](https://support.claude.com/en/articles/17154008-monthly-api-credits-for-max-and-team-plans)
can fund eligible API-key requests; they do not add a fifth connection type.
Gemini follows its [API-key contract](https://ai.google.dev/gemini-api/docs/api-key)
and [API billing](https://ai.google.dev/gemini-api/docs/billing).

### Draft validation and explanations

Use a separate, versioned `ExperimentDraft` with the profile, user-selected
dataset reference, import fields, units, catalog model/variant, evaluation
settings, declared assumptions, and missing fields. Its state is `needs_input`,
`ready`, or `unsupported`. The LLM's `ready` flag cannot establish validity.

Resolve the draft through the same configuration, dataset, capability, and
experiment preparation checks as manual setup. Unknown model IDs, invented
repositories, unsupported horizons, and missing units or dates must not become
executable configurations. Show the final resolved summary and use the normal
human execution confirmation. Drafts stay in memory until accepted.

Model explanations use catalog capabilities and maintained documentation.
Result explanations use a bounded summary of artifacts accepted by the existing
forecast or finance validators. Preserve exact numbers, units, undefined-value
reasons, and evaluation periods. Describe documented model differences without
inventing benchmark rankings. Render authoritative local values separately from
assistant prose; optional assistant annotations never overwrite results.

### Credentials and data protection

- Keep persistent API keys and OAuth tokens in an approved OS credential vault,
  with a hidden-input, session-only option when persistence is unavailable.
  Do not use the current visible wizard question input for secrets.
- Keep credentials out of experiment YAML, drafts, manifests, results, command
  arguments, numerical-worker environments, logs, diagnostics, and exports.
  Redact transport errors and tracing output on failure as well as success.
- For ChatGPT sign-in, validate state, nonce, PKCE, identity, and granted scopes;
  preserve account separation and handle token renewal and sign-out safely.
- Build outgoing context from an explicit allowlist. Show what will leave the
  machine and which connection receives it. Raw observations, full prediction
  tables, source YAML, filesystem paths, and environment dumps are excluded by
  default; dataset references are resolved locally, not by the LLM.
- Treat prompts, dataset descriptions, names, and model output as untrusted.
  Validate draft structure locally and sanitize terminal output. No generated
  file paths, code, tool calls, or embedded instructions gain execution authority.
- Bound input, received output, elapsed time, and request count. Allow at most one
  inference request per submitted message, cancellation, and manual fallback;
  use no automatic inference retry/repair loops or connection switching.
  Authentication refresh and model discovery are separate bounded operations.
  Provider-specific output controls apply only where supported. Local limits
  are not a guaranteed monetary cap.

Provider dependencies stay optional and load only when selected. Explicit
commands and the guided workflow remain usable offline with no assistant keys
or packages. Saved numerical results remain readable without any LLM runtime.

## Implementation order and reviewable deliverables

Each work package is a focused change with its own acceptance evidence. P2, P3,
and P4 can proceed separately after P1; P6 can proceed independently of model
inference work. Integrate through the [contribution workflow](../CONTRIBUTING.md).

| Package | Deliverable | Completion evidence |
| --- | --- | --- |
| P1 Catalog and checkpoint status | Generalize family/variant metadata, compatibility checks, verification records, and active terminal selection; add compatible Bolt Base metadata. Audit optional runtime constraints. | Offline catalog and wizard tests preserve both baselines and show the correct five-family structure as integrations become available, truthful checkpoint status, and refusal of mismatched architectures before allocation. |
| P2 Chronos-2 | Curated Small/standard profiles, safe adapter, provenance, and terminal/config integration. | Contract tests plus one bounded real Small checkpoint check, cached replay, cleanup, and saved-result use without its runtime. |
| P3 TiRex-2 | Curated checkpoint profiles, safe univariate adapter, and terminal/config integration. | Equivalent contract and representative real-check evidence; no fictitious size choices. |
| P4 TTM | Resolve and document native-point task compatibility first; implement explicit variants and causal scaling. | Old-record regression checks, hand-checked scaling/inverse scaling, future-value isolation, and one bounded real variant check. |
| P5 Profiles and guided setup | Add the five usage profiles and their branching questions/reports across manual setup, configuration, and saved history. | Tiny examples and wizard paths cover every profile, cancellation/back navigation, explicit defaults, and unchanged numerical behavior. |
| P6 Connections and credential boundary | Implement the four connection modes, optional transports, secure credential storage, account/billing display, and error handling. | Fake-transport and fake-secret tests pass before separately authorized real connection checks. |
| P7 Natural-language setup and explanations | Implement the versioned draft, clarification, normal validators/review, and grounded explanations in the terminal. | End-to-end fake-provider tests cover intent preservation, missing fields, unsupported future targets, invalid responses, explanations, and manual fallback. |
| P8 Release qualification | Update guides/examples, build source and wheel, and run the installed workflow and configured CI. | Current release checklist passes with exact source/artifact identities and documented representative/variant verification status. |

**P1–P5 are complete locally.** TTM adds three exact-context R2 MSE profiles,
history-only scaling, explicit native-point comparisons, and a representative
CPU check under the approved shared runtime. See the [contract](ttm-contract.md)
and [P4 evidence](p4-ttm-verification.md). P5 adds the five usage profiles across
configuration, guided setup, and saved presentation; see its [guide](usage-profiles.md)
and [verification](p5-profiles-verification.md). P6 connections and credential
boundaries are next, followed by P7 language assistance. Publication of
this milestone remains separate from implementation and package publication.

## Release acceptance

- [x] Users can select all five forecasting families and their approved variants
  through `/new`, with metadata also available through model listing/help.
- [x] Each new adapter has one recorded representative real inference check.
  Larger compatible variants are selectable without mandatory individual
  inference runs and are labelled accurately.
- [ ] Manual and assisted setup both expose all five usage profiles and produce
  equivalent validated settings for the same selections.
- [ ] Existing schema-1 configurations, datasets, completed forecasts, comparisons,
  and finance reports remain readable with their prior meanings and hashes.
- [ ] The four LLM connection modes have explicit authentication and billing
  selection; fake-secret tests show no credential persistence in experiment
  artifacts, worker logs, or exports.
- [ ] Drafting preserves user intent, requests missing information, and cannot
  bypass normal validation or execution confirmation.
- [ ] Explanations use validated evidence and cannot alter forecasts, metrics,
  thresholds, or saved records.
- [ ] Offline tests cover all profiles and four fake transports, malformed and
  truncated output, quota/authentication errors, timeouts, cancellation, prompt
  injection, unsupported actions, and zero network access while disabled.
- [ ] Source/wheel contents, clean installed workflows, documentation, formatting,
  linting, and the supported baseline CI matrix pass for the release candidate.

## Deferred work

Large comparative benchmarks and forecasting-advantage claims remain postponed.
DeepSeek, Ollama, other local/open-weight LLM connections, and further provider
authentication methods belong to later releases. Model training/fine-tuning,
autonomous coding, shell agents, new trading policies, live feeds, prospective
forecasts without actuals, panels, covariates, probability scoring, and a hosted
web application are outside this release's scope.

Resource use and publication follow the existing permission boundaries in
[CONTRIBUTING.md](../CONTRIBUTING.md). This plan does not install runtimes,
download weights, connect accounts, make paid requests, bump the version, or
publish either release.

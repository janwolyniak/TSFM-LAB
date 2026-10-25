# Writing a model adapter

TSFM Lab's Python evaluator accepts any trusted local object implementing the
[ForecastModel protocol](../src/tsfm_lab/models/base.py). A custom adapter can use
the numerical pipeline directly today. The YAML commands and interactive setup
accept only the explicitly supported adapter IDs: `last_value`, `seasonal_naive`,
`chronos_bolt`, and `timesfm_2p5`. There is no automatic plugin discovery or
arbitrary import path in configuration.

## Run the educational example

[custom_adapter.py](../examples/custom_adapter.py) implements `LinearDrift` using
NumPy and the existing contracts. From a source checkout with the core dependencies
available:

```sh
PYTHONPATH=src python3 examples/custom_adapter.py
```

If the package is installed, `python examples/custom_adapter.py` also works from
the repository root. No terminal or model runtime is required. The example prints
JSON; it does not create experiment directories.

The example forecasts the original synthetic target `demand`, in
`synthetic_units`, sampled hourly in UTC. It uses context 8, horizon 4 observations,
stride 4, and MASE lag 4, giving four windows and 16 forecast cases. See the
[dataset description and license](../examples/datasets/README.md).

For a context of length L, endpoint drift is `(last - first) / (L - 1)`. Future
step h is `last + h * drift`. With history `[1, 2, 4, 7]`, this gives `[9, 11, 13]`
for steps 1 through 3. It requires at least two observations. This is educational
arithmetic for checking the adapter interface; its synthetic scores do not
establish forecasting quality on real data.

`LinearDrift` remains in the example file. To evaluate a class from your own
trusted module, construct it in Python and pass it to the same runner:

```python
result = run_experiment(dataset, LinearDrift(), spec)
print(result.scores.overall)
print(result.predictions)
```

Here `dataset` is a validated `Dataset` and `spec` is an `EvaluationSpec`, as in
the runnable example. `linear_drift_example` is deliberately absent from the
catalog, factory, YAML schema, and interactive model choices.

## Implement the six required methods

The protocol is structural: inheritance from a framework base class is not
required. Use these exact signatures:

| Method | Responsibility |
| --- | --- |
| `metadata() -> ModelMetadata` | Return adapter identity/version, family, and truthful capabilities without loading weights or importing a heavy runtime. |
| `configuration() -> dict` | Return JSON-serializable inference options and resolved model identity. The runner snapshots this before loading. |
| `validate(spec: EvaluationSpec) -> None` | Reject unsupported context/horizon/task choices before inference. Do not allocate model resources. |
| `load() -> None` | Import optional inference libraries, verify/load the selected local weights, and initialize resources once per run. |
| `predict(requests: tuple[ForecastRequest, ...]) -> tuple[ForecastOutput, ...]` | Produce one aligned output per request using only permitted historical inputs. |
| `unload() -> None` | Release resources; make cleanup safe after a partial load and repeated calls. |

The dataset contract validates values and the time grid at construction. Before
entering its loading section, the runner resolves window bounds, checks model
capabilities, and calls the adapter's validation method. It calls `load` once and
currently sends one request per `predict` call for each rolling origin. The batch-shaped interface
does not imply batched execution by the runner. Inside the loading/inference
section, `unload` runs through a `finally` path after successful inference, load
failure, inference failure, or cancellation. Validation failures before this
section do not call `unload`, which is why validation must allocate no resources.

The arithmetic example needs no loaded resources, so its load/unload methods are
no-ops. A checkpoint adapter should keep a clear loaded state, reject prediction
before loading, and clean up resources if initialization fails. Avoid network
access, prints, prompts, and terminal-framework imports inside prediction logic.
Progress and cancellation belong to the shared runner/application layer.

## Preserve the forecast contract

| Value | Current representation |
| --- | --- |
| Historical target values | Detached read-only float64 vector `[L]`, in the imported target's units. |
| Historical timestamps | Ordered UTC-aware index `[L]`; its last timestamp equals `origin`. |
| Requested target timestamps | UTC-aware index `[H]` on the same regular grid, strictly after the origin. |
| Returned point values | Finite float64 vector `[H]` in the original target units. |
| Identity | Copy `request_id`, `series_id`, `origin`, and the exact requested target timestamps. Set the actual adapter ID. |

The evaluator retains future actual values separately; they are absent from
`ForecastRequest`. Fit any request-dependent transformation on permitted history
and invert it before returning original-unit forecasts. Do not keep a reference
to the full dataset, fit normalization on the evaluation split, silently truncate
context, extend a horizon through an undocumented rollout, or replace failed
cases with a smaller intersection of available forecasts.

The runner checks each output's request/model identity, exact timestamps, shape,
finiteness, and point statistic before scoring. `ForecastOutput` copies its point
buffer, and the runner snapshots it again before later predictions can reuse a
working array. Treat historical and timestamp buffers as immutable. These
contracts prevent accidental leakage by conforming adapters; they do not isolate
hostile local Python code.

The initial task requests a median point forecast. Deterministic outputs such as
the drift example are admissible and must retain `point_statistic="deterministic"`.
For probabilistic models, use the requested statistic and label it accurately;
never relabel a median as a mean. Quantile/sample scoring and output contracts
remain later work. See [evaluation methodology](evaluation.md) and
[core contracts](contracts.md) for chronology, metric pooling, and undefined MASE.

## Declare capabilities and provenance

`ModelCapabilities` describes minimum/maximum context, maximum horizon, target
count, covariates, output types, and devices. The current engine requires a
univariate point adapter supporting CPU. A model's broader native capabilities
do not add multivariate targets, covariates, or accelerators to this engine.
Publish only the subset the adapter actually supports and verifies.

The TimesFM 2.5 adapter exposes CPU/float32, batch size 1, contexts up to 16,256,
and horizons up to 128. It extracts the genuine 0.5 quantile rather than relabelling
the mean. The model version is separate from its `timesfm==3.0.2` runtime package;
the optional `[timesfm]` extra and [constraints](../requirements-timesfm.txt)
select that tested API. See [TimesFM usage](timesfm.md). An incompatible global
Torch default dtype fails without the adapter changing process-wide settings.

Use a stable adapter ID and change its adapter version when inference semantics
change. Keep model implementation identity distinct from checkpoint identity.
The example uses family `example`; the runner consequently records its default
source as `local_python`, rather than claiming it is a built-in catalog entry.

The runner's source provenance identifies the TSFM Lab checkout. Adapter code in
another repository needs its own recorded revision and local-change status in
the adapter configuration; arbitrary external Python source is not automatically
fingerprinted.

The runner also recognizes optional hooks outside the required protocol:

- A `source` attribute/property can supply the model source. Ordinary local Python
  adapters can use `local_python`. The `local` and `huggingface` checkpoint-source
  schemas have explicit per-family validation for Chronos-Bolt and TimesFM 2.5.
  An adapter/repository mismatch is rejected before loading weights.
- `runtime_configuration() -> dict` records actual device, dtype, batch size, and
  relevant seed/determinism choices. Without it, the current defaults are CPU,
  float64, and batch size 1. Describe the actual model arithmetic dtype, even when
  normalized input/output arrays use float64.
- `environment_metadata() -> dict[str, str]` records additional installed runtime
  distribution versions. Use package metadata rather than importing frameworks
  just to read versions. This hook is called after inference and cleanup, so it
  must not depend on a live loaded model or replace the core environment fields.

For pretrained integrations, record repository, requested revision, resolved
commit SHA, consumed-file identities, adapter version, runtime configuration,
and relevant dependency versions. Local checkpoint identity needs a file/checksum
manifest. The existing [checkpoint manager](../src/tsfm_lab/models/checkpoints.py),
[family profiles](../src/tsfm_lab/models/checkpoint_profiles.py),
[Chronos adapter](../src/tsfm_lab/models/chronos_bolt.py), and
[TimesFM adapter](../src/tsfm_lab/models/timesfm.py) show the supported
policy: metadata inspection first, native-limit checks, verified local-only safe
weight loading, and explicit downloads managed separately from experiments.
New families must preserve reload of existing records and their fingerprints.

Display checkpoint/model-code license information separately from the platform's
[Apache-2.0 license](../LICENSE). Do not assume model weights or datasets inherit
the platform license. The educational example and bundled synthetic data use
the repository license.

## Propose a supported integration

A local Python example does not register a new YAML or terminal adapter. A
supported integration needs a focused change and agreement under
[CONTRIBUTING.md](../CONTRIBUTING.md), including these existing boundaries:

| Boundary | Files to review |
| --- | --- |
| User input, source rules, and runtime options | [Configuration schema](../src/tsfm_lab/config.py). |
| Explicit construction and checkpoint resolution | [Model factory](../src/tsfm_lab/models/factory.py). |
| Discovery metadata and compatible repositories | [Catalog](../src/tsfm_lab/models/catalog.py). |
| Guided choices and validation | [Inline setup](../src/tsfm_lab/repl/wizard.py) and [form validation](../src/tsfm_lab/tui/forms.py). |
| Reproducible checkpoint records and offline reload | [Checkpoint records](../src/tsfm_lab/models/checkpoints.py) and [artifact validation](../src/tsfm_lab/artifact_validation.py). |
| Optional dependencies and distribution | [Package metadata](../pyproject.toml), development constraints, and configured CI. |

Add contract tests with manually known outputs, native context/horizon failures,
wrong-length or shifted forecasts, finite values, and cleanup after failure.
Change future observations to verify that an earlier forecast stays unchanged;
later rolling forecasts may legitimately change once those observations enter
history. Keep heavy inference optional in routine tests, and separately record a
small real runtime smoke test before claiming an integration works on a device.
Static inspection and mocked model tests do not establish forecasting advantage.

The supplied example tests cover known endpoint arithmetic, a linear trend with
exact zero error, temporal perturbation, minimum-context rejection, and complete
offline script execution with model and terminal imports blocked.

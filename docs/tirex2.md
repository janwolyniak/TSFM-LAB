# TiRex-2 checkpoints and inference

The optional `tirex_2` adapter evaluates one regular univariate series on CPU,
using float32 model arithmetic and the native median quantile. Four curated
repositories share the 82,501,744-parameter architecture. Their training and
forecast profiles differ; they are not four model sizes.

## Curated profiles

| Choice | Repository | Exact reference | Maximum context | Maximum horizon |
| --- | --- | --- | ---: | ---: |
| Default | `NX-AI/TiRex-2` | `30788a311699acba7e6393ad9700ea7b7cb9bb2d` | 2,048 | 320 |
| FEV benchmark | `NX-AI/TiRex-2-fevbench` | `acd6153b8ee2179f38cfd3cc4fd64a3c045a424f` | 2,048 | 320 |
| GiftEval zero-shot | `NX-AI/TiRex-2-gifteval-zs` | `c94872d1c8caacbe7755cbfa01bc5c4c5c5ddb7f` | 8,192 | 928 |
| GiftEval pretrain | `NX-AI/TiRex-2-gifteval-pretrain` | `54f88ac06ccfba9fe3e727aa8856b39775481f0f` | 8,192 | 928 |

The default Safetensors file is 330,038,664 bytes; each other profile's file is
330,038,640 bytes. Download size does not describe inference RAM use. Repository
lookup reports that profile's limits, while adapter lookup reports the maximum
across supported profiles. The actual inspected configuration determines the
limits enforced during an experiment.

All four model repositories declare Apache-2.0. Their pinned native JSON
configurations define a nine-quantile grid from 0.1 to 0.9, including the median.
See the official pinned configurations for
[default](https://huggingface.co/NX-AI/TiRex-2/blob/30788a311699acba7e6393ad9700ea7b7cb9bb2d/config.json),
[FEV](https://huggingface.co/NX-AI/TiRex-2-fevbench/blob/acd6153b8ee2179f38cfd3cc4fd64a3c045a424f/config.json),
[GiftEval zero-shot](https://huggingface.co/NX-AI/TiRex-2-gifteval-zs/blob/c94872d1c8caacbe7755cbfa01bc5c4c5c5ddb7f/config.json), and
[GiftEval pretrain](https://huggingface.co/NX-AI/TiRex-2-gifteval-pretrain/blob/54f88ac06ccfba9fe3e727aa8856b39775481f0f/config.json).

## Optional runtime and notices

**Built with technology from NXAI**

The model-weight license and runtime-dependency terms are separate. TiRex-2's
optional `mlstm-kernels==2.0.6` dependency uses the
[NXAI Community License](https://github.com/NX-AI/mlstm_kernels/blob/v2.0.6/LICENSE),
including attribution, redistribution, and additional commercial terms. Section
2 addresses consolidated annual revenue above €100 million together with
incorporation into a commercial product or service. Read the full terms for your
use. The project ships the [dependency license](../licenses/mlstm-kernels-2.0.6.txt)
and [NOTICE](../NOTICE); terminal model information also exposes these runtime
notices. TSFM-LAB's own code remains Apache-2.0.

Install this optional runtime only when needed:

```sh
python -m pip install -c requirements-tirex2.txt '.[cli,tirex2]'
tsfm models info tirex_2
tsfm models info NX-AI/TiRex-2-fevbench --json
```

The supported stack pins TiRex-2 0.4.0, Torch 2.8.x, Hub 0.35.3, Safetensors
0.6.x, FlashRNN 1.0.8, mLSTM Kernels 2.0.6, Ninja 1.13.2, and Narwhals 2.13.0,
with NumPy 2.1.3 or newer below 3. The CPU path uses native recurrent kernels
with compilation disabled. Linux's upstream dependency graph can include
accelerator packages even when choosing CPU inference; use the platform's
explicit constrained environment and inspect its dependencies before installing.

P3 updates the shared Hub pin from 0.34.4 to 0.35.3 for all checkpoint adapters;
it retains Torch 2.8 and Transformers 4.55. Existing-family CPU regressions and
the exact verification environment belong in the [P3 record](p3-tirex2-verification.md).
This does not upgrade an existing global installation automatically. Model
listing, configuration validation, and saved-result inspection require none of
the TiRex or recurrent-kernel packages.

## Explicit pull and terminal selection

Normal experiments use already cached or standalone files. Downloading is an
explicit operation:

```sh
tsfm models pull NX-AI/TiRex-2 --revision 30788a311699acba7e6393ad9700ea7b7cb9bb2d --cache-dir model-cache
tsfm models cache --cache-dir model-cache --json
tsfm run examples/configurations/tirex2-main.yaml --json
```

The [example](../examples/configurations/tirex2-main.yaml) uses the bundled
synthetic historical series with the same median task and scoring rules as the
baselines. Its paths are relative to the YAML file. Use a matching cache in the
pull and configuration. To select another profile, change both repository and
revision to that profile's table entry and explicitly pull its files first.

Use `/new tirex_2` in the persistent terminal. The existing repository question
offers all four profiles and their limits. It preserves the canonical mixed-case
repository spelling, sets that choice's pinned SHA, and retains an explicit SHA
override when returning to the same profile. Changing the repository selects
its own default SHA. Setup does not download or load weights and retains the
normal review/execute confirmation.

A standalone configuration uses `model.source: local`, `adapter_id: tirex_2`,
and `model.path` pointing to the copied files. Omit repository, revision, cache,
and offline fields. Copy actual bytes out of a Hub snapshot before treating it
as standalone; the resolver restricts Hub symlinks to their verified cache.

## Forecast and loading contract

The resolver recognizes the reviewed native JSON configuration directly. It
validates the complete permitted profile before runtime import/allocation,
including fixed activation classes, dimensions, scaling, and output quantiles.
Unexpected configuration changes fail. Derived checkpoint records identify the
TiRex-2 adapter without rewriting the consumed native JSON or its file hashes.

The adapter rechecks consumed files, constructs the model directly, reads only
Safetensors on CPU, expands documented shared parameter aliases, and requires
strict parameter loading. It does not use the upstream loader's legacy YAML or
pickle fallback, dynamic activation selection, or automatic Hub resolution.
Flex Attention and compilation are disabled for the supported CPU path.

Each request supplies detached history with no actual future values or
covariates. The native forecast returns nine quantile channels; the adapter
selects 0.5 explicitly, checks output dimensions/finiteness, and returns aligned
float64 median forecasts in original target units. Scales, differencing, and
inverse transformations use only supplied history. Over-limit contexts and
horizons fail before the upstream code can silently truncate them.

The checkpoint's fixed forecast settings are preserved and recorded:

| Profile | `tta_sign_flip` | `tta_diff` | `pad_context` |
| --- | --- | --- | --- |
| Default / FEV | false | true | true |
| GiftEval profiles | true | false | true |

Padding supplies the native model's missing-history representation; it does not
introduce future observations. These settings are not user-tuned experiment
options in this integration. Saved-result validation binds them to the recorded
profile and native bounds. Median prediction does not sample. The runner's
normal cleanup and cancellation rules apply.

## Verification scope

P3's representative target is the default checkpoint. The FEV and GiftEval
choices receive metadata/API compatibility checks without separate inference
runs. Only completed checks enter project evidence; inspect the selected
checkpoint's actual status through `models info`, `/models`, and its saved
provenance. See the [P3 verification record](p3-tirex2-verification.md) for
executed checks, exact identities, qualification results, and remaining gaps.

Routine tests use inert weights and fake runtimes. The separately opted-in real
CPU test uses existing authorized files and never downloads weights. No model
ranking, forecast-quality claim, GPU/MPS support, multivariate/covariate feature,
ONNX backend, live feed, or prospective forecasting mode is added here.

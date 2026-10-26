# Chronos-2

The optional `chronos_2` adapter evaluates one regular univariate series with
Chronos-2 Small or standard. It supports CPU float32 inference, context lengths
1–8,192 and horizons 1–1,024. It returns the checkpoint's native 0.5 quantile in
original target units. It does not expose upstream panels, covariates, joint
cross-series learning, or longer-horizon heuristic extensions.

## Checkpoints and runtime

| Choice | Repository | Pinned reference | Parameters | Safetensors bytes |
| --- | --- | --- | ---: | ---: |
| Small | `autogluon/chronos-2-small` | `ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a` | 27,934,624 | 111,749,048 |
| Standard | `amazon/chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` | 119,477,664 | 477,930,472 |

Both references declare Apache-2.0 and the `Chronos2Model` architecture with
`Chronos2Pipeline`. The supported native horizon is 64 output patches × 16
observations. Their Small/standard quantile grids differ; both include the
median. See the pinned [Small configuration](https://huggingface.co/autogluon/chronos-2-small/blob/ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a/config.json)
and [standard configuration](https://huggingface.co/amazon/chronos-2/blob/29ec3766d36d6f73f0696f85560a422f50e8498c/config.json).
Weight-file sizes describe download/storage, not runtime RAM requirements.

Use the same optional runtime as Chronos-Bolt:

```sh
python -m pip install -c requirements-chronos.txt '.[cli,chronos]'
tsfm models info chronos_2
tsfm models info autogluon/chronos-2-small --json
```

The adapter checks Chronos 2.2.0, Torch 2.8.x, Transformers 4.55.x, Hub 0.35.3,
Safetensors 0.6.x, and Accelerate 1.12.x. These remain optional; catalog lookup,
configuration validation, and saved-result reload/export do not import them.
Torch's existing default dtype must be float32; the adapter fails otherwise
without changing that process-wide setting. CPU device scopes protect against
an ambient accelerator default. Other devices and dtypes are unsupported.

## Explicit download and experiment

Only `models pull` downloads weights. A normal run requires the selected files
already present in its cache or a compatible standalone directory:

```sh
tsfm models pull autogluon/chronos-2-small --revision ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a --cache-dir model-cache
tsfm models cache --cache-dir model-cache --json
tsfm run examples/configurations/chronos2-small.yaml --json
```

The [example configuration](../examples/configurations/chronos2-small.yaml) uses
the bundled original synthetic data. It is a historical evaluation with observed
targets; it does not forecast an unobserved future period. Paths resolve relative
to the YAML file. Use the same cache directory in the pull and configuration.
To select standard, change the repository and revision together to the table's
standard reference; pulling its larger weights is a separate explicit action.

In the persistent terminal, `/models` lists both choices. `/new chronos_2`
starts guided setup with Small or standard selection, pinned revision defaults,
native-bound checks, and the normal review/execute confirmation. Variant changes
select the corresponding default revision; returning to the same variant keeps
an explicit revision override. Model selection never pulls weights.

For a standalone copy, set `model.source: local`, `model.adapter_id: chronos_2`,
and `model.path` to its directory; omit repository, revision, cache, and offline
fields. Hub snapshot symlinks are allowed only inside their verified cache;
copy actual file bytes before treating a snapshot as a standalone directory.
See the [shared safe-checkpoint rules](chronos.md#standalone-local-checkpoints).

## Forecast and loading contract

The adapter validates the architecture, reviewed dimensions, patch sizes,
arcsinh normalization, time encoding, quantile grid, and native limits before
weight loading. Unknown configuration fields and aliases that could override
validated dimensions fail. File hashes are checked again immediately before
loading. Loading calls the model class directly, uses only local Safetensors,
disables remote code, and rejects missing, unexpected, or mismatched parameters;
it bypasses upstream automatic adapter discovery.

Every request supplies detached historical values. The pipeline uses
`batch_size=1`, `cross_learning=false`, `context_length=8192`, and
`limit_prediction_length=true`. Requests beyond the native bounds fail; history
is never silently shortened. Upstream instance normalization and its inverse
use only supplied context. The adapter selects the exact native median channel
from `[1, quantiles, horizon]` outputs, validates all values and dimensions, and
returns detached float64 forecasts aligned to the requested timestamps.
The pipeline's higher-level output named `mean` is not used.

Resolved runtime records include the two fixed forecasting flags, the CPU/dtype,
batch size, and seed. Prediction is deterministic and does not sample; the seed
is recorded as unused. The runner releases model and pipeline references after
success, failure, or cancellation. Saved records keep schema-1 checkpoint
fingerprints and the separate project-verification snapshot.

## Verification and limits

Small has a bounded real CPU integration check; standard is selectable after
metadata/API compatibility checks without an individual inference run. The
[P2 verification record](p2-chronos2-verification.md) distinguishes these facts
and identifies the exact checked runtime and files. See the
[catalog guide](model-catalog.md) for reference and selected-file status.
The checks establish integration, rather than forecast quality or a ranking.

The opt-in test never downloads anything:

```sh
TSFM_CHRONOS2_CACHE=model-cache TSFM_CHRONOS2_REVISION=ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a python -m pytest -q tests/test_chronos2_smoke.py
```

Alternatively, set `TSFM_CHRONOS2_CHECKPOINT` to a standalone directory. The
test checks finite aligned medians, exact cached replay, unchanged predictions
after future actuals change, and cleanup. Routine CI uses inert checkpoint files
and fake runtimes, and skips this separately authorized real-model check.

Implementation references: the pinned upstream
[pipeline](https://github.com/amazon-science/chronos-forecasting/blob/75a2dfe228e39d55283f477c01399b9e53d89661/src/chronos/chronos2/pipeline.py)
and [model](https://github.com/amazon-science/chronos-forecasting/blob/75a2dfe228e39d55283f477c01399b9e53d89661/src/chronos/chronos2/model.py).

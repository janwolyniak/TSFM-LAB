# TimesFM 2.5 checkpoints and inference

The optional `timesfm_2p5` adapter evaluates the explicit TimesFM 2.5 Torch model
on CPU. It shares the baseline/Bolt rolling evaluation, saved records, comparison,
and export workflows. Catalog lookup and saved-result reload need no model runtime.

## Model, runtime, and supported task

The curated checkpoint is
[`google/timesfm-2.5-200m-pytorch`](https://huggingface.co/google/timesfm-2.5-200m-pytorch),
whose model card declares Apache-2.0. Its pinned revision is
`1d952420fba87f3c6dee4f240de0f1a0fbc790e3`. The Safetensors file contains
925,181,104 bytes with SHA-256
`2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8`.
The [pinned configuration](https://huggingface.co/google/timesfm-2.5-200m-pytorch/blob/1d952420fba87f3c6dee4f240de0f1a0fbc790e3/config.json)
is inspected before any explicit weight download.

The tested runtime is **`timesfm==3.0.2`**, with Torch 2.8.x, Hub 0.35.3, and
Safetensors 0.6.x. Package version 3.0.2 and model version 2.5 are different
identities. Only its `TimesFM_2p5_200M_torch` class is used; TimesFM 3.0 models and
their separate licenses are outside this adapter. See the
[official runtime release](https://github.com/google-research/timesfm/releases/tag/v3.0.2)
and [constraints](../requirements-timesfm.txt).

| Property | Supported behavior |
| --- | --- |
| Task | One regular univariate series, original target units, median point scoring. |
| Context | 1–16,256 observations; no silent history truncation. |
| Horizon | 1–128 observations; one native output patch, no autoregressive extension. |
| Runtime | CPU, float32, batch size 1, Torch compilation disabled. |
| Checkpoints | The curated exact cached revision or a compatible standalone local directory. |

The native configuration reports context 16,384, but upstream rounds context to
a multiple of 32 and reserves a 128-step output patch. The supported rectangle
therefore caps historical context at 16,256. Short inputs receive masked left
padding, with the actual historical values preserved. These bounds follow the
[pinned runtime implementation](https://github.com/google-research/timesfm/blob/v3.0.2/src/timesfm/timesfm_2p5/timesfm_2p5_torch.py).

## Install separately and inspect offline

In the intended environment, when dependency installation is authorized:

```sh
python -m pip install -c requirements-dev.txt -c requirements-timesfm.txt '.[cli,parquet,timesfm]'
tsfm models info timesfm_2p5 --json
tsfm models info google/timesfm-2.5-200m-pytorch --json
```

This installs the runtime without weights. The library imports only when loading
a selected model. An older editable TimesFM 2.0.0 installation is rejected: the
supported 3.0.2 wrapper owns a separate model per adapter instance and has a
verified loading/inference API. Missing or untested runtime versions fail clearly.

The default Torch dtype must remain float32. Upstream creates internal cache and
statistics tensors using that default; the adapter rejects another default at
load and prediction instead of changing process-wide settings. Parameters and
inputs are also explicitly placed on CPU with float32 arithmetic.

## Explicitly download, then run from cache

The following is an explicit approximately 925 MB weight download:

```sh
tsfm models pull google/timesfm-2.5-200m-pytorch --revision 1d952420fba87f3c6dee4f240de0f1a0fbc790e3 --cache-dir ./model-cache --json
tsfm run examples/configurations/timesfm-2p5.yaml --json
```

The [example configuration](../examples/configurations/timesfm-2p5.yaml) uses
`../../model-cache` relative to its own directory. Omit `model.cache_dir` to use
the configured global Hub cache instead. Offline execution requires an exact
lowercase 40-character SHA and complete cached files; it does not resolve a
moving branch or download missing files. Moving revisions may be resolved only
when explicitly permitted by the selected source configuration.

Local files use the same adapter:

```yaml
model:
  source: local
  adapter_id: timesfm_2p5
  path: ../../my-timesfm-2p5-checkpoint
runtime:
  device: cpu
  dtype: float32
  batch_size: 1
  seed: 0
```

The standalone directory must contain its compatible `config.json` and
`model.safetensors`; `README.md` is optional. Local files must stay inside the
selected directory. Hub snapshot links are checked against their permitted blob
store. Remote-code mappings, redirected loading, unsupported architectures,
quantization and PEFT sidecars fail. A Bolt adapter cannot select TimesFM weights,
and the TimesFM adapter cannot select Bolt weights.

Only recorded files are loaded. The adapter calls the pinned runtime's local
Safetensors loader, which uses strict parameter loading and evaluation mode;
it does not call `from_pretrained` or any automatic Hub loader.

## Median semantics and reproducibility

The runtime returns a point array `[B,H]` and an array `[B,H,10]` containing the
mean in channel 0 and quantiles 0.1–0.9 in channels 1–9. The adapter selects
**channel 5, the median**, verifies that the separate point output agrees with
it, and rejects wrong shapes or nonfinite values. Normalized saved points are
float64 in the original target units; no distributional score is inferred.

The fixed forecast options are recorded with every run: input normalization and
flip invariance enabled; positivity inference, continuous quantile head, crossing
repair, and backcast disabled. The seed is recorded but unused by this
deterministic path. The adapter records actual runtime versions and the consumed
checkpoint files' hashes. A model-card license is recorded only when the consumed
card supplies it; catalog license metadata does not fill in absent file evidence.

The family-aware checkpoint validator preserves existing schema-1 Bolt records
and fingerprints. Completed TimesFM records reload, compare, and export without
model weights or model libraries. Comparison still requires the same dataset,
cases, context, horizon, metric definitions, and scaling convention.

## Verification

Routine tests use fake weights and runtimes. They cover median-versus-mean
selection, native bounds, input immutability, future-value separation, safe local
loading, independent model instances, error cleanup, version/default-dtype
rejection, configuration/UI selection, and saved-record integrity.

For an existing global cached revision, the optional real CPU smoke test is:

```sh
TSFM_TIMESFM_REVISION=1d952420fba87f3c6dee4f240de0f1a0fbc790e3 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_timesfm_smoke.py -q
```

Set `TSFM_TIMESFM_CACHE_DIR` for a different existing Hub cache, or use
`TSFM_TIMESFM_CHECKPOINT=/absolute/standalone/path` instead of the revision.
These tests never pull weights. The [second-family verification record](milestone6-verification.md)
documents the actual cached CPU run, replay, and their limits. Synthetic
integration checks do not establish a forecasting advantage on real data.

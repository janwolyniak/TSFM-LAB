# Chronos Bolt checkpoints and inference

Milestone 3 adds the optional `chronos_bolt` adapter, a curated checkpoint catalog,
explicit model pulls, cache-only experiment execution, and standalone local
checkpoints. The initial runtime is CPU with float32 model arithmetic; normalized
point forecasts remain float64 in the original target units.

## Supported models and runtime

The curated Hub repositories are
[`amazon/chronos-bolt-tiny`](https://huggingface.co/amazon/chronos-bolt-tiny),
[`amazon/chronos-bolt-small`](https://huggingface.co/amazon/chronos-bolt-small), and
[`amazon/chronos-bolt-base`](https://huggingface.co/amazon/chronos-bolt-base).
The official cards list Apache-2.0. A checkpoint record derives its license from
its consumed README when available and reports `null` otherwise; catalog license
metadata and an inspected local file's license are separate fields of evidence.
Other Hub repositories and Chronos variants are outside this adapter.
The initial CPU integration was verified with Tiny. A subsequent explicitly
authorized Small run verified the pinned checkpoint, offline replay, local
loading, saved records, comparison, and exports; see the
[expanded verification record](milestone6-verification.md). Other devices remain
unverified.
Base uses the compatible Bolt architecture and has metadata/API compatibility
checks; no Base weight download or individual inference test was required for
its addition. See the [catalog and verification guide](model-catalog.md) for its
pinned reference, exact file identities, and project-status labels.

The initial supported runtime uses Chronos forecasting 2.2.0, PyTorch 2.8.x,
Transformers 4.55.x, and Hugging Face Hub 0.35.3. The exact development runtime is
listed in [requirements-chronos.txt](../requirements-chronos.txt). Runtime checks
refuse incompatible versions. This bounded matrix avoids silently changing
inference/loading semantics with a third-party upgrade.

When installing dependencies is appropriate for your environment:

```sh
python -m pip install -c requirements-chronos.txt -e '.[cli,chronos]'
```

The `hub` extra suffices for checkpoint downloads without inference; neither it
nor the Chronos runtime is needed for baseline evaluation or saved-result reload.
There is no automatic installation, upgrade, CUDA, MPS, quantized loading, model
training, or remote-code execution.

## Inspect and explicitly pull

```sh
tsfm models list
tsfm models info chronos_bolt --json
tsfm models info amazon/chronos-bolt-tiny --json
tsfm models info amazon/chronos-bolt-base --json
tsfm models cache --cache-dir ./model-cache --json
```

Catalog information is offline and imports no inference runtime. Cache inspection
reads only supported repository snapshots, reports incomplete snapshots, and does
not delete or download files.
The guided terminal lists Tiny, Small, and Base under `/new chronos_bolt` and
defaults a newly selected variant to its curated reference SHA. Users can edit
that revision explicitly. The [Base example](../examples/configurations/chronos-bolt-base.yaml)
is a cache-only configuration; running it requires separately obtained weights.

Pulling is an explicit download operation:

```sh
tsfm models pull amazon/chronos-bolt-tiny --revision a0e552de83495b5c28c14c71c374f3e33280b340 --cache-dir ./model-cache --json
```

This pinned tiny revision has a 34,622,352-byte Safetensors weight file, plus a
small configuration and model card. A moving branch/tag may instead be supplied
to `--revision`: the command resolves it once to an immutable commit SHA, inspects
the configuration first, then downloads only `config.json`, `model.safetensors`,
and an available `README.md` at that SHA. It uses the Hub cache rather than a
second weight store. The JSON response includes requested and resolved revisions,
consumed file hashes/sizes, native limits, license, path, and checkpoint fingerprint.

## Run from cached files

[chronos-bolt-tiny.yaml](../examples/configurations/chronos-bolt-tiny.yaml) selects
the pinned revision and uses the same synthetic target, units, rolling origins,
context, horizon, and MASE scale as the baseline examples. It uses the repository's
ignored `model-cache` directory, matching the pull command above. Omit
`model.cache_dir` to use the Hub's configured global cache. Relative cache paths
in YAML resolve against the YAML file's directory.

```yaml
model:
  source: huggingface
  adapter_id: chronos_bolt
  repository: amazon/chronos-bolt-tiny
  revision: a0e552de83495b5c28c14c71c374f3e33280b340
  offline: true
  cache_dir: ../../model-cache
runtime:
  device: cpu
  dtype: float32
  batch_size: 1
  seed: 0
```

```sh
tsfm run examples/configurations/chronos-bolt-tiny.yaml --json
```

The normal run path never downloads checkpoint files. With `offline: true`, an
exact lowercase 40-character SHA and a complete cached snapshot are required;
no metadata request is made. Setting `offline: false` permits resolving a moving
revision online, but the resolved files must already be cached. The adapter itself
always loads a verified local directory with `local_files_only=True`.

The actual checkpoint determines native context and prediction limits. Official
tiny configuration currently specifies 2048 observations of context and a
64-observation native horizon. Requests above native limits fail; the adapter does
not truncate context or extend horizons autoregressively. Future target values
never enter the adapter.

## Standalone local checkpoints

Use a standalone directory containing the complete `config.json` and
`model.safetensors`; an optional README supplies license metadata:

```yaml
model:
  source: local
  adapter_id: chronos_bolt
  path: ../../my-bolt-checkpoint
runtime:
  device: cpu
  dtype: float32
```

Local paths resolve against the configuration file. Consumed files must remain
inside the selected directory; Hub snapshots use their separately verified
snapshot/blob structure. A standalone copy of a Hub checkpoint works as a local
model. Local records do not claim a Hub commit revision.

Checkpoint verification rejects unsupported architecture/pipeline configuration,
remote-code mappings, alternative config/weight redirects, quantization, PEFT
sidecars, missing safe weights, and changed consumed files. A file being named
`model.safetensors` does not alone establish compatibility: strict runtime loading
also rejects missing, unexpected, or mismatched parameters. No unsafe checkpoint
deserialization fallback is enabled.

## Forecast semantics and records

The adapter calls the installed pipeline's quantile prediction method for level
0.5 and validates exact `[batch, horizon, 1]` shape and finite values. It uses that
median explicitly: the pinned upstream Bolt implementation misleadingly names its
second median output `mean`. No distributional metric or uncertainty interval is
inferred from the point output. Dropout is disabled through evaluation mode;
there is no sampling, and the recorded seed is unused.

Saved Python and CLI results include the same resolved model identity, native
capabilities, CPU/dtype, adapter version, and model-runtime versions. File hashes
are rechecked before loading. Completed artifacts can later be shown, compared,
or exported without the weights or model libraries; persisted identity validation
does not reread checkpoint files.

## Verification scope

Routine tests use fake Hub clients, tiny synthetic files, and fake model runtimes.
They cover revision races, offline behavior, unsafe/unsupported configurations,
checkpoint mutation, strict weight loading, forecast alignment, and recorded
identity. They require no Torch installation or model download.

The real CPU smoke test is explicitly opt-in and uses existing standalone local
weights:

```sh
TSFM_CHRONOS_CHECKPOINT=/absolute/local/checkpoint PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/test_chronos_smoke.py -q
```

That test verifies local integration, finite aligned median outputs, and recorded
runtime/checkpoint metadata. It passed after an explicitly authorized pinned Tiny
download. Cached replay, a standalone local copy, and installed CLI execution also
passed. These checks do not establish accuracy superiority. See the
[verification record](milestone3-verification.md) for evidence and limits.
The later [Small verification](milestone6-verification.md) is a separate record;
the original milestone's counts and artifact identities remain historical.

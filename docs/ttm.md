# Granite TTM R2 and R2.1 checkpoints and inference

The optional `ttm_r2` adapter evaluates a regular univariate series using CPU
float32 inference. Its MSE head produces a **native point forecast**. Training
with MSE does not establish that this output is a calibrated conditional mean;
it is never labelled median. Select the explicit `native_point` evaluation task.

## Profiles and installation

Eight branches of `ibm-granite/granite-timeseries-ttm-r2` are selectable:

| Profile | Required context | Native horizon | Exact revision | Weight bytes |
| --- | ---: | ---: | --- | ---: |
| `512-96-r2` | 512 | 96 | `d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f` | 3,240,592 |
| `1024-96-r2` | 1,024 | 96 | `1a382618f3c5b63836281ad632e54724e55e85a5` | 11,879,480 |
| `1536-96-r2` | 1,536 | 96 | `f6ed09d67cf42e4877f7fa33d98326911385693d` | 12,344,144 |
| `512-192-r2` | 512 | 192 | `25f4a00a25e1d0b67c40a10b06645d72e2266929` | 3,634,200 |
| `512-336-r2` | 512 | 336 | `0009f51f52c87473e60ff3a03cd8633ee85d61e8` | 4,224,600 |
| `512-720-r2` | 512 | 720 | `bcfc4a497430a228afcbaf378f18f316642ece2a` | 5,799,000 |
| `52-16-ft-r2.1` (weekly) | 52 | 16 | `8c9c1d742ec5893d6419e3aa2ab21ed23e28df85` | 319,440 |
| `90-30-ft-r2.1` (daily) | 90 | 30 | `6e5cb8ee51e0634a45637490f5db43148b2fa6be` | 3,496,760 |

The five S6 additions have configuration, fake-runtime and workflow evidence;
their **real inference qualification remains pending**. See the
[S6 TTM record](s6-ttm-verification.md). R2.1 changes the patching architecture and
requires a frequency-prefix tensor; its daily/weekly profiles are not model sizes.

The 512 profile has 805,280 parameters. Larger parameter counts are not verified;
the repeated main-branch count in repository metadata is not used. These profiles
change model dimensions and patch sizes as well as context. Weight download size
is not a RAM estimate. Weights and Granite runtime declare Apache-2.0.

```sh
python -m pip install -c requirements-ttm.txt '.[cli,ttm]'
tsfm models info ttm_r2
tsfm models info ibm-granite/granite-timeseries-ttm-r2 --variant 1024-96-r2 --json
```

The coordinated runtime uses Granite 0.3.4, Torch 2.8, Transformers 4.56.0,
Tokenizers 0.22.0, scikit-learn 1.7.2, Filelock 3.20.3, and Hub 0.35.3.
Install into your chosen environment explicitly. Core/CLI setup, metadata, and
saved-result operations do not require Granite, Torch, or other model runtimes.
The [P4 record](p4-ttm-verification.md) describes the isolated validation and
existing-model regressions. Optional TiRex dependencies retain their separate
[license notices](../NOTICE).

## Explicit acquisition and terminal setup

```sh
tsfm models pull ibm-granite/granite-timeseries-ttm-r2 --variant 512-96-r2 --cache-dir model-cache
tsfm run examples/configurations/ttm-512-96.yaml --json
```

`--variant` selects that profile's pinned SHA. It cannot be combined with
`--revision`; an explicit revision can instead be used by itself. Omitting both
retains the existing pull behavior of resolving `main`. Model inspection defaults
to the first curated profile. Normal experiment setup and execution never pull
weights automatically. The example's paths are relative to its YAML file.

Use `/new ttm_r2` in the persistent terminal. Select **Checkpoint profile** and
the native-point task. Changing profile sets its pinned SHA and required context;
returning to the same profile preserves an explicit SHA override. The default
short demo is replaced by a bundled 1,600-observation hourly synthetic series;
custom dataset paths are preserved. For R2.1, select your own daily or weekly
dataset and declare `1D` or `7D` respectively; the hourly demonstration is not
silently resampled. Setup retains review, back, and cancellation.

YAML stores the actual repository and exact revision, with
`evaluation.point_statistic: native_point`; no new variant field is required.
For another profile, update context and revision, and ensure evaluation begins
after enough history. The horizon may be 1 through that profile's native horizon.
Future mode uses the same bounds and cadence with no holdout requirement; select
`mode: future_forecast`, configuration schema 2 and a `forecast` block as described
in the [future guide](future-forecasting.md). Local mode uses
`model.source: local`, `adapter_id: ttm_r2`, and a standalone `model.path`, omitting
Hub-specific fields. Copy actual files from a cached snapshot for standalone use.

## Forecast contract

TTM R2 requires a declared fixed interval between one minute and one hour,
inclusive. The R2.1 daily profile requires exactly one fixed UTC day and the
weekly profile exactly seven fixed UTC days. Use `7D`, not an anchored calendar
alias such as `W`; business days, holidays and local DST calendars are not a
regular UTC grid. Unsupported cadence is refused before allocation and, in
guided preparation, before weight transfer. Data is never resampled automatically.
Exact historical context is required: no padding or truncation. MAE checkpoints,
unlisted profiles, covariates, multivariate forecasts, and fine-tuning remain
outside this integration.

Each request standardizes only its detached past window using float64 mean and
population standard deviation; exactly constant history uses scale 1. The model
receives float32 `[1, context, 1]`. Its native `scaling: std` stage remains active
and reverses its own transform. R2.1 additionally receives `freq_token` as a
one-element CPU int64 tensor: `8` for daily or `9` for weekly. This fixed choice
is recorded in checkpoint and runtime identities. `.prediction_outputs` must be
finite with shape `[1, native_horizon, 1]`. The adapter takes the first requested
H values and reverses the
external transform once, yielding float64 predictions in original target units.
Unsafe normalization or inverse results fail. No future values or labels enter
the model. The native head and `prediction_filter_length` are never resized.

The complete reviewed configuration is checked before loading. Local-only
Safetensors loading refuses missing, unexpected, mismatched, or unsafe weights.
Inference is eager, deterministic, and independent per request, with normal
runner cleanup. Checkpoint schema 2 fingerprints the native point and scaling
semantics; existing schema-1 checkpoint identities remain unchanged.

## Comparisons and scope

The native-point task permits matching-case comparisons with baselines and the
other curated families. Actual output labels remain `native_point` for TTM,
`median` for the other pretrained adapters, and `deterministic` for baselines.
`controlled_native_point_v1` records each run's labels and discloses mixed
statistics. It makes no common mean/median claim. Tasks and context lengths must
match; old median-task runs cannot enter this comparison silently. Metrics and
chronological evaluation rules remain unchanged.

OHLC reports continue to require a median-task parent, including when a
native-point run happened to use a median-producing adapter or baseline.

Only the original `512-96-r2` profile has a bounded real CPU check. Other profiles receive
complete configuration/API checks and fake-runtime workflow tests, with no
individual inference claim. See the [approved contract](ttm-contract.md) and
[P4 evidence](p4-ttm-verification.md), followed by the
[S6 TTM evidence](s6-ttm-verification.md). Synthetic checks establish integration,
not forecasting quality, model rankings, or financial usefulness.

The pinned [model card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/blob/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/README.md)
requires external scaling; the [native implementation](https://github.com/ibm-granite/granite-tsfm/blob/4d4c6cf0e345c985d84f6292a4f397caf5e2ba78/tsfm_public/models/tinytimemixer/modeling_tinytimemixer.py)
defines the preserved inner scaler and output behavior.

To opt into the bounded real test with an already acquired standalone checkpoint:

```sh
TSFM_TTM_CHECKPOINT=/absolute/path/to/checkpoint python -m pytest -q -m ttm_smoke tests/test_ttm.py
```

This test never downloads weights. It exercises real replay, future-actual
isolation, original-unit native-point output, and model cleanup with network and
compiler guards.

# S6 TTM profile expansion verification

Post-release follow-up: see [2 November model qualification](v0.3.0-model-qualification.md)
and the [0.3.1 repair record](v0.3.1-verification.md). The original evidence below
retains its date and scope; later checks do not rewrite the 0.3.0 artifacts.

Status on 29 October 2026: five additional TTM profiles implemented and checked
with official configuration fixtures, inert local weights and fake native tensor
responses. **Real inference, installed-runtime checks and platform qualification
for these additions remain pending.** No installation or model-weight download
was performed for this work. The existing `512-96-r2` P4 evidence remains distinct.

## Retained profile decision

Add native R2 horizons 192, 336 and 720 at context 512, plus the R2.1 MSE weekly
52/16 and daily 90/30 profiles. Keep the existing Granite 0.3.4 runtime and all
existing dependency pins. The release's additional-family decision is separate.

The [official model card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)
identifies R2.1 daily/weekly support and required external scaling. The
[pinned runtime profile list](https://github.com/ibm-granite/granite-tsfm/blob/4d4c6cf0e345c985d84f6292a4f397caf5e2ba78/tsfm_public/resources/model_paths_config/ttm.yaml)
includes these branches. The
[pinned preprocessor](https://github.com/ibm-granite/granite-tsfm/blob/4d4c6cf0e345c985d84f6292a4f397caf5e2ba78/tsfm_public/toolkit/time_series_preprocessor.py)
assigns daily token 8 and weekly token 9; the
[pinned model implementation](https://github.com/ibm-granite/granite-tsfm/blob/4d4c6cf0e345c985d84f6292a4f397caf5e2ba78/tsfm_public/models/tinytimemixer/modeling_tinytimemixer.py)
requires `freq_token` for prefix-tuned configurations. Code, runtime and selected
weights declare Apache-2.0; no licensing conclusion is inherited by other models.

| Profile | Exact revision | Config bytes | Weight bytes |
| --- | --- | ---: | ---: |
| `52-16-ft-r2.1` | `8c9c1d742ec5893d6419e3aa2ab21ed23e28df85` | 1,647 | 319,440 |
| `90-30-ft-r2.1` | `6e5cb8ee51e0634a45637490f5db43148b2fa6be` | 1,648 | 3,496,760 |
| `512-192-r2` | `25f4a00a25e1d0b67c40a10b06645d72e2266929` | 1,651 | 3,634,200 |
| `512-336-r2` | `0009f51f52c87473e60ff3a03cd8633ee85d61e8` | 1,651 | 4,224,600 |
| `512-720-r2` | `bcfc4a497430a228afcbaf378f18f316642ece2a` | 1,651 | 5,799,000 |

Configuration SHA-256 identities, in the same order:

```text
2276979b72771bf1c460356b9211fa24479a0546d5c03ddf52cccdb0f86c2b44
abf1b5a26903f1d6b8eb62e66a72bf85bb0ced591b5ffaa655ec031b4d1ec326
012025d3a082d44c593f5f23a0b623be427e628d19094e58e22121941d42f1c8
1c25e5da9d2151047511fd6062ab93ee323a05774a879bbdda856bda406058f6
2b3f031ae0c7a33fb32621feea6381ae63c9c26d170621c3a765fc9d483835c2
```

Weight SHA-256 identities reported by the official Hub file manifests, in the
same order; these were **not verified against downloaded weights**:

```text
5a017bde4dd785b7d366ac200c46bceea356cbf68f7c2b719f73d5b7370d7168
17cdd864fa419e5aad8665789a92b6347e97f137ac2ff3c605887225f503f491
748437cbdbe19030679445aa12737bd14a983ee8d8aa9e8610e317275e0547d5
05e55d45980ad904de5fd2ff5c41b401234073e450acf09fc6152bb99016ced8
db3c51b1321c8f87cd0cfb10a48f896741a314349519e1a14714160bbabe574b
```

## Implemented contract

R2 profiles retain the one-minute-to-one-hour cadence policy. R2.1 weekly and
daily accept exactly seven or one fixed UTC days respectively. Anchored calendar
weeks, business days and DST-dependent local calendars remain unsupported.
Request timestamps are checked again at the direct adapter boundary. Guided
metadata review rejects an incompatible cadence before weight transfer, including
a custom revision whose identity was unknown during initial setup.

The complete configuration allowlist captures architecture, patching, dimensions,
loss, preprocessing and frequency-prefix behavior. No unsafe or partial model
load is introduced: CPU float32, local-only Safetensors, disabled remote code,
strict missing/unexpected/mismatched-key refusal and eager inference remain.
The adapter checks the complete native output shape, then returns the requested
prefix in original units through the existing causal float64 external inverse.
R2.1 adds its fixed frequency token to checkpoint and runtime provenance.

Old TTM point dictionaries and runtime options are unchanged. Historical
schema-2 checkpoints and earlier numerical fingerprints keep their meanings.
The catalog and setup obtain profile-specific bounds/cadence from selected
revisions, without importing an inference runtime. The metadata-only model-form
prototype uses a valid profile cadence before a dataset is selected.

## Executed evidence

The new `tests/test_ttm_expansion.py` covers exact configuration byte hashes,
all five profile bounds and truthful unverified labels, strict fake loading,
native horizons and original-unit inverse, required CPU int64 frequency tokens,
configuration/review/direct-request cadence refusals, forged profile rejection,
historical and future persisted outputs, reload/export without checkpoint access,
and wizard selection into future mode. Existing TTM tests retain native-point,
causality, strict-loading, cleanup, saved comparison and old-record checks.

Executed locally with Python 3.13.6 and the existing development environment:

```sh
python -m pytest -q tests/test_ttm.py tests/test_ttm_expansion.py tests/test_ttm_workflow.py tests/test_model_setup.py tests/test_model_workflow.py tests/test_catalog.py tests/test_future_core.py tests/test_future_setup.py tests/test_future_application.py tests/test_tui_forms.py tests/test_tui_wizard.py tests/test_release_compatibility.py tests/test_project_verification.py
```

Result: **374 passed, 3 skipped**. Skips retain explicitly optional installed
runtime or UI coverage; the TTM real smoke requires authorized local weights.
Black and Ruff pass for all changed Python files, and `git diff --check` passes.
The additional command below passes **117 tests**, covering every selected
settings profile, metadata preparation and future configuration:

```sh
python -m pytest -q tests/test_model_settings.py tests/test_checkpoint_preparation.py tests/test_future_config.py
```

## Remaining qualification

Run representative bounded real CPU inference for the new R2 horizon behavior
and each R2.1 architecture/profile, including cached replay, causal isolation,
safe loading, cancellation and output reload. Retain exact installed versions,
source/weight identities and platform details. Larger/profile status remains
separate; no accuracy, memory, speed, native-Windows or maintainer-inference claim
is inferred from fake-runtime success.

# S6 TOTO 2 decision and offline verification

On 29 October 2026 the owner approved TOTO 2 as the one additional family,
restricted to Python 3.12/3.13 and offline implementation/tests. No installation,
weight download, native-runtime import, real inference or platform qualification
was performed. The [TTM expansion](s6-ttm-verification.md) is a separate feature.

## Compatibility decision and identities

Choose `Datadog/Toto-2.0-4m` at
`8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9`, with `toto-2==2.0.0` from runtime
release `44ea4e88852228039564aa3e76fac26aafac0803`.
[Published metadata](https://pypi.org/pypi/toto-2/2.0.0/json) admits Torch 2.8 and
NumPy 2.1.3 but requires Python ≥3.12. The owner accepted the Python 3.11
exclusion; the numerical core keeps Python 3.11 and no environment manager is added.

[Uni2TS 2.0.0](https://pypi.org/pypi/uni2ts/2.0.0/json) instead requires Torch
<2.5, NumPy 1.26 and SciPy 1.11, conflicting with the shared stack;
[Moirai 2 weights](https://huggingface.co/Salesforce/moirai-2.0-R-small) use
CC-BY-NC-4.0. [TOTO 1 runtime 0.2.0](https://pypi.org/pypi/toto-ts/0.2.0/json)
pins Torch 2.7, NumPy 1.26.4, Transformers 4.52.1 and scikit-learn 1.5.0,
also conflicting. The reviewed
[TOTO 2 release requirements](https://github.com/DataDog/toto/blob/44ea4e88852228039564aa3e76fac26aafac0803/toto2/pyproject.toml)
are the reference; moving `main` has different requirements. Pin GluonTS 0.16.2,
because [0.17 metadata](https://pypi.org/pypi/gluonts/0.17.0/json) requires
Torch ≥2.10. The [candidate constraints](../requirements-toto2.txt) are metadata
compatible; actual installation/import checks remain pending.

The [pinned config](https://huggingface.co/Datadog/Toto-2.0-4m/blob/8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9/config.json)
is retained unmodified: 592 bytes, SHA-256
`7a926d130e401ab0c5fdb3564f46c8d917bd05c7b3ae26b9c22d2da2ef01d2d8`.
The official Hub manifest reports `model.safetensors` as 16,582,848 bytes and
SHA-256 `316660d5afb47943e531f39242e0b02ca0b8bb73be5709dfe07ca80dfce9805e`;
this was not verified against downloaded weights.

Selected model/runtime declare Apache-2.0. The
[official dd-unit-scaling source notice](https://github.com/DataDog/toto/blob/44ea4e88852228039564aa3e76fac26aafac0803/dd_unit_scaling/dd_unit_scaling/__init__.py)
states Apache-2.0; an absent PyPI license field is not treated as evidence.

## Implemented contract and evidence

Context 32–2,048 in multiples of 32 and horizon 1–128 are conservative project
limits, not upstream architecture limits. Existing schema-1 record fields carry
those bounds for this adapter; earlier checkpoint identities remain unchanged.
The [reviewed implementation](https://github.com/DataDog/toto/blob/44ea4e88852228039564aa3e76fac26aafac0803/toto2/toto2/model.py)
defines preserved causal scaling, `asinh`/inverse, quantile sorting and median
feedback. Decode blocks are fixed at 32; fallback scaling and real-space caps are
disabled. Infinity repair is overridden with refusal. Inputs contain only
observed history with no padding/imputation, and the complete quantile tensor is
validated before selecting the median. The loader bypasses the Hub mixin and
uses reviewed dataclass construction plus strict local Safetensors.

Python guards precede resolution, preparation, transfers and native imports.
Saved execution provenance requires Python 3.12/3.13; detached reload is allowed
on Python 3.11. Historical guided review checks only consumed history for
float32 representability before metadata/weights.

`tests/test_toto2.py` and `tests/test_toto2_workflow.py` check configuration hashes,
strict safe loading, native tensor shape/order/finiteness, fixed decoding,
nonfinite refusal, request/grid preservation, input bounds, runtime/Python guards,
optional-import isolation, historical/future outputs, causality, saved-provenance
tampering, reload/export without weights or model imports, guided fake
acquisition/cache replay, wizard correction, cancellation/failure records and
managed worker completion/reaping. `tests/toto2_helpers.py` supplies inert weights
and fake tensors. It simulates Python 3.12 and candidate package versions even
when core tests use another interpreter; those values are fixture provenance.

Executed with the existing Python 3.13.6 development environment:

```sh
python -m pytest -q tests/test_toto2.py tests/test_toto2_workflow.py tests/test_runtime_diagnostics.py
```

Result: **71 passed**. The additional existing-family/shared-boundary regression
command passed **601 tests, 2 optional skips**:

```sh
python -m pytest -q tests/test_catalog.py tests/test_model_workflow.py tests/test_model_settings.py tests/test_model_setup.py tests/test_runtime_diagnostics.py tests/test_assistant_evidence.py tests/test_optional_boundaries.py tests/test_ttm.py tests/test_ttm_expansion.py tests/test_ttm_workflow.py tests/test_chronos2.py tests/test_timesfm.py tests/test_tirex2.py tests/test_future_core.py tests/test_future_application.py tests/test_checkpoint_preparation.py
```

Black/Ruff and `git diff --check` passed for the final changed Python files.
Local documentation links and parsing the shipped future YAML example passed;
the example was not executed with a real model. The
configured real-model smoke checks remain skipped without authorized local
weights. No real-runtime regression or accuracy result is implied.

## Pending qualification

Authorize an isolated runtime installation/import check, resolve actual dependency
or platform failures, then retain bounded real CPU inference, cache replay,
causality, native inverse behavior, cancellation/recovery and artifact reload with
exact source/runtime/weight identities. Record OS/Python/terminal versions for
each advertised platform. S6/S9 real inference and Windows qualification remain
open; fake-runtime checks establish neither forecast quality nor native support.

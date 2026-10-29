# Datadog TOTO 2

The optional `toto_2` adapter selects `Datadog/Toto-2.0-4m` and produces
univariate median forecasts. It requires Python **3.12 or 3.13**. Python 3.11 can
inspect the catalog and reload/export valid saved results, but cannot resolve,
prepare or run this model. Configuration, loading and workflows have offline
fixture/fake-runtime checks; installed imports, real inference and native
platform qualification remain pending. See the [S6 record](s6-toto2-verification.md).

## Checkpoint and bounds

| Setting | Integration contract |
| --- | --- |
| Exact revision | `8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9` |
| Context | 32–2,048 observations, multiples of 32 |
| Horizon | 1–128 observations |
| Data | One finite regular UTC series, any positive fixed cadence |
| Output | Native sorted quantile 0.5, labelled `median` |
| Tasks | `median` or explicit mixed-output `native_point` |
| Weight file | 16,582,848 bytes; not a RAM estimate |

These context/horizon bounds are conservative **project integration limits**,
not upstream architectural limits. For this adapter, the existing checkpoint
fields named `native_context_length` and `native_prediction_length` carry
2,048/128 so shared validators enforce those limits. Larger variants and other
TOTO generations are not curated.

## Optional setup

These commands describe the integration interface. Real-runtime installation
and inference qualification remain pending; installation and weight acquisition
are separate explicit actions:

```sh
python -m pip install -c requirements-toto2.txt '.[cli,toto2]'
tsfm models info Datadog/Toto-2.0-4m --json
tsfm models pull Datadog/Toto-2.0-4m --variant 4m
tsfm run examples/configurations/toto2-future.yaml --json
```

The example uses the bundled 32-row synthetic series and cached weights, with
future accuracy unavailable. Historical evaluation requires additional
observations beyond the context. Use `/new toto_2` in the terminal, choose
**2.0 4M**, review fixed preprocessing, choose a dataset and mode, then confirm
the reviewed configuration. Invalid patch lengths are reprompted. Assistant
drafting uses the same validation and cannot authorize preparation or inference.

The candidate [constraints](../requirements-toto2.txt) coordinate TOTO 2.0.0,
Torch 2.8, Hub 0.35.3, Safetensors 0.6, NumPy 2.1.3, GluonTS 0.16.2,
dd-unit-scaling 0.1.0, unit-scaling 0.3.5, Einops 0.8.2, Jaxtyping 0.3.11,
Lightning/pytorch-lightning 2.4.0, SciPy 1.15.3 and Matplotlib 3.10.8.
GluonTS 0.17 requires a newer Torch stack and is excluded. Package-metadata
compatibility does not establish successful installation/imports.

## Numerical and loading behavior

Only detached historical context enters the model as float32 `[1,1,context]`,
with an all-observed mask and zero series identifier. No external scaling,
history padding, imputation, resampling or clipping is applied. Native causal
standard-deviation scaling, `asinh` and inverse remain active. Decode blocks are
fixed at 32, with native median feedback between blocks. Runtime provenance
records `has_missing_values=False`, `scaler_fallback_min_obs=0`,
`quantile_real_cap_k=0.0`, and the native normalization/sorting policies.

Upstream replaces infinities during inverse transformation. This integration
overrides that hook to reject nonfinite quantiles before feedback or sorting.
The full `[9,1,1,horizon]` tensor must be ordered and finite; index four becomes
an original-unit median output with preserved request identity and timestamps.

The complete exact 4M configuration is checked. Loading constructs the reviewed
installed dataclass/model directly, reads verified local Safetensors and loads
parameters strictly. It never calls the upstream Hub mixin, enables remote code
or falls back to unsafe weights. CPU float32, eager inference and batch size one
are fixed. Saved execution must identify Python 3.12/3.13; reload does not require
the current interpreter or installed runtime to match that execution environment.

Selected [weights](https://huggingface.co/Datadog/Toto-2.0-4m) and
[runtime code](https://github.com/DataDog/toto/tree/44ea4e88852228039564aa3e76fac26aafac0803/toto2)
declare Apache-2.0, separately from TSFM-LAB and optional dependency licenses.
The [dd-unit-scaling source notice](https://github.com/DataDog/toto/blob/44ea4e88852228039564aa3e76fac26aafac0803/dd_unit_scaling/dd_unit_scaling/__init__.py)
also states Apache-2.0 despite its empty PyPI license field. No upstream code or
weights are bundled. No accuracy, speed, RAM or native-platform claim follows
from the offline tests.

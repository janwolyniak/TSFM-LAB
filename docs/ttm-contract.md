# Proposed TTM integration contract

Status: **proposal awaiting owner decisions** on 26 October 2026. This document
does not authorize dependency installation, runtime changes, or inference. No
TTM adapter or new evaluation task has been implemented. It makes P4's required
contract step reviewable under the [release plan](next-release-plan.md).

## Initial checkpoint choices

Curate three R2, MSE-trained, non-frequency-token profiles from the same
`ibm-granite/granite-timeseries-ttm-r2` repository:

| Profile | Exact revision | Required history | Native horizon | Safetensors bytes |
| --- | --- | ---: | ---: | ---: |
| 512→96 | `d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f` | 512 | 96 | 3,240,592 |
| 1024→96 | `1a382618f3c5b63836281ad632e54724e55e85a5` | 1,024 | 96 | 11,879,480 |
| 1536→96 | `f6ed09d67cf42e4877f7fa33d98326911385693d` | 1,536 | 96 | 12,344,144 |

All model weights declare Apache-2.0. The smallest profile has 805,280 stored
parameters. Larger counts remain unverified; upstream repository API metadata
repeats the main profile's count for other branches, so it must not be copied.
Stored weight size is not a runtime RAM estimate. The larger profiles also
change dimensions and patch sizes; their complete configurations need validation.

These are branches of one repository. Catalog and terminal setup must expose an
explicit profile selection and pinned revision default. YAML continues to store
the actual repository and exact revision. Changing profile resets its default
revision; returning to the same profile preserves a deliberate override. Normal
setup/runs do not download weights or choose a different profile automatically.

The 512→96 config and weights are already cached and their local hashes match:

- Config: 1,563 bytes, SHA-256
  `5e2367547c103e92cb8ebc63cbd5ad4d7bf83facad5a2a2ec261fa770ed659d5`.
- Weights: SHA-256
  `a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710`.

The proposed initial frequency policy is a declared regular interval between
one minute and one hour inclusive, reflecting the R2 model card's stated scope.
Reject unsupported intervals before model allocation; never resample or convert
units automatically. Daily/weekly R2.1, MAE-trained variants, other native
horizons, covariates, multivariate forecasts, and fine-tuning are deferred.

Sources: pinned [512→96](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/blob/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/config.json),
[1024→96](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/blob/1a382618f3c5b63836281ad632e54724e55e85a5/config.json),
[1536→96](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/blob/f6ed09d67cf42e4877f7fa33d98326911385693d/config.json),
and the [model card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/blob/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/README.md).

## Forecast and causal scaling

Require exactly the chosen profile's historical context length. Reject shorter
or longer context rather than padding or accepting upstream truncation.
Generate the checkpoint's native 96 predictions, then explicitly select the
first requested `H`, where `1 ≤ H ≤ 96`. Do not change
`prediction_filter_length`, resize a head, roll predictions, or use an automatic
`get_model()` fallback.

The model card requires independent external standard scaling. For each request,
compute the mean and population standard deviation using only its detached
historical values, in float64. Use scale 1 for an exactly constant series. Feed
the resulting standardized history as CPU float32 `[1, context, 1]`, with no
future values or labels. Reject nonfinite normalization or inverse results.

Preserve the checkpoint's native `scaling: std` stage as well: the pinned
implementation computes its own historical location/scale and reverses that
inner transform. Its `.prediction_outputs` are therefore in external-standardized
units. Reverse the external transform once to recover original target units.
Do not remove the native scaler or interpret its output as already being in the
original dataset units. Record both scaling stages and the external policy ID.

Use the direct model class, local-only Safetensors, strict loading diagnostics,
CPU float32, evaluation/inference mode, and normal cleanup. Reject missing,
unexpected, mismatched, random-initialized, or unsafe weights. The native output
must have exactly `[1, 96, 1]`, with finite values. No distribution or samples
are fabricated from this point head.

Implementation source: [pinned Granite 0.3.4 model](https://github.com/ibm-granite/granite-tsfm/blob/4d4c6cf0e345c985d84f6292a4f397caf5e2ba78/tsfm_public/models/tinytimemixer/modeling_tinytimemixer.py).

## Point task and comparisons — proposed decision

TTM's MSE-trained head produces a **native point forecast**. MSE training alone
does not verify a conditional mean, and the output must not be labelled median.
Add an explicit `native_point` evaluation task beside the existing `median`
task. The existing task remains the default for existing models and retains its
old output acceptance rules, records, and hashes.

Recommended policy: the new task permits comparison of each model's declared
point output on identical cases and metric definitions. Retain actual prediction
labels: TTM `native_point`, existing pretrained adapters `median`, and baselines
`deterministic`. Explain that these outputs can target different statistical
functionals. Do not claim they all estimate a verified mean or median.

Require matching evaluation tasks and a distinct comparison policy identifier,
`controlled_native_point_v1`. Old median-task runs are never silently mixed with
new native-point runs. MAE, RMSE, MASE, case alignment, aggregation, and fixed
pre-evaluation MASE scaling remain numerically unchanged. Terminal setup exposes
the selected task; manual YAML is validated rather than silently rewritten.

The narrower alternative is to admit only TTM and deterministic baselines to the
new task. Owner selection is required before implementing either policy.

TTM needs point-only nested checkpoint metadata with a new serialization version
covering native output semantics, MSE loss, release/profile, frequency policy,
exact native bounds, and normalization. Preserve schema-1 checkpoint field sets,
fingerprint bytes, old environment records, and verification snapshots. The
containing YAML/result formats can remain schema 1; no old file is rewritten.

## OHLC eligibility — separate owner decision

The bounded recommendation is to keep OHLC reports on the existing median task.
TTM forecasts can still be evaluated on financial price series; they would not
yet enter the OHLC report. A generic native-point task using a baseline or a
median-producing adapter must not accidentally bypass this policy: validate the
parent evaluation task during both report creation and independent report reload.

Alternatively, the owner may explicitly extend OHLC eligibility to native-point
runs. That change must retain actual output labels and all existing positive
price, origin/entry/exit timing, threshold, fee, slippage, short financing,
capital, and accounting rules. It must not reinterpret an MSE output as a median.
No such extension is assumed by this proposal.

## Runtime — proposed decision and resource request

Use `granite-tsfm==0.3.4`, which supports Python 3.11–3.13 and Torch 2.8. Keep
Hub 0.35.3 and NumPy 2.1.3. A coordinated shared runtime requires Transformers
4.56.0, Tokenizers 0.22.0, scikit-learn 1.7.2, and Filelock 3.20.3, plus the
missing Granite dataset dependencies. This preserves one inference interpreter;
separate family environments would require a new execution/routing design.

The static required closure passed for 72 packages / 142 requirements; it has
not been executed. Ten macOS/Python 3.13 wheels total 27,839,485 bytes, about
28 MB, without a Torch download. The representative TTM weights can be reused
from cache. No additional restrictive dependency license was found in this audit.
See the [pinned runtime requirements](https://github.com/ibm-granite/granite-tsfm/blob/4d4c6cf0e345c985d84f6292a4f397caf5e2ba78/pyproject.toml).

Test first in a temporary environment, preserving global packages. Its inherited
`mlxtend` is unrelated to forecasting and conflicts with the older NumPy and
required scikit-learn; do not repair it incidentally or claim that the whole
inherited environment is dependency-clean. Verify the model-required closure
and retain the separate clean core/CLI package check.

After installation approval, run one bounded cached TTM CPU check and brief
replay checks for Bolt Tiny/Small, Chronos-2 Small, TimesFM, and TiRex-2. No paid
compute, benchmark campaign, fine-tuning, or larger checkpoint download is included.

## Acceptance evidence after contract approval

- Hand-check external scaling/inverse scaling, constant histories, native output
  prefix selection, exact context, frequency refusal, and full profile validation.
- Prove that changing future actuals cannot change earlier predictions or scalers.
- Verify native-point labels, explicit comparison policy, cross-task refusal,
  selected OHLC eligibility, and unchanged old median records/reports/hashes.
- Exercise all three profiles through catalog, terminal selection, YAML,
  strict fake loaders, cancellation/back navigation, and saved-result operations.
- Run the representative cached CPU inference/replay/cleanup check; mark larger
  profiles compatibility checked without claiming individual inference.
- Complete configured tests, formatting/linting, docs/examples, source/wheel
  contents, license files, and clean installed workflows before a local commit
  for the implementation. Publication remains a separate approval.

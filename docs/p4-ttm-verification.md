# P4 TTM integration verification

Date: 26 October 2026. Scope: the approved TTM R2 contract and coordinated
shared runtime, with local CPU inference only. This is integration evidence,
not a benchmark or evidence of forecast superiority. Source version remains
0.1.0; package publication is separate.

## Checkpoint and inference

The representative checkpoint is
`ibm-granite/granite-timeseries-ttm-r2` at
`d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f`, reused from the existing cache.
No model weights were downloaded during P4 verification.

| Consumed file | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 1,563 | `5e2367547c103e92cb8ebc63cbd5ad4d7bf83facad5a2a2ec261fa770ed659d5` |
| `model.safetensors` | 3,240,592 | `a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710` |

The direct model check found 805,280 parameters and empty missing, unexpected,
mismatched, and error loading diagnostics. It retained native `scaling: std`,
standardized each supplied history in float64 using population standard
deviation, generated `[1,96,1]`, selected the first four values, and reversed
the external scaling once. Constant-history scale 1, exact replay, future-actual
isolation, released model references, and unchanged Torch default dtype/matmul
precision passed. Network and compiler guards recorded zero attempts.

The real application workflow used the original 1,600-row hourly synthetic
[fixture](../examples/datasets/ttm-seasonal.csv), context 512, horizon/stride 4,
targets 512–527, and MASE lag 24. Cached, replayed, and standalone runs each
produced four successful windows and 16 finite aligned native-point cases.
Predictions were exactly equal across those routes. Baseline comparisons used
`controlled_native_point_v1` with actual labels retained. A separate real Bolt
Tiny run in the same task passed mixed native-point/median comparison. Mutating
future actuals left the first real application forecast unchanged. CSV, JSON, and Parquet
exports reloaded to identical prediction values. No model-quality conclusion
follows from this constructed series.

## Shared runtime

The temporary environment reused existing Torch dependencies and installed the
approved official wheels. Seventeen runtime wheels / 4,030 package files matched
their wheel bytes. The required closure of 72 distributions / 142 dependency
requirements passed. Ten new/replacement wheels totalled 27,839,485 bytes.

| Component | Version |
| --- | --- |
| Python | 3.13.6, macOS arm64 |
| Granite TSFM | 0.3.4 |
| Torch | 2.8.0 |
| Transformers / Tokenizers | 4.56.0 / 0.22.0 |
| Hub / Safetensors / Accelerate | 0.35.3 / 0.6.2 / 1.12.0 |
| scikit-learn / Filelock | 1.7.2 / 3.20.3 |
| NumPy / Pandas | 2.1.3 / 2.3.3 |

Five cached regressions passed under this runtime: Bolt Tiny, Bolt Small,
Chronos-2 Small, TimesFM 2.5 200M, and TiRex-2 main. Each used one C32/H4
origin plus exact replay, retained median labels and timestamps, produced finite
original-unit values, and released adapter state. Network/compiler attempts
remained zero; process defaults were preserved. The checks took about 5.7 seconds
after imports; this is not a comparative inference-speed measurement.

Global package versions and inspected package/metadata hashes were unchanged.
The inherited environment has unrelated `mlxtend` conflicts with NumPy and
scikit-learn; it is not claimed to be globally dependency-clean. No unrelated
package was repaired. Core/CLI package qualification uses a separate clean
environment without model runtimes. See [constraints](../requirements-ttm.txt).

## Contract and workflow verification

Routine tests use inert weights and fake runtimes for all three full reviewed
configurations. They cover strict loading, causal scaling/inverse arithmetic,
native horizon prefixes, exact context, minute-to-hour frequency refusal,
task/output labels, profile selection and SHA overrides, back/cancel behavior,
saved schema-2 validation, and runtime-free reload/comparison/export.

New native-point comparisons retain every actual output label and reject mixed
evaluation tasks. Known adapter labels are bound to their declarations. Existing
median-task output acceptance, checkpoint schema-1 field sets/fingerprint bytes,
and old comparison representation are preserved. OHLC derivation and independent
report reload both reject native-point parents, including baseline/median-output
runs, without changing timing, cost, or accounting rules.

The source/wheel verification scripts require the new backend, pure profile/task
modules, licensed fixture, example, constraints, and guides in their appropriate
artifacts. Installed qualification checks all seven catalog entries, three TTM
profiles, lazy optional imports, schema-2 saved operations without weights, and
the existing median/finance arithmetic fixtures.

The full offline suite passed **1,686 tests**, with five opt-in model tests
skipped, in 58.81 seconds. The real TTM smoke passed separately in 2.44 seconds,
including exact replay, future-actual isolation, and weak-reference cleanup.
Independent code review found no actionable correctness/security/regression
issue. Black checked 127 Python files; Ruff and whitespace checks passed. All
nine example configurations parsed; the two TTM fixture copies match bytewise.
Documentation checks covered 37 Markdown files, 267 local links, and 53 shell
blocks, with no errors.

The preparation source archive contained 202 files / 56 required source paths;
the wheel contained 73 entries and the three declared license files. Its core
dependencies remain NumPy and Pandas. All ten installed qualification stages
passed in 16.428 seconds in the isolated core/CLI/Parquet environment, with model
and kernel libraries absent; that environment's dependency consistency passed.
A separate isolated process reloaded the real saved TTM result, compared it with
baselines, and re-exported all three formats exactly while optional model imports
and current project evidence lookup were blocked. An older real schema-1
median-task TiRex result also reloaded successfully.

Final source/wheel hashes and installed-byte identities are retained beside the
final artifacts in local verification reports. They are checked independently
of source tests; neither wheel contents nor a Git commit is inferred from the
historical working-tree check.

## Evidence and limits

The checked-in project registry records the representative check against exact
config/weight hashes, tested dirty source provenance, and runtime versions.
New runs snapshot that evidence independently of checkpoint identity. The two
larger profiles remain **compatibility checked, individual inference not run**.
Their full configurations and terminal/fake-runtime paths are tested; no larger
parameter count, inference result, or performance claim is inferred.

Detailed local evidence is retained under
`/private/tmp/tsfm-next-models-01a174c4/p4-verification/`: `verification.json`,
`direct-ttm-cpu-check.json`, `existing-adapters-p4-runtime-regression.json`,
`runtime-audit.json`, `installed-required-closure.json`,
`additional-application-checks.json`, `ttm-smoke.xml`, and qualification reports.
Temporary paths are a local audit trail and are not dependencies of saved runs.
The [example](../examples/configurations/ttm-512-96.yaml), opt-in local smoke
test, and package scripts provide the reproducible workflows. The approved
[contract](ttm-contract.md) and pinned sources in the [guide](ttm.md) describe
the boundaries: no benchmark campaign, larger download, training, paid compute,
accelerators, covariates, or package publication is included.

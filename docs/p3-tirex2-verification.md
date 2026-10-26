# P3 TiRex-2 verification

Prepared locally on 26 October 2026. P3 adds `tirex_2` with four curated
training/forecast profiles, safe CPU float32 median inference, terminal/YAML
selection, native-bound checks, and saved-result use without model libraries.
It updates the shared Hub pin to 0.35.3 and adds the required optional-runtime
license notices. The package version remains 0.1.0; no release is published.
See the [TiRex-2 guide](tirex2.md) and [next release plan](next-release-plan.md).

## Pinned profiles and file identities

All four checkpoints have 82,501,744 parameters and Apache-2.0 model weights.
They are distinct repositories and training profiles, not model-size tiers.

| Repository | Revision | Context / horizon | Config SHA-256 | Weights SHA-256 |
| --- | --- | --- | --- | --- |
| `NX-AI/TiRex-2` | `30788a311699acba7e6393ad9700ea7b7cb9bb2d` | 2,048 / 320 | `e591152ebc5bd7245b5258cd26a52b57c1f6cfe60d6743111856ab0b12f7425e` | `17760e4eee4c14e0d8bad8b9d9ce6f5076aaeeb9d956a88b8e95cf1493a1c176` |
| `NX-AI/TiRex-2-fevbench` | `acd6153b8ee2179f38cfd3cc4fd64a3c045a424f` | 2,048 / 320 | same as default | `18c177e40b8d5d8e6ff487cb9be3e3ff2880412dec12a46da0b3261b83825c77` |
| `NX-AI/TiRex-2-gifteval-zs` | `c94872d1c8caacbe7755cbfa01bc5c4c5c5ddb7f` | 8,192 / 928 | `4ff442a0eb80e86b4205b750a6e7ffb794b17130627d7d14b99140c5bfb999ea` | `fe452780212fcd27bb5c822c88eac97334200a35dd5f74fc0e4317fb31a37ccb` |
| `NX-AI/TiRex-2-gifteval-pretrain` | `54f88ac06ccfba9fe3e727aa8856b39775481f0f` | 8,192 / 928 | same as GiftEval zero-shot | `b184d7ea3847a7cd769b24fedeb8daaae95094cb06aa58470ac6003da8410371` |

Default/FEV configs contain 1,639 bytes; GiftEval configs contain 1,668 bytes.
The default Safetensors file has 330,038,664 bytes; each other file has
330,038,640 bytes. Only the default weights were downloaded and locally hashed
for P3. Other weight identities come from pinned upstream file manifests;
their configurations, architecture, license, preprocessing, native limits, and
loader compatibility were inspected. No separate inference result is claimed
for those profiles. The guide links each pinned official configuration.

Default's consumed config/weights/card fingerprint was
`c2dfa83efdaaf7f6598816a95d0f75ee0941fe65f2af8936210920561cf62220`.
The standalone copy had the same fingerprint. Old schema-1 checkpoint fields,
fingerprint serialization, and historical verification evidence are unchanged.
The native TiRex configuration has no Transformers architecture declaration;
reviewed-profile recognition derives its adapter metadata without rewriting it.

## Runtime decision and permissions

The owner approved a coordinated Hub 0.35.3 pin, existing-model CPU regressions,
required NXAI notices, a temporary dependency setup, the approximately 330 MB
default checkpoint download, and brief CPU checks. A single model interpreter
continues to serve all families; no environment manager or interpreter routing
was introduced. The global installation was preserved.

The isolated test environment uses Python 3.13.6/macOS arm64, Torch 2.8.0,
Hub 0.35.3, TiRex-2 0.4.0, Safetensors 0.6.2, FlashRNN 1.0.8, mLSTM Kernels
2.0.6, Ninja 1.13.2, NumPy 2.1.3, Narwhals 2.13.0, einops 0.8.1, and
matplotlib 3.10.8. Pristine Chronos 2.2.0 and TimesFM 3.0.2 packages were used
for the shared-Hub regressions; Transformers remains 4.55.0.
All 564 installed package files across seven official wheels matched their
wheel bytes. Those wheels totalled 1,674,661 bytes, including the reused Chronos
wheel and TimesFM package. No shared dependency was silently upgraded.

The required inference dependency closure passed for 48 packages. Whole-environment
`pip check` reports the same inherited, unrelated issue in both global and
temporary environments: `mlxtend 0.24.0` requests NumPy ≥2.3.5 while the existing
NumPy is 2.1.3. Neither package was changed to repair that unrelated environment.
This check does not establish that the entire inherited global environment is
dependency-clean. The separate core/CLI installed-wheel check is isolated.

**Built with technology from NXAI**

The optional `mlstm-kernels` dependency uses the NXAI Community License, with
attribution/redistribution provisions and additional commercial terms. The owner
approved its explicit inclusion. [NOTICE](../NOTICE) and the
[complete dependency license](../licenses/mlstm-kernels-2.0.6.txt) ship in source
and wheel metadata; catalog, terminal, and guide notices distinguish these terms
from Apache-2.0 model weights and project code.

## Eager CPU execution boundary

Upstream runtime 0.4.0's public `compile=False` only skips selected recurrent
compilation. Its residual-block class is unconditionally decorated with
`torch.compile`. The first direct check completed in 4.8 seconds before this
was identified, and is retained as an initial implicit-compilation attempt.
Recursive compiler disabling around prediction alone then failed under an
explicit compilation-attempt guard, before graph tracing.

The final adapter restores ordinary `nn.Module.__call__` and `_call_impl` on
two instance-local residual-block subclasses, then recursively disables
compilation for its prediction callable. It does not patch upstream files,
global classes, forward implementations, or process compiler configuration.
A guarded real check confirmed unchanged parameter/buffer identities, state
keys, forward methods/hooks, default dtype, and matmul precision, with zero
Dynamo compilation entries. Model/predict references were released afterwards.
The two C32/H4 prediction passes took about 0.13 seconds after construction;
these timings are diagnostic, not speed benchmarks.

## Executed experiment and regression checks

The project adapter's guarded opt-in smoke passed in **3.36 seconds**: one
C32/H4 origin, four finite aligned native medians, exact replay, future-actual
isolation, zero connection/compilation attempts, unchanged dtype/matmul settings,
and released runtime references.

The complete default-checkpoint workflow used the original bundled seasonal
series with C8/H4/stride4, unchanged scoring, and four successful origins/16
forecast cases. Cached replay and standalone-file execution matched exactly.
Baseline comparison accepted matching forecast cases. CSV, JSON, and Parquet
exports reloaded with identical prediction values. Socket/DNS/HTTP and Dynamo
guards recorded no attempts. The bounded workflow took about 2.95 seconds.

A separate core/CLI/Parquet process loaded the real saved TiRex result,
compared it with the baselines, and exported all three formats while optional
model/kernel imports and lookup of current project evidence were blocked.
The saved historical snapshot remained authoritative.

Hub 0.35.3 regressions passed for Bolt Tiny, Bolt Small, Chronos-2 Small, and
TimesFM 2.5. Each used existing weights, one C32/H4 origin, exact replay, four
finite aligned medians, cleanup, unchanged dtype/matmul precision, and zero
network attempts. The four checks took 4.57 seconds in total; no additional
weights were downloaded.

The offline suite passed **1,535 tests**, with four opt-in real-model tests
skipped, in 55.89 seconds. TiRex's real test and the existing-model checks were
executed separately as above. Fake-runtime tests cover both native profiles,
all four repository choices, native endpoints, dynamic-activation/configuration
rejection before allocation, strict alias restoration, eager execution, cleanup,
runtime gates, malformed outputs, detached independent history, persisted flag
tampering, canonical repository case, and saved-result operations without models.
Independent source/configuration review found no actionable correctness issues.

Black checked 117 Python files; Ruff and whitespace checks passed. All eight
example configurations parsed. Documentation checks covered 34 Markdown files,
241 local links, and 49 shell blocks. Model information and notices were
inspected at 80 columns.

Source archive and wheel inspection passed: 189 source files, 47 required
source paths, and 69 wheel entries, including the exact three declared license
files. The project's core dependencies remain NumPy/pandas. The preparation
wheel was installed without dependencies into the isolated core/CLI/Parquet
environment; all nine installed workflow stages passed in 13.372 seconds, with
model/kernel libraries absent. That environment's dependency consistency passed.
Final source/wheel hashes and installed-byte identities are recorded alongside
the final artifacts rather than inferred from source tests.

## Evidence and limitations

The representative project check records P3's working tree based on `6fd7186`,
marked dirty, with exact adapter/profile hashes in execution evidence. It does
not claim that the baseline commit contains P3. Other checkpoints remain
compatibility checked without individual inference. Historical checks retain
the runtime/source versions that were actually used.

Detailed disposable evidence is under
`/private/tmp/tsfm-next-models-01a174c4/p3-verification/`: native configs/cards and
file manifests, runtime/license audits, required dependency closure, direct
compiler-boundary checks, marked smoke XML, experiment records, no-runtime
exports, and shared-Hub regressions. Weights remain outside Git.

The evidence establishes bounded integration, without a model ranking, general
forecast-quality claim, runtime RAM estimate, GPU/MPS support, benchmark campaign,
multivariate/covariate support, streaming state, or ONNX backend. No package
release/tag is published. Publishing or merging P3 remains subject to separate
owner approval under [CONTRIBUTING.md](../CONTRIBUTING.md).

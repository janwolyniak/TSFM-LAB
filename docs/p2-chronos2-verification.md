# P2 Chronos-2 verification

Verified locally on 26 October 2026. This package adds the optional
`chronos_2` adapter and curated Small/standard choices to the existing shell,
guided terminal, checkpoint, and saved-result workflows. It does not change
the package version or publish a release. See the [usage guide](chronos2.md).

## Exact checkpoint identities

| Choice | Repository / revision | File | Bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| Small | `autogluon/chronos-2-small@ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a` | `config.json` | 969 | `f2780468adc8b16322aa0b6e28b26476c9401ecf64ab53292b6d2b0d3f423722` |
| Small | same | `model.safetensors` | 111,749,048 | `492290ae82bb89f9769e3479ce90b3179de1f33e600c34daa0352531538b23cd` |
| Standard | `amazon/chronos-2@29ec3766d36d6f73f0696f85560a422f50e8498c` | `config.json` | 1,067 | `ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031` |
| Standard | same | `model.safetensors` | 477,930,472 | `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42` |

Small's files were explicitly downloaded and their hashes checked. Standard's
configuration, upstream weight manifest, architecture, license, dimensions,
native limits, and loader API were inspected; no standard weights were downloaded
or used for this milestone. Both profiles are Apache-2.0. Source:
[pinned Small files](https://huggingface.co/autogluon/chronos-2-small/tree/ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a),
[pinned standard files](https://huggingface.co/amazon/chronos-2/tree/29ec3766d36d6f73f0696f85560a422f50e8498c).

The consumed Small config/weights/card fingerprint was
`f03383c379b048d9da13720d9c4510b6b73a0db3d483444bde075b66b05633da`.
Its standalone copy retained that fingerprint. Historical checkpoint record
shapes and fingerprint serialization remain unchanged.

## Runtime and source provenance

The owner's existing Chronos installation contained local CPU DataLoader changes.
It was preserved. The owner authorized a pristine official Chronos 2.2.0 wheel
in a temporary environment reusing existing dependencies, the approximately
112 MB Small download, and a brief CPU check. The 72,363-byte wheel had SHA-256
`39a26ddaad1193054e42cec239fe87d97344d2d65cacb3fd6ae4a75e9e851ae5`.
All 15 installed Chronos package files matched the wheel; the four Chronos-2
implementation files also matched official upstream commit
`75a2dfe228e39d55283f477c01399b9e53d89661`.

The check used Python 3.13.6 on macOS arm64, Torch 2.8.0, Transformers 4.55.0,
Hub 0.34.4, Safetensors 0.6.2, Accelerate 1.12.0, NumPy 2.1.3, pandas 2.3.3,
PyYAML 6.0.2, and PyArrow 23.0.1. No shared dependency was upgraded. Model
arithmetic was CPU float32 with one Torch thread; the default dtype was unchanged.

Project inference was checked in the P2 working tree based on `7fab984`, marked
dirty in its historical evidence. That source field does not claim the baseline
commit contains P2. The temporary execution evidence additionally records the
tested adapter/profile file hashes. The retained project-check entry identifies
the actual source state and runtime; it is not a signed attestation.

## Executed integration checks

The bundled original synthetic seasonal series was evaluated with context 8,
horizon 4, stride 4, and the unchanged median/MASE contract. Small completed four
origins and 16 finite, aligned forecast cases with no failed windows.

- A second exact-cache run reproduced predictions bit for bit.
- A standalone file copy produced exactly the same predictions and fingerprint.
- Last-value and seasonal-naive runs used identical forecast cases; controlled
  comparison accepted them.
- CSV, JSON, and Parquet exports reloaded with identical prediction values.
- Socket/DNS/HTTP connection audit hooks blocked network use throughout the
  workflow; the recorded attempt list was empty.
- A separate core/CLI/Parquet environment reloaded the real saved Small result,
  compared it, and exported all three formats with model-runtime imports blocked.
  It also rejected any lookup of today's project evidence, proving reload uses
  the saved snapshot.
- The opt-in C32/H4 smoke test produced finite native medians, exact replay, and
  identical predictions after changing future actuals. The runner released its
  pipeline and Torch references after each run.
- A direct upstream check confirmed native 0.5-channel semantics, a 1,025-step
  refusal, and release of model/pipeline objects.

The final bounded workflow took about 3 seconds after interpreter startup.
The separate adapter smoke passed in 2.90 seconds. These are diagnostic wall
times, not performance benchmarks. Official upstream CPU pin-memory warnings
were retained; no upstream code was patched for the checks.

## Contract, UI, and package checks

The offline tests cover both native quantile grids, strict load diagnostics,
identity changes before allocation, malformed output, detached inputs,
independent requests, native bounds, default dtype/device behavior, cleanup,
and rejection of altered dimensions, patch/normalization settings, and aliases.
Workflow tests cover both variants, pinned revision defaults, explicit overrides,
back navigation, YAML/factory use without downloads, historical snapshots,
forecast-option tampering, and saved-result operations without optional runtimes.

The final complete suite passed **1,400 tests**, with three opt-in real-model
tests skipped, in 53.57 seconds. The Chronos-2 smoke was run separately as above;
the unchanged Bolt/TimesFM opt-in checks were not repeated for this adapter.
Black checked 112 Python files, Ruff and whitespace checks passed, all seven
example configurations parsed, and documentation checks covered 32 Markdown
files, 212 local links, and 46 shell blocks. Independent source/configuration
review found no actionable correctness issues. Terminal model information was
inspected at 80 columns.

Source archive and wheel contents passed inspection: 178 source files,
39 required source paths, 65 wheel entries, and only NumPy/pandas as core
dependencies. The exact preparation wheel was installed without dependencies
into the existing isolated core/CLI/Parquet environment. All nine installed
workflow checks passed in 11.01 seconds, with Torch, Chronos, Hub, TimesFM,
Transformers, and Textual absent. Dependency consistency passed. Final artifact
hashes and installed byte identities are recorded alongside the built artifacts;
they are not inferred from source tests.

The initial regression run exposed five old assertions that treated Chronos-2
as unavailable. They were updated for the expanded catalog. An initial installed
verifier invocation selected a global launcher through PATH and correctly
refused to proceed; selecting the isolated environment's launcher passed without
changing application code. Routine test collection's JUnit property warnings
and the upstream CPU pin-memory warnings do not change the completed results.

## Limits and reproduction

Small is representative integration evidence for this adapter. Standard remains
**Metadata/API compatibility checked; individual inference not run**. No runtime
RAM estimate, GPU/MPS verification, forecasting-advantage claim, real-data model
ranking, multivariate/covariate support, or benchmark campaign is established.
Routines do not download weights; see the [explicit pull and smoke commands](chronos2.md).

Detailed disposable evidence is under
`/private/tmp/tsfm-next-models-01a174c4/p2-verification/`: pinned configurations,
model/runtime audits, download identities, source-file hashes, real run records,
workflow verification, no-runtime exports, and smoke XML. Large weights and
experiment outputs remain outside Git. Publication requires the owner's separate
approval under [CONTRIBUTING.md](../CONTRIBUTING.md).

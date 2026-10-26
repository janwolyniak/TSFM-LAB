# P1 catalog verification

Date: 26 October 2026. Scope: the first implementation package in the
[next release plan](next-release-plan.md). Package version remains `0.1.0`;
this work does not publish a package release.

## Implemented behavior

The shared registry describes implemented adapter families, curated variants,
native limits, and runtime requirements. Chronos-Bolt Base joins Tiny and Small
in the catalog and terminal setup. A changed variant defaults to its curated
reference SHA; revisiting the same variant preserves an explicit revision.
Baselines and TimesFM 2.5 remain selectable. Chronos-2, TiRex-2, and TTM are
future adapter packages and are not advertised as runnable here.

Catalog and setup distinguish representative project inference checks from the
selected checkpoint's own status. Base has metadata/API compatibility checks,
without a Base weight download or inference test. Exact reference identities
and the optional runtime audit are documented in the
[catalog guide](model-catalog.md).

New pretrained runs capture a detached `provenance.project_verification` snapshot
once before model loading. It binds the selected adapter/checkpoint and exact
configuration/weight identities, while retaining source/runtime information for
historical representative checks. Reload validates that snapshot independently
of the current project evidence registry. Old records without the field display
its absence; checkpoint schema-1 fields, fingerprint serialization, comparison
identity, and existing numerical metric definitions remain unchanged.

## Executed verification

Local execution used Python 3.13.6 on macOS. No new dependencies or model weights
were installed or downloaded. Real smoke tests reused existing local/cached
weights and installed runtimes; Base remained outside real inference testing.

| Check | Result |
| --- | --- |
| Complete configured suite | 1,303 passed, 2 opt-in model smoke tests skipped; 52.58 seconds. |
| Project verification regression tests | 30 passed, including detached hook snapshots, malformed metadata refusal before allocation, exact-file evidence matching, legacy records, and forecast/comparison/finance reload without checkpoints or model runtimes. |
| Base and registry tests | 12 passed using the exact pinned Base configuration and inert weights: local/cached resolution, dispatch, family mismatch refusal, stable fingerprint, and no manufactured inference claim. |
| Terminal and shell checks | 77 scoped tests passed; human Base information and `/models` inspected at 80 columns. Variant SHA defaults, back navigation, persisted status, and legacy no-retrofit behavior pass. Pull/cache JSON shapes remain unchanged. |
| Real Chronos-Bolt Small smoke | One origin/four aligned finite median outputs passed in 3.77 seconds using the existing standalone copy. |
| Real TimesFM 2.5 smoke | One origin/four aligned finite median outputs passed in 1.98 seconds through exact cached-revision resolution. |
| Formatting and lint | Black checked 111 Python files; Ruff and whitespace checks passed. |
| Documentation and example | Local links/anchors, fences, and shell syntax checked. The cache-only Base YAML example passes configuration validation without loading weights. |
| Packaging | Source archive and wheel built from it passed content/metadata inspection, including registry/evidence modules and the Base example; NumPy/pandas remain the only numerical-core dependencies. |
| Clean installed wheel | Installed the built project wheel without dependency changes into the existing isolated core/CLI/Parquet environment. `pip check` and all nine installed-verifier stages passed in 10.519 seconds, including offline Base metadata and finance regression checks. |
| Independent review | No outstanding actionable findings in the core registry, provenance snapshot boundaries, or catalog guide. |

The first manual smoke invocations incorrectly classified Hub snapshots as
standalone directories. The file-boundary guard rejected those symlinks before
inference; corrected standalone/cached paths passed without changing the guard.

## Artifact identity and remaining work

The initial distribution described above was built from the preparation tree.
Final source and wheel artifacts are rebuilt from the committed task revision,
and the final installed artifact is checked again. External build manifests
record their hashes without embedding the containing archive's own hash here.
The pre-existing untracked `PLAN.md`, checkpoints, environments, generated
results, and temporary evidence remain outside Git and the distributions.

P2 can begin with the existing Chronos runtime. The TiRex-2 Hub requirement and
TTM Torch/Transformers requirements conflict with the current shared stack;
isolation or a coordinated dependency change must be resolved before P3/P4.
No environment manager or shared-runtime upgrade is introduced by P1.

Project verification snapshots preserve internal consistency, not signed
maintainer attestations. Passing these checks establishes bounded integration
and compatibility behavior, not forecast quality, RAM requirements, speed,
model rankings, or economic value. Publication and merge follow the maintained
[contribution workflow](../CONTRIBUTING.md).

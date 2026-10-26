# Model catalog and project verification

The catalog separates an adapter family from its curated checkpoint variants.
An adapter defines the supported runtime, input/output contract, and limits;
a variant identifies particular upstream weights. A larger variant can be
selectable after metadata and loader-profile checks without a separate real
inference test. Its own status must say what was checked.

Catalog lookup works offline without importing optional model libraries or
loading weights:

```sh
tsfm models list --json
tsfm models info chronos_bolt --json
tsfm models info amazon/chronos-bolt-base --json
```

The active terminal exposes this information through `/models`. Use
`/new chronos_bolt` or `/new timesfm_2p5` to choose a family and checkpoint during
guided setup. See the [terminal guide](tui.md) and [CLI guide](cli.md) for the
complete workflow. Last value and seasonal naive remain available as baselines.

The Python API keeps `catalog_info(identifier)` for adapter or repository lookup.
Adapter entries add `variants` and `runtime_requirements`. Repository lookup
preserves the adapter's `display_name` and adds `variant_display_name`;
`checkpoint_info(repository)` returns the variant's own `display_name`.
CLI repository information also exposes that variant under `selected_checkpoint`.

## Curated choices

| Adapter | Variant / repository | Context / horizon supported by the adapter | Safetensors bytes | Recorded project inference |
| --- | --- | --- | ---: | --- |
| `chronos_bolt` | Tiny: `amazon/chronos-bolt-tiny` | 2,048 / 64 | 34,622,352 | Recorded for the reference checkpoint. |
| `chronos_bolt` | Small: `amazon/chronos-bolt-small` | 2,048 / 64 | 190,888,824 | Recorded for the reference checkpoint. |
| `chronos_bolt` | Base: `amazon/chronos-bolt-base` | 2,048 / 64 | 821,203,576 | Metadata/API compatibility checked; individual inference not run. |
| `timesfm_2p5` | 2.5 200M: `google/timesfm-2.5-200m-pytorch` | 16,256 / 128 | 925,181,104 | Recorded for the reference checkpoint. |

Weight-file size describes stored bytes, not required runtime memory. Catalog
status applies to its listed reference revision. A different requested revision
or standalone local checkpoint is inspected separately before execution.
`parameter_count` is the stored parameter total, including additional heads;
TimesFM's 200M model name is retained even though its stored total is larger.

The pretrained adapters currently support CPU float32 median forecasts, with
batch size one. TimesFM records its native context of 16,384, while the adapter
reserves room for its output patch and accepts at most 16,256 historical values.
Its native quantile head does not expand the adapter's 128-step horizon.
See the [Chronos guide](chronos.md) and [TimesFM guide](timesfm.md).

### Bolt Base reference

The curated reference is
`amazon/chronos-bolt-base@5d9f166d69f47aef3401367a7b842e78fe97b121`.
Its [pinned configuration](https://huggingface.co/amazon/chronos-bolt-base/blob/5d9f166d69f47aef3401367a7b842e78fe97b121/config.json)
declares `ChronosBoltModelForForecasting` with `ChronosBoltPipeline`, context
2,048, horizon 64, patch size/stride 16, and deciles including the median.
The [pinned model card](https://huggingface.co/amazon/chronos-bolt-base/blob/5d9f166d69f47aef3401367a7b842e78fe97b121/README.md)
declares Apache-2.0.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 1,123 | `46f7e7e695569f67b06178faa4f236c281392945aa2d1bdf844442e57155530b` |
| `model.safetensors` | 821,203,576 | `31f875483a3215bc6880a0837ea608a13ce55f88ad90538c3cdd0b29aeb60b36` |

These are configuration inspection and upstream file-manifest identities.
The Base weights were not downloaded for this catalog addition and Base inference
was not run. Tiny/Small's recorded execution checks are representative adapter
evidence; they do not become a Base execution result. An actual Base run still
requires the optional runtime, available local weights, validated file identities,
and strict parameter loading. Normal experiment execution never downloads weights;
an explicit `models pull` is a separate operation.

## Reading verification status

Reference choices display **Reference checkpoint inference check recorded** or
**Metadata/API compatibility checked; individual inference not run**. After
inspecting selected files, the display is **Project inference check recorded for
these config/weight files** or **Compatibility checked; project inference check
not recorded for these files**.

Compatibility and real inference answer separate questions:

- `compatibility.status: checked` with scope `metadata_and_loader_profile`
  means the checkpoint passed the implemented configuration and loader-profile
  checks. It does not establish runtime memory use, speed, or forecast quality.
- `project_inference_status: recorded` means the project has retained execution
  evidence for the selected checkpoint's consumed configuration and weight
  identities. `selected_checkpoint_checks` contains that evidence.
- `project_inference_status: not_recorded` means no matching project execution
  evidence was recorded. `adapter_representative_checks` can still identify
  checks performed with another checkpoint using the same adapter.

Each retained check identifies its repository/revision, configuration and weight
hashes/sizes, source provenance when known, runtime versions, machine, device,
dtype, executed checks, and documentation reference. Unknown source fields stay
`null`; a dirty source checkout stays marked dirty. A local copy can match
recorded file identities without claiming to have been loaded from a Hub revision.

The historical Tiny, Small, and TimesFM checks each exercised four successful
windows and 16 forecast cases on the bundled synthetic series, with exact offline
cached replay. Their records document the checks actually run:
[Tiny](milestone3-verification.md) and
[Small/TimesFM](milestone6-verification.md).
They establish bounded integration evidence, without a model ranking, general
accuracy claim, device certification, or benchmark result.

### Saved snapshots and older records

For new pretrained-model runs, `provenance.project_verification` captures the
verification snapshot before loading the model. The saved snapshot describes
what was recorded at that point. Reload validates that historical snapshot against
the saved checkpoint identity; it does not substitute today's evidence registry.
The schema-1 checkpoint record and fingerprint retain their existing meaning.

A user's completed experiment is evidence that their configured run completed.
It does not add a maintainer check to the project registry or change a saved
`not_recorded` snapshot into `recorded`. The run's results and checkpoint/runtime
provenance remain the place to inspect that user's execution.

Older saved results may have no `project_verification` field. The display is
**Project verification was not recorded**. Absence does not mean that inference
failed or that today's selected checkpoint has never been checked. Loading,
comparison, finance reporting, and export preserve the old record without adding
or rewriting verification fields. Built-in baselines do not require checkpoint
verification snapshots.
These self-contained provenance checks establish internal consistency; they are
not signed maintainer attestations or authentication of a rewritten workspace.

## Future optional-runtime gates

Chronos-2, TiRex-2, and TTM are planned integrations, not adapters made available
by P1. The 26 October 2026 audit compares declared dependency requirements with
the current Python 3.11–3.13, Torch 2.8, Transformers 4.55, and Hub 0.34.4 stack.
Declared compatibility is not an executed integration test.

**Chronos-2 (P2).** Existing `chronos-forecasting==2.2.0` includes
`Chronos2Pipeline`; Python ≥3.10, Torch ≥2.2/<3, and Transformers ≥4.41/<5 admit
the current stack. Its distinct architecture/profile, safe local loading, and
representative Small check still need implementation.
See the [pinned runtime requirements](https://github.com/amazon-science/chronos-forecasting/blob/75a2dfe228e39d55283f477c01399b9e53d89661/pyproject.toml).

**TiRex-2 (P3).** `tirex-2==0.4.0` admits Python ≥3.11 and Torch ≥2.8, but
requires Hub ≥0.35.3, conflicting with 0.34.4. It also adds recurrent-kernel
dependencies. Resolve isolation or a coordinated Hub change before installation;
require safe files and the explicit CPU path. Its published checkpoint variants
are not size tiers.
See the [pinned runtime requirements](https://github.com/NX-AI/tirex-2/blob/614f46fd7c60e332d8a32e525403b341c2c1ce1e/pyproject.toml).

**TTM (P4).** `granite-tsfm==0.3.4` admits Python 3.11–3.13/Torch <2.9 but pins
Transformers 4.56.0. Audited release `0.3.11` requires Torch ≥2.10/<2.12 and
Transformers ≥4.57.6/<6. Settle runtime isolation or a coordinated dependency
change, then explicit native context/horizon/frequency, causal scaling, and
native-point task semantics. Do not relabel a point head as a verified median.
See the [pinned 0.3.4 requirements](https://github.com/ibm-granite/granite-tsfm/blob/4d4c6cf0e345c985d84f6292a4f397caf5e2ba78/pyproject.toml)
and [pinned 0.3.11 requirements](https://github.com/ibm-granite/granite-tsfm/blob/dfe846e6c1f1b0086b4a7b0831249ac8aeb7dcd7/pyproject.toml).

P1 does not install these runtimes or introduce an environment manager. P2 can
start with the current compatible runtime. Resolve the conflicting execution
design before P3/P4 implementation; shared dependency changes
need existing-adapter regression checks. TimesFM 2.0 and 3.0 are separate model
generations outside the current 2.5 adapter, rather than larger variants covered
by its verification. Follow the [next release plan](next-release-plan.md) for
the remaining scope and acceptance gates.

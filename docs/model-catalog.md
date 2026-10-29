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
`/new chronos_bolt`, `/new timesfm_2p5`, `/new chronos_2`, or `/new tirex_2` to choose a family and checkpoint during
guided setup. See the [terminal guide](tui.md) and [CLI guide](cli.md) for the
complete workflow. Last value and seasonal naive remain available as baselines.

The Python API keeps `catalog_info(identifier)` for adapter or repository lookup.
Its optional `variant` argument and `checkpoint_info(repository, variant=None)`
select a particular curated branch when one repository contains several profiles.
Adapter entries add `variants`, `runtime_requirements`, and `runtime_notices`.
Each variant carries its effective `max_context` and `max_horizon`. Repository lookup
preserves the adapter's `display_name` and adds `variant_display_name`;
`checkpoint_info(repository)` returns the variant's own `display_name`.
CLI repository information also exposes that variant under `selected_checkpoint`.
Repository capabilities use that variant's bounds; adapter capabilities describe
the maximum across available variants. This distinction matters for TiRex-2's
different training profiles. Existing single-profile choices inherit their
adapter's limits.

## Curated choices

| Adapter | Variant / repository | Context / horizon supported by the profile | Safetensors bytes | Recorded project inference |
| --- | --- | --- | ---: | --- |
| `chronos_bolt` | Tiny: `amazon/chronos-bolt-tiny` | 2,048 / 64 | 34,622,352 | Recorded for the reference checkpoint. |
| `chronos_bolt` | Small: `amazon/chronos-bolt-small` | 2,048 / 64 | 190,888,824 | Recorded for the reference checkpoint. |
| `chronos_bolt` | Base: `amazon/chronos-bolt-base` | 2,048 / 64 | 821,203,576 | Metadata/API compatibility checked; individual inference not run. |
| `timesfm_2p5` | 2.5 200M: `google/timesfm-2.5-200m-pytorch` | 16,256 / 128 | 925,181,104 | Recorded for the reference checkpoint. |
| `chronos_2` | Small: `autogluon/chronos-2-small` | 8,192 / 1,024 | 111,749,048 | Recorded for the reference checkpoint. |
| `chronos_2` | Standard: `amazon/chronos-2` | 8,192 / 1,024 | 477,930,472 | Metadata/API compatibility checked; individual inference not run. |
| `tirex_2` | Default: `NX-AI/TiRex-2` | 2,048 / 320 | 330,038,664 | Recorded for the reference checkpoint. |
| `tirex_2` | FEV benchmark: `NX-AI/TiRex-2-fevbench` | 2,048 / 320 | 330,038,640 | Metadata/API compatibility checked; individual inference not run. |
| `tirex_2` | GiftEval zero-shot: `NX-AI/TiRex-2-gifteval-zs` | 8,192 / 928 | 330,038,640 | Metadata/API compatibility checked; individual inference not run. |
| `tirex_2` | GiftEval pretrain: `NX-AI/TiRex-2-gifteval-pretrain` | 8,192 / 928 | 330,038,640 | Metadata/API compatibility checked; individual inference not run. |
| `ttm_r2` | `ibm-granite/granite-timeseries-ttm-r2`, `512-96-r2` | exactly 512 / 96 | 3,240,592 | Representative CPU check recorded. |
| `ttm_r2` | Same repository, `1024-96-r2` | exactly 1,024 / 96 | 11,879,480 | Compatibility checked; individual inference not run. |
| `ttm_r2` | Same repository, `1536-96-r2` | exactly 1,536 / 96 | 12,344,144 | Compatibility checked; individual inference not run. |
| `ttm_r2` | Same repository, `512-192-r2` | exactly 512 / 192 | 3,634,200 | Compatibility checked; real inference pending. |
| `ttm_r2` | Same repository, `512-336-r2` | exactly 512 / 336 | 4,224,600 | Compatibility checked; real inference pending. |
| `ttm_r2` | Same repository, `512-720-r2` | exactly 512 / 720 | 5,799,000 | Compatibility checked; real inference pending. |
| `ttm_r2` | Same repository, `52-16-ft-r2.1` | exactly 52 / 16; fixed weekly | 319,440 | Compatibility checked; real inference pending. |
| `ttm_r2` | Same repository, `90-30-ft-r2.1` | exactly 90 / 30; fixed daily | 3,496,760 | Compatibility checked; real inference pending. |

Weight-file size describes stored bytes, not required runtime memory. Catalog
status applies to its listed reference revision. A different requested revision
or standalone local checkpoint is inspected separately before execution.
`parameter_count` is the stored parameter total, including additional heads;
TimesFM's 200M model name is retained even though its stored total is larger.

Chronos, TimesFM, and TiRex adapters support CPU float32 median forecasts, with
batch size one. TimesFM records its native context of 16,384, while the adapter
reserves room for its output patch and accepts at most 16,256 historical values.
Its native quantile head does not expand the adapter's 128-step horizon.
See the [Bolt guide](chronos.md), [Chronos-2 guide](chronos2.md),
[TimesFM guide](timesfm.md), [TiRex-2 guide](tirex2.md), and [TTM guide](ttm.md).
TTM uses a native-point task and exact context. R2 requires minute-to-hour cadence;
the selected R2.1 daily/weekly profile requires exactly `1D`/`7D`. Variant metadata
and setup expose that profile's cadence; unresolved local/custom revisions are
rechecked against their actual complete configuration before inference. See the
[S6 TTM verification limits](s6-ttm-verification.md).
Use `models info ibm-granite/granite-timeseries-ttm-r2 --variant 1024-96-r2`
to inspect a nondefault branch. YAML records its repository and pinned SHA.

TiRex-2's choices are training profiles of the same architecture, not size tiers.
The model-weight license remains Apache-2.0. Its separate optional runtime
notices identify mLSTM Kernels' NXAI Community License and display **Built with
technology from NXAI**. Review the [runtime terms](tirex2.md#optional-runtime-and-notices),
[NOTICE](../NOTICE), and shipped [license copy](../licenses/mlstm-kernels-2.0.6.txt)
before using that optional runtime. The numerical core does not acquire those
dependencies.

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
Chronos-2 Small's separate [P2 check](p2-chronos2-verification.md) uses a pristine
Chronos 2.2.0 runtime and records its own exact configuration/weight identities.
These checks establish bounded integration evidence, without a model ranking, general
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

## Optional-runtime decisions

Chronos-2 is implemented in P2; P3 adds TiRex-2 and its four training profiles.
P3 updates the shared Hub pin to 0.35.3 while retaining the Python 3.11–3.13,
Torch 2.8, and Transformers 4.55 stack. The original 26 October audit used Hub
0.34.4 and exposed the conflict resolved below. Declared compatibility and actual
execution remain distinct; see the [P3 record](p3-tirex2-verification.md) for
qualification and existing-family regression evidence.

**Chronos-2 (P2).** Existing `chronos-forecasting==2.2.0` includes
`Chronos2Pipeline`; Python ≥3.10, Torch ≥2.2/<3, and Transformers ≥4.41/<5 admit
the current stack. P2 adds its distinct architecture/profile, safe local loading,
and representative Small check without changing the shared dependency pins.
See the [pinned runtime requirements](https://github.com/amazon-science/chronos-forecasting/blob/75a2dfe228e39d55283f477c01399b9e53d89661/pyproject.toml).

**TiRex-2 (P3).** The owner approved a coordinated Hub 0.35.3 pin and isolated
validation before changing any existing installation. `tirex-2==0.4.0` uses
Torch 2.8 with the pinned recurrent dependencies and explicit eager CPU path.
Its native JSON and Safetensors load directly with strict parameters; unsafe
legacy loading is excluded. The NXAI-licensed mLSTM dependency is optional and
its notices are exposed separately from the model-weight license.
See the [pinned runtime requirements](https://github.com/NX-AI/tirex-2/blob/614f46fd7c60e332d8a32e525403b341c2c1ce1e/pyproject.toml).

**TTM (P4).** `granite-tsfm==0.3.4` admits Python 3.11–3.13/Torch <2.9 but pins
Transformers 4.56.0. Audited release `0.3.11` requires Torch ≥2.10/<2.12 and
Transformers ≥4.57.6/<6. The owner approved the coordinated 0.3.4 runtime with
Transformers 4.56, Tokenizers 0.22, scikit-learn 1.7.2, and Filelock 3.20.3.
P4 implements three explicit context profiles, causal scaling, and native-point
task semantics; see the [P4 record](p4-ttm-verification.md). Do not relabel a point head as a verified median.
See the [pinned 0.3.4 requirements](https://github.com/ibm-granite/granite-tsfm/blob/4d4c6cf0e345c985d84f6292a4f397caf5e2ba78/pyproject.toml)
and [pinned 0.3.11 requirements](https://github.com/ibm-granite/granite-tsfm/blob/dfe846e6c1f1b0086b4a7b0831249ac8aeb7dcd7/pyproject.toml).

P1 introduced no environment manager. P2's original verification used an isolated
pristine Chronos package. P3 retains one managed inference interpreter and
coordinates the shared Hub pin with existing-adapter regression checks. P4
retains that interpreter design and qualifies the shared Transformers update. TimesFM 2.0 and 3.0 are separate model
generations outside the current 2.5 adapter, rather than larger variants covered
by its verification. Follow the [next release plan](next-release-plan.md) for
the remaining scope and acceptance gates.

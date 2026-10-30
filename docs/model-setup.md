# Model-specific experiment setup

Status: implemented in the **unreleased v0.3.0 S2 development checkout**. The
published v0.2.0 package retains its previous setup questions. This stage builds
on [first-launch preferences](onboarding.md). The later S3 checkout adds
[guided automatic preparation](guided-preparation.md). S4 adds
[future forecasting](future-forecasting.md) alongside historical evaluation.

## Guided journey

Start `/new guided` or `/new ADAPTER_ID`. An adapter argument sets the initial
choice and remains editable. `/new describe` reviews assistant suggestions through
the same questions and validation. Choose:

1. Generic, Finance, Energy, Temperature, or Weather usage.
2. A friendly model family, then its generation where there is a choice.
3. A baseline or curated variant, described according to its actual meaning.
4. Forecast task, applicable baseline season, context, and horizon. Required
   preprocessing and runtime settings are explained as fixed requirements.
5. A local dataset, column mapping where needed, regular cadence, timezone, units,
   and usage-specific measurement declarations. Inspect original-unit values.
6. Historical evaluation name, stride, MASE lag, and explicit target bounds.
7. The validated summary and final execution confirmation, which defaults to no.

| Family | Generation | Choice |
| --- | --- | --- |
| Baselines | Deterministic | Last value or Seasonal naive. |
| Chronos | Bolt | Tiny, Small, or Base model size. |
| Chronos | 2 | Small or Standard model size. |
| TimesFM | 2.5 | The curated 200M model size. |
| TiRex | 2 | Default, FEV benchmark, GiftEval zero-shot, or GiftEval pretrain **training profile**. |
| IBM Granite TTM | R2 | Native 512/96, 1024/96, or 1536/96 context/horizon profile. |

Generation is shown directly when only one is available. A training profile is
not a model size. Checkpoint verification labels describe the selected reference;
representative checks for other variants remain separate evidence.

The S2 boundary uses an already cached checkpoint or a compatible local
directory. S3 adds an `automatic` source that downloads reviewed missing files only
after final confirmation; `cached` stays offline. No setup path installs runtimes.
Choose **Advanced
checkpoint options? → yes** to inspect repository/revision details, enter a custom
exact SHA, or change the cache directory. A changed repository or profile resets
its revision; revisiting the same identity preserves a custom SHA. A known TTM
revision must agree with the selected native profile. An unknown SHA remains
unverified until the actual checkpoint is inspected.

## Constraints and data checks

Editable context/horizon questions display model bounds and retain valid defaults.
TTM requires its exact native context, the `native_point` task, and supported
minute/hour cadence. Its context-only standardization and native scaling are
required; setup offers no switch that disables them. Other families retain their
own native preprocessing. User units are preserved without hidden conversion,
clipping, resampling, or imputation.

Data inspection rejects invalid timestamps, gaps, unsupported cadence, and
insufficient history before checkpoint access. Oversized horizons are rejected
against known model limits before dataset work. Original-unit previews show the
first and last rows with UTC timestamps, including legitimate negative values.
Actual local checkpoint metadata can refine catalog limits; deliberate values
that conflict are shown for correction. Final preparation rechecks the complete
dataset, evaluation grid, and native checkpoint requirements without loading an
inference runtime or allocating a model.

`/back` revisits choices and `/cancel`, Escape, or Ctrl+C stops setup. Compatible
numeric choices, units, and point semantics survive back navigation. Changing
family/generation/profile invalidates checkpoint-dependent metadata. Changing the
dataset refreshes generated date bounds; deliberate bounds remain visible and
must pass validation. Category/model changes discard stale parsed data so new
dataset choices can be reviewed. Invalid data can be retried at the dataset step.

After dataset preview, select historical evaluation or future forecasting.
Historical mode requires holdout observations; future mode requires only its
consumed context and resolves origin to the last observation. Switching mode
preserves compatible data/model choices and recollects mode-specific settings.
Any switch invalidates a prepared plan. Future accuracy remains unavailable
because future actuals are absent; see the [future guide](future-forecasting.md).

## Shared Python and configuration contract

`tsfm_lab.models.settings.model_settings` returns an immutable metadata-only
description used by manual setup, assistant validation, YAML loading, and Python
configuration construction. It reads no checkpoint files and imports no optional
inference framework. For example:

```python
from tsfm_lab.models.settings import model_settings

settings = model_settings(
    "ttm_r2",
    repository="ibm-granite/granite-timeseries-ttm-r2",
    variant="1024-96-r2",
)
assert settings.capabilities.min_context == settings.capabilities.max_context == 1024
assert settings.supported_tasks == ("native_point",)
```

Known reference revisions select exact curated bounds. Unknown revisions and
local directories initially use the adapter envelope; final checkpoint checks
remain authoritative. `validate_settings` checks task, context, horizon, seasonal
baseline requirements, and cadence. Configuration errors now surface earlier.

Schema version 1, YAML fields, resolved configuration, numerical identity, saved
records, and finance semantics are unchanged for valid configurations. Unsupported
preprocessing/model options remain rejected. See [configuration](cli.md),
[catalog](model-catalog.md), [language assistance](language-assistance.md), and
[the S2 verification record](s2-model-setup-verification.md).

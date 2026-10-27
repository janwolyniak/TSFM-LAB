# First launch and preferences

Status: implemented for the **unreleased v0.3.0 S1 development stage**. The
published v0.2.0 package does not include these questions or commands. Package
metadata stays at 0.2.0 until release qualification.

Start `tsfm` or `tsfm ui` in a terminal. A fresh settings directory offers optional
setup. Choose `skip` to reach the prompt and suppress the offer on later launches;
`/setup` revisits it. `/cancel`, Escape, Ctrl+C, or Ctrl+D during the preferences
questions discards edits. If setup has never been completed or skipped, the offer
returns on the next launch. Manual forecasting needs no assistant connection.
Redirected commands and scripts never enter onboarding or load these preferences.

## Saved choices

Setup reviews these nonsecret preferences before saving them together:

| Choice | Effect |
| --- | --- |
| Workspace directory | Base for relative dataset, checkpoint, and export paths in guided setup. Initially the launch directory. |
| Results directory | Initial saved-run directory for subsequent interactive sessions. Initially `results` inside the workspace. |
| Model cache directory | Default for cached pretrained choices in guided setup; initially follows the Hugging Face cache environment. |
| Category | Initial Generic, Finance, Energy, Temperature, or Weather selection in guided setup. |
| Export format | Used for `/export ID PATH` when `PATH` has no extension. Initially CSV. |

Relative directory entries resolve against the selected workspace. `/back`
revisits choices. Saving probes writable directories or their nearest existing
parents without creating the selected workspace, results, or cache directories.
Skip persists only the dismissal without probing unused work directories.
Existing results and checkpoints stay in place; settings never migrate them.
The effective results directory is checked again before interactive execution.

`/settings` shows the settings file and saved choices, then offers editing.
`/setup` also shows runtime diagnostics and offers optional assistant setup and
the bundled offline baseline demonstration. The demonstration retains its own
review and confirmation, original data, and numerical settings.

Runtime diagnostics inspect installed package versions against the catalog. They
identify missing or incompatible optional packages without installing, importing
inference frameworks, downloading weights, or allocating models. Matching package
metadata does not establish that imports, inference, or native Windows execution
work. Actual model checks keep their existing verification labels.

## Overrides and compatibility

`tsfm ui --workspace DIRECTORY` overrides the saved results directory for that
session, including after editing preferences. `/workspace DIRECTORY` changes the
current results view without rewriting saved preferences. A subsequent `/setup`
or settings edit reapplies the launch option if one was supplied.

Guided choices and assistant suggestions override preference defaults and still
pass ordinary validation and final confirmation. A saved YAML configuration keeps
its own input paths, cache selection, output directory, and numerical identity;
`/run CONFIG.yaml` and `tsfm run CONFIG.yaml` do not merge preferences into it.
Scriptable history/export commands keep their existing defaults and options.

An explicit export extension takes priority over the saved format; an explicit
format argument takes priority over both. For example, `/export latest forecast`
uses the preference, `/export latest forecast.csv` selects CSV, and
`/export latest forecast.csv json` explicitly selects JSON. Files are never
overwritten. Relative interactive exports resolve inside the workspace.

## Storage and optional connections

Preferences use versioned JSON at `~/.config/tsfm-lab/preferences.json` on Linux
and macOS, or `%LOCALAPPDATA%\TSFM-LAB\preferences.json` on Windows. Set
`TSFM_LAB_SETTINGS_DIR` to use a different settings directory, including a
disposable directory for tests. The cache default respects `HF_HUB_CACHE`,
`HUGGINGFACE_HUB_CACHE`, `HF_HOME`, then `XDG_CACHE_HOME`, with the usual
`~/.cache/huggingface/hub` fallback.

Malformed or unsupported preferences fail with an actionable message and remain
untouched. Restore a known valid copy or choose another settings directory;
ordinary shell commands remain available. Saving uses an atomic replacement so
failed writes preserve the previous preferences.

Assistant setup opens the existing [connection menu](connections.md). Credentials
remain in that menu's session/native-vault boundary and are excluded from
preferences, run artifacts, and logs. Selecting setup does not send an assistant
request. Connection, runtime, and checkpoint verification retain their separate
permission and qualification requirements.

See the [v0.3.0 plan](release-v0.3.0-plan.md) and
[S1 verification record](s1-onboarding-verification.md) for remaining stages and
the limits of this local implementation evidence.

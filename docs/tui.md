# Interactive terminal

The optional Textual application shares validated configurations, evaluation,
experiment storage, comparison rules, and exports with the shell workflow.
It does not download checkpoints. Start with the original bundled synthetic data;
no model runtime, network access, credentials, or GPU is needed for baselines.

## Start

After choosing to install the optional dependencies in your environment:

```sh
python -m pip install -c requirements-tui.txt -e '.[tui,parquet]'
tsfm ui --workspace results
```

Omit `parquet` if only CSV/JSON are needed. `tsfm` without arguments opens the same
application with workspace `results` when both input/output are terminals.
Redirected `tsfm` prints help. Redirected `tsfm ui` exits with an actionable error.
`tsfm --help`, `--version`, and named shell commands never launch the application.
The numerical core does not import Textual.

## Offline keyboard walkthrough

Use a terminal at least 80 columns by 24 rows. `Tab` and `Shift+Tab` move focus;
`Enter` activates buttons. Arrow keys navigate lists and selection menus. Longer
forms scroll as focus moves. The footer shows keys available on the current screen.

1. At home, optionally edit the workspace directory. Activate **New experiment**
   (or press `n` while focus is outside the workspace input).
2. In **Model**, keep **Last value**, then activate **Next**.
3. In **Data**, activate **Use demo dataset**, wait for its inspection, then
   activate **Next**. It fills the path, columns, hourly frequency, UTC timezone,
   and `synthetic_units`. Your current model selection is retained.
4. In **Evaluation**, keep context 8, horizon 4, stride 4, and MASE seasonal lag 4.
   Suggested bounds select the last 16 observations. Activate **Next**.
5. Review the task, four windows, 16 forecast cases, bounds, and workspace.
   Activate **Run**. Invalid inputs remain on their step with an error.
6. Watch stages and completed-window progress. When the worker stops successfully,
   results open with target/units/frequency/horizon, overall and per-lead scores,
   a short forecast table, and model/provenance details.
7. Scroll or tab to the export controls, choose a new path/format, and activate
   **Export** (`Ctrl+E` also exports). Existing files are preserved. `Escape`
   returns to home.

The last-value demo has MAE 2.5, RMSE about 2.915476, and MASE 2.5. Repeat with
**Seasonal naive**, period 4, for scores 1.0 on all three metrics. These are
arithmetic checks on [constructed data](../examples/datasets/README.md), with
four horizons of four observations, and do not establish model advantage.
The installed package includes the same original CSV as the source example.

## History, comparison, and model selection

Home offers **Browse models**, **Experiment history**, and **Compare experiments**.
The model browser is an offline catalog. Arrow keys and `Enter` select an adapter
for a new wizard. Chronos-Bolt requires an exact cached revision or a standalone
local checkpoint directory; loading happens only after **Run**. The UI uses CPU.
Use the [Chronos shell guide](chronos.md) for explicit downloads and runtime setup.

History lists persisted records. Arrow keys choose a row, `Enter` opens it, `r`
refreshes, and `Escape` returns. `Space` selects completed runs; select two or more
and press `c` to compare. **Compare experiments** opens this same selection screen.
Comparisons enforce the [shared compatibility policy](cli.md); different tasks
cannot be presented as a comparable leaderboard. Failed/cancelled or corrupt
records show status/errors instead of completed scores. Viewing, comparison, and
export reload saved files without inference.

## Cancellation and errors

`Escape` or **Cancel run** requests cancellation. The screen continues to say
**Cancellation requested** until the child has stopped. `q` outside inputs or
`Ctrl+Q` requests application exit; an active experiment follows the same cleanup.
Cooperative cancellation checks surround model loading, each forecast window,
scoring, and saving. Native inference may remain busy until a call returns.
After three seconds the controller requests termination, then uses a kill
fallback after another 0.25 seconds if necessary.

The controller reaps the child before reporting its final state and checks the
persisted manifest off the UI thread. If completion won the race, the completed
record is preserved. Otherwise known active runs become cancelled or failed.
An abrupt crash before the worker reports its UUID can leave an incomplete record
for inspection; it cannot appear completed. Failed runs display a bounded tail
of worker diagnostics. Child output goes to a temporary log so model-library
messages do not overwrite the screen; closing the controller removes that log.
Cancelled/failed experiment directories remain in the workspace.

The application manages one local run at a time. It has no resume facility,
automatic checkpoint pull, charting workspace, model training, or job server.
Headless interactions and rendered 80×24/120×36 layouts are covered by the
[local verification record](milestone4-verification.md).

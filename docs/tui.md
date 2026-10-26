# Interactive terminal

TSFM opens a persistent local command prompt inside your terminal. Commands,
questions, progress, and results form one transcript in normal terminal scrollback.
It uses the terminal's background and a small amount of accent color. There are
no home-screen buttons, tabs, or full-screen forms. Assistant connections are
optional; local forecasting does not require one.

## Start

Install the optional CLI dependencies in your chosen environment when authorized:

```sh
python -m pip install -c requirements-dev.txt -e '.[cli,parquet]'
tsfm
```

`tsfm ui --workspace results` starts the same session with an explicit workspace.
Omit `parquet` for CSV/JSON only. Both input and output must be terminals;
redirected `tsfm` prints help and redirected `tsfm ui` reports an error. Named
shell commands and JSON output keep their existing behavior. The numerical core
imports neither the prompt editor nor any terminal framework.

## A first session

Type `/demo`, then answer `y` to run the original synthetic baseline. A short
summary appears before confirmation, progress stays inline, and the results are
printed when the worker stops. The prompt returns below the output.

```text
› /demo
  ...task summary...
  Run experiment? [no] › y
  ...progress and results...
› /history
› /show latest
› /export latest forecasts.csv
```

The example above illustrates the command sequence; ellipses stand for the actual
validated task and result output. The bundled hourly demand series has 32
observations in `synthetic_units`. The default task uses context 8, horizon 4,
stride 4, MASE lag 4, and four origins covering 16 forecast cases. Last-value MAE
is 2.5, RMSE about 2.915476, and MASE 2.5. `/demo seasonal_naive` uses period 4
and scores 1.0 on all three metrics. These are arithmetic checks on
[constructed data](../examples/datasets/README.md), not model advantage claims.

## Commands

| Command | Behavior |
| --- | --- |
| `/help` | Show commands and keyboard shortcuts. |
| `/demo [last_value\|seasonal_naive]` | Review and confirm the bundled offline experiment. |
| `/new [adapter]` | Ask setup questions one at a time, then review and confirm. |
| `/run CONFIG.yaml` | Execute a saved configuration and use its configured results directory. |
| `/models` | Print the offline model catalog. |
| `/connections` | Manage optional assistant connections, hidden credentials, and an explicit connection check. |
| `/history` | List saved records in the current workspace. |
| `/show ID` | Show verified scores, per-lead metrics, and a short forecast preview. |
| `/compare ID ID [ID…]` | Enforce the shared matching-task comparison policy. |
| `/export ID PATH [csv\|json\|parquet]` | Export saved predictions to a new file; existing files are preserved. |
| `/workspace [DIRECTORY]` | Show or change the session workspace without creating it. |
| `/status` | Show the current workspace and last run in this session. |
| `/exit` or `/quit` | Return to the surrounding shell. |

Commands also accept their names without the leading slash. IDs accept a full
UUID or an unambiguous prefix; `latest` selects the latest verified completed
experiment. Quote paths with spaces. Tab completes command names, known run IDs,
and paths for run/workspace/export; Up/Down recall command history. Command
history lives only in memory for the current session. Setup answers have separate
input history, so they do not pollute command recall.

## Guided setup

`/new` first asks for a usage profile, then a model and any checkpoint details,
then a dataset, profile-specific measurements/units, mapping fields
when using your own file, name, evaluation settings, and explicit timestamp bounds.
Empty answers accept the displayed default. Validation errors reprompt the field
or step. `/back` returns to the previous setup step; `/cancel`, Escape, or Ctrl+C
aborts setup. Nothing runs or creates a workspace until the final confirmation.
`Run experiment?` defaults to **no**. Custom datasets need declared target units,
frequency, and timezone handling; no gaps or missing values are silently repaired.

Enter `/connections` at any setup question to manage optional connections and
return to that question. The [connection guide](connections.md) explains the four
account/billing routes, storage, explicit model choice, and checks. Natural-language
setup and explanations arrive in P7.

Chronos-Bolt and TimesFM 2.5 use CPU/float32 and an already cached exact revision
or compatible standalone local checkpoint. Select them with `/new chronos_bolt`
or `/new timesfm_2p5`, or choose the model during `/new`. Setup validates the
adapter/repository pairing, local architecture metadata, and supported
context/horizon limits without loading model runtimes or downloading weights.

TimesFM supports contexts up to 16,256 and horizons up to 128, with batch size 1.
The runtime must use Torch's float32 default dtype; an incompatible process
setting fails without being changed. Use the [Chronos guide](chronos.md) or
[TimesFM guide](timesfm.md) for the optional runtime, explicit pulls, and checkpoint
rules. The offline `/demo` command remains limited to the bundled baselines.

Use Generic, Finance, Energy, Temperature, or Weather. Domain examples are
original synthetic data, with negative temperatures preserved. Custom data paths
and deliberate numeric settings survive back navigation; switching from an
example clears fictional measurement declarations. Hourly seasonal examples
are shown as suggestions, with lag choices left explicit. The original quick
`/demo` keeps its existing configuration and numbers.

History and results show profile, measurement context, and error units. An
eligible close-price Finance result offers the existing OHLC command with every
required input named, without executing a report or supplying trading rules.
See [usage profiles](usage-profiles.md) for exact metadata and examples.

## Interrupts and saved results

During inference, Ctrl+C requests cooperative cancellation. Progress says the
worker is stopping until it has actually stopped. After three seconds the
controller requests termination, with a kill fallback after another 0.25 seconds.
A completed record wins a late cancellation race. Otherwise known active records
become cancelled/failed; incomplete records are retained and never acquire scores.
Ctrl+C at an idle prompt clears the current input; Ctrl+D at an empty main prompt
or `/exit` ends the session. A run owns the prompt until its terminal state.

The worker ignores terminal SIGINT and receives cancellation through its event.
The main monitor defers SIGINT handling to safe poll boundaries and restores the
previous signal handler after cleanup. Child logs cannot overwrite the display;
failed runs show a bounded diagnostic tail. Stored terminal control characters
are printed visibly rather than interpreted as terminal instructions.

Viewing, comparison, and export reload checked artifacts without inference. The
session invokes local TSFM commands; it does not execute arbitrary shell commands
or interpret natural language with an AI model.

## Design and compatibility

The interaction direction draws from command prompts, slash-command discovery,
and keyboard controls documented by [Codex](https://learn.chatgpt.com/docs/developer-commands),
[Claude Code](https://code.claude.com/docs/en/interactive-mode), and
[OpenClaw](https://docs.openclaw.ai/cli/tui). TSFM adapts these interaction patterns
to deterministic local experiment commands. The line editor uses
[prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/asking_for_input.html).

The former Textual screens remain in the source tree for regression coverage and
are not launched by `tsfm`. The older `[tui]` extra remains compatible, while
`[cli]` is sufficient for the new prompt. See the
[rework verification record](terminal-rework.md) for evidence and limits.

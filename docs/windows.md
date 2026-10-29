# Native Windows candidate

The v0.3.0 development checkout targets **Windows 11 x64, Python 3.11–3.13,
PowerShell and Windows Terminal**, without WSL. Portability code and native CI
jobs are implemented. **Native Windows qualification is still pending**: no
Windows 11 machine, hosted Windows job, model inference or native credential
vault was exercised in this implementation session. See the
[S8 verification record](s8-windows-verification.md).

## Platform evidence

| Workflow | macOS | Linux | Native Windows |
| --- | --- | --- | --- |
| Core, baseline workers, historical/future records and exports | Local Python 3.13.6 checks recorded in S8 | Python 3.11–3.13 CI configured; current candidate not exercised here | Python 3.11–3.13 CI configured; execution pending |
| Interactive keyboard, completion, scrollback and Ctrl+C | Existing POSIX terminal tests retained | POSIX terminal tests configured | Private console-event test implemented; Windows Terminal keyboard checks pending |
| Optional pretrained runtimes | Individual family evidence and limits remain in their guides | Per-family native qualification pending for this candidate | Every family remains unqualified; metadata alone is insufficient |
| Assistant protocols | Offline fake-provider coverage | Same tests configured | Same tests configured; native vault/browser checks pending |

The hosted `windows-latest` runner tests its supplied Windows Server image. Even
a successful job would not replace the separate Windows 11/Windows Terminal
interactive check. Windows ARM, GPU, MPS and network filesystems are outside this
target. CPU is the first qualification path.

## Candidate PowerShell setup

These commands are prepared and reviewed for PowerShell; execution on native
Windows remains pending. Use the development source checkout for these features,
not the older v0.2.0 artifacts. A Python 3.12 environment also accommodates the
optional TOTO family. No activation or execution-policy change is required.

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -c requirements-dev.txt '.[cli,parquet,connections]'
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\tsfm.exe --version
& .\.venv\Scripts\python.exe -m tsfm_lab --help
& .\.venv\Scripts\tsfm.exe ui --workspace '.\saved results'
```

Install dependencies only within the operator's authorized environment. In the
interactive terminal, `/setup` and `/settings` select subsequent directories;
`/demo` runs the bundled baseline, and `/new guided` selects historical or future
mode. Review the displayed configuration and confirm the run. A pretrained run
requires its separately installed runtime and explicit reviewed download consent.
The forecasting workflow requires no assistant key.

Drive-qualified local paths work in YAML, Python configuration and slash commands.
Use single-quoted YAML strings for backslashes, or forward slashes in native execution
configuration. Quote paths containing spaces in the terminal:

```text
/run "C:\Forecast lab\My experiments\forecast.yaml"
/workspace "C:\Forecast lab\Łódź\saved results"
/export latest "C:\Forecast lab\My experiments\forecast.csv"
```

Backslashes remain literal in Windows slash commands, including a trailing
backslash inside quotes. URL schemes, drive-relative paths such as `C:results`,
UNC/device paths, alternate data streams and reserved/ambiguous Windows names
are refused for active configuration. Use an ordinary local disk. NTFS supports
the hard-link publication used to refuse concurrent export overwrites; other
filesystems are not qualified. Redirected Windows command output uses UTF-8.

Preferences default to `%LOCALAPPDATA%\TSFM-LAB`; connection metadata resides
under its `connections` directory. Keys remain in an explicitly selected native
credential vault or in session memory. Vault failure never writes plaintext
credentials. The Windows Credential Manager backend and browser sign-in still
need native verification; session-only storage remains an explicit choice.

## Forecasting runtimes

| Family | Candidate runtime | Native Windows status |
| --- | --- | --- |
| Chronos-Bolt / Chronos-2 | `requirements-chronos.txt`, Torch 2.8 | [Windows x64 Torch wheels exist](https://pypi.org/project/torch/2.8.0/#files); installation/import/inference/cache replay pending |
| TimesFM 2.5 | `requirements-timesfm.txt`, TimesFM 3.0.2 | Metadata reviewed; installation/import/inference/cache replay pending |
| TiRex-2 | `requirements-tirex2.txt`, FlashRNN 1.0.8, Windows-only Triton 3.4.0.post21 | Candidate constraint only; installation/native kernel imports/eager CPU inference pending |
| IBM Granite TTM R2/R2.1 | `requirements-ttm.txt` | Metadata reviewed; installation/import and each distinct profile behavior pending |
| TOTO 2 4M | `requirements-toto2.txt`, Python 3.12–3.13 only | No Python 3.11 runtime; all real/native inference pending |

Use the model-specific guide and constraints when installing an optional family.
No family is silently dropped or replaced with WSL. A failed native dependency
requires an explicit support decision before release.

FlashRNN's [published dependency metadata](https://pypi.org/pypi/flashrnn/1.0.8/json)
selects `triton-windows` on Windows. The conditional constraint fixes
`3.4.0.post21`; its [published files](https://pypi.org/project/triton-windows/3.4.0.post21/#files)
include Windows x64 Python 3.11–3.13 wheels, and the
[upstream compatibility table](https://github.com/triton-lang/triton-windows#pytorch)
pairs Torch 2.8 with Triton 3.4. Runtime diagnostics and TiRex loading enforce
this candidate version on Windows. This is a dependency decision, not evidence
that a CPU forecast has run. TiRex's
[upstream compiler guidance](https://nx-ai.github.io/tirex-2/faq/) needs checking
against this adapter's eager, compilation-disabled CPU path on the actual host.

## Records copied between platforms

Saved historical and future records can retain absolute POSIX or Windows source,
checkpoint and output locations without accessing those original locations.
Validation keeps file identities, numerical fingerprints, grids and unavailable
future accuracy intact. Legacy checkpoint metadata also retains previously
accepted UNC or redundant-separator spelling as inert provenance. Active
configuration has stricter local-disk requirements.

Detached future configurations use the same strict field, model, measurement,
origin and runtime rules with a lexical path resolver. They are marked read-only
internally. Planning, preparation and managed execution refuse that object before
filesystem/model work; explicitly load a native configuration to run again.
Opening or exporting a record does not migrate or rewrite its saved bytes.

## Remaining native qualification

Record Windows edition/build, Python version/architecture, PowerShell version,
Windows Terminal version and each installed runtime. In Windows Terminal check
first launch/skip/revisit, `/new` in both modes, completion, arrow history,
scrollback, cancellation before/during inference, return to prompt and exit.
Confirm no worker remains and no cancelled record appears completed. Check
drive paths, spaces, non-ASCII names, locked files and overwrite refusal.

The automated console fixture broadcasts a real
[Windows Ctrl+C event](https://learn.microsoft.com/en-us/windows/console/generateconsolectrlevent)
inside a private disposable console. It never broadcasts into the user's or
pytest's console. It is distinct from a terminal keyboard/ConPTY test. POSIX
signal/PTY cases keep explicit platform exclusions; ordinary worker cancellation,
forced-stop, record reconciliation and failure tests remain enabled on Windows.
Symlink tests skip only Windows error 1314 when the runner lacks that privilege;
regular-copy caches and ordinary-file refusal tests remain active. Record any
such skips when qualifying a host.

The configured nine-job CI matrix runs Black/Ruff, tests, source/wheel builds,
fresh wheel installation, dependency checks and isolated installed workflows.
Its cross-platform helper checks installed bytes against the exact built wheel.
Hosted CI remains blocked by account billing and was not rerun as part of S8.

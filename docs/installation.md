# Installing TSFM Lab

Install a supplied source checkout, source archive, or wheel into a dedicated
Python environment. The current package version is `0.1.0`; it has not been
published to PyPI. Commands such as `pip install tsfm-lab==0.1.0` from a package
index are therefore not an installation route for this release candidate.

Package metadata requires Python 3.11 or newer. The configured test matrix covers
Python 3.11, 3.12, and 3.13 on Linux and macOS. The TimesFM milestone's merged
`main` revision `b8d5647` passed all six jobs in
[run 40240609625](https://github.com/janwolyniak/TSFM-LAB/actions/runs/40240609625).
That routine matrix uses mocked models and does not establish optional model
inference on every platform. The later finance candidate has its own
[verification record](milestone7-verification.md).
Local model checks use Python 3.13 on macOS and are documented separately in the
[Chronos](chronos.md) and [TimesFM](timesfm.md) guides. See the
[release verification record](milestone5-verification.md) for prior installation
evidence; the build helper emits artifact hashes in a separate manifest.

## Install from a checkout or source archive

Use a checkout containing `pyproject.toml`, `src/tsfm_lab`, and the documentation.
For a published source revision, clone
[the project repository](https://github.com/janwolyniak/TSFM-LAB):

```sh
git clone https://github.com/janwolyniak/TSFM-LAB.git
cd TSFM-LAB
```

Check out the published commit or tag named by its maintainer before installing.
The locally prepared implementation is not made available remotely merely by
building a release candidate. For a supplied source archive, unpack it and enter
the directory containing `pyproject.toml` instead.

From that directory, on macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.txt '.[cli,parquet]'
python -m pip check
tsfm --version
```

The constraints file selects the recorded runtime dependency versions; it does not
install development tools unless a requested extra needs them. This is a regular
project installation, so the environment uses an installed package rather than
importing files from the checkout through an editable path hook.

Dependency installation can access the package index. The baseline workflow
itself needs no network connection, model checkpoint, API key, or AI service.

## Install a supplied wheel

Create and activate an environment as above. Install the wheel using its actual
local path; the example filename is the current package version:

```sh
python -m pip install '/absolute/path/tsfm_lab-0.1.0-py3-none-any.whl[cli,parquet]'
python -m pip check
tsfm --version
```

Replace `/absolute/path/` with the supplied artifact's location. The wheel
provides the package and bundled demonstration data. Documentation and the
repository's example YAML files are available in the source distribution; they
are not assumed to exist in the current directory after a wheel installation.

The dependency groups let users choose a smaller installation:

| Installation | Functionality |
| --- | --- |
| Base package | Python numerical core and CSV ingestion. |
| `cli` | Scriptable commands and the persistent interactive terminal. |
| `parquet` | Parquet input and export, using PyArrow. |
| `hub` | Explicit curated checkpoint downloads and cache inspection. |
| `chronos` | The pinned optional CPU Chronos-Bolt inference runtime. |
| `timesfm` | The pinned optional TimesFM 2.5 CPU inference runtime. |
| `dev` | pytest, Black, and Ruff for contributors. |

`cli` is sufficient for the active interactive terminal. The older `tui` extra
installs the retained Textual screens for regression coverage; it is not required
for the current prompt-based interface. Adding `chronos`, `timesfm`, or `hub`
does not pull model weights automatically.

## Find the installed command

The command is lowercase `tsfm`. Activating the environment places its generated
console script on `PATH`:

```sh
command -v tsfm
python -c 'import sys; print(sys.executable)'
python -m tsfm_lab --version
```

The first two paths should refer to the same environment. `python -m tsfm_lab`
is the equivalent entry point when a shell is resolving a different `tsfm`.
Using `.venv/bin/tsfm` explicitly also selects the local environment.

Editable installs such as `-e '.[dev,cli,parquet]'` are for development. A local
Python 3.13 macOS check encountered an editable `.pth` path that was ignored in
`.venv`, despite installation metadata and the console script being present.
If that environment reports `ModuleNotFoundError: tsfm_lab`, use the regular
project install or supplied wheel above. This observation is specific to the
checked environment; it is not a claim that editable installs fail generally.

## Run the offline demonstration

With an interactive terminal:

```sh
tsfm ui --workspace results
```

Then enter:

```text
/demo
```

Review the summary and answer `y` to `Run experiment?`, which defaults to `no`.
The bundled last-value example uses 32 hourly observations in `synthetic_units`,
context 8, horizon 4, stride 4, and four rolling origins. It produces 16 forecast
cases with MAE 2.5, RMSE about 2.915476, and MASE 2.5. These are deterministic
checks on constructed data, not evidence of pretrained-model advantage.

Use `/history`, `/show latest`, and `/export latest results/exports/forecasts.csv`
in the same session. Output remains in normal terminal scrollback. See the
[interactive guide](tui.md) for inline setup, commands, and cancellation.

## Run saved YAML experiments

From the root of the source checkout or unpacked source archive:

```sh
tsfm run examples/configurations/last-value.yaml --json
tsfm run examples/configurations/seasonal-naive.yaml --json
tsfm experiments list --workspace results --json
```

The run commands return UUIDs. Use an actual returned UUID in the following
command, replacing `RUN_ID`:

```sh
tsfm experiments show RUN_ID --workspace results --json
tsfm export RUN_ID --workspace results --output results/exports/forecasts.csv --json
```

Exports refuse an existing destination, so choose a new filename for another
export. YAML dataset/output paths resolve against the configuration file's
directory; `--workspace` resolves against the current working directory. The
[CLI guide](cli.md) explains custom configurations and comparison rules.

The same `[cli]` installation supports the separate
[hypothetical OHLC workflow](finance-extension.md), including saved-report
verification and CSV equity export. Python finance APIs require only the base
NumPy/pandas installation; optional inference runtimes are unnecessary.

The repository ignores `.venv/`, `results/`, `model-cache/`, and build artifacts.
The supplied YAML examples write under `results/`; placing exports under
`results/exports/` also keeps them outside tracked project files. Custom data and
export locations need their own deliberate storage or ignore choices.

## Add CPU Chronos inference separately

Start with the baseline demonstration before installing a model runtime. The
Chronos adapter accepts only its bounded CPU/float32 configuration and requires
compatible Chronos forecasting 2.2.0, PyTorch 2.8.x, Transformers 4.55.x, and
Hugging Face Hub 0.35.3. The supplied
[runtime constraints](../requirements-chronos.txt) record exact versions.

Follow the [Chronos guide](chronos.md) to install that optional group, inspect the
catalog, explicitly pull an exact checkpoint revision when a download is intended,
and run from cached or standalone local files. Normal runs never install
dependencies, download checkpoint files, or enable remote model code. Baseline
results, history, comparison, and export remain usable without the model runtime.

## Add CPU TimesFM inference separately

The `timesfm_2p5` adapter uses TimesFM model 2.5 through the explicit Torch API in
runtime package `timesfm==3.0.2`, with PyTorch 2.8.x, Hugging Face Hub 0.35.3, and
Safetensors 0.6.x. The [TimesFM constraints](../requirements-timesfm.txt) record
the tested versions. From the source root, install the optional runtime only
when dependency installation is authorized:

```sh
python -m pip install -c requirements-timesfm.txt '.[cli,timesfm]'
tsfm models info timesfm_2p5 --json
```

Select the curated `google/timesfm-2.5-200m-pytorch` repository or a compatible
standalone local checkpoint. The adapter requires CPU/float32, batch size 1,
contexts up to 16,256 observations, and horizons up to 128. It fails if Torch's
process-wide default dtype is not float32; it never changes that setting.
Follow the [TimesFM guide](timesfm.md) for explicit pulls, cache-only experiments,
median semantics, and the bounded local verification evidence.

## Add CPU TiRex-2 inference separately

TiRex-2 offers four training/forecast profiles through `tirex_2`. The shared
checkpoint runtime now pins Hub 0.35.3; Torch 2.8 and Transformers 4.55 stay
unchanged. Update the chosen model environment explicitly using the current
constraints when upgrading the source; no command upgrades global packages
automatically. Saved forecasts remain readable without model libraries.

**Built with technology from NXAI**

The optional kernel dependency has separate NXAI Community License terms;
read [NOTICE](../NOTICE), the [complete license](../licenses/mlstm-kernels-2.0.6.txt),
and the [TiRex-2 runtime guide](tirex2.md#optional-runtime-and-notices).
The [TiRex-2 constraints](../requirements-tirex2.txt) specify the tested CPU
environment. Its explicit install is:

```sh
python -m pip install -c requirements-tirex2.txt '.[cli,tirex2]'
tsfm models info tirex_2
```

This optional stack adds recurrent-kernel packages; Linux can also resolve
accelerator dependencies from their declared platform requirements. Core/CLI
installation and routine CI require none of them. Follow the guide for an
explicit pinned checkpoint pull and the four profile-specific native bounds.

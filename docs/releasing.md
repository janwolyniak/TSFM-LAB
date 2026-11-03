# Building, verifying, and publishing v0.3.1

On 2 November 2026 the owner authorized a **0.3.1 patch release after the
approved checks pass**, preserving the original v0.3.0 tag, assets and checksums.
Scope includes isolated optional-runtime installs, pinned checkpoint downloads,
bounded local CPU checks, and curated GitHub Pages publication at
`https://janwolyniak.github.io/TSFM-LAB/` while keeping source private.
No paid APIs, cloud/GPU inference, training or PyPI upload is included.
Windows 11/Windows Terminal, live providers, native vaults and browser sign-in
remain pending by owner decision. See the [patch notes](release-v0.3.1.md),
[verification record](v0.3.1-verification.md) and
[model evidence](v0.3.0-model-qualification.md). Older release records retain
their original dates, artifacts and verification limits.

## Review and freeze

Follow the [contribution workflow](../CONTRIBUTING.md): review the whole diff,
repair findings, run applicable tests/lint/format checks, inspect documentation
and publish a pull request with the self-review/validation summary. Merge using a
merge commit after the owner's authorization. Do not approve your own pull request
or treat an unavailable CI check as passed.

Keep package metadata in `pyproject.toml` and `src/tsfm_lab/__init__.py`, the
CITATION version/date/URL, README badges, current installation/release guides and
the curated public artifact consistent with **0.3.1**. Retain older release and
stage records as historical evidence. Open plan criteria requiring unperformed
qualification stay open with the release scope explained.

Before tagging, record the exact merged main revision. Build and verify artifacts
from its committed files; a same-version development wheel is insufficient.
Any packaged source edit invalidates the previous source identity and requires a
new commit/build and affected checks. Do not commit directly to main as routine
release preparation.

## Local checks and build

Use existing approved environments and the [README checks](../README.md#checks).
Record Python, OS, dependency/build-tool versions, commands, outcomes and skipped
checks. Routine checks require no model download, credentials, GPU or paid API.
Use one numerical CPU thread for disposable fixtures, as in the current evidence.

The builder uses the installed setuptools backend and pip, creates no build
isolation environment, and installs no dependencies. Ensure the build tools
already satisfy the repository's constraints. From the committed source root,
choose a new output directory:

```sh
TSFM_RELEASE_REVISION="$(git rev-parse HEAD)"
python scripts/build_distribution.py --out-dir /tmp/tsfm-v0.3.1-dist
```

The helper builds the source archive and then its wheel using `--no-index`,
`--no-deps` and `--no-build-isolation`. Building from the archive exposes missing
source files. Retain the JSON artifact/content report and hashes. The expected
files are `tsfm_lab-0.3.1.tar.gz` and `tsfm_lab-0.3.1-py3-none-any.whl`.
Do not claim byte-reproducible rebuilds without comparing repeated build bytes.

Inspect metadata, Python requirement, license/notices, explicit extras, entry
point, modules, bundled synthetic data/styles, guide/constraint/example sources
and verifier fixtures. `PLAN.md`, Git internals, environments, caches, checkpoints,
user data, secrets and authoring outputs must be absent. Manifest exclusion is
additional protection, not a substitute for inspecting intended source files.

The source list uses [setuptools distribution controls](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html)
through `MANIFEST.in`. Wheels and source archives are distinct artifacts; see the
[Python packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

## Verify an ordinary installation outside the checkout

Use a disposable environment without system/user site packages and a regular
installation of the exact wheel with CLI/Parquet dependencies available.
Dependency installation needs its own approved scope. Current qualification
copies an existing clean dependency environment and installs only the local wheel
with `--no-index --no-deps --no-cache-dir --force-reinstall`; it does not claim
fresh dependency resolution. Preserve original environments.

For an approved fresh install, the ordinary procedure is:

```sh
TSFM_SOURCE_ROOT="$PWD"
TSFM_SMOKE_ROOT="$(mktemp -d)"
python3 -m venv "$TSFM_SMOKE_ROOT/environment"
"$TSFM_SMOKE_ROOT/environment/bin/python" -m pip install -c "$TSFM_SOURCE_ROOT/requirements-dev.txt" '/absolute/path/tsfm_lab-0.3.1-py3-none-any.whl[cli,parquet]'
"$TSFM_SMOKE_ROOT/environment/bin/python" -m pip check
cd "$TSFM_SMOKE_ROOT"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PATH="$TSFM_SMOKE_ROOT/environment/bin:$PATH" "$TSFM_SMOKE_ROOT/environment/bin/python" -I "$TSFM_SOURCE_ROOT/scripts/verify_distribution.py" --wheel /absolute/path/tsfm_lab-0.3.1-py3-none-any.whl
```

The exact `--wheel` argument is mandatory. `-I` and the verifier reject source or
editable import leakage, wrong executable selection and inherited system site
packages. Installed package and metadata bytes must match the wheel before any
workflow checks. `pip check` checks dependency metadata; application execution
is a separate check. See the [Windows guide](windows.md) for native `Scripts`
executable paths and pending Windows-specific acceptance.

All 15 stages must pass: catalog/entry points, fixture arithmetic, baselines,
history/comparison/exports, validation failures, four-case costed finance,
native-point metadata fixture, five categories, seven fake assistant routes,
onboarding/preferences, guided historical/future baseline workers, optional import
boundaries and old saved records. No step establishes real pretrained inference,
live provider/vault compatibility, predictive superiority or trading profit.

## Bind artifacts to the committed source

From the source root, bind the source and wheel to the merged revision:

```sh
python scripts/verify_release_identity.py --revision "$TSFM_RELEASE_REVISION" --source /absolute/path/tsfm_lab-0.3.1.tar.gz --wheel /absolute/path/tsfm_lab-0.3.1-py3-none-any.whl
```

This compares packaged source bytes with Git and wheel runtime bytes with the
source archive. An unrelated untracked file may make the builder's dirty flag
true; the separate identity check must establish what entered the artifacts.
The installed verifier also executes the sealed pre-profile compatibility fixture
without rewriting it; its [provenance](../tests/fixtures/release_legacy_v1/README.md)
remains historical.

## Record qualification limits

The nine-job Python 3.11–3.13 Linux/macOS/Windows core matrix stays enabled.
Retain actual current-candidate job results and installed-wheel reports. Earlier
pre-start billing failures and the frozen 0.3.0 Windows failures are historical;
neither qualifies a repaired candidate. Do not change billing or protections.

Real-model evidence is specific to its exact package, checkpoints, runtime and
macOS CPU host. Keep frozen-wheel checks separate from patch-wheel replays.
Windows Server automated execution does not certify Windows 11/Terminal,
optional Windows model runtimes, live providers, native vaults or browser flows.
Carry all owner-deferred checks explicitly in the release manifest and body.

## Publish and verify the GitHub Release

After the reviewed merge, source identity, content inspection and isolated wheel
workflows pass, generate `verification.json` containing the exact merged revision,
artifact hashes, local results, review/fix record, commands/environment and the
unperformed checks. Hash the final source, wheel and verification manifest into
`SHA256SUMS`; do not include a self-referential hash of the sums file.

Create tag `v0.3.1` at `TSFM_RELEASE_REVISION` and publish those four exact assets
using the [release notes](release-v0.3.1.md), completed with revision, check counts,
qualification limits and artifact hashes. Keep the repository private. The separately authorized Pages workflow publishes
only curated documentation; verify its anonymous HTTP results and file identities.
Publishing a GitHub Release remains distinct from a PyPI upload.

Verify the remote tag resolves to the frozen merged revision and download the
uploaded assets into a new disposable directory. All downloaded hashes must match
`SHA256SUMS` and the local artifact bytes; the release body/manifest must identify
the same revision. Report the release URL and remaining verification limits.

# Building, verifying, and publishing v0.2.0

For v0.3.0's additional native Windows gate and candidate PowerShell workflow,
see the [Windows guide](windows.md) and [S8 verification](s8-windows-verification.md).
The v0.2.0 authorization and CI exception below remain specific to that release.

A release candidate consists of a reviewed source revision, its source archive
and wheel, and a verification record that identifies exactly what was exercised.
On 27 October 2026, the owner authorized finalizing **v0.2.0**, its Git tag, and
its GitHub Release assets, with hosted CI skipped because the account allowance
was exhausted. This authorization is already in force for this release; no
additional publication approval is needed. The distribution destination is the
[v0.2.0 GitHub Release](https://github.com/janwolyniak/TSFM-LAB/releases/tag/v0.2.0).
It includes the wheel, source distribution, `verification.json`, and `SHA256SUMS`; it does not include
a PyPI upload. Other releases and dependency/resource changes retain the
[contribution workflow's permission boundaries](../CONTRIBUTING.md).

The release sequence is:

1. Finalize the v0.2.0 version, notes, documentation, and integrated source;
   commit that complete state and record its exact revision.
2. Build a fresh source archive and its wheel from that committed revision.
3. Verify artifact identity, archive contents, and an ordinary isolated wheel
   installation; retain hashes, check reports, and the CI exception.
4. Create tag `v0.2.0` at that verified revision and publish the four exact assets
   through the authorized GitHub Release. Verify the tag target and uploaded hashes.

## Prepare the source and environment

Use the final integrated revision with a clean task-owned diff, version `0.2.0`
in `pyproject.toml` and `src/tsfm_lab/__init__.py`, and current documentation.
Run applicable local checks and record hosted CI as waived, not passed. Follow
the maintained
[development and merge workflow](../CONTRIBUTING.md), rather than creating an
independent release branch policy here.

Keep the README's source-version badge consistent with both version fields.
Its release-status badge and text describe publication separately. The static
metadata badges work for this private repository; the native GitHub Actions badge
tracks `main` checks for users with repository access.

Record the Git revision, package version, Python version, operating system,
installed build tools, dependency constraints, and exact commands. Refer to the
[README checks](../README.md#checks) for formatting, linting, and tests. Routine
release checks must not depend on a model download, credentials, or GPUs. A real
Chronos smoke test is a separately identified check using an explicitly selected
local checkpoint, as described in the [model guide](chronos.md).

The build helper uses the already installed setuptools backend and pip. It does
not require the `build` package, create a build-isolation environment, or install
dependencies. Ensure the chosen build environment already has the versions
declared by the repository's build requirements and constraints.

## Build the source archive and wheel

After freezing the committed source, record its revision and choose a fresh
output directory from the repository root:

```sh
TSFM_RELEASE_REVISION="$(git rev-parse HEAD)"
python scripts/build_distribution.py --out-dir /tmp/tsfm-v0.2.0-dist
```

The helper builds a source archive, then invokes pip to unpack it and build a
wheel from that source in a disposable build directory, using `--no-deps`,
`--no-build-isolation`, and `--no-index`. It validates archive contents and package
metadata, then prints JSON containing artifact paths and SHA-256 identities.
Building the wheel from the source archive checks that an omitted source file
cannot be hidden by the original checkout.

The source file list is maintained in `MANIFEST.in` using the
[setuptools distribution controls](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html).
See the [Python packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
for the distinction between source archives, wheels, and publication.

For the current version, the expected artifact names are
`tsfm_lab-0.2.0.tar.gz` and `tsfm_lab-0.2.0-py3-none-any.whl`. Treat the helper's
actual metadata and reported paths as authoritative when the version changes.

A reproducible procedure fixes source revision, tools, dependency versions, and
build inputs. Record the build timestamp or `SOURCE_DATE_EPOCH` when it is used.
Compare hashes from repeated builds before claiming byte-identical output;
successful builds alone do not establish that stronger property.

## Inspect the artifacts

Check the helper's content/metadata validation and review both archives before
delivery:

- The package metadata has the intended name, version, Python requirement,
  license, entry point, and optional dependency groups.
- The installed command is `tsfm`; its entry point selects the package's module.
- The wheel contains the Python modules and packaged demonstration CSV. Retained
  stylesheet resources must be present if the older Textual screens are included.
- The source archive contains the setup files, documentation, synthetic
  examples, constraints, and verification scripts needed to build and verify it
  independently.
- `PLAN.md`, Git internals, local environments, checkpoints, caches, experiment
  results, private datasets, reference repositories, secrets, and temporary
  authoring outputs are absent.

The untracked project brief is local input, not a release artifact. File exclusion
checks do not replace reviewing the intended source files for sensitive content.
Do not stage ignored outputs merely to make a package build succeed.

## Verify an ordinary installation outside the checkout

Create a new environment without system site packages and install the exact wheel
with its `cli,parquet` extras. The installation may resolve dependencies; use an
approved installation scope and record the resolved versions. The following
example uses a fresh disposable directory and the source checkout only to locate
the constraints and verification script:

```sh
TSFM_SOURCE_ROOT="$PWD"
TSFM_SMOKE_ROOT="$(mktemp -d)"
python3 -m venv "$TSFM_SMOKE_ROOT/environment"
"$TSFM_SMOKE_ROOT/environment/bin/python" -m pip install -c "$TSFM_SOURCE_ROOT/requirements-dev.txt" '/absolute/path/tsfm_lab-0.2.0-py3-none-any.whl[cli,parquet]'
"$TSFM_SMOKE_ROOT/environment/bin/python" -m pip check
cd "$TSFM_SMOKE_ROOT"
PATH="$TSFM_SMOKE_ROOT/environment/bin:$PATH" "$TSFM_SMOKE_ROOT/environment/bin/python" -I "$TSFM_SOURCE_ROOT/scripts/verify_distribution.py" --wheel /absolute/path/tsfm_lab-0.2.0-py3-none-any.whl
```

Replace the wheel path with the just-built artifact. Keep the environment and
output record until the results have been reviewed. `-I` ignores `PYTHONPATH` and
the current-directory import path, so a source checkout cannot accidentally
satisfy package imports. Use a regular wheel installation, not an editable
installation, for this acceptance check.

The `--wheel` option checks installed package bytes against that wheel before
the workflow checks. A same-version older installation is not sufficient. The
verifier also
requires a virtual environment without inherited system site packages.

The verification script exercises the installed command and bundled resources
using disposable baseline data and artifacts. It checks the numerical example,
configuration-driven execution, history/reload, compatible comparison, and
exports without a pretrained model or network request. It must fail rather than
fall back to importing this repository's `src` directory. Record its actual
checks and output with the artifact hashes.
It also runs a small OHLC fixture through finance backtest, checks independent
costed accounting, then moves the sources and verifies detached report reload,
CSV equity export, and overwrite refusal. This bounded check uses no model
weights and does not establish a profitable trading rule.

`pip check` verifies dependency metadata compatibility; it does not execute the
application. The installed smoke test verifies its exercised workflows; it does
not establish every optional model runtime or platform combination.

## Bind artifacts to the committed source

Verify the source archive and wheel from the frozen source against its recorded
revision. Run this command from the source root used above:

```sh
cd "$TSFM_SOURCE_ROOT"
python "$TSFM_SOURCE_ROOT/scripts/verify_release_identity.py" --revision "$TSFM_RELEASE_REVISION" --source /absolute/path/tsfm_lab-0.2.0.tar.gz --wheel /absolute/path/tsfm_lab-0.2.0-py3-none-any.whl
```

The identity check compares packaged files with the selected Git revision and
checks that the wheel's package modules and resources match the source archive.
Its artifact hashes identify the files to retain beside the installation report.
Unrelated untracked files can make the builder's `dirty` field true; the separate
byte comparison establishes which committed files actually entered the artifacts.
Do not describe an uncommitted build as an exact committed candidate.

Run the retained pre-profile fixture against the installed wheel as well:

```sh
"$TSFM_SMOKE_ROOT/environment/bin/python" -I "$TSFM_SOURCE_ROOT/tests/fixtures/release_legacy_v1/verify.py"
```

This uses synthetic configurations, saved forecasts, comparisons, and a sealed
finance report produced by the documented older revision. It checks current
readers against old bytes, including dataset identity and preservation of the
input files. See the fixture's [provenance and reproduction instructions](../tests/fixtures/release_legacy_v1/README.md).

The [v0.2.0 release notes](release-v0.2.0.md) and
[P8 qualification record](p8-release-verification.md) document the included scope,
local qualification, and the release's verification limits. A source change
after freezing invalidates the previous artifact identity: commit the final
change, record the new revision, rebuild, and repeat affected checks before tagging.

## Record the verification ceiling

The configured CI matrix covers Python 3.11–3.13 on Linux and macOS. The owner
explicitly waived hosted CI for the final v0.2.0 release on 27 October 2026.
Record this as an exception with local check evidence; skipped jobs are not
successful checks and the final release's platform matrix remains unverified.
Keep the configured workflow and repository protections intact. The existing
integration exception uses `[skip ci]` in the authorized release commits.

Assistant qualification uses simulated responses and fake credentials. Live
provider/account compatibility and native OS credential-vault behavior remain
unqualified. No real credentials, new model downloads, or additional model
inference are required to qualify these release artifacts.

Keep prior milestone records as historical evidence. A previous wheel's hash,
installed-module check, or Chronos smoke result does not identify a newly built
wheel. Record the new source/artifact identity and rerun the affected checks for
the final candidate. Distinguish a baseline install check from the separately
bounded Chronos integration evidence; neither supports a general forecasting or
trading performance claim.

## Publish the authorized GitHub Release

The local candidate is ready for the authorized publication when its acceptance
criteria are met, artifact contents and hashes are reviewed, installation
instructions work from the delivered source/wheel, and all material limitations
are recorded. Include any unrun remote CI/platform checks in that review.

Use the [v0.2.0 notes](release-v0.2.0.md) and finalize their GitHub Release body
with the precise source revision, local check results, hosted-CI waiver, and
artifact hashes. Keep historical milestone records and the v0.1.0 preparation
notes as historical evidence. A GitHub Release does not imply PyPI publication.

Create `verification.json` after the source freeze, recording the exact commit,
artifact identities, local checks, and CI exception. Generate `SHA256SUMS` for
`tsfm_lab-0.2.0.tar.gz`, `tsfm_lab-0.2.0-py3-none-any.whl`, and `verification.json`
using their final bytes and filenames. Create tag `v0.2.0` at
`TSFM_RELEASE_REVISION`, then attach those three files and `SHA256SUMS` to the
GitHub Release. Do not substitute earlier preparation builds
or GitHub's automatically generated source snapshots for the checked assets.

After publication, verify the remote tag resolves to the frozen revision,
all uploaded asset hashes match `SHA256SUMS`, and the release body identifies
the same revision and evidence. Report the GitHub Release URL and the verification
limits. No additional approval step is required for this already authorized release.

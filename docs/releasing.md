# Building and verifying a release candidate

A release candidate consists of a reviewed source revision, its source archive
and wheel, and a verification record that identifies exactly what was exercised.
The steps below prepare local artifacts. Uploading packages, publishing a GitHub
release, pushing a tag, or publishing branches requires separate owner
authorization under [CONTRIBUTING.md](../CONTRIBUTING.md).

## Prepare the source and environment

Use the intended committed revision with a clean task-owned diff, the agreed
version in `pyproject.toml` and `src/tsfm_lab/__init__.py`, current documentation,
and the repository's configured checks passing. Follow the maintained
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

From the repository root, choose a fresh output directory:

```sh
python scripts/build_distribution.py --out-dir /tmp/tsfm-dist
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
`tsfm_lab-0.1.0.tar.gz` and `tsfm_lab-0.1.0-py3-none-any.whl`. Treat the helper's
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
"$TSFM_SMOKE_ROOT/environment/bin/python" -m pip install -c "$TSFM_SOURCE_ROOT/requirements-dev.txt" '/absolute/path/tsfm_lab-0.1.0-py3-none-any.whl[cli,parquet]'
"$TSFM_SMOKE_ROOT/environment/bin/python" -m pip check
cd "$TSFM_SMOKE_ROOT"
PATH="$TSFM_SMOKE_ROOT/environment/bin:$PATH" "$TSFM_SMOKE_ROOT/environment/bin/python" -I "$TSFM_SOURCE_ROOT/scripts/verify_distribution.py"
```

Replace the wheel path with the just-built artifact. Keep the environment and
output record until the results have been reviewed. `-I` ignores `PYTHONPATH` and
the current-directory import path, so a source checkout cannot accidentally
satisfy package imports. Use a regular wheel installation, not an editable
installation, for this acceptance check.

The verification script exercises the installed command and bundled resources
using disposable baseline data and artifacts. It checks the numerical example,
configuration-driven execution, history/reload, compatible comparison, and
exports without a pretrained model or network request. It must fail rather than
fall back to importing this repository's `src` directory. Record its actual
checks and output with the artifact hashes.

`pip check` verifies dependency metadata compatibility; it does not execute the
application. The installed smoke test verifies its exercised workflows; it does
not establish every optional model runtime or platform combination.

## Record the verification ceiling

The configured CI matrix covers Python 3.11–3.13 on Linux and macOS. The local
release check exercises the actual Python 3.13 macOS environment named in
[milestone5-verification.md](milestone5-verification.md). Do not describe the
remaining matrix as verified until its jobs have passed for the release candidate.

Keep prior milestone records as historical evidence. A previous wheel's hash,
installed-module check, or Chronos smoke result does not identify a newly built
wheel. Record the new source/artifact identity and rerun the affected checks for
the final candidate. Distinguish a baseline install check from the separately
bounded Chronos integration evidence; neither supports a general forecasting or
trading performance claim.

## Review before publication

The local candidate is ready for a publication decision when its acceptance
criteria are met, artifact contents and hashes are reviewed, installation
instructions work from the delivered source/wheel, and all material limitations
are recorded. Include any unrun remote CI/platform checks in that review.

Present the precise artifacts, source revision, verification record, and intended
publication destination to the owner. Publication approval is separate from
approval to implement, install dependencies, build local artifacts, or run tests.
Until that approval is given, retain the candidate locally and report its status
without uploading, tagging, or claiming that PyPI/GitHub publication has occurred.

For the first GitHub Release, start from the
[prepared release notes](release-v0.1.0.md). Once publication is authorized,
finalize the README's release badge/status and the changelog, remove the notes'
preparation-status paragraph, and change their documentation links from `main`
to `v0.1.0`. A GitHub Release does not imply PyPI publication. Do not use a public
Shields GitHub-release lookup for this private repository.

Review, commit, and integrate that final wording before freezing the release
revision. Rebuild the source distribution and wheel from that exact revision
and repeat the affected installation checks; the preparation candidate's hashes
do not identify these new artifacts. Create tag `v0.1.0` at the verified revision
and attach its wheel, source distribution, and `SHA256SUMS` to the authorized
GitHub Release. Include the exact source revision, check results, and artifact
hashes in the release body so the tag and downloadable files agree.

# TSFM Lab v0.3.0

Released 30 October 2026 through GitHub. Python 3.11–3.13 remains the application
target; TOTO 2 requires Python 3.12 or 3.13. This release includes the reviewed
implementation and local offline evidence. It does not certify unexecuted
native platforms, optional model runtimes or live assistant services.

## Included changes

- First-launch onboarding, saved nonsecret directory/category/export preferences,
  and `/setup`/`/settings` for later changes.
- Model-specific guided setup across all five usage categories, shared option
  validation and dataset previews before preparation or inference.
- Reviewed checkpoint preparation, exact-revision/cache verification, managed
  progress/cancellation, and checked SVG/provenance outputs with exact paths.
- Separate future forecasting from the final observation, causal consumed
  context, original-unit predictions and detached reload/export. Accuracy remains
  unavailable while future actuals are absent; historical comparison and finance
  validators refuse future records.
- DeepSeek, installed local Ollama and explicit compatible endpoints, alongside
  the four existing assistant routes. Endpoint-bound credentials, outgoing review
  and ordinary local draft/execution validation remain enforced.
- Five additional TTM profiles: daily/weekly R2.1 and R2 native horizons
  192/336/720. TOTO 2 4M adds one median-output family with explicit project bounds,
  safe loading and a Python 3.12–3.13 requirement.
- Windows path/quoting, detached cross-platform records, portable locking and
  executable paths, plus nine configured Linux/macOS/Windows CI jobs.
- Curated static documentation generation with a complete capability matrix,
  category examples, crawl files and citation metadata. Hosting is separate.
- Release review fixed an infinite ancestor traversal when onboarding/settings
  validates an unavailable directory root, including an absent Windows drive.

Old configurations and saved historical/finance artifacts remain readable.
The numerical scope remains one regular univariate series, fixed UTC intervals,
complete horizons, declared units, no hidden imputation or model optimization.

## Install and verify

Download `tsfm_lab-0.3.0-py3-none-any.whl` or `tsfm_lab-0.3.0.tar.gz` from the
[GitHub Release](https://github.com/janwolyniak/TSFM-LAB/releases/tag/v0.3.0).
`verification.json` records the exact merged revision, artifact identities,
review/local checks and qualification limits. Validate the files using
`SHA256SUMS` before installation. The repository remains private; access does not
change as part of publication. No PyPI upload is included.

```sh
python -m pip install './tsfm_lab-0.3.0-py3-none-any.whl[cli,parquet]'
python -m pip check
tsfm --version
tsfm ui --workspace results
```

The baseline demonstration and manual forecasting need no assistant connection
or model weights. Optional runtime installation and checkpoint downloads remain
explicit; see the [installation guide](installation.md).

## Verification limits

The owner requested review/fixes, merge/push and release publication on
30 October 2026 after the pre-release qualification limits were reported. The
prior instruction to use offline tests and defer real inference remains in force.

- The [release verification record](v0.3.0-verification.md) retains local macOS
  Python 3.11–3.13 test/lint and exact installed-wheel evidence. Installed workflows
  exercise actual baseline workers and fake assistant transports, without model
  downloads, real credentials or new runtime installations.
- Hosted GitHub jobs cannot start because of account billing/spending limits.
  They are unavailable checks, not successful tests. Publication proceeds under
  the owner's release request with this exception recorded; no account settings
  or protection rules are changed.
- New TTM/TOTO installed-runtime compatibility and representative real CPU
  inference remain unqualified. Catalog compatibility/fake-runtime evidence is
  separate from real inference and from forecast accuracy.
- Windows 11/Windows Terminal, native locks/console control events, optional-model
  execution and native credential-vault/browser flows remain unverified on native
  hosts. Portable/simulated tests do not certify those paths.
- Live-provider behavior remains unqualified; fake transports verify the exercised
  protocols and local boundaries. Static documentation is built locally and has
  not been deployed to a confirmed public host.

No benchmark ranking, financial advantage, GPU/MPS support, panels, covariates,
training or live-feed support is claimed. Code uses Apache-2.0; runtime and model
weight licenses remain separate, including the bundled TiRex runtime notices.

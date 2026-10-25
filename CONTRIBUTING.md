# Contributing to TSFM-LAB

This guide defines how to propose, implement, validate, review, and merge changes.
It applies to code, tests, research, documentation, and automated coding agents.
Keep contributions focused, reproducible, and easy to review.

## Project direction and current tooling

TSFM-LAB is being developed as a local-first platform for evaluating pretrained
time-series models. Keep model adapters, datasets, evaluation logic, domain
extensions, and terminal interfaces separate. Discuss changes to architecture,
scope, public interfaces, dependencies, or evaluation methodology before
implementing them. Do not add cloud infrastructure, a web backend, or model
training as incidental parts of another change.

The repository currently has no Python package configuration, automated test
suite, formatter configuration, or CI workflow. This guide establishes workflow
rules; it does not imply that those checks or GitHub protections are already
configured. When tooling is introduced, record the exact setup and check commands
in [README.md](README.md) and configure them in the repository. Prefer Black for
Python formatting, Ruff for linting, and pytest for tests unless an agreed project
decision establishes alternatives. Do not install or upgrade dependencies without
the repository owner's authorization.

Read applicable `AGENTS.md` instructions and current architecture or setup
documentation before editing the relevant area. Keep shared contribution rules
here rather than copying them into multiple instruction files.

## Before starting work

1. Define the problem, intended behavior, scope, and acceptance criteria in an
   issue, task description, or agreed discussion. Small, clear fixes do not need
   a separate issue.
2. Inspect the working tree and current branch. Preserve unrelated local changes,
   untracked files, experiment outputs, and work belonging to other contributors.
3. Check existing issues, branches, and pull requests to avoid duplicate work.
4. Start one focused branch from the current `main` using the workflow below.
5. If a discovery changes the agreed scope or approach, request a decision before
   implementing that expansion; continue independent work where possible.

## Branching policy

### Permanent and temporary branches

- `main` is the integration branch and should remain usable. Completed work enters
  it through reviewed pull requests. Do not commit directly to `main`.
- Use short-lived topic branches for individual fixes, features, documentation
  changes, and experiments. Keep one coherent purpose per branch.
- There is no permanent `develop` branch. Do not introduce additional integration
  or release branches without an agreed need.
- Branch from `main` unless work explicitly depends on an unmerged topic branch.
  For a dependent branch, document its base and merge order in the pull request.
  Merge the prerequisite first, then ensure the dependent pull request targets
  `main` and contains only the remaining changes.
- Open a draft pull request early for work that benefits from feedback. A draft
  is not approval to merge unfinished work.

### Branch names

Use `<type>/<short-description>` or `<type>/<issue-number>-<short-description>`.
Use lowercase ASCII letters, digits, and hyphens in the description, with one
slash separating the type. Use a bare issue number when one exists; do not invent
one. Prefer a short description of the outcome over a person's name or date.

| Type | Use | Example |
| --- | --- | --- |
| `feat` | New user-facing functionality | `feat/42-rolling-origin-evaluation` |
| `fix` | Correct incorrect behavior | `fix/57-forecast-timestamp-alignment` |
| `docs` | Documentation only | `docs/contributing-guide` |
| `refactor` | Internal restructuring without intended behavior changes | `refactor/adapter-registry` |
| `test` | Tests or test infrastructure | `test/forecast-contracts` |
| `perf` | Performance improvements | `perf/batch-window-inference` |
| `build` | Packaging or dependency configuration | `build/optional-model-dependencies` |
| `ci` | Continuous integration workflows | `ci/python-checks` |
| `chore` | Maintenance outside the categories above | `chore/ignore-local-artifacts` |
| `experiment` | Bounded research or feasibility work | `experiment/chronos-cpu-baseline` |
| `hotfix` | Urgent correction requiring expedited review | `hotfix/result-cache-corruption` |

Avoid vague names such as `updates`, `test2`, or `final`, personal prefixes, and
multiple unrelated objectives in one branch. An experiment branch follows the
same review and validation rules if any of its work is proposed for `main`.
A hotfix uses the normal merge strategy and must document any validation gap.

### Start a branch

The examples below assume a clone with write access and a remote named `origin`.
Replace the example branch and issue number with the actual task. Fork
contributors should fetch `upstream/main` as their base and publish topic branches
to their own `origin`; open the pull request against this repository's `main`.

```sh
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c feat/42-rolling-origin-evaluation
```

Run branch-switching and integration commands with a clean working tree. If work
is already present, inspect ownership and preserve it before proceeding. Do not
automatically stash, reset, clean, or discard another person's changes. If
`--ff-only` fails, inspect the divergence and resolve its cause; do not force the
local branch to match the remote.

### Keep a branch current

Before requesting final review, integrate the latest target branch and rerun the
affected checks. Use a merge to update a published or shared branch:

```sh
git fetch origin
git switch feat/42-rolling-origin-evaluation
git merge origin/main
```

Rebasing is allowed for unpublished commits on a private local branch. Do not
rebase a shared branch or rewrite published history as routine cleanup. Never
force-push `main`. Rewriting a published topic branch requires explicit maintainer
approval and coordination with its contributors; if approved, use
`--force-with-lease` rather than `--force`. Request review again when the reviewed
content changes.

## Implementation and commits

- Follow existing code conventions; favor explicit, readable implementations and
  avoid unrelated refactoring or formatting.
- Keep optional model dependencies isolated. Do not require every inference
  framework for basic platform use.
- Validate input schemas, timestamps, shapes, horizons, and model capabilities at
  their boundaries. Fail clearly when an operation is unsupported.
- Update relevant documentation when behavior, configuration, interfaces, or
  workflows change. Explain compatibility effects and migration steps where needed.
- Add regression tests for corrected behavior and meaningful tests for new logic.
  Do not add tests that merely repeat an implementation or require large downloads
  for routine validation.
- Never commit credentials, tokens, private datasets, downloaded checkpoints,
  virtual environments, or local reference repositories. Respect `.gitignore`;
  ignored files are not automatically safe to publish or delete.

Use focused commits with imperative summaries. New commits should use
`<type>(<optional-scope>): <summary>`, using the branch types above except
`experiment` and `hotfix`: use `feat`, `fix`, `test`, or `docs` as appropriate for
the actual change. Omit the parentheses when there is no useful scope.

Examples: `feat(evaluation): add rolling-origin splits`,
`fix(metrics): align targets before scoring`, and
`docs: document contribution workflow`.

Use the body to explain motivation, material tradeoffs, and issue references when
needed. Do not claim checks passed unless they were run. Stage specific paths or
hunks, inspect the staged diff, and commit only task-owned changes:

```sh
git diff
git add CONTRIBUTING.md
git diff --cached --check
git diff --cached
git commit -m "docs: document contribution workflow"
```

The staging command is an example for a documentation task; use your actual
paths. Automated agents should create a local commit after verified work when
ownership is clear. If edits overlap another contributor's work, ask before
staging or committing those files.

## Validation and research integrity

Run the repository-configured checks relevant to the change before declaring it
complete. For code, run targeted tests and configured formatting/linting checks;
include type, build, CLI, and integration checks when the affected interface
requires them. Broaden checks when failures or shared-interface changes warrant
it. Routine tests should use small, disposable fixtures without production access,
paid services, GPUs, or mandatory model downloads.

For documentation-only changes, inspect the full diff and verify links, examples,
structure, and any configured documentation checks. Execute examples when their
behavior changes. Render and visually inspect documents or figures when layout
is part of the deliverable. If tooling does not exist or a check cannot run,
state that limitation explicitly; static inspection is not runtime validation.

Evaluation contributions must additionally satisfy the following:

- State the target, units, sampling frequency, forecast horizon, and evaluation
  population. Keep chronological splits explicit and prevent leakage through
  preprocessing, context construction, tuning, or target alignment.
- Fit learned preprocessing on permitted training history only. Keep holdouts
  separate from model and protocol selection; disclose reuse of exposed data.
- Compare models on matching datasets, splits, horizons, information sets, and
  metric definitions. Include relevant simple baselines and explain exclusions.
- Record the source revision, model/checkpoint revision, data provenance or
  identifier, configuration, seeds, environment, hardware, and metric aggregation
  needed to reproduce results. Resolve moving model references to exact revisions.
- Separate measured results from assumptions and expected benefits. Report failed
  runs, limitations, and negative results when they affect the conclusion.
- Keep forecast accuracy and distribution calibration distinct from financial
  utility. Trading evaluations must specify signal timing, execution prices,
  transaction costs, turnover, and capital assumptions without future information.
- Keep raw/processed data and large outputs out of Git. Track small licensed
  fixtures, contracts, configurations, and concise evidence summaries as needed.
  Document how to reproduce omitted artifacts and any access restrictions.
- Do not silently enable untrusted remote model code or unsafe checkpoint loading.
  Document model and data license constraints; see the repository's
  [LICENSE](LICENSE) for the project code license.

Get owner authorization before dependency installation, paid APIs or compute,
large downloads or training workloads, external-system changes, destructive
operations, or operations expected to exceed ten minutes. Authorization to edit
locally does not automatically authorize publishing a branch, opening a pull
request, or merging it. Existing authorization remains valid within its scope.

## Pull requests and review

Once publication is authorized, publish the intended topic branch explicitly:

```sh
git push --set-upstream origin feat/42-rolling-origin-evaluation
```

Use a descriptive pull request title following the commit-summary format. Target
`main` unless the documented dependency requires a temporary different base.
Include:

- The problem, intended behavior, and related issue or acceptance criteria.
- The final changes and relevant design choices or compatibility effects.
- Exact checks run and their outcomes, including checks skipped and why.
- Evidence for research claims, with reproduction instructions where applicable.
- Remaining limitations and follow-up work that is outside this change.

Before marking a pull request ready, inspect the entire diff for unrelated files,
secrets, generated artifacts, and undocumented behavior. Resolve review comments
through code changes or a clear technical response. Prefer follow-up commits to
rewriting published commits. Review findings should prioritize correctness, data
integrity, regressions, security, and missing verification.

At least one maintainer must review before merging. Authors should not approve
their own changes. For a sole-maintainer repository, explicitly record a
self-review and validation summary in the pull request when independent review
is unavailable. Automated agents must not approve their own pull requests or
treat successful checks as merge authorization.

## Merge policy

Use **Create a merge commit** on GitHub or `git merge --no-ff` locally. Preserve
the topic branch's commits and a visible merge boundary. Do not use squash merges,
rebase merges, or fast-forward integration into `main`. Keep intermediate commits
reviewable; do local cleanup before the first publication when useful.

A maintainer may merge only when:

- The agreed scope is complete, documentation is current, and the diff is focused.
- Review is complete and blocking conversations are resolved.
- Applicable checks pass for the current merge candidate, with any unavailable
  checks and accepted exceptions explicitly recorded.
- The branch includes the latest target-branch changes and has no conflicts.
- Publication and merge authorization have been given by the owner or a delegated
  maintainer. An automated agent needs explicit authorization for these actions.

Prefer merging through the reviewed pull request. An explicitly authorized local
merge should reference that pull request and use a clean working tree:

```sh
git fetch origin
git switch main
git pull --ff-only origin main
git merge --no-ff --no-commit feat/42-rolling-origin-evaluation
```

Inspect the merged diff and run applicable checks before completing the merge.
If checks fail, repair the topic branch and repeat review/validation as needed;
use `git merge --abort` to abandon an uncommitted merge when appropriate. Do not
publish a broken merge. Once the candidate is verified:

```sh
git commit -m "Merge feat/42-rolling-origin-evaluation into main (PR #123)"
git push origin main
```

Replace the PR number with the actual reference. If `main` advances before the
push, fetch, integrate the new remote changes without rewriting published
history, and validate the resulting candidate again. Never force-push to bypass
a rejection or branch protection.

Resolve conflicts by understanding both changes; do not blindly select one side.
Rerun affected tests and obtain renewed review when the resolution changes
behavior. Delete local or remote topic branches only after confirming they are
merged, contain no remaining work, and deletion is authorized. Do not delete an
unmerged experiment merely because its pull request is closed.

Revert a faulty published change with a new reviewed revert commit rather than
resetting `main`. Reverting a merge requires selecting the correct parent and
checking effects on later dependent changes; explain the rollback in the pull
request and rerun relevant checks.

### Recommended GitHub protections

Maintainers should configure `main` to require pull requests, review resolution,
and applicable status checks, and block force pushes and deletion. Require fresh
review after substantive updates where reviewer availability allows it. Enable
merge commits and disable squash/rebase merge options; do not require linear
history, which conflicts with this policy. Configure required check names only
after their workflows exist. These are configuration recommendations, not claims
about the current remote settings; changing settings requires authorization.

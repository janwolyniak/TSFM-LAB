# Explicit OHLC finance verification

Date: 26 October 2026. Scope: the separately agreed finance extension over
checked saved price forecasts, after the TimesFM/Small/fev milestone merged.
Package version remains the unpublished `0.1.0` candidate.

## Agreed behavior

The owner selected lead 1, stride 1, next-bar open entry, and that bar's close
exit. Signals compare forecasted target close with observed origin close.
Positions are unit long, flat, or short after entry costs, with full liquidation
every bar. Explicit capital, threshold, fees, slippage, short financing, and
signal latency accompany each report. Entry open must be strictly later than
origin close plus latency. The complete original price series must match saved
dataset identity and all execution closes.

The extension keeps forecast metrics unchanged, adds price-return and three-way
direction diagnostics, and reports strategy, cash, and costed buy-and-hold paths.
Reports contain raw selected cases, copied parent metadata/artifact identities,
data and implementation provenance, resolved rules, and recalculated results.
JSON reload and CSV export work without original files or inference runtimes.
See the [finance contract](finance-extension.md) for formulas and limitations.

## Executed checks

Local verification uses Python 3.13.6 on macOS. No dependencies, checkpoint
downloads, broker connections, or paid compute were added for finance.

| Check | Result |
| --- | --- |
| Complete configured test suite | 1,248 passed, 2 opt-in real-model smoke tests skipped; 49.55 seconds. The skipped tests require explicit model runtime/checkpoint selection and were exercised for the preceding model milestone. |
| Final report/CLI/package regressions | 84 passed in 5.76 seconds after the explicit quote-unit reload fix and validated-snapshot CSV export. This includes the added regression beyond the earlier complete-suite count. |
| Black and Ruff | All 106 Python files passed formatting/lint checks, including the bounded research scripts. |
| Accounting oracles | Long/short/cash conservation, entry sizing after costs, both transaction legs, per-bar short financing, repeated-position round trips, short high insolvency, finite arithmetic, and masked input rejection pass. |
| Timing and signals | Explicit OHLC ordering/grid/bounds, strict latency and touching-bar refusal, exact threshold ties, lead-1/stride-1 alignment, and unchanged signals after future-open perturbation pass. |
| Checked reports | Strict JSON/duplicate/nonfinite handling, fingerprints, parent grid/counts and lead-1 forecast metrics, financial recalculation, tampering detection, atomic publication/failure injection, and no-overwrite checks pass. |
| Runtime independence | A fresh process blocks model, terminal, and YAML imports while reloading and exporting a saved report. Source files and the parent workspace are moved away in the CLI/installed-verifier fixtures. |
| Documented synthetic example | The committed twelve-session OHLC fixture produces eight forecast cases and signals `[-1,1,-1,1,-1,1,-1,1]`; forecast MAE is 1.25, RMSE is sqrt(1.75), and MASE is 0.5. Checked report reload and eight-row CSV export pass. |
| Distribution inspection | Source archive and wheel built from that archive pass content/metadata checks: 161 source files, 32 required source paths, 60 wheel entries, finance modules/examples/guides included, and only NumPy/pandas in the numerical core. |
| Clean installed wheel | Installed the local wheel without dependency changes into the existing isolated core/CLI/Parquet environment. `pip check` passes and all nine installed-verifier stages pass in 8.869 seconds, including four-case independent finance arithmetic, detached reload/export, and overwrite refusal. No model runtime, Hub, or Textual is present. |
| Documentation | 27 Markdown files, 170 local links/anchors, and 42 shell blocks pass structural/link/syntax checks. External finance references resolve to the cited SEC explanations. |
| Independent review | Static review found no outstanding actionable issues after fixing quote-unit reload consistency and correcting the example's model-option nesting. |

The checks above describe the verified preparation tree. Final artifacts are
rebuilt after committing the task changes and checked with the installed verifier
again. Build manifests outside Git identify source revisions and archive checksums;
this document cannot embed the hash of an archive that contains itself.

## Limits and publication

The fixture is hand-constructed, with a fictional daily calendar including
weekends. Its returns establish accounting behavior and no market advantage.
This policy refuses continuous touching bars even at zero latency; real data
must not be shifted to create a false execution gap. The simulator does not
model live orders, borrow availability, margin rules, corporate actions, market
impact, or within-bar drawdown. Volatility is unannualized sample standard
deviation; drawdown uses period-end equity including initial capital.

Hashes and internal consistency checks are not signatures. Detached reports
cannot reconstruct full original datasets or authenticate individual prediction
rows solely from a retained checksum. Source creation validates the original
workspace; reload verifies the embedded evidence and current calculation policy.

The preceding model milestone merged in
[PR #2](https://github.com/janwolyniak/TSFM-LAB/pull/2) at `b8d5647`, whose six
post-merge jobs passed in
[run 40240609625](https://github.com/janwolyniak/TSFM-LAB/actions/runs/40240609625).
Finance remote CI and integration require their own candidate checks and owner
authorization. No GitHub/PyPI package release was published. The pre-existing
untracked `PLAN.md`, generated reports, environments, and evidence stay outside
Git.

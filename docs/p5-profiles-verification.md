# P5 usage profiles verification

Date: 26 October 2026. Scope: Generic, Finance, Energy, Temperature, and Weather workflow metadata,
guided setup, examples, and saved presentation under the approved
[release plan](next-release-plan.md). No numerical forecasting or finance
accounting policy is changed. Package publication remains separate.

The implementation validates a detached optional schema-1 `workflow` block.
Missing metadata retains its old serialization and displays as Generic.
Profile-only comparisons retain the same dataset fingerprints, prediction
values, metric artifacts, and policy. Target units and all actual task settings
remain comparison constraints.

Both terminal setup paths preserve deliberate numeric settings, custom dataset
paths, back navigation, cancellation, execution review, and model restrictions.
Domain examples provide 1,600 original synthetic hourly observations; negative
temperatures remain intact. Fictional example declarations are cleared when
moving to custom data, and units/measurement semantics are asked explicitly.

Saved metadata is checked in the initial configuration and result before
completion and during reload. Semantic mismatches fail before output/status
mutation. The public setup accessor checks its own file checksum and returns
a detached snapshot without original data or optional runtimes. Existing
manifests, old results, comparison JSON, and forecast export columns keep their
prior representation.

Finance close-price runs offer an explicit existing-command handoff with every
required timing/cost placeholder. The manual report contract and median-only
eligibility remain unchanged. Forecast-only finance, negative non-price data,
and negative temperature forecasts remain usable without clipping.

Focused tests cover strict profile/options/units, both setup interfaces, metadata
binding and tamper refusal, old-record byte preservation, cross-profile numerical
equivalence under both tasks, source-free CSV/JSON/Parquet reload/export,
resource lifetimes, and unchanged manual OHLC rules. Source/wheel checks require
the new modules, fixtures, examples, and guides. The installed workflow adds all
five profile runs, negative temperatures, profile-only comparison, and reload
after source removal, with model libraries absent.

The full offline suite passed **1,900 tests**, with five opt-in model tests
skipped, in 71.23 seconds. They are unchanged model checks outside P5's metadata
scope. A further 15 compatibility cases passed for every usage profile with
each exact TTM context, using inert local checkpoints and real bundled data,
with runtime imports/network blocked. Formatting checked 129 Python files;
linting and whitespace checks passed. All 14 example configurations parsed.
Documentation checks covered 39 Markdown files, 280 local links, and 54 shell
blocks without errors. Independent core/UI
review found and repaired initial-metadata preflight handling and a Rich-markup
rendering issue; regression tests pass, with no remaining actionable findings.
The packaging review strengthened the installed check from aggregate equality
to exact per-case actual/prediction equality against source and relabelled runs.

All five actual example YAMLs ran in an isolated installed environment and each
produced 16 forecast cases. The finance example generated, independently reloaded,
and exported its 16-case OHLC report under explicit demonstration assumptions;
network guards recorded no attempts. Older real TTM schema-2 and TiRex schema-1
saved results reloaded through the new installed package, including exact saved
TTM CSV/JSON/Parquet re-exports without model imports/current evidence lookup.

The preparation source archive contained 224 files / 70 required paths; the wheel
contained 79 entries and the three existing license files. Its only mandatory
core dependencies remain NumPy and Pandas. All eleven installed workflow stages
passed in 23.288 seconds, with model/kernel libraries absent; the clean
environment's dependency consistency passed. No dependency was installed or
upgraded for P5; only the project wheel was replaced in the temporary verifier.

Final source/wheel hashes and installed-byte identities are recorded beside the
final artifacts, independently of source tests and the historical working-tree
provenance. Detailed local evidence
lives under `/private/tmp/tsfm-next-models-01a174c4/p5-verification/`; temporary
paths are an audit trail, not dependencies of saved experiments.

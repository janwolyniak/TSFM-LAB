# TSFM-LAB 0.3.1

Qualification and repair patch, 2 November 2026. The owner authorized bounded
isolated real-model checks, private-source GitHub Pages publication, review,
merge and patch publication. The original **v0.3.0 tag, assets and checksums are
preserved**. This release does not include a PyPI upload.

## Repairs

- Windows export durability now uses a writable descriptor before publication.
  Unknown named-user homes are rejected before creating output directories.
- Windows worker cancellation watches for an available control byte before
  reading, avoiding the native startup hang observed during NumPy DLL loading.
  Closing the controller pipe still cancels the child.
- Clean workers preserve the two nonsecret Windows architecture fields needed
  by Python 3.11's platform metadata. Credentials remain excluded. Onboarding,
  workspace, export and cache inputs validate raw named homes before expansion.
- LF checkout rules preserve byte-pinned checkpoint metadata; cross-platform
  fixtures retain exact CSV/YAML bytes. Oversized malformed-input tests use short
  case names that fit Windows environment limits without reducing their payloads.
- ChatGPT-plan requests use the required Responses message-item array. The
  API-key route retains its existing input and output-token contract.
- The documented TOTO constraints pin SciPy 1.14.1, which passed import and real
  CPU checks on Python 3.12 and 3.13. The prior SciPy 1.15.3 wheel failed to load
  on the qualification host. The application already admitted 1.14.1.
- CI retains bounded failure diagnostics and installed-wheel reports. Curated
  documentation publication uses the selected GitHub Pages URL with private source.

## Evidence and remaining scope

All five new TTM profiles and TOTO passed bounded real CPU checks on macOS ARM64
against the frozen 0.3.0 wheel. Checks cover native/original-unit agreement,
causal contexts, future output, cache replay, cancellation/recovery and detached
exports. See the [model record](v0.3.0-model-qualification.md) and
[patch verification](v0.3.1-verification.md) for exact artifacts and executed
patch checks. These are runtime/contract checks, not forecast or trading claims.

Windows 11/Windows Terminal, interactive console handling, optional models on
Windows, all live assistant providers, native credential vaults and browser
sign-in **remain pending by owner decision**. Hosted Windows Server CI is a
separate automated check. Offline provider tests do not establish live acceptance.

The GitHub Release carries the source archive, wheel, `verification.json` and
`SHA256SUMS`. Its manifest binds the packages and checks to the exact merged
revision. The original [0.3.0 record](v0.3.0-verification.md) remains historical.

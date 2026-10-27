# Frozen schema-1 compatibility fixture

These are original small synthetic artifacts written by commit
`0ef0064ee80c83e90c6d9c927e0f5c5af4ede1a2`, the merged TTM/native-point
implementation before usage profiles and language assistance. Current writers
did not create or repair the retained run records or finance report.

The fixture contains eight invented hourly OHLC observations, two baseline YAML
configurations, six aligned forecasts per baseline, the old controlled-comparison
response, and a sealed costed finance report. `identity.json` records the exact
source revision, Git source-archive hash, generating environment, dataset and
report fingerprints, and SHA-256 hashes of all 17 original files. UUIDs, elapsed
times, creation timestamps and temporary source paths are retained as originally
written. This fixture proves compatibility for these contracts and does not
establish coverage of every possible historical record or forecasting quality.

To reproduce with a clone containing that commit and already installed core/CLI
dependencies, use a fresh output directory:

```sh
python scripts/generate_legacy_fixture.py /tmp/tsfm-legacy-reproduction
```

The generator extracts the historical package with `git archive`, runs that
package in an isolated Python process, and copies its unmodified artifacts out
before disposing of the temporary source and input location. It installs nothing
and makes no network requests. A new generation has new run identities/timing;
do not replace the frozen fixture merely to update those incidental fields.

The current-code check verifies config parsing and the exact historical dataset
fingerprint, then copies saved artifacts to a detached temporary workspace and
loads/exports forecasts, compares runs, and validates the sealed finance report.
The original input locations are absent. It checks original file hashes before
and after reading and never invokes inference or an assistant.

Run the source regression with `python -m pytest tests/test_release_compatibility.py`.
To check an ordinary installed wheel outside the checkout, use its interpreter:

```sh
/absolute/path/to/environment/bin/python -I /absolute/path/to/fixture/verify.py
```

The probe does not add a source checkout to the import path. The normal installed
distribution verifier calls the same probe against the archive's retained fixture.

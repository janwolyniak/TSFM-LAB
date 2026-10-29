# TTM S6 configuration fixtures

These small Apache-2.0 configuration files were retrieved from the official
`ibm-granite/granite-timeseries-ttm-r2` repository on 29 October 2026. No model
weights are included. Tests create inert local weight bytes and fake runtime
tensors; successful fixture tests are not real model execution evidence.

Exact revisions and configuration/file hashes are retained in the
[S6 verification record](../../../docs/s6-ttm-verification.md). Tests verify the
original configuration bytes against the recorded SHA-256 values before checking
their implemented profiles. Upstream license information is on the
[model card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2).

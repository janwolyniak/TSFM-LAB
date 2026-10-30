# S5 additional assistant connections

Implemented on 29 October 2026: DeepSeek, explicit compatible API roots, and
installed local Ollama. They share the existing selection, protected credentials,
preview approval, bounded transport, draft validation and grounded explanation
services. The four prior routes retain their protocols and state fields. Existing
schema-1 connection files remain readable; only new routes require an endpoint.

The conservative compatible protocol uses `/models` and nonstreaming
`/chat/completions`, system/user text and `max_tokens`. It does not infer support
for tools, streaming or vendor-specific structured-output fields. DeepSeek uses
its fixed official destination. Local Ollama uses `/api/tags`, `/api/show` and
`/api/chat`, explicit `:local` model source intent, completion-capability checking
and refusal of remote/cloud metadata. No server start, model pull or fallback
request exists. See [connection contracts and upstream references](connections.md).

Endpoint credentials are bound inside protected storage to the route and root.
Replacement clears discovery/model state and pending approvals. An edited public
endpoint cannot forward a saved key. All requests use disabled redirects,
disabled environment proxies, byte/time limits and cancellation. No credentials
are copied into numerical configurations or artifacts.

Verification uses fake credentials, HTTP responses, stores and assistant replies:

- Full suite before the final local-source refinement: **3,061 passed, 15 skipped**.
- Final focused connection/assistant regressions: **426 passed**, including **57**
  new cases covering all three protocols through actual draft/explanation
  services, malformed/truncated output, unsupported completion, error redaction,
  authentication/quota/timeout, decline/cancellation, endpoint replacement,
  credential binding, local restart, remote-alias refusal and embedding refusal.
- Repository Ruff and Black checks pass. A focused independent review prompted
  the explicit local-source and completion-capability checks before publication.

No real provider request, paid API, local Ollama installation/server/model,
native credential-vault access or external connection qualification was performed.
The owner requested offline implementation and deferred real inference. Generic
compatible endpoints are not universally certified by these protocol tests.

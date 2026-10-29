# Optional assistant connections

P6 adds connection setup and an explicit connection check. P7 adds optional
[natural-language setup and explanations](language-assistance.md). Forecasting,
saved records, comparisons, finance reports, and exports work without a connection.

| Connection | Credentials | Usage method |
| --- | --- | --- |
| ChatGPT plan | Official browser sign-in for an eligible account and grant | ChatGPT plan usage |
| OpenAI API key | OpenAI Platform API key | Separate OpenAI API billing |
| Anthropic API key | Claude Console API key; explicit workspace ID if required | Anthropic API billing |
| Google Gemini API key | Gemini Developer API key | Gemini API project quotas and billing |
| DeepSeek API key | DeepSeek API key | DeepSeek API billing |
| Local Ollama | No cloud key; an installed local server and model | Local compute |
| Compatible endpoint | A separate key for the explicit API root | Selected endpoint's usage and billing |

These are distinct routes. A limit, error, or unavailable model never changes the
selected account, provider, model, or billing route automatically. Consumer Claude
or Google subscription login is not included.

The three additional routes are development features for v0.3.0; they are absent
from the published v0.2.0 package. See [S5 verification](s5-connections-verification.md).

## Additional destinations

DeepSeek uses its fixed `https://api.deepseek.com` destination, `/models` discovery
and nonstreaming `/chat/completions`. The compatible route requires an explicit
API root, such as `https://host.example/v1`, supporting those same two paths and
the bounded Chat Completions subset: system/user text, `max_tokens` and
`stream: false`. Tools, reasoning controls, provider-specific structured-output
fields and streaming are not requested. A successful reply must contain exactly
one assistant message with `finish_reason: stop`; truncation or tool calls fail.
Compatibility is tested per protocol, not promised for every server claiming it.
Both drafting and explanations still pass their ordinary strict local validators.

API roots require HTTPS except for explicit HTTP loopback destinations. Credentials
in URLs, query strings, fragments, redirects and unsafe path segments are refused.
The destination appears in model discovery and every outgoing-context review.
Replacing an endpoint requires entering its own key and discovering/selecting its
model again; it invalidates pending approvals. Keys are bound to the saved
destination and route inside protected storage. Editing public endpoint metadata
cannot forward a previously stored key to another destination.

The Ollama route accepts only an HTTP loopback root (default
`http://127.0.0.1:11434`). It discovers installed models with `/api/tags`, checks the
selected model with `/api/show`, and uses nonstreaming `/api/chat` with a bounded
`num_predict` allowance. Cloud names and remote-model metadata are refused.
Requests use Ollama's explicit `:local` source suffix; a server must support that
contract. Unsupported servers fail rather than triggering a retry with an
unqualified name. Inspection must advertise `completion`; embedding-only models
cannot be used for assistance. Returned remote metadata is rejected as well.
Run a trusted local-only Ollama server with cloud access disabled, as described in
[Ollama's cloud controls](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features).
The application never pulls a model, starts a server or changes its configuration.
Disconnected/missing servers leave manual forecasting available. The nonsecret
local route can be reused after restart without a vault or cloud key; `replace`
changes its root. It never falls back to an external provider.

These protocols follow the official [DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)
and Ollama [local model discovery](https://docs.ollama.com/api/tags),
[model inspection](https://docs.ollama.com/api-reference/show-model-details), and
[chat API](https://docs.ollama.com/api/chat). Real-provider qualification remains
separate from the retained fake-provider evidence.

## Setup

When installing dependencies is authorized, add the optional runtime in your
chosen environment. From a source checkout:

```sh
python -m pip install -e '.[cli,connections]'
tsfm connections manage
```

In an existing `tsfm` session, use `/connections`. You can enter `/connections`
at any `/new` setup question, then return to that question without losing the
experiment draft. The menu offers:

1. `add`: choose a connection type, a nonsecret name, and credential storage.
   API keys use a separate hidden prompt without history, completion, or clipboard.
2. `select`: explicitly choose the active connection. Adding one does not select it.
3. `sign-in`: for a selected ChatGPT-plan connection, explicitly open official
   browser authorization. Use separate connections for separate accounts.
4. `models`: explicitly discover models for the active credentials, then choose
   an identifier. No model is selected automatically. Listing is not inference.
5. `test`: review the exact fixed connection-check context, provider, usage method,
   model, and local limits. Sending defaults to **no**. A confirmed check makes
   one inference request; the reply is displayed only when fully accepted.
6. `replace` or `disconnect`: replace an API key, renew plan sign-in, or disable a
   connection and remove its credentials. Replacement clears model selection.
   Sign-out attempts remote revocation once and reports whether it was confirmed;
   local disconnection does not depend on that confirmation.

Ctrl+C cancels a connection action and returns to the terminal. No automatic
inference retries, repair requests, or fallback providers are used. An interrupted
or failed request can still have consumed provider usage. Retry only by reviewing
and confirming a new check. A changed connection invalidates an outstanding preview.

Read nonsecret saved metadata without a prompt, vault access, or network request:

```sh
tsfm connections list --json
```

Both commands accept `--state-directory DIRECTORY`, separate from experiment
results. Default metadata lives at `~/.config/tsfm-lab/connections` on Unix-like
systems and `%LOCALAPPDATA%/TSFM-LAB/connections` on Windows. Metadata contains
labels, provider/model choices, account/registration identifiers, and random
credential references. It contains no API keys or OAuth tokens. Its saved status
is unverified; only explicit discovery or a check accesses credentials or the provider.

## Credential storage and isolation

`session` keeps credentials in memory until the connection-manager process exits.
Its nonsecret profile remains; reconnect or replace the key in a later session.
`vault` uses a directly selected native keyring backend: macOS Keychain, Windows
Credential Locker, or Linux Secret Service. It rejects automatic backend chains,
unknown backends, and plaintext alternatives. If unavailable or locked, storage
fails and the user must explicitly choose session storage; no credential file is
created. Persistent credentials rely on the OS vault and its access policy.

Connection metadata is bounded, strictly validated, locked for updates, and
atomically replaced. POSIX files/directories require private permissions;
symlinks and special files are refused. Pending vault deletions remain recorded
for retry; disconnected credentials cannot be used even if removal fails.

Secrets stay out of experiment configurations, results, exports, terminal command
arguments, and diagnostics. HTTP errors use canned messages; provider error
bodies are not displayed. Successful public metadata/text is checked for reflected
credentials. Credentials are excluded from text context; this is not a guarantee
against a malicious provider already receiving its own authentication material.

Managed numerical runs start a fresh isolated interpreter with an explicit clean
environment before imports. Assistant credentials and arbitrary parent environment
variables are absent. Maintained platform/cache/thread controls pass through;
explicit offline controls are preserved. Credential objects and connection state
are never included in numerical configurations or worker startup payloads.
Managed workers also disable implicit Hub token lookup. IPC and returned log tails
are bounded; native temporary model logs keep their existing append-until-close behavior.

## Request and sign-in contracts

The P6 check sends only its displayed constant text and instructions. It does not
read datasets, paths, YAML, observations, forecasts, environment variables, or saved
results for outgoing context. The reply is untrusted literal terminal text. P6
does not execute generated content or turn it into an experiment.

Provider HTTP uses TLS verification, fixed provider origins, no redirects, no
environment proxy/credential settings, and zero transport retries. Requests have
byte limits, timeouts, and cancellation. Cancellation cannot preempt every blocking
OS DNS operation instantly. Local limits are not guaranteed monetary caps.

ChatGPT-plan sign-in follows the official [registration and PKCE flow](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
and [profile/session contract](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions).
The loopback listener starts before the browser, and returned state, nonce,
signature, issuer, audience, timestamps, registration, account, and scopes are
validated. Refresh replacement is serialized. A rotated token set awaiting
identity validation is saved protected and remains unusable for inference; an
uncertain failed refresh cannot replay the old refresh token. Disconnection keeps
nonsecret registration identity for safe later sign-in.

Plan inference obeys the [preview restrictions](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations):
input is an array, `store:false`, `stream:true`, and unsupported output-token fields
are omitted. The whole stream must end with an accepted completed response;
partial text, truncation, tools, refusals, quota errors, and interrupted streams
are rejected. OpenAI API-key requests use their separate request contract.
Anthropic uses its [Messages API](https://platform.claude.com/docs/en/api/messages/create)
and [versioning contract](https://platform.claude.com/docs/en/api/versioning); Gemini
uses [generateContent](https://ai.google.dev/api/generate-content) and the
[API-key header](https://ai.google.dev/gemini-api/docs/api-key).

## Qualification boundary

Qualification uses fake credentials, signed synthetic identity tokens, simulated
provider transports, and injected credential vaults. It does not log in to an
account, make real provider requests, or exercise a real OS vault. Native vault
availability, real account eligibility/limits, and live provider behavior remain
unqualified until separately authorized checks. See the
[P6 verification record](p6-connections-verification.md).

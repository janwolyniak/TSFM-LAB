# P6 connections verification

Date: 26 October 2026. Scope: four explicit assistant connections, protected
credential boundaries, terminal setup, and the separately approved clean-environment
numerical worker change under the [release plan](next-release-plan.md).

The implementation provides distinct ChatGPT-plan, OpenAI API-key, Anthropic
API-key, and Gemini API-key routes. Selection, discovery, assistant model choice,
and the fixed connection check are explicit. A connection cannot change provider,
account, workspace, storage, model, or billing route after a preview is confirmed.
No fallback route, automatic inference retry, or model choice is introduced.
Natural-language setup and explanations remain P7.

Credentials use a hidden, history-free prompt with a disabled clipboard and cleared
input buffer. Session credentials remain in memory; persistent credentials use an
explicit native vault backend. Public metadata uses separate random references,
strict schemas, private POSIX permissions, bounded reads, locks, and atomic writes.
Rollback and deferred deletion keep disconnected credentials unusable when a vault
operation fails. No credential plaintext file is created.

Synthetic sign-in tests exercise PKCE, state, nonce, issued registration retention,
single-use attempts, exact loopback redirect/host matching, signed RSA identities,
issuer/audience/subject/time validation, scopes, rotation, and sign-out. A protected
write-ahead marker prevents replay after an interrupted refresh exchange. Pending
identity recovery retains rotated credentials through temporary failure without
spending another refresh token. Invalid identities stay blocked. The listener binds
before the browser opens, handles slow callback headers within its deadline, and
closes on cancellation or failure.

Fake provider tests cover exact URLs, headers, encoded request bounds, pagination,
credential reflection, malformed/truncated/tool/refusal output, missing permissions,
expired credentials, authentication/quota/network errors, slow headers and streams,
deadlines, and cancellation. Plan output is accepted only after a complete valid
SSE response, with `store:false`, `stream:true`, and supported request fields.
HTTPX also runs against its in-memory mock transport; no provider endpoint is used.

Real local subprocess checks verify credentials are absent before project imports,
static isolated startup arguments, unchanged parent environment during simultaneous
launches, retained cache/offline controls, progress/cancellation, zero exit status,
truthful record settlement, bounded IPC/backpressure, partial pipe writes, and
cleanup after constructor or thread-start interruption. Startup, events, queues,
and returned log tails are bounded. Native stdout/stderr temporary log files retain
their prior append-until-close behavior and do not have a new hard on-disk size cap.
Implicit Hub token lookup is disabled in managed workers.

Terminal tests cover hidden input, route display, explicit model selection, exact
preview consent, stale/replayed previews, disconnect/replacement, literal output,
and returning from connections to the current `/new` question. Saved-record and
CSV/JSON export checks demonstrate that configured fake credentials remain absent.

The configured Linux/macOS CI matrix installs the optional connection runtime to
exercise synthetic signatures and in-memory HTTPX. Its separate clean installed
wheel gate blocks provider/model dependencies, retaining proof of offline operation.
Remote P6 CI has not run at this local verification stage.

The full offline source suite passed **2,300 tests**, with five opt-in real-model
checks skipped, in 75.48 seconds. Formatting checked 153 Python files; Ruff and
whitespace checks passed. After adding the installed-worker/path-resolution gate,
all 18 distribution-verifier/build tests passed. Documentation checks validated
39 root/docs Markdown files, 273 relative links, 55 shell blocks, and the CI YAML.
Independent provider, authentication, and worker reviews produced regression
fixes for route tampering, slow network/callback input, interrupted refresh, pending
identity recovery, sign-in cancellation, partial pipe writes, and startup cleanup.
No actionable review finding remains.

The preparation source archive contained 250 files and 88 required paths; its wheel
contained 95 entries and all three existing license files. Only NumPy and Pandas
remain mandatory core dependencies. All **13 installed workflow stages** passed
in 24.504 seconds in an isolated Python 3.13.6 environment without any provider,
model, kernel, or Textual runtime. These include all four simulated connections,
one confirmed inference per route, nonsecret files, zero implicit optional imports,
and an actual clean worker completing with zero exit status. Dependency consistency
passed. Only the project wheel was replaced in the disposable verifier environment.

Final committed source/archive/wheel hashes and exact installed-byte identities
are recorded beside the final artifacts, separately from preparation provenance.
Local evidence is kept under
`/private/tmp/tsfm-next-models-01a174c4/p6-verification/`; it is an audit trail,
not a dependency of saved experiments.

No real keys, provider calls, account login, browser launch, OS-vault access,
dependency installation/upgrade, paid compute, model download, package release,
or P6 remote publication is part of this qualification. Native-vault behavior,
real account eligibility/limits, and live provider compatibility remain unqualified
pending separate authorization. Blocking OS DNS resolution may outlive cancellation;
synchronous transport calls inside an already running event loop fail safely.

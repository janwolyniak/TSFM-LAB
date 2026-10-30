# Natural-language setup and explanations

Language assistance is optional. Guided setup, saved YAML runs, results, finance
reports, comparisons, and exports continue to work offline without a connection.
The forecasting model and the assistant model are separate choices.

In the v0.3.0 S2 checkout, proposals share the
[model-specific settings contract](model-setup.md) with manual setup, YAML, and
Python configuration. Known variant limits are checked locally; unsupported or
conflicting suggestions remain visible for correction in the ordinary guide.

## Describe an experiment

Start `tsfm`, enter `/new`, and choose `describe` rather than the default `guided`.
You can also enter `/new describe` directly. `/new guided` and `/new ADAPTER_ID`
open the ordinary guide. `/demo` retains its existing confirmation.

1. Choose a local CSV or Parquet dataset and explicitly select historical
   evaluation or future forecasting. The path stays on this machine.
2. Describe the selected task in plain words, for example:
   “Evaluate the historical temperature series in Celsius with context 128,
   horizon 24 and stride 24. Use Chronos-2 Small.”
3. Review the assistant connection, account/workspace, model, usage method,
   exact outgoing instructions/message, and local limits. Sending defaults to **no**.
4. Inspect the proposed settings, locally detected missing fields/errors, and
   any assumptions. Choose `review` to visit the ordinary questions, `clarify`
   to submit another explicitly confirmed message, or `guided` to use manual setup.
5. Review every resolved setting and confirm the ordinary `Run experiment?`
   question, which also defaults to **no**. Sending a message never starts a run.

Select and discover an assistant model through `/connections` first. That menu
is also reachable during ordinary setup. See [connection setup](connections.md).
Credentials belong in its hidden prompt, never in an assistant message.

The S4 checkout supports [future forecasting](future-forecasting.md) from the
last observed timestamp using separately validated schema-2 drafts. Human mode
selection governs every preview and resolution; a provider cannot switch modes.
Training, panels, covariates, generated code and new trading rules remain outside
the execution contract. Historical evaluation is never silently substituted for
a future request. Changing mode requires a fresh preview and confirmation.

Clarification retains bounded original user intent and earlier choices. It can
fill missing fields, but a response that changes a prior nonblank choice is
rejected. Use guided review to edit an existing choice. Omitted/blank replacements
cannot erase earlier settings. Guided fallback retains the locally selected file.
After six retained messages or the local text budget, use guided review rather
than silently dropping earlier constraints.

## Explain checked facts

In the active terminal:

```text
/explain model chronos_2
/explain model chronos_2 timesfm_2p5
/explain run latest
/explain finance saved-report.json
```

Choose up to three model identifiers. Model explanations use the maintained
catalog, including actual output semantics and the difference between representative
adapter checks and individually tested checkpoints. They cannot establish a model
ranking or predict performance on your data.

Experiment explanations require a completed record that passes full saved-artifact
validation. Finance explanations require a saved report that passes its fingerprint
and accounting checks. They need no original data or inference runtime. Unverified
pending/failed records and corrupt reports are refused.

Future explanations use only validated metadata: origin and context/target
bounds, N/L/H, units/cadence, model/settings and actual point label. Accuracy is
explicitly unavailable. Observation values, prediction values, row tables, value
summaries and paths remain excluded from outgoing explanation evidence.

Checked local facts appear separately from optional assistant wording. Exact
numbers, units, undefined-value reasons, evaluation periods, assumptions, and
verification labels remain authoritative. Explanation replies must cite supplied
facts and pass structural, numeric, and privacy checks. These checks cannot prove
the semantic truth of arbitrary prose: the displayed wording is an **untrusted
interpretation**, not an independent fact check or trading recommendation.
Explanations never alter scores, costs, forecasts, reports, or saved files.

## Information and request limits

Outgoing context is constructed explicitly. Setup sends the bounded goal and
retained user clarifications, an opaque dataset reference, supported catalog choices,
the draft contract, and previous decoded semantic choices. It does not send the
local dataset path, file contents, observations, or terminal history. Explanations
send only the displayed projection of validated/catalog facts. Names, free profile
descriptions, machine paths, raw forecasts, equity curves, YAML, and environment
metadata are excluded. Machine paths and obvious credential formats are rejected
even when placed inside an otherwise allowed text field.

Every message has its own preview and single-use confirmation, tied to the selected
connection/account/model/workspace and current state. Cancellation, edited previews,
changed connections, expired approvals, incomplete replies, malformed JSON, unknown
fields/references, and unsupported actions produce no accepted draft/explanation.
There are no automatic inference retries, repair messages, or provider/billing
fallbacks. Another request always needs a new preview and confirmation. Failed or
cancelled requests can still consume provider usage; local limits are not monetary caps.

Drafts remain in memory and cannot supply paths, revisions, runtime settings,
commands, or output destinations. Local code resolves the selected dataset and
curated checkpoint identity. A draft's `ready` claim is advisory; local status means
its candidate fields were checked, and complete normal preparation and human review
are still required. Missing units, dates, import mappings, or model choices must be
resolved rather than inferred from an assistant's status label.

Provider-specific structured-output features are not assumed across the four routes.
All responses use the existing completed-text boundary and strict local decoding.
The [official OpenAI guidance](https://developers.openai.com/api/docs/guides/structured-outputs)
also distinguishes schema adherence from correctness of generated content.

P7 qualification uses fake credentials, simulated HTTP, local disposable fixtures,
and no paid requests. Live provider compatibility and native-vault availability
remain separately unqualified. See the [P7 verification record](p7-language-verification.md).

# S7 public documentation preparation evidence

Status on 29 October 2026: repository-local static artifact implemented and
checked. **No host registration, deployment, repository visibility change or
public reachability/indexability check was performed.** Candidate URLs remain
subject to host/URL confirmation before publication. The repository stays private.

## Scope and retained boundary

The build reads only curated `docs/public/guide.md`, `docs/public/style.css`,
package metadata and citation fields, plus a public catalog projection. Existing
private documentation, datasets, configurations containing user paths, model
weights, secrets and experiment outputs are not recursively copied. The output
manifest contains relative input names and hashes, not machine/checkout paths.

The guide covers setup/preferences and the offline demonstration, reviewed
model-specific setup, six pretrained adapters/two baselines and every curated
variant, original-unit outputs, all five category baseline examples, historical
versus future semantics, unavailable future accuracy, finance eligibility and
limitations, citation information and S1–S7 unreleased notes. It distinguishes
source package 0.2.0 from unreleased 0.3.0 additions, reference inference from
configuration compatibility, TOTO's Python restriction and pending qualification.

The static artifact includes accessible text/navigation, responsive local CSS,
canonical/social metadata, JSON-LD matching visible package metadata, robots,
sitemap, optional navigation-only `llms.txt` and a completion manifest. Raw
Markdown HTML is disabled. Metadata is escaped, heading IDs are deterministic,
section links resolve, and builds refuse unsafe URLs and existing destinations.

## Executed checks

The focused suite exercises output reproducibility/hashes, local anchors/assets,
crawl/canonical/structured metadata, exact catalog coverage, omission of private
canaries and runtime-machine provenance, HTML escaping, duplicate heading IDs,
unsafe input/URL refusal, no overwrites and a subprocess with model/network
imports blocked. It executes Generic, Finance, Energy, Temperature and Weather
CLI baseline examples plus the future baseline with unique disposable outputs.

Executed with the existing Python 3.13.6 environment:

```sh
python -m pytest -q tests/test_public_docs.py
python -m black --check scripts/build_public_docs.py tests/test_public_docs.py
python -m ruff check scripts/build_public_docs.py tests/test_public_docs.py
```

Result: **32 passed**; Black/Ruff and `git diff --check` pass. The six example
cases ran through actual CLI dispatch with disposable relocated datasets/output
settings. No pretrained runtime or weights were used. Changed Markdown local
links passed a filesystem check. External reachability was not inferred from
those checks; private source links retain their access requirement.

The generator built six static files for candidate
`https://janwolyniak.github.io/TSFM-LAB/` without hosting/network operations.
The generated manifest retains exact source/public-catalog/output hashes and
explicit candidate/private/not-deployed status. It covers all 21 baseline and
curated checkpoint rows. A separate local browser visual review follows these
generator checks; it does not establish public reachability or platform support.

The primary review served only the generated directory on loopback and opened
the guide in the in-app browser. The default-width page and model table were
visually inspected; section navigation reached the model matrix. A 390-pixel
responsive check reported a 390-pixel document, a single-column layout and
16-pixel body text, with the wide model table contained in its keyboard-focusable
horizontal scroller. A scroll hint was added after review. These are local
browser checks, not native Windows or public-host checks.

## Remaining gates

The chosen candidate URL is not a confirmed host. Deployment and subsequent
public link/reachability/crawl checks remain pending. Search indexing and AI
citations are not guaranteed outcomes, following
[Google's guidance](https://developers.google.com/search/docs/appearance/ai-features).
New real-model/native-provider qualification, native Windows checks and exact
release-candidate packaging/CI are separately open under S3/S6/S8/S9. This
documentation artifact does not turn those limits into accepted execution claims.

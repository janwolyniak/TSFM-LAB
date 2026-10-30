# Curated static documentation

S7 prepares an indexable static documentation artifact for the 0.3.0
release. This is **local preparation only**. The repository remains
private; no host is registered, no visibility setting changes, and no deployment
or public reachability/indexability claim is made.

## Build and inspect

Use the existing development/CLI environment. The generator uses the already
available `markdown-it-py` parser and PyYAML; it introduces no frontend framework,
JavaScript runtime or dependency installation step.

From the checkout root, choose an explicit candidate HTTPS directory URL:

```sh
python scripts/build_public_docs.py \
  --base-url https://janwolyniak.github.io/TSFM-LAB/ \
  --out-dir build/public-docs
```

That URL is an illustrative candidate, not a selected/verified host. Confirm the
host and public URL before deployment. The base URL must have no credentials,
port, query, fragment, private/loopback host or unsafe path. A missing base URL is
an error. The output defaults to `build/public-docs` and must not already exist,
even as an empty directory or symlink. Choose a new output directory for another
build; the generator never overwrites a previous artifact.

The directory contains only:

- `index.html`: semantic text documentation, stable section navigation, generated
  model/profile table, canonical/social metadata and `SoftwareApplication` JSON-LD.
- `style.css`: responsive local styles, system fonts, keyboard focus/skip link,
  reduced-motion and print behavior; no external fonts or executable JavaScript.
- `robots.txt` and `sitemap.xml`: crawl settings for the supplied candidate URL.
- `llms.txt`: optional navigation, with no special indexing claim.
- `build-manifest.json`: safe input/output hashes and explicit
  `built_not_deployed`, candidate-URL and private-repository status.

The manifest is written last as the completion marker. Builds use no timestamps,
random identities, local absolute paths or runtime-machine provenance; identical
curated inputs/catalog/base URL produce identical output bytes. A partially
failed build does not have a completed manifest.

Preview only that generated directory with a local static server. Do not serve
the repository root. Visual inspection and public deployment checks are separate
from the generator's offline checks; see [S7 evidence](s7-public-docs-verification.md).

## Content and privacy boundary

Only four explicit source inputs are read: `docs/public/guide.md`,
`docs/public/style.css`, `pyproject.toml` and `CITATION.cff`. The maintained
catalog supplies a whitelisted public projection of model/variant bounds,
cadence, point output and evidence status. Raw representative-check provenance,
machine details, credentials, private paths and experiment records are omitted.
The generator does not recursively publish existing `docs/`, copy datasets or
exports, or fetch remote material. It refuses common private-path/key-assignment
patterns in curated text; this is an additional guard, not a complete secret
scanner or permission to add private material to public inputs.

Markdown raw HTML is disabled. Generated text/metadata is escaped; heading IDs
are unique and deterministic, section links are checked, and image dependencies
are refused. Catalog compatibility and selected-reference inference are shown
separately. Source package/version/citation metadata identifies 0.3.0; release publication
is distinct from documentation deployment and qualification limits remain visible. No PyPI, model accuracy/ranking or native-platform qualification is invented.

## Publication remains separate

Before deployment, confirm the selected host/public URL and public audience,
review the generated directory and manifest, rebuild for that URL, then use an
authorized hosting workflow. Do not make the source repository public merely to
host documentation. Verify reachability, canonical URLs, links, crawl/index rules
and any infrastructure access controls after deployment; local checks alone
cannot establish them. No such deployment is authorized by this local build.

[Google's AI search guidance](https://developers.google.com/search/docs/appearance/ai-features)
emphasizes ordinary search eligibility, useful text, internal links, crawl access
and structured data matching visible content. It requires no special AI file or
schema. Crawl/index inclusion and AI citations are not guaranteed; `llms.txt` is
optional navigation only.

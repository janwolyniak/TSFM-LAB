# Curated static documentation

S7 prepares a curated static documentation artifact. On 2 November 2026 the
owner approved GitHub Pages at `https://janwolyniak.github.io/TSFM-LAB/` while
keeping the source repository private. Pages is configured for workflow-based
publication. A successful build alone does not establish deployment or public
reachability; the [patch verification record](v0.3.1-verification.md) records those
checks separately.

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

Without `--publication-approved` the generator retains preview/candidate status.
The approved deployment build adds that flag and uses the selected URL above.
The base URL must have no credentials,
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
- `build-manifest.json`: safe input/output hashes and private-repository status.
  Preview builds say `built_not_deployed`; approved builds say
  `built_for_deployment` and `owner_confirmed_for_publication`. Neither status
  claims that a deployment or public HTTP check has succeeded.

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
separately. Source package/version/citation metadata identifies 0.3.1; release publication
is distinct from documentation deployment and qualification limits remain visible. No PyPI, model accuracy/ranking or native-platform qualification is invented.

## Approved publication workflow

The [public documentation workflow](https://github.com/janwolyniak/TSFM-LAB/blob/main/.github/workflows/public-docs.yml) builds
only the six generated files, retains read-only repository access during build,
and grants Pages/OIDC deployment permissions only to its deployment job on
`main`. It checks the configured Pages base URL against the owner-approved URL.
Pull requests build the artifact without deploying it. Source visibility stays
private; public documentation exposes the deliberately curated content.

After deployment, anonymously request all six files, verify their manifest hashes,
canonical/social/structured URLs, internal anchors, sitemap and crawl rules.
Check that repository visibility is still private. Retain the deployment revision,
workflow URL and actual HTTP outcomes. These checks do not establish indexing.

[Google's AI search guidance](https://developers.google.com/search/docs/appearance/ai-features)
emphasizes ordinary search eligibility, useful text, internal links, crawl access
and structured data matching visible content. It requires no special AI file or
schema. Crawl/index inclusion and AI citations are not guaranteed; `llms.txt` is
optional navigation only.

"""Build only curated, public-safe documentation; never register or deploy a host."""

import argparse
import hashlib
import html
import ipaddress
import json
import re
import sys
import tomllib
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit
from xml.sax.saxutils import escape as xml_escape

import yaml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsfm_lab.models.catalog import catalog_entries  # noqa: E402

PUBLIC_INPUTS = (
    "docs/public/guide.md",
    "docs/public/style.css",
    "pyproject.toml",
    "CITATION.cff",
)
REPOSITORY = "https://github.com/janwolyniak/TSFM-LAB"
PLANNED_RELEASE = "0.3.0"
_PRIVATE = re.compile(
    r"/Users/|/home/|[A-Za-z]:[\\/]Users[\\/]|\bsk-[A-Za-z0-9_-]{16,}|(?i:api[_-]?key|access[_-]?token|password)\s*[:=]\s*\S+"
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _text(value):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 16384
        or _PRIVATE.search(value)
    ):
        raise ValueError(
            "Curated documentation contains invalid text or a private path/credential pattern"
        )
    return value


def _reference(value):
    value = _text(value)
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("Public references require a safe HTTPS URL")
    return value


def validate_base_url(value):
    """Normalize an explicit candidate directory URL without network access."""
    if (
        not isinstance(value, str)
        or value != value.strip()
        or any(ord(char) < 33 for char in value)
    ):
        raise ValueError("Supply an explicit safe HTTPS documentation base URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or "?" in value
        or "#" in value
        or "\\" in value
        or parsed.port is not None
        or re.fullmatch(r"[A-Za-z0-9.-]+", parsed.hostname) is None
    ):
        raise ValueError(
            "Documentation base URL must use HTTPS without credentials, port, query or fragment"
        )
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith((".localhost", ".local")) or ".." in host:
        raise ValueError("Choose the intended public documentation hostname")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("Choose the intended public documentation hostname")
    path = unquote(unquote(parsed.path))
    if (
        "\\" in path
        or "//" in path
        or any(part in {".", ".."} for part in path.split("/"))
        or any(ord(char) < 33 for char in path)
        or any(char in path for char in "?#<>\"'")
    ):
        raise ValueError("Documentation base URL contains an unsafe path")
    return urlunsplit(("https", host, parsed.path.rstrip("/") + "/", "", ""))


def _read_inputs(root):
    result = {}
    for name in PUBLIC_INPUTS:
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError("Curated documentation input escapes the source directory")
        raw = path.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("Curated documentation input is unexpectedly large")
        result[name] = raw
    return result


def _metadata(inputs):
    package = tomllib.loads(inputs["pyproject.toml"].decode())["project"]
    citation = yaml.safe_load(inputs["CITATION.cff"])
    version = _text(package["version"])
    if (
        re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)", version) is None
        or citation["version"] != version
    ):
        raise ValueError("Package and citation versions must agree")

    def authors(people):
        return [
            " ".join((_text(person["given-names"]), _text(person["family-names"])))
            for person in people
        ]

    preferred = citation["preferred-citation"]
    return {
        "version": version,
        "description": _text(package["description"]),
        "authors": authors(citation["authors"]),
        "citation_title": _text(preferred["title"]),
        "citation_authors": authors(preferred["authors"]),
        "citation_url": _reference(preferred["url"]),
    }


def public_catalog():
    """Project only public fields; omit runtime-machine and experiment provenance."""
    rows = []
    cadence = {
        "any_fixed": "Declared fixed UTC interval",
        "minute_hour_fixed_v1": "1 minute–1 hour",
        "daily_fixed_v1": "Exactly 1D",
        "weekly_fixed_v1": "Exactly 7D",
    }
    for entry in catalog_entries():
        for variant in entry["variants"] or [None]:
            caps = entry["capabilities"]
            low = variant["min_context"] if variant else caps["min_context"]
            high = variant["max_context"] if variant else caps["max_context"]
            horizon = variant["max_horizon"] if variant else caps["max_horizon"]
            context = (
                f"Exactly {low:,}"
                if low == high
                else f"{low:,}–{high:,}" if high else f"At least {low}"
            )
            if entry["adapter_id"] == "toto_2":
                context += "; multiples of 32 (project bounds)"
            policy = (
                variant.get("frequency_policy", entry["frequency_policy"])
                if variant
                else entry["frequency_policy"]
            )
            rows.append(
                {
                    "adapter_id": entry["adapter_id"],
                    "model": _text(entry["display_name"]),
                    "variant": (
                        _text(variant["display_name"]) if variant else "Built-in"
                    ),
                    "repository": variant["repository"] if variant else None,
                    "revision": variant["reference_revision"] if variant else None,
                    "context": context,
                    "horizon": str(horizon) if horizon else "Dataset/task limits",
                    "cadence": cadence[policy],
                    "point": entry["point_statistic"],
                    "compatibility": (
                        "Curated configuration/profile"
                        if variant
                        else "Built-in deterministic"
                    ),
                    "inference": (
                        (
                            "Reference check recorded"
                            if variant["project_inference_status"] == "recorded"
                            else "Not recorded"
                        )
                        if variant
                        else "Offline workflow checked"
                    ),
                }
            )
    return rows


def _matrix(rows):
    escape = html.escape
    result = [
        '<div class="catalog-scroll" tabindex="0" role="region" aria-label="Model and checkpoint capability matrix"><table><thead><tr>'
    ]
    result.extend(
        f'<th scope="col">{name}</th>'
        for name in (
            "Model / profile",
            "Context / horizon",
            "Cadence",
            "Point output",
            "Profile compatibility",
            "Reference inference",
        )
    )
    result.append("</tr></thead><tbody>")
    for row in rows:
        identity = f'<strong>{escape(row["model"])}</strong><small>{escape(row["variant"])}</small>'
        if row["repository"]:
            identity += f'<small><code>{escape(row["repository"])}</code></small><details><summary>Pinned reference</summary><code>{escape(row["revision"])}</code></details>'
        flag = "recorded" if row["inference"] != "Not recorded" else "pending"
        result.append(
            f'<tr data-adapter="{escape(row["adapter_id"])}"><td>{identity}</td><td>{escape(row["context"])} / {escape(row["horizon"])}</td><td>{escape(row["cadence"])}</td><td><code>{escape(row["point"])}</code></td><td>{escape(row["compatibility"])}</td><td class="evidence {flag}">{escape(row["inference"])}</td></tr>'
        )
    return "".join(result) + "</tbody></table></div>"


def _citation(meta):
    escape = html.escape
    return f'<div class="citation"><p><strong>TSFM Lab.</strong> {escape(", ".join(meta["authors"]))}. Source package {escape(meta["version"])}. Apache-2.0.</p><p>Following CITATION.cff, cite the working paper and identify the TSFM Lab version or commit used:</p><p>{escape(", ".join(meta["citation_authors"]))}. <a href="{escape(meta["citation_url"], quote=True)}">{escape(meta["citation_title"])}</a>. Unpublished working paper.</p></div>'


def render_guide(markdown, rows, meta):
    _text(markdown)
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    markdown = markdown.replace("{{SOURCE_VERSION}}", meta["version"])
    tokens, headings, used, references = (
        parser.parse(markdown),
        [],
        {"main", "title"},
        [],
    )
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            title = tokens[index + 1].content
            slug = (
                re.sub(
                    r"[^a-z0-9]+",
                    "-",
                    unicodedata.normalize("NFKD", title)
                    .encode("ascii", "ignore")
                    .decode()
                    .lower(),
                ).strip("-")
                or "section"
            )
            identity, suffix = slug, 2
            while identity in used:
                identity = f"{slug}-{suffix}"
                suffix += 1
            used.add(identity)
            token.attrSet("id", identity)
            if token.tag == "h2":
                headings.append((identity, title))
        for child in token.children or []:
            if child.type == "link_open":
                target = child.attrGet("href")
                if target.startswith("#"):
                    references.append(unquote(target[1:]))
                else:
                    _reference(target)
            if child.type == "image":
                raise ValueError(
                    "Public guide must not load external or private image assets"
                )
    if any(target not in used for target in references):
        raise ValueError("Public guide contains an unresolved section link")
    rendered = parser.renderer.render(tokens, parser.options, {})
    for marker, value in (
        ("MODEL_MATRIX", _matrix(rows)),
        ("CITATION", _citation(meta)),
    ):
        placeholder = "<p>{{" + marker + "}}</p>"
        if rendered.count(placeholder) != 1:
            raise ValueError(f"Public guide requires exactly one {marker} placeholder")
        rendered = rendered.replace(placeholder, value)
    return rendered, headings


def _page(content, headings, meta, base):
    escape = html.escape
    title = "TSFM Lab — local forecasting guide"
    description = escape(meta["description"], quote=True)
    structured = {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "TSFM Lab",
        "description": meta["description"],
        "softwareVersion": meta["version"],
        "url": base,
        "applicationCategory": "DeveloperApplication",
        "license": "https://www.apache.org/licenses/LICENSE-2.0",
        "softwareRequirements": "Python 3.11+ core; TOTO 2 requires Python 3.12 or 3.13",
        "author": [{"@type": "Person", "name": name} for name in meta["authors"]],
    }
    encoded = (
        _json(structured)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    nav = "".join(
        f'<li><a href="#{identity}">{escape(name)}</a></li>'
        for identity, name in headings
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><meta name="description" content="{description}"><meta name="robots" content="index,follow"><meta name="referrer" content="no-referrer">
<link rel="canonical" href="{escape(base, quote=True)}"><link rel="stylesheet" href="style.css">
<meta property="og:title" content="{title}"><meta property="og:description" content="{description}"><meta property="og:url" content="{escape(base, quote=True)}"><meta property="og:type" content="website">
<script type="application/ld+json">{encoded}</script></head><body>
<a class="skip" href="#main">Skip to documentation</a>
<header class="masthead"><a class="wordmark" href="#main" aria-label="TSFM Lab documentation">TSFM / LAB</a><div class="status"><span>Source package {escape(meta["version"])}</span><span class="development">{PLANNED_RELEASE} unreleased</span><span>Candidate docs · not deployed</span></div></header>
<div class="layout"><aside class="sidebar"><nav aria-label="Documentation sections"><h2>In this guide</h2><ul>{nav}</ul></nav><p class="note">Local forecasting.<br>Explicit assumptions.<br>Inspectable outputs.<br><br>Source access remains private.</p></aside>
<main class="content" id="main"><section class="hero" aria-labelledby="title"><p class="eyebrow">Local time-series forecasting</p><h1 id="title">TSFM Lab</h1><p class="lead">{description}</p><div class="actions"><a class="button primary" href="#quickstart">Start with an offline baseline</a><a class="button" href="#model-matrix">Inspect supported models</a></div><p class="publication">Development documentation, prepared locally. New behavior is unreleased; qualification limits are visible throughout this guide.</p></section>{content}</main></div>
<footer class="footer"><p>Candidate documentation URL: <code>{escape(base)}</code>. Host/URL confirmation, deployment and public reachability checks remain separate.</p><p>Source repository access is private: <a href="{REPOSITORY}">authorized checkout</a>. TSFM Lab code is Apache-2.0; optional runtime and weight licenses remain separate.</p></footer></body></html>"""


def build_docs(out_dir, base_url, *, root=ROOT):
    """Build a new directory from curated inputs; no existing directory is reused."""
    base = validate_base_url(base_url)
    output, root = Path(out_dir).absolute(), Path(root).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "Documentation output already exists; choose a new directory"
        )
    inputs = _read_inputs(root)
    style = _text(inputs["docs/public/style.css"].decode())
    if re.search(r"@import\b|url\s*\(", style, flags=re.I):
        raise ValueError(
            "Public styles must use local system fonts and no remote assets"
        )
    meta, rows = _metadata(inputs), public_catalog()
    content, headings = render_guide(
        inputs["docs/public/guide.md"].decode(), rows, meta
    )
    if not {"quickstart", "model-matrix"} <= {identity for identity, _ in headings}:
        raise ValueError(
            "Public guide must retain the quickstart and model-matrix sections"
        )
    payloads = {
        "index.html": _page(content, headings, meta, base).encode(),
        "style.css": inputs["docs/public/style.css"],
        "robots.txt": f"User-agent: *\nAllow: /\nSitemap: {base}sitemap.xml\n".encode(),
        "sitemap.xml": f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>{xml_escape(base)}</loc></url></urlset>\n'.encode(),
        "llms.txt": (
            "# TSFM Lab\n\nLocal forecasting and reproducible evaluation.\nSource package "
            + meta["version"]
            + "; 0.3.0 unreleased. Candidate documentation; not deployed.\n\n"
            + "\n".join(f"- [{name}]({base}#{identity})" for identity, name in headings)
            + "\n"
        ).encode(),
    }
    manifest = {
        "schema_version": 1,
        "status": "built_not_deployed",
        "base_url": base,
        "url_status": "candidate_requires_host_confirmation",
        "repository_access": "private",
        "source_package_version": meta["version"],
        "planned_release": PLANNED_RELEASE,
        "release_status": "unreleased",
        "privacy_policy": "curated_inputs_only_v1",
        "inputs": {name: _hash(raw) for name, raw in inputs.items()},
        "catalog_sha256": _hash(_json(rows).encode()),
        "outputs": {
            name: {"bytes": len(raw), "sha256": _hash(raw)}
            for name, raw in payloads.items()
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        with (output / name).open("xb") as target:
            target.write(raw)
    # The manifest is the final completion marker; partial writes never claim success.
    with (output / "build-manifest.json").open("x", encoding="utf-8") as target:
        target.write(_json(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        required=True,
        help="Explicit candidate HTTPS documentation directory URL",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "build" / "public-docs")
    args = parser.parse_args()
    try:
        manifest = build_docs(args.out_dir, args.base_url)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(
        _json(
            {
                "status": manifest["status"],
                "output_directory": str(args.out_dir.absolute()),
                "base_url": manifest["base_url"],
            }
        ),
        end="",
    )


if __name__ == "__main__":
    main()

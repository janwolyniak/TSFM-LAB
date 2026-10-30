"""Curated static documentation is reproducible, crawlable and free of private inputs."""

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tsfm_lab.cli import app
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.models.catalog import catalog_entries

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_public_docs", ROOT / "scripts/build_public_docs.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
BASE = "https://example.invalid/tsfm-lab/"


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids, self.links, self.tags, self.scripts, self.adapters = (
            [],
            [],
            [],
            [],
            [],
        )
        self.meta, self.text, self.canonical = {}, [], None
        self.script = None
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if "href" in attrs:
            self.links.append(attrs["href"])
        if attrs.get("rel") == "canonical":
            self.canonical = attrs["href"]
        if tag == "meta":
            self.meta[attrs.get("name", attrs.get("property"))] = attrs.get("content")
        if tag == "script":
            self.script = {"attrs": attrs, "text": ""}
            self.scripts.append(self.script)
        if "data-adapter" in attrs:
            self.adapters.append(attrs["data-adapter"])

    def handle_endtag(self, tag):
        if tag == "script":
            self.script = None

    def handle_data(self, text):
        self.text.append(text)
        if self.script is not None:
            self.script["text"] += text


@pytest.fixture
def curated_root(tmp_path):
    root = tmp_path / "source"
    for name in builder.PUBLIC_INPUTS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    (root / "docs/private.md").write_text("/Users/private/PUBLIC_DOCS_SECRET_CANARY")
    (root / "results").mkdir()
    (root / "results/private.csv").write_text("PRIVATE_DATASET_CANARY")
    return root


def test_reproducible_complete_artifact_links_crawl_and_public_manifest(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    first = builder.build_docs(one, BASE)
    assert builder.build_docs(two, BASE) == first
    assert {p.name for p in one.iterdir()} == {
        "index.html",
        "style.css",
        "robots.txt",
        "sitemap.xml",
        "llms.txt",
        "build-manifest.json",
    }
    assert all(p.read_bytes() == (two / p.name).read_bytes() for p in one.iterdir())
    page = Page((one / "index.html").read_text())
    assert len(page.ids) == len(set(page.ids))
    for link in page.links:
        if link.startswith("#"):
            assert link[1:] in page.ids
        elif not link.startswith("https://"):
            assert (one / link).is_file()
    assert page.canonical == page.meta["og:url"] == BASE
    assert page.meta["robots"] == "index,follow"
    structured = json.loads(page.scripts[0]["text"])
    assert structured["@type"] == "SoftwareApplication"
    assert structured["softwareVersion"] == first["source_package_version"] == "0.3.0"
    assert structured["description"] == page.meta["description"]
    assert structured["description"] in "".join(page.text)
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert citation["preferred-citation"]["title"] in "".join(page.text)
    assert len(page.scripts) == 1 and page.scripts[0]["attrs"] == {
        "type": "application/ld+json"
    }
    assert not {"iframe", "img", "form"}.intersection(page.tags)
    assert "Sitemap: " + BASE + "sitemap.xml" in (one / "robots.txt").read_text()
    sitemap = ET.parse(one / "sitemap.xml")
    assert sitemap.find(".//{*}loc").text == BASE
    assert (
        first["status"] == "built_not_deployed"
        and first["repository_access"] == "private"
    )
    assert first["url_status"] == "candidate_requires_host_confirmation"
    assert first["release_status"] == "released"
    assert set(first["inputs"]) == set(builder.PUBLIC_INPUTS)
    for name, identity in first["outputs"].items():
        raw = (one / name).read_bytes()
        assert identity == {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    assert (
        "https://example.invalid/tsfm-lab/#quickstart" in (one / "llms.txt").read_text()
    )


def test_catalog_matrix_matches_variants_without_transplanting_machine_provenance(
    tmp_path, curated_root, monkeypatch
):
    entries = copy.deepcopy(catalog_entries())
    for entry in entries:
        entry["private_path"] = "/Users/private/PUBLIC_DOCS_SECRET_CANARY"
        for variant in entry["variants"]:
            variant["representative_checks"] = [
                {"runtime": {"machine": "PRIVATE_MACHINE_CANARY"}}
            ]
    monkeypatch.setattr(builder, "catalog_entries", lambda: entries)
    out = tmp_path / "public"
    builder.build_docs(out, BASE, root=curated_root)
    expected = Counter(
        {entry["adapter_id"]: len(entry["variants"]) or 1 for entry in entries}
    )
    assert Counter(Page((out / "index.html").read_text()).adapters) == expected
    assert len(entries) == 8 and expected["ttm_r2"] == 8
    rows = builder.public_catalog()
    assert (
        next(row for row in rows if row["adapter_id"] == "toto_2")["inference"]
        == "Not recorded"
    )
    joined = "\n".join(p.read_text() for p in out.iterdir())
    assert not any(
        word in joined
        for word in (
            "/Users/",
            "PUBLIC_DOCS_SECRET_CANARY",
            "PRIVATE_DATASET_CANARY",
            "PRIVATE_MACHINE_CANARY",
            str(curated_root),
            str(out),
        )
    )
    assert (
        "Python 3.12 or 3.13" in joined and "Future accuracy is unavailable" in joined
    )
    assert "The source package version is 0.3.0" in joined
    assert "0.3.0 unreleased" not in joined


@pytest.mark.parametrize(
    "url",
    [
        None,
        "http://example.com/",
        "//example.com/",
        "https://user:password@example.com/",
        "https://example.com/?x=1",
        "https://example.com/#part",
        "https://example.com:443/",
        "https://localhost/",
        "https://127.0.0.1/",
        "https://example.com/a/../b",
        "https://example.com/%2e%2e/",
        "https://example.com/%252e%252e/",
        "https://example.com/a\\b",
        "https://example.com/\n",
    ],
)
def test_unsafe_base_url_is_refused_before_output_creation(tmp_path, url):
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        builder.build_docs(output, url)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_existing_destination_is_never_overwritten(tmp_path, kind):
    output = tmp_path / "existing"
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_text("preexisting")
    else:
        try:
            output.symlink_to(tmp_path / "missing")
        except (OSError, NotImplementedError):
            pytest.skip(
                "Host cannot create symlinks; file/directory overwrite refusal is covered separately"
            )
    with pytest.raises(FileExistsError):
        builder.build_docs(output, BASE)
    if kind == "file":
        assert output.read_text() == "preexisting"


def test_metadata_html_and_raw_markdown_are_escaped_and_heading_ids_unique(
    tmp_path, curated_root
):
    project = curated_root / "pyproject.toml"
    description = 'Inspect <script>alert("metadata")</script> & outputs'
    project.write_text(
        re.sub(
            r"^description = .*?$",
            lambda match: "description = " + json.dumps(description),
            project.read_text(),
            flags=re.M,
        )
    )
    guide = curated_root / "docs/public/guide.md"
    guide.write_text(
        guide.read_text()
        + '\n## Repeat\n\n## Repeat\n\n## Repeat-2\n\n<script>alert("raw")</script>\n'
    )
    output = tmp_path / "safe"
    builder.build_docs(output, BASE, root=curated_root)
    page = Page((output / "index.html").read_text())
    assert len(page.scripts) == 1
    assert len(page.ids) == len(set(page.ids))
    assert (
        page.meta["description"]
        == json.loads(page.scripts[0]["text"])["description"]
        == description
    )
    assert "&lt;script&gt;" in (output / "index.html").read_text()


@pytest.mark.parametrize(
    "addition",
    [
        "\n[broken](#missing)\n",
        "\n![private](https://example.com/private.png)\n",
        "\n/Users/private/sensitive.csv\n",
        "\napi_key=secret-value\n",
    ],
)
def test_bad_curated_content_fails_before_publication_marker(
    tmp_path, curated_root, addition
):
    guide = curated_root / "docs/public/guide.md"
    guide.write_text(guide.read_text() + addition)
    output = tmp_path / "refused"
    with pytest.raises(ValueError):
        builder.build_docs(output, BASE, root=curated_root)
    assert not output.exists()


def test_builder_needs_no_optional_model_runtime_or_network(tmp_path):
    code = """
import importlib.abc,runpy,socket,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','toto2','chronos','timesfm','tirex2','tsfm_public','huggingface_hub','safetensors','httpx','keyring'}:
            raise AssertionError('Unexpected optional import '+fullname)
def forbidden(*a,**kw): raise AssertionError('Unexpected network access')
sys.meta_path.insert(0,Block())
socket.create_connection=forbidden
socket.socket.connect=forbidden
script,out=sys.argv[1:]
sys.argv=[script,'--base-url','https://example.invalid/docs/','--out-dir',out]
runpy.run_path(script,run_name='__main__')
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(ROOT / "scripts/build_public_docs.py"),
            str(tmp_path / "isolated"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert json.loads(completed.stdout)["status"] == "built_not_deployed"


def test_styles_cannot_introduce_remote_assets(tmp_path, curated_root):
    style = curated_root / "docs/public/style.css"
    style.write_text(
        style.read_text() + '\n@import url("https://example.com/font.css");\n'
    )
    output = tmp_path / "no-remote-style"
    with pytest.raises(ValueError, match="no remote assets"):
        builder.build_docs(output, BASE, root=curated_root)
    assert not output.exists()


@pytest.mark.parametrize(
    "name",
    [
        "usage-generic",
        "usage-finance",
        "usage-energy",
        "usage-temperature",
        "usage-weather",
        "future-baseline",
    ],
)
def test_public_baseline_examples_execute_with_disposable_unique_outputs(
    tmp_path, name
):
    source = ROOT / "examples/configurations" / f"{name}.yaml"
    raw = yaml.safe_load(source.read_text())
    raw["dataset"]["path"] = str((source.parent / raw["dataset"]["path"]).resolve())
    output = tmp_path / name / "results"
    raw["output"]["directory"] = str(output)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    completed = CliRunner().invoke(app, ["run", str(path), "--json"])
    assert completed.exit_code == 0, completed.output
    result = json.loads(completed.stdout)
    store = ExperimentStore(output)
    assert store.get_manifest(result["run_id"])["status"] == "completed"
    if name == "future-baseline":
        saved, frame, observations = store.load_future_result(result["run_id"])
        assert saved["evaluation"]["status"] == "unavailable" and len(frame) == 4
        assert len(observations) == 8
    else:
        saved, frame = store.load_result(result["run_id"])
        assert saved["configuration"]["workflow"]["profile"] == name.removeprefix(
            "usage-"
        )
        assert (
            len(frame) == 16 and saved["metrics"]["overall"]["mae"]["value"] is not None
        )

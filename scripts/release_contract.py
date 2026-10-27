"""Bounded stdlib checks for this project's source manifest and package metadata."""

import ast
import configparser
import fnmatch
import re
import shlex
from collections import Counter
from email.parser import BytesParser
from pathlib import PurePosixPath


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _match(path, pattern):
    parts, patterns = PurePosixPath(path).parts, PurePosixPath(pattern).parts
    return len(parts) == len(patterns) and all(
        fnmatch.fnmatchcase(part, rule)
        for part, rule in zip(parts, patterns, strict=True)
    )


def source_names(committed, configuration):
    """Apply only the manifest directives maintained by this repository."""
    expected = {
        name
        for name in committed
        if name.startswith("src/tsfm_lab/")
        or name
        in {"README.md", "pyproject.toml", "MANIFEST.in", "setup.py", "setup.cfg"}
    }
    for pattern in configuration["project"].get("license-files", []):
        expected.update(name for name in committed if _match(name, pattern))
    for raw in committed["MANIFEST.in"].decode().splitlines():
        tokens = shlex.split(raw, comments=True)
        if not tokens:
            continue
        action, *arguments = tokens
        require(
            action in {"include", "recursive-include", "exclude", "global-exclude"},
            f"Unsupported source manifest directive: {action}",
        )
        require(arguments, f"Empty source manifest directive: {action}")
        if action == "recursive-include":
            require(
                len(arguments) >= 2, "recursive-include needs a directory and patterns"
            )
            directory, *patterns = arguments
            chosen = {
                name
                for name in committed
                if name.startswith(directory.rstrip("/") + "/")
                and any(
                    fnmatch.fnmatchcase(PurePosixPath(name).name, p) for p in patterns
                )
            }
        elif action == "global-exclude":
            chosen = {
                name
                for name in committed
                if any(
                    fnmatch.fnmatchcase(PurePosixPath(name).name, p) for p in arguments
                )
            }
        else:
            chosen = {
                name for name in committed if any(_match(name, p) for p in arguments)
            }
        if action in {"exclude", "global-exclude"}:
            expected.difference_update(chosen)
        else:
            expected.update(chosen)
    return expected


def _name(value):
    require(
        bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value)), "Invalid package name"
    )
    return re.sub(r"[-_.]+", "-", value).lower()


def _specifiers(value):
    result = []
    for raw in value.strip().removeprefix("(").removesuffix(")").split(","):
        raw = raw.strip()
        if raw:
            require(
                re.fullmatch(r"(?:===|==|!=|~=|>=|<=|>|<)\s*[A-Za-z0-9*+.!_-]+", raw),
                f"Unsupported dependency constraint: {value}",
            )
            result.append(re.sub(r"\s+", "", raw))
    return tuple(sorted(result))


def _requirement(value, extra=None):
    requirement, separator, marker = value.partition(";")
    if separator:
        match = re.fullmatch(
            r"""\s*extra\s*==\s*(["'])([A-Za-z0-9._-]+)\1\s*""", marker
        )
        require(
            match is not None and extra is None,
            f"Unsupported dependency marker: {value}",
        )
        extra = _name(match[2])
    match = re.fullmatch(
        r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[([A-Za-z0-9._, -]+)\])?\s*(.*?)\s*",
        requirement,
    )
    require(match is not None, f"Unsupported dependency requirement: {value}")
    extras = tuple(
        sorted(
            _name(item.strip()) for item in (match[2] or "").split(",") if item.strip()
        )
    )
    return _name(match[1]), extras, _specifiers(match[3]), extra


def metadata_contract(raw, configuration):
    """Check complete dependency and extra semantics against committed pyproject."""
    metadata, project = BytesParser().parsebytes(raw), configuration["project"]
    for key in ("Name", "Version", "Requires-Python", "License-Expression"):
        require(
            len(metadata.get_all(key, [])) == 1,
            f"Missing/duplicate package metadata: {key}",
        )
    require(
        _name(metadata["Name"]) == _name(project["name"]),
        "Package metadata name differs",
    )
    require(
        metadata["Version"] == project["version"], "Package metadata version differs"
    )
    require(
        _specifiers(metadata["Requires-Python"])
        == _specifiers(project["requires-python"]),
        "Package Python requirement differs",
    )
    require(
        metadata["License-Expression"] == project["license"], "Package license differs"
    )
    expected = Counter(_requirement(item) for item in project.get("dependencies", []))
    groups = project.get("optional-dependencies", {})
    for extra, requirements in groups.items():
        expected.update(_requirement(item, _name(extra)) for item in requirements)
    actual = Counter(
        _requirement(item) for item in metadata.get_all("Requires-Dist", [])
    )
    require(
        actual == expected,
        "Package dependency metadata differs from committed pyproject",
    )
    extras = [_name(item) for item in metadata.get_all("Provides-Extra", [])]
    require(
        Counter(extras) == Counter(_name(item) for item in groups),
        "Package extra metadata differs",
    )
    return {"name": _name(metadata["Name"]), "version": metadata["Version"]}


def wheel_contract(files, configuration=None):
    roots = {name.split("/", 1)[0] for name in files if ".dist-info/" in name}
    require(len(roots) == 1, "Wheel needs exactly one metadata directory")
    prefix = roots.pop() + "/"
    required = {
        prefix + name for name in ("METADATA", "WHEEL", "RECORD", "entry_points.txt")
    }
    require(required <= files.keys(), "Wheel is missing mandatory package metadata")
    metadata = BytesParser().parsebytes(files[prefix + "METADATA"])
    require(
        len(metadata.get_all("Name", [])) == len(metadata.get_all("Version", [])) == 1,
        "Wheel has missing or duplicate identity metadata",
    )
    name, version = metadata.get("Name", ""), metadata.get("Version", "")
    require(
        _name(name) == "tsfm-lab"
        and bool(re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", version)),
        "Wheel is not a versioned tsfm-lab package",
    )
    require(
        prefix == f"tsfm_lab-{version}.dist-info/",
        "Wheel metadata directory identity differs",
    )
    require(
        all(name.startswith(("tsfm_lab/", prefix)) for name in files),
        "Wheel has files outside the supported package and metadata directories",
    )
    if configuration is not None:
        metadata_contract(files[prefix + "METADATA"], configuration)
    entry_contract(files[prefix + "entry_points.txt"], configuration)
    wheel = BytesParser().parsebytes(files[prefix + "WHEEL"])
    require(wheel.get("Wheel-Version") == "1.0", "Unsupported wheel format")
    return prefix, {"name": "tsfm-lab", "version": version}


def entry_contract(raw, configuration=None):
    entry = configparser.ConfigParser(interpolation=None)
    entry.optionxform = str
    entry.read_string(raw.decode())
    project = (
        configuration["project"]
        if configuration
        else {"scripts": {"tsfm": "tsfm_lab.__main__:main"}}
    )
    expected = dict(project.get("entry-points", {}))
    for group, field in (
        ("console_scripts", "scripts"),
        ("gui_scripts", "gui-scripts"),
    ):
        if project.get(field):
            require(group not in expected, f"Conflicting entry point group: {group}")
            expected[group] = project[field]
    actual = {section: dict(entry.items(section)) for section in entry.sections()}
    require(
        not entry.defaults() and actual == expected,
        "Package entry point metadata differs",
    )


def version_contract(raw, version):
    values = []
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            values.append(ast.literal_eval(node.value))
    require(values == [version], "Committed package __version__ differs from metadata")

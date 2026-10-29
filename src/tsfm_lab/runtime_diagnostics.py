"""Offline package-metadata checks without importing optional inference runtimes."""

import re
from importlib.metadata import PackageNotFoundError, version

from tsfm_lab.models.registry import ADAPTER_PROFILES


def _release(value: str) -> tuple[int, ...]:
    # The catalog qualifies final releases. Do not imply prerelease compatibility.
    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)*)(?:\.post[0-9]+)?(?:\+[a-zA-Z0-9._-]+)?", value
    )
    if match is None:
        raise ValueError("Unqualified release version")
    parts = tuple(int(part) for part in match[1].split("."))
    return parts + (0,) * max(0, 3 - len(parts))


def _matches(installed: str, requirement: str) -> bool:
    """Match only the reviewed release constraints used by the local catalog."""
    try:
        release = _release(installed)
        for clause in requirement.split(","):
            match = re.fullmatch(r"(==|>=|<)([0-9]+(?:\.[0-9]+)*)", clause)
            if match is None:
                return False
            operator, wanted = match.groups()
            if operator == "==" and installed != wanted:
                return False
            if operator == ">=" and release < _release(wanted):
                return False
            if operator == "<" and release >= _release(wanted):
                return False
    except (TypeError, ValueError):
        return False
    return True


def runtime_diagnostics() -> list[dict]:
    """Describe baseline availability and optional runtime metadata compatibility.

    An available pretrained entry means its installed distribution versions match
    the catalog. It does not certify imports, checkpoint loading, or native
    platform support. This function never downloads, installs, or loads models.
    """
    entries = [
        {
            "adapter_id": adapter,
            "label": label,
            "status": "available",
            "detail": "Built-in CPU baseline; no optional model runtime required.",
        }
        for adapter, label in (
            ("last_value", "Last value"),
            ("seasonal_naive", "Seasonal naive"),
        )
    ]
    installed = {}
    unreadable = set()
    for profile in ADAPTER_PROFILES:
        for package, _ in profile.runtime_requirements:
            if package in installed:
                continue
            try:
                installed[package] = version(package)
            except PackageNotFoundError:
                installed[package] = None
            except (OSError, ValueError, UnicodeError):
                installed[package] = None
                unreadable.add(package)
    for profile in ADAPTER_PROFILES:
        missing, incompatible = [], []
        if profile.adapter_id == "toto_2":
            from tsfm_lab.models.toto2_profile import ensure_supported_python

            try:
                ensure_supported_python()
            except ImportError as exc:
                entries.append(
                    {
                        "adapter_id": profile.adapter_id,
                        "label": profile.display_name,
                        "status": "incompatible",
                        "detail": str(exc),
                    }
                )
                continue
        for package, requirement in profile.runtime_requirements:
            found = installed[package]
            if package in unreadable:
                incompatible.append(f"{package}: cannot read package metadata")
            elif found is None:
                missing.append(f"{package}{requirement}")
            elif not _matches(found, requirement):
                incompatible.append(f"{package}: found {found}, requires {requirement}")
        status = (
            "missing" if missing else "incompatible" if incompatible else "available"
        )
        detail = []
        if missing:
            detail.append("Missing packages: " + ", ".join(missing))
        if incompatible:
            detail.append("Incompatible packages: " + "; ".join(incompatible))
        if not detail:
            detail.append("Installed package versions match the catalog")
        detail.append(
            "Metadata check only; imports, loading, and platform support unverified"
        )
        entries.append(
            {
                "adapter_id": profile.adapter_id,
                "label": profile.display_name,
                "status": status,
                "detail": ". ".join(detail) + ".",
            }
        )
    return entries

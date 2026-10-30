"""Execution-only platform candidates, separate from detached record contracts."""

import sys


def platform_requirements(adapter_id):
    if adapter_id == "tirex_2" and sys.platform == "win32":
        return (("triton-windows", "==3.4.0.post21"),)
    return ()


def optional_provenance_packages(adapter_id):
    """Accept observations from another host without requiring its runtime here."""
    return {"triton-windows"} if adapter_id == "tirex_2" else set()

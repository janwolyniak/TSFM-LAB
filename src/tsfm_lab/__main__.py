"""Console entry point that leaves the optional CLI outside numerical imports."""

import sys


def _configure_windows_output():
    # The Windows console already uses its Unicode API. Redirected CLI output
    # uses explicit UTF-8 so non-ASCII paths survive pipes and saved diagnostics.
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure") and not stream.isatty():
                stream.reconfigure(encoding="utf-8")


def main() -> None:
    _configure_windows_output()
    try:
        from tsfm_lab.cli import app
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "typer",
            "rich",
            "yaml",
            "click",
            "shellingham",
            "annotated_doc",
        }:
            raise
        print(
            "The tsfm command requires the optional tsfm-lab[cli] dependencies.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    app()


if __name__ == "__main__":
    main()

"""Console entry point that leaves the optional CLI outside numerical imports."""

import sys


def main() -> None:
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

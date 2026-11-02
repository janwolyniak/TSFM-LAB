"""Export checked forecasts without running a model or replacing existing files."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.run_modes import read_completed


class ExportError(ValueError):
    """The requested export format is unsupported."""


def export_predictions(
    run_id: str,
    workspace: str | Path,
    destination: str | Path,
    *,
    format: str | None = None,
) -> dict:
    target = Path(destination).expanduser().absolute()
    selected = format or target.suffix.lower().lstrip(".")
    if selected not in {"csv", "parquet", "json"}:
        raise ExportError("Export format must be csv, parquet, or json")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Export destination already exists: {target}")
    _, predictions = read_completed(ExperimentStore(workspace), run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".tsfm-export-", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        if selected == "parquet":
            try:
                predictions.to_parquet(temporary, index=False, engine="pyarrow")
            except ImportError as exc:
                raise ImportError("Parquet export requires tsfm-lab[parquet]") from exc
        elif selected == "csv":
            predictions.to_csv(temporary, index=False, lineterminator="\n")
        else:
            rows = predictions.copy()
            for column in ("origin", "target_timestamp"):
                rows[column] = rows[column].map(lambda value: value.isoformat())
            temporary.write_text(
                json.dumps(rows.to_dict(orient="records"), allow_nan=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
        # Windows' durability operation requires a writable file descriptor.
        with temporary.open("r+b") as file:
            os.fsync(file.fileno())
        # Publishing a hard link is atomic and refuses a concurrent existing path.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "run_id": run_id,
        "path": str(target),
        "format": selected,
        "rows": len(predictions),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }

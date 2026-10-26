"""Compact scrollback reports over the shared catalog and verified run records."""

import re
from pathlib import Path
from uuid import UUID

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text

from tsfm_lab.comparison import compare_runs
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.exports import ExportError, export_predictions
from tsfm_lab.models.catalog import catalog_entries
from tsfm_lab.models.verification import verification_label
from tsfm_lab.repl.text import literal


def _metric(value: dict) -> str:
    if value["value"] is None:
        reason = str(value.get("reason") or "unavailable").replace("_", " ")
        return f"undefined ({reason})"
    return f"{value['value']:.6g}"


def _utc(value) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _canonical_id(value: str) -> bool:
    try:
        return str(UUID(value)) == value
    except (ValueError, TypeError, AttributeError):
        return False


class RecordCommands:
    """Render literal, borderless reports; never initialize or execute a model."""

    def __init__(self, console: Console, workspace: Path):
        self.console = console
        self.workspace = Path(workspace).expanduser().resolve()

    def _store(self) -> ExperimentStore:
        return ExperimentStore(Path(self.workspace).expanduser())

    def _print(self, value: str) -> None:
        self.console.print(Text(literal(value)), soft_wrap=True)

    def _table(self, columns, rows) -> None:
        table = Table(box=None, pad_edge=False, header_style="", collapse_padding=True)
        for label in columns:
            table.add_column(Text(literal(label)), overflow="fold")
        for row in rows:
            table.add_row(*(Text(literal(value)) for value in row))
        self.console.print(table)

    def _task(self, dataset: dict, evaluation: dict) -> None:
        frequency = dataset["frequency"]
        if isinstance(frequency, int):
            frequency = str(pd.Timedelta(frequency, unit="ns"))
        self._print(
            f"Target: {dataset['target_name']} · Units: {dataset['units']} · Frequency: {frequency}"
        )
        self._print(
            f"Horizon: {evaluation['horizon']} observations · Context: {evaluation['context_length']} · Stride: {evaluation['stride']}"
        )
        self._print(f"Forecast task: {evaluation['point_statistic']}")
        self._print(
            f"Targets: {evaluation['evaluation_start']} to {evaluation['evaluation_end']} (end excluded)"
        )

    def models(self) -> list[dict]:
        entries = catalog_entries()
        self._print("Supported models")
        self._table(
            ("Model", "ID", "Context", "Horizon", "Device", "License"),
            (
                (
                    entry["display_name"],
                    entry["adapter_id"],
                    entry["capabilities"]["max_context"] or "any",
                    entry["capabilities"]["max_horizon"] or "any",
                    ", ".join(entry["capabilities"]["devices"]),
                    entry["license"],
                )
                for entry in entries
            ),
        )
        self._print(
            "Pretrained adapters require their installed runtime and a cached or local checkpoint."
        )
        for entry in entries:
            for notice in entry.get("runtime_notices", []):
                self._print(notice)
            variants = entry.get("variants", [])
            for variant in variants:
                self._print(f"{variant['display_name']} · {variant['repository']}")
                if len(entry["repositories"]) < len(variants):
                    self._print(f"  Checkpoint profile: {variant['variant']}")
                context = (
                    f"Context exactly {variant['min_context']}"
                    if variant["min_context"] == variant["max_context"]
                    else f"Context up to {variant['max_context']}"
                )
                self._print(
                    f"  {context} · Horizon up to {variant['max_horizon']} observations"
                )
                if variant["weight_bytes"] is not None:
                    self._print(
                        f"  Weight files: {variant['weight_bytes'] / 1_000_000:.1f} MB"
                    )
                self._print("  Reference checkpoint:")
                self._print(f"    {variant['verification_label']}")
            if variants:
                for check in variants[0]["representative_checks"]:
                    self._print(
                        f"  Adapter reference: {check['repository']} @ {check['revision'][:12]} · {check['device']}/{check['dtype']}"
                    )
        return entries

    def history(self) -> list[dict]:
        store = self._store()
        records = store.list_runs()
        if not records:
            self._print(
                f"No saved experiments in {store.workspace}. Run /demo to create an offline example."
            )
            return records
        self._print(f"Experiments in {store.workspace}")
        self._table(
            ("ID", "Experiment", "Status", "Created UTC"),
            (
                (
                    record["run_id"][:8],
                    record.get("name", ""),
                    record["status"],
                    (
                        _utc(record["created_at"])
                        if record.get("created_at")
                        else "unknown"
                    ),
                )
                for record in records
            ),
        )
        self._print(
            "Use /show <id> to inspect an experiment; IDs accept a unique prefix."
        )
        return records

    def resolve_id(self, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise StoreError(
                "Choose an experiment ID, a unique prefix, or latest. Use /history."
            )
        selected = value.strip().lower()
        store = self._store()
        if selected != "latest":
            try:
                exact = str(UUID(selected))
            except ValueError:
                exact = None
            if exact is not None:
                if not (store.root / exact).exists():
                    raise StoreError(
                        f"No saved experiment {selected}. Use /history to find an ID."
                    )
                store.get_manifest(exact)
                return exact
            if re.fullmatch(r"[0-9a-f-]{1,35}", selected) is None:
                raise StoreError(
                    "Enter a UUID, a hexadecimal ID prefix, or latest. Use /history."
                )
        records = store.list_runs()
        if selected == "latest":
            completed = [
                record for record in records if record["status"] == "completed"
            ]
            if not completed:
                raise StoreError(
                    "No completed experiments. Use /history to inspect runs, or /demo to create one."
                )
            return completed[0]["run_id"]
        matches = [
            record["run_id"]
            for record in records
            if _canonical_id(record["run_id"]) and record["run_id"].startswith(selected)
        ]
        if not matches:
            raise StoreError(
                f"No experiment matches {selected!r}. Use /history to find an ID."
            )
        if len(matches) != 1:
            examples = ", ".join(matches[:5])
            raise StoreError(
                f"Ambiguous experiment prefix {selected!r}: {examples}. Use a longer prefix."
            )
        return matches[0]

    def show(self, run_id: str) -> dict:
        resolved = self.resolve_id(run_id)
        store = self._store()
        manifest = store.get_manifest(resolved)
        if manifest["status"] != "completed":
            self._print(f"{manifest['name']} · {resolved} · {manifest['status']}")
            error = manifest.get("error")
            if isinstance(error, dict):
                stage = f" ({error['stage']})" if error.get("stage") else ""
                self._print(
                    f"{error.get('message') or error.get('type') or 'No error details recorded'}{stage}"
                )
            elif error:
                self._print(str(error))
            return {"manifest": manifest}
        result, predictions = store.load_result(resolved)
        configuration = result["configuration"]
        self._print(f"{manifest['name']} · {resolved}")
        self._task(configuration["dataset"], configuration["evaluation"])
        self._print(
            f"Forecast output: {', '.join(sorted(predictions['point_statistic'].unique()))}"
        )
        model, runtime = configuration["model"], configuration["runtime"]
        self._print(
            f"Model: {model['adapter_id']} · {runtime['device']} · {runtime['dtype']} · Completed"
        )
        self._table(
            ("Metric", "Value"),
            (
                (name.upper(), _metric(value))
                for name, value in result["metrics"]["overall"].items()
            ),
        )
        names = list(result["metrics"]["overall"])
        leads = sorted(
            result["metrics"]["by_lead"].items(), key=lambda item: int(item[0])
        )
        self._print("Metrics by forecast lead")
        self._table(
            ("Lead", *(name.upper() for name in names)),
            (
                (lead, *(_metric(values[name]) for name in names))
                for lead, values in leads[:12]
            ),
        )
        if len(leads) > 12:
            self._print(f"Showing the first 12 of {len(leads)} forecast leads.")
        self._print(
            f"Forecast preview · first {min(12, len(predictions))} of {len(predictions)} cases · UTC"
        )
        self._table(
            ("Origin", "Target", "Lead", "Actual", "Forecast"),
            (
                (
                    _utc(row.origin),
                    _utc(row.target_timestamp),
                    row.lead,
                    f"{row.actual:.6g}",
                    f"{row.prediction:.6g}",
                )
                for row in predictions.head(12).itertuples()
            ),
        )
        checkpoint = model["options"].get("checkpoint")
        if checkpoint:
            self._print(
                f"Checkpoint: {checkpoint.get('repository') or checkpoint['path']}"
            )
            self._print(
                f"Revision: {checkpoint.get('resolved_revision') or 'local files'} · Fingerprint: {checkpoint['fingerprint']}"
            )
            verification = result["provenance"].get("project_verification")
            if verification is None:
                self._print("Project verification was not recorded.")
            else:
                self._print(f"Selected checkpoint: {verification_label(verification)}")
                for check in verification["adapter_representative_checks"]:
                    self._print(
                        f"Adapter reference: {check['repository']} @ {check['revision'][:12]} · {check['device']}/{check['dtype']}"
                    )
        self._print(f"Export full forecasts with /export {resolved[:8]} <path.csv>.")
        return {"manifest": manifest, "result": result, "predictions": predictions}

    def compare(self, run_ids: list[str]) -> dict:
        resolved = [self.resolve_id(value) for value in run_ids]
        result = compare_runs(resolved, self._store().workspace)
        self._print(
            "Controlled comparison · matching forecast tasks and observed cases"
        )
        self._task(result["task"]["dataset"], result["task"]["evaluation"])
        native = "point_semantics" in result
        if native:
            self._print(result["point_semantics"])
        names = list(result["runs"][0]["metrics"])
        self._table(
            (
                "Experiment",
                "Model",
                *(("Output",) if native else ()),
                "Precision",
                *(name.upper() for name in names),
            ),
            (
                (
                    run["name"],
                    run["model"]["adapter_id"],
                    *((", ".join(run["output_statistics"]),) if native else ()),
                    run["runtime"]["dtype"],
                    *(_metric(run["metrics"][name]) for name in names),
                )
                for run in result["runs"]
            ),
        )
        return result

    def export(self, run_id: str, path: str | Path, format: str | None = None) -> dict:
        resolved = self.resolve_id(run_id)
        try:
            result = export_predictions(
                resolved, self._store().workspace, path, format=format
            )
        except (OSError, ImportError) as exc:
            raise ExportError(str(exc)) from exc
        self._print(f"Exported {result['rows']} forecast cases to {result['path']}")
        return result

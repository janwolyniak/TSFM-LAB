"""Offline model browsing and responsive views over verified experiment records."""

import asyncio
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Footer,
    Input,
    Select,
    Static,
)

from tsfm_lab.comparison import compare_runs
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.exports import export_predictions
from tsfm_lab.models.catalog import catalog_entries
from tsfm_lab.tui.view_data import (
    error_text,
    metric_text,
    preview_rows,
    record_details,
    task_summary,
)


def _row(table: DataTable, *values, key=None) -> None:
    table.add_row(*(Text(str(value)) for value in values), key=key)


class _View(Screen):
    BINDINGS = [Binding("escape", "back", "Back")]
    DEFAULT_CSS = """
    .view-title { height: auto; margin: 1 1 0 1; text-style: bold; color: $accent; }
    .view-body { height: 1fr; padding: 0 1; }
    .view-actions { dock: bottom; height: 3; margin-bottom: 1; padding: 0 1; }
    .view-actions Button { margin-right: 1; min-width: 10; }
    .view-section { height: auto; }
    .view-message { height: auto; margin-bottom: 1; }
    .view-heading { height: auto; margin-top: 1; text-style: bold; }
    .view-table { height: 8; margin-bottom: 1; }
    #model-table { height: 6; }
    #history-table { height: 12; }
    #overall-metrics { height: 6; }
    #forecast-preview { height: 14; }
    #export-controls { height: 3; }
    #export-format { width: 18; margin-right: 1; }
    #export-button { min-width: 16; }
    """

    def action_back(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#view-back")
    def _back(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_back()


class ModelScreen(_View):
    """Select from the small installed-adapter catalog without probing a network."""

    def compose(self) -> ComposeResult:
        yield Static("Models", classes="view-title")
        with VerticalScroll(classes="view-body"):
            yield DataTable(id="model-table", cursor_type="row", classes="view-table")
            yield Static(
                "Select a model to see its requirements.",
                id="model-details",
                markup=False,
            )
        with Horizontal(classes="view-actions"):
            yield Button("Back", id="view-back")
            yield Button("Use model", id="use-model", variant="primary", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#model-table", DataTable)
        table.add_columns("Model", "Context", "Horizon", "Device")
        self.entries = []
        try:
            self.entries = catalog_entries()
            for entry in self.entries:
                caps = entry["capabilities"]
                _row(
                    table,
                    entry["display_name"],
                    caps["max_context"] or "any",
                    caps["max_horizon"] or "any",
                    ", ".join(caps["devices"]),
                    key=entry["adapter_id"],
                )
            self.query_one("#use-model", Button).disabled = not self.entries
            self._details(0)
        except Exception as exc:
            self.query_one("#model-details", Static).update(
                Text(f"Models unavailable: {exc}")
            )
        table.focus()

    def _details(self, index: int) -> None:
        if not 0 <= index < len(self.entries):
            return
        entry = self.entries[index]
        requirement = "Built in; ready for offline evaluation."
        if entry["source"] != "builtin":
            requirement = "Requires installed Chronos dependencies and a cached or local checkpoint."
        if entry["adapter_id"] == "seasonal_naive":
            requirement += " Context must contain a complete season."
        self.query_one("#model-details", Static).update(
            Text(
                f"{entry['display_name']} · License: {entry['license']}\n"
                f"Point forecast: {entry['point_statistic']} · Target count: {entry['capabilities']['target_count']}\n"
                f"{requirement}"
            )
        )

    @on(DataTable.RowHighlighted, "#model-table")
    def _highlight(self, event: DataTable.RowHighlighted) -> None:
        self._details(event.cursor_row)

    @on(Button.Pressed, "#use-model")
    @on(DataTable.RowSelected, "#model-table")
    def _choose(self) -> None:
        index = self.query_one("#model-table", DataTable).cursor_row
        if 0 <= index < len(self.entries):
            self.dismiss(self.entries[index]["adapter_id"])


class HistoryScreen(_View):
    BINDINGS = [
        *_View.BINDINGS,
        Binding("space", "toggle_selection", "Select"),
        Binding("c", "compare_selected", "Compare"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(self, workspace: Path):
        super().__init__()
        self.workspace = Path(workspace).expanduser().resolve()
        self.records: dict[str, dict] = {}
        self.order: list[str] = []
        self.selected: set[str] = set()

    def compose(self) -> ComposeResult:
        yield Static("Experiment history", classes="view-title")
        with VerticalScroll(classes="view-body"):
            yield Static(
                "Loading saved experiments…",
                id="history-status",
                classes="view-message",
                markup=False,
            )
            yield DataTable(id="history-table", cursor_type="row", classes="view-table")
        with Horizontal(classes="view-actions"):
            yield Button("Back", id="view-back")
            yield Button("Refresh", id="history-refresh")
            yield Button("View", id="history-view", disabled=True)
            yield Button("Compare", id="history-compare", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#history-table", DataTable).add_columns(
            "", "Experiment", "Status", "ID", "Created UTC"
        )
        self.action_refresh()

    def action_refresh(self) -> None:
        self.selected.clear()
        self.query_one("#history-compare", Button).disabled = True
        self.run_worker(
            self._refresh(), group="history-read", exclusive=True, exit_on_error=False
        )

    async def _refresh(self) -> None:
        status = self.query_one("#history-status", Static)
        status.update("Loading saved experiments…")
        try:
            records = await asyncio.to_thread(ExperimentStore(self.workspace).list_runs)
            self.records = {record["run_id"]: record for record in records}
            self.order = list(self.records)
            self._populate()
            status.update(
                f"{len(records)} saved experiments. Space selects completed runs; C compares selections."
            )
            self.query_one("#history-view", Button).disabled = not records
            self.query_one("#history-table", DataTable).focus()
        except Exception as exc:
            self.records.clear()
            self.order.clear()
            self.selected.clear()
            self.query_one("#history-table", DataTable).clear()
            status.update(Text(f"History unavailable: {exc}"))
            self.query_one("#history-view", Button).disabled = True
            self.query_one("#history-compare", Button).disabled = True

    def _populate(self) -> None:
        table = self.query_one("#history-table", DataTable)
        cursor = table.cursor_row
        table.clear()
        for run_id in self.order:
            record = self.records[run_id]
            _row(
                table,
                "✓" if run_id in self.selected else "",
                record.get("name", ""),
                record["status"],
                run_id[:8],
                record.get("created_at", "")[:16].replace("T", " "),
                key=run_id,
            )
        if self.order:
            table.move_cursor(row=min(cursor, len(self.order) - 1))
        self.query_one("#history-compare", Button).disabled = len(self.selected) < 2

    def _current(self) -> str | None:
        index = self.query_one("#history-table", DataTable).cursor_row
        return self.order[index] if 0 <= index < len(self.order) else None

    def action_toggle_selection(self) -> None:
        run_id = self._current()
        if run_id and self.records[run_id]["status"] == "completed":
            self.selected.symmetric_difference_update({run_id})
            self._populate()
            self.query_one("#history-status", Static).update(
                f"{len(self.selected)} completed experiments selected."
            )
        else:
            self.query_one("#history-status", Static).update(
                "Select completed experiments for comparison."
            )

    def action_compare_selected(self) -> None:
        if len(self.selected) >= 2:
            self.app.push_screen(
                ComparisonScreen(
                    self.workspace,
                    [run_id for run_id in self.order if run_id in self.selected],
                )
            )

    @on(DataTable.RowSelected, "#history-table")
    @on(Button.Pressed, "#history-view")
    def _open(self) -> None:
        if run_id := self._current():
            self.app.push_screen(ResultsScreen(self.workspace, run_id))

    @on(Button.Pressed, "#history-refresh")
    def _refresh_button(self) -> None:
        self.action_refresh()

    @on(Button.Pressed, "#history-compare")
    def _compare_button(self) -> None:
        self.action_compare_selected()


class ResultsScreen(_View):
    BINDINGS = [*_View.BINDINGS, Binding("ctrl+e", "export", "Export", priority=True)]

    def __init__(self, workspace: Path, run_id: str):
        super().__init__()
        self.workspace, self.run_id = Path(workspace).expanduser().resolve(), run_id
        self.completed = False

    def compose(self) -> ComposeResult:
        yield Static("Experiment results", classes="view-title")
        with VerticalScroll(classes="view-body"):
            yield Static(
                "Loading experiment…",
                id="result-summary",
                classes="view-message",
                markup=False,
            )
            yield Static("", id="result-status", classes="view-message", markup=False)
            with Vertical(id="completed-result", classes="view-section"):
                yield Static("Overall metrics", classes="view-heading")
                yield DataTable(id="overall-metrics", classes="view-table")
                yield Static("Metrics by forecast lead", classes="view-heading")
                yield DataTable(id="lead-metrics", classes="view-table")
                yield Static(
                    "First 12 forecast cases · timestamps in UTC",
                    classes="view-heading",
                )
                yield DataTable(id="forecast-preview", classes="view-table")
                with Collapsible(title="Model and reproducibility", collapsed=True):
                    yield Static("", id="run-details", markup=False)
                yield Static("Export forecasts to a new file", classes="view-heading")
                yield Input(
                    str(self.workspace / "exports" / f"{self.run_id}.csv"),
                    id="export-path",
                )
                with Horizontal(id="export-controls"):
                    yield Select(
                        [("CSV", "csv"), ("JSON", "json"), ("Parquet", "parquet")],
                        value="csv",
                        allow_blank=False,
                        id="export-format",
                    )
                    yield Button("Export", id="export-button", disabled=True)
                yield Static(
                    "Existing files are preserved.",
                    id="export-status",
                    classes="view-message",
                    markup=False,
                )
        with Horizontal(classes="view-actions"):
            yield Button("Back", id="view-back")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#completed-result").display = False
        self.run_worker(
            self._load(), group="result-read", exclusive=True, exit_on_error=False
        )

    def _read(self):
        store = ExperimentStore(self.workspace)
        manifest = store.get_manifest(self.run_id)
        return (
            (manifest, *store.load_result(self.run_id))
            if manifest["status"] == "completed"
            else (manifest, None, None)
        )

    async def _load(self) -> None:
        try:
            manifest, result, frame = await asyncio.to_thread(self._read)
            if result is None:
                self.query_one("#result-summary", Static).update(
                    Text(f"{manifest['name']} · {manifest['status']}")
                )
                self.query_one("#result-status", Static).update(
                    Text(error_text(manifest.get("error")))
                )
                return
            config = result["configuration"]
            self.query_one("#result-summary", Static).update(
                Text(task_summary(config["dataset"], config["evaluation"]))
            )
            self.query_one("#result-status", Static).update(
                Text(f"{manifest['name']} · Completed · {len(frame)} forecast cases")
            )
            overall = self.query_one("#overall-metrics", DataTable)
            overall.add_columns("Metric", "Value")
            for name, value in result["metrics"]["overall"].items():
                _row(overall, name.upper(), metric_text(value))
            names = list(result["metrics"]["overall"])
            leads = self.query_one("#lead-metrics", DataTable)
            leads.add_columns("Lead", *(name.upper() for name in names))
            for lead, values in sorted(
                result["metrics"]["by_lead"].items(), key=lambda item: int(item[0])
            ):
                _row(leads, lead, *(metric_text(values[name]) for name in names))
            preview = self.query_one("#forecast-preview", DataTable)
            preview.add_columns(
                "Origin UTC", "Target UTC", "Lead", "Actual", "Forecast"
            )
            for row in preview_rows(frame):
                _row(preview, *row)
            self.query_one("#run-details", Static).update(Text(record_details(result)))
            self.completed = True
            self.query_one("#completed-result").display = True
            self.query_one("#export-button", Button).disabled = False
        except Exception as exc:
            self.query_one("#result-summary", Static).update("Experiment unavailable")
            self.query_one("#result-status", Static).update(
                Text(f"Cannot show verified results: {exc}")
            )

    def action_export(self) -> None:
        if self.completed and not self.query_one("#export-button", Button).disabled:
            self.run_worker(
                self._export(),
                group="result-export",
                exclusive=True,
                exit_on_error=False,
            )

    @on(Button.Pressed, "#export-button")
    @on(Input.Submitted, "#export-path")
    def _export_button(self) -> None:
        self.action_export()

    async def _export(self) -> None:
        button = self.query_one("#export-button", Button)
        button.disabled = True
        status = self.query_one("#export-status", Static)
        try:
            path = self.query_one("#export-path", Input).value
            if not path.strip():
                raise ValueError("Enter a destination file path.")
            selected = self.query_one("#export-format", Select).value
            result = await asyncio.to_thread(
                export_predictions, self.run_id, self.workspace, path, format=selected
            )
            status.update(Text(f"Saved {result['rows']} forecasts to {result['path']}"))
        except Exception as exc:
            status.update(Text(f"Export unavailable: {exc}"))
        finally:
            button.disabled = False


class ComparisonScreen(_View):
    def __init__(self, workspace: Path, run_ids: list[str]):
        super().__init__()
        self.workspace, self.run_ids = Path(workspace).expanduser().resolve(), list(
            run_ids
        )

    def compose(self) -> ComposeResult:
        yield Static("Controlled comparison", classes="view-title")
        with VerticalScroll(classes="view-body"):
            yield Static(
                "Checking forecasting tasks…",
                id="comparison-summary",
                classes="view-message",
                markup=False,
            )
            yield Static(
                "", id="comparison-status", classes="view-message", markup=False
            )
            yield DataTable(id="comparison-table", classes="view-table")
        with Horizontal(classes="view-actions"):
            yield Button("Back", id="view-back")
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(
            self._load(), group="comparison-read", exclusive=True, exit_on_error=False
        )

    async def _load(self) -> None:
        try:
            result = await asyncio.to_thread(compare_runs, self.run_ids, self.workspace)
            task = result["task"]
            self.query_one("#comparison-summary", Static).update(
                Text(task_summary(task["dataset"], task["evaluation"]))
            )
            table = self.query_one("#comparison-table", DataTable)
            names = list(result["runs"][0]["metrics"])
            table.add_columns(
                "Experiment", "Model", "Precision", *(name.upper() for name in names)
            )
            for run in result["runs"]:
                _row(
                    table,
                    run["name"],
                    run["model"]["adapter_id"],
                    run["runtime"]["dtype"],
                    *(metric_text(run["metrics"][name]) for name in names),
                )
            self.query_one("#comparison-status", Static).update(
                "Matching forecast tasks and observed cases."
            )
        except Exception as exc:
            self.query_one("#comparison-summary", Static).update(
                "Comparison unavailable"
            )
            self.query_one("#comparison-status", Static).update(Text(str(exc)))

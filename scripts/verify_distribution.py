"""Verify an installed wheel in isolation; never install or download dependencies.

Run with the clean environment's bin directory on PATH:
    python -I scripts/verify_distribution.py
"""

import csv
import hashlib
import importlib
import importlib.metadata
import importlib.resources
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

_CHECKOUT = Path(__file__).resolve().parents[1]
_OPTIONAL = (
    "torch",
    "chronos",
    "timesfm",
    "transformers",
    "huggingface_hub",
    "textual",
)
_COLUMNS = {
    "request_id",
    "series_id",
    "adapter_id",
    "origin",
    "target_timestamp",
    "lead",
    "actual",
    "prediction",
    "point_statistic",
}
_BLOCK_OPTIONAL = """import importlib.abc
import json
import sys

blocked = {'torch', 'chronos', 'timesfm', 'transformers', 'huggingface_hub', 'textual'}

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, Block())
"""


class VerificationError(RuntimeError):
    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def require(condition, message, **details):
    if not condition:
        raise VerificationError(message, **details)


def strict_json(text):
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    try:
        require("\x1b" not in text, "Machine-readable stdout contains terminal escapes")
        return json.loads(text, parse_constant=nonfinite)
    except ValueError as exc:
        raise VerificationError(
            f"Expected undecorated JSON stdout: {exc}", stdout_tail=text[-2000:]
        ) from exc


def installed_environment():
    require(
        sys.flags.isolated == 1,
        "Run the verifier as python -I scripts/verify_distribution.py",
    )
    require(
        sys.prefix != sys.base_prefix,
        "Use a virtual environment with the built wheel installed",
    )
    try:
        module = importlib.import_module("tsfm_lab")
    except ImportError as exc:
        raise VerificationError(
            "The tsfm-lab wheel is not installed in this environment; install the built wheel with [cli,parquet] before verification"
        ) from exc
    require(
        getattr(module, "__file__", None),
        "Installed tsfm_lab has no file-backed module origin",
    )
    origin = Path(module.__file__).resolve()
    require(
        not origin.is_relative_to(_CHECKOUT),
        "Refusing source-checkout imports; install the built wheel instead of an editable checkout",
        module_origin=str(origin),
    )
    require(
        not origin.is_relative_to(Path(__file__).resolve().parent),
        "Refusing imports from the verifier scripts directory",
        module_origin=str(origin),
    )
    require(
        origin.is_relative_to(Path(sys.prefix).resolve()),
        "The imported package does not belong to this virtual environment",
        module_origin=str(origin),
        environment=sys.prefix,
    )
    distribution = importlib.metadata.distribution("tsfm-lab")
    direct = distribution.read_text("direct_url.json")
    require(
        not direct or not json.loads(direct).get("dir_info", {}).get("editable", False),
        "Editable installations do not verify wheel contents",
    )
    expected = Path(sys.executable).parent / "tsfm"
    discovered = shutil.which("tsfm")
    require(
        expected.is_file() and os.access(expected, os.X_OK),
        "The installed wheel did not generate this environment's tsfm executable",
        expected_launcher=str(expected),
    )
    require(
        discovered is not None and Path(discovered).resolve() == expected.resolve(),
        "PATH selects a different tsfm executable; prepend this environment's bin directory",
        expected_launcher=str(expected),
        discovered_launcher=discovered,
    )
    entries = [
        entry
        for entry in distribution.entry_points
        if entry.group == "console_scripts" and entry.name == "tsfm"
    ]
    require(
        len(entries) == 1 and entries[0].value == "tsfm_lab.__main__:main",
        "Installed console entry point is missing or incorrect",
    )
    configuration = Path(sys.prefix) / "pyvenv.cfg"
    system_site = None
    if configuration.is_file():
        for line in configuration.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip().lower() == "include-system-site-packages":
                system_site = value.strip().lower() == "true"
    return expected, {
        "version": distribution.version,
        "module_origin": str(origin),
        "launcher": str(expected),
        "python": sys.version.split()[0],
        "interpreter": sys.executable,
        "isolated": True,
        "isolated_install": True,
        "virtual_environment": sys.prefix,
        "system_site_packages": system_site,
        "optional_modules_available": {
            name: importlib.util.find_spec(name) is not None for name in _OPTIONAL
        },
    }


def canonical_rows(rows):
    result = []
    for row in rows:
        require(
            set(row) == _COLUMNS,
            "Exported prediction columns differ from the public schema",
            columns=sorted(row),
        )
        origin = datetime.fromisoformat(str(row["origin"]).replace("Z", "+00:00"))
        target = datetime.fromisoformat(
            str(row["target_timestamp"]).replace("Z", "+00:00")
        )
        require(
            origin.tzinfo is not None and target.tzinfo is not None,
            "Exported timestamps lost their timezone",
        )
        lead, actual, forecast = (
            int(row["lead"]),
            float(row["actual"]),
            float(row["prediction"]),
        )
        require(
            lead > 0 and math.isfinite(actual) and math.isfinite(forecast),
            "Exported numerical values are invalid",
        )
        result.append(
            (
                str(row["series_id"]),
                str(row["adapter_id"]),
                origin.astimezone(UTC).isoformat(),
                target.astimezone(UTC).isoformat(),
                lead,
                actual,
                forecast,
                str(row["point_statistic"]),
            )
        )
    return result


class Verifier:
    def __init__(self, launcher, root):
        self.launcher, self.root = launcher, root
        self.check = "initialization"
        self.passed = []
        self.deadline = time.monotonic() + 120
        self.environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME"}
        }
        self.environment.update(
            NO_COLOR="1",
            TERM="dumb",
            PYTHONNOUSERSITE="1",
            PYTHONSAFEPATH="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            HF_HOME=str(root / "unused-hub"),
        )

    def command(self, *arguments, expected=0, python=False):
        command = (
            [sys.executable, "-I", *arguments]
            if python
            else [str(self.launcher), *map(str, arguments)]
        )
        remaining = self.deadline - time.monotonic()
        require(
            remaining > 0, "Distribution verification exceeded its 120-second budget"
        )
        try:
            result = subprocess.run(
                command,
                cwd=self.root,
                env=self.environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=min(30, remaining),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise VerificationError(
                "Installed command timed out",
                command=command,
                timeout_seconds=min(30, remaining),
            ) from exc
        require(
            result.returncode == expected,
            "Installed command returned an unexpected exit code",
            command=command,
            expected_exit=expected,
            actual_exit=result.returncode,
            stdout_tail=result.stdout[-2000:],
            stderr_tail=result.stderr[-3000:],
        )
        require(
            "Traceback (most recent call last)" not in result.stderr,
            "Installed command exposed a traceback",
            stderr_tail=result.stderr[-3000:],
        )
        return result

    def json_command(self, *arguments):
        return strict_json(self.command(*arguments, "--json").stdout)

    def done(self, name):
        self.passed.append(name)

    def config(self, adapter, dataset, workspace):
        model = {"source": "builtin", "adapter_id": adapter}
        if adapter == "seasonal_naive":
            model["options"] = {"seasonal_period": 4}
        return {
            "schema_version": 1,
            "name": f"installed-{adapter}",
            "model": model,
            "dataset": {
                "path": str(dataset),
                "timestamp_column": "timestamp",
                "target_column": "demand",
                "frequency": "1h",
                "units": "synthetic_units",
            },
            "evaluation": {
                "context_length": 8,
                "horizon": 4,
                "stride": 4,
                "seasonal_period": 4,
                "evaluation_start": "2026-01-01T16:00:00Z",
                "evaluation_end": "2026-01-02T08:00:00Z",
                "metrics": ["mae", "rmse", "mase"],
            },
            "output": {"directory": str(workspace)},
        }

    def save_config(self, name, configuration):
        # JSON syntax is a YAML subset, preserving paths exactly on both platforms.
        path = self.root / f"{name}.yaml"
        path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
        return path

    def run(self, evidence):
        self.entry_points(evidence)
        self.catalog()
        dataset, start, values = self.dataset()
        workspace = self.root / "workspace"
        identifiers = self.baselines(dataset, workspace)
        self.records(workspace, identifiers)
        self.exports(workspace, identifiers, start, values)
        self.invalid_configuration(dataset)
        self.optional_boundaries()
        return {
            "baseline_run_ids": identifiers,
            "forecast_cases_per_run": 16,
            "checks": self.passed,
        }

    def entry_points(self, evidence):
        self.check = "entry_points"
        require(
            self.command("--version").stdout.strip() == evidence["version"],
            "CLI and installed package versions disagree",
        )
        for arguments in (("--help",), ()):
            output = self.command(*arguments).stdout
            require(
                "Usage:" in output
                and all(
                    name in output
                    for name in (
                        "run",
                        "models",
                        "datasets",
                        "experiments",
                        "compare",
                        "export",
                    )
                )
                and not any(
                    marker in output
                    for marker in (
                        "Session closed",
                        "TSFM LAB  ",
                        "/demo to try it",
                        "› ",
                    )
                ),
                "Redirected help/no-argument invocation did not return command help",
                stdout_tail=output[-2000:],
            )
        refusal = self.command("ui", expected=2)
        require(
            "terminal" in refusal.stderr.lower(),
            "Redirected UI command did not explain its terminal requirement",
        )
        self.done("installed_entry_points_and_redirected_routing")

    def catalog(self):
        self.check = "catalog"
        catalog = self.json_command("models", "list")
        require(
            {item["adapter_id"] for item in catalog}
            == {"last_value", "seasonal_naive", "chronos_bolt", "timesfm_2p5"},
            "Installed model catalog differs from supported adapters",
        )
        for identifier, adapter in (
            ("last_value", "last_value"),
            ("amazon/chronos-bolt-tiny", "chronos_bolt"),
            ("google/timesfm-2.5-200m-pytorch", "timesfm_2p5"),
        ):
            require(
                self.json_command("models", "info", identifier)["adapter_id"]
                == adapter,
                "Model information lookup returned a different adapter",
            )
        cache = self.root / "empty-cache"
        require(
            self.json_command("models", "cache", "--cache-dir", cache) == []
            and not cache.exists(),
            "Empty cache inspection created files or required downloads",
        )
        self.done("offline_catalog_without_checkpoints")

    def dataset(self):
        self.check = "bundled_dataset"
        raw = (
            importlib.resources.files("tsfm_lab.resources")
            .joinpath("seasonal.csv")
            .read_bytes()
        )
        dataset = self.root / "seasonal.csv"
        dataset.write_bytes(raw)
        source = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        require(
            len(source) == 32,
            "Bundled demonstration dataset has a different observation count",
        )
        start = datetime(2026, 1, 1, tzinfo=UTC)
        values = []
        for index, row in enumerate(source):
            expected_value = 10 + index // 4 + (0, 5, -3, 2)[index % 4]
            require(
                datetime.fromisoformat(row["timestamp"])
                == start + timedelta(hours=index)
                and float(row["demand"]) == expected_value,
                "Bundled synthetic data no longer matches its documented generation formula",
            )
            values.append(expected_value)
        self.done("bundled_resource_and_independent_fixture_arithmetic")

        return dataset, start, values

    def baselines(self, dataset, workspace):
        self.check = "baseline_runs"
        identifiers = []
        for adapter, expected in (
            ("last_value", (2.5, math.sqrt(8.5), 2.5)),
            ("seasonal_naive", (1, 1, 1)),
        ):
            response = self.json_command(
                "run",
                self.save_config(adapter, self.config(adapter, dataset, workspace)),
            )
            require(
                response["status"] == "completed",
                "A baseline experiment did not complete",
            )
            require(
                response["counts"]
                == {
                    "attempted_windows": 4,
                    "successful_windows": 4,
                    "failed_windows": 0,
                    "forecast_cases": 16,
                    "unique_target_timestamps": 16,
                },
                "Baseline window/case counts changed",
            )
            require(
                all(
                    math.isclose(
                        response["metrics"][name]["value"],
                        value,
                        rel_tol=1e-12,
                        abs_tol=0,
                    )
                    for name, value in zip(
                        ("mae", "rmse", "mase"), expected, strict=True
                    )
                ),
                "Baseline metrics disagree with independent arithmetic",
                adapter=adapter,
                metrics=response["metrics"],
            )
            identifiers.append(response["run_id"])
        self.done("two_baselines_four_windows_sixteen_cases")

        return identifiers

    def records(self, workspace, identifiers):
        self.check = "experiment_records"
        history = self.json_command("experiments", "list", "--workspace", workspace)
        require(
            {item["run_id"] for item in history} == set(identifiers)
            and all(item["status"] == "completed" for item in history),
            "Installed experiment history omitted or corrupted a baseline run",
        )
        for identifier in identifiers:
            record = self.json_command(
                "experiments", "show", identifier, "--workspace", workspace
            )
            require(
                record["manifest"]["status"]
                == record["result"]["status"]
                == "completed",
                "Reloaded experiment is incomplete",
            )
        comparison = self.json_command(
            "compare", *identifiers, "--workspace", workspace
        )
        require(
            comparison["compatible"] is True and len(comparison["runs"]) == 2,
            "Controlled baseline comparison failed",
        )
        self.done("history_show_and_controlled_comparison")

    def exports(self, workspace, identifiers, start, values):
        self.check = "prediction_exports"
        exports, csv_rows = {}, None
        for format in ("csv", "json", "parquet"):
            path = self.root / f"forecasts.{format}"
            result = self.json_command(
                "export", identifiers[0], "--workspace", workspace, "--output", path
            )
            require(
                result["rows"] == 16
                and hashlib.sha256(path.read_bytes()).hexdigest() == result["sha256"],
                "Export count or checksum is incorrect",
                format=format,
            )
            exports[format] = path
            if format == "csv":
                csv_rows = canonical_rows(
                    list(csv.DictReader(io.StringIO(path.read_text())))
                )
            elif format == "json":
                require(
                    canonical_rows(strict_json(path.read_text())) == csv_rows,
                    "JSON and CSV exports disagree",
                )
        expected = []
        for first in (16, 20, 24, 28):
            for lead in range(1, 5):
                expected.append(
                    (
                        "series_0",
                        "last_value",
                        (start + timedelta(hours=first - 1)).isoformat(),
                        (start + timedelta(hours=first + lead - 1)).isoformat(),
                        lead,
                        float(values[first + lead - 1]),
                        float(values[first - 1]),
                        "deterministic",
                    )
                )
        require(
            csv_rows == expected,
            "CSV predictions or temporal alignment differ from the synthetic reference",
        )
        reload_code = "from tsfm_lab.parquet import load_parquet; import json,sys; d=load_parquet(sys.argv[1], timestamp_column='target_timestamp', target_column='prediction', frequency='1h'); print(json.dumps({'timestamps':[t.isoformat() for t in d.timestamps], 'values':d.values.tolist()}))"
        reloaded = strict_json(
            self.command("-c", reload_code, str(exports["parquet"]), python=True).stdout
        )
        require(
            reloaded
            == {
                "timestamps": [row[3] for row in expected],
                "values": [row[6] for row in expected],
            },
            "Parquet reload lost prediction values or target timestamps",
        )
        seasonal_path = self.root / "seasonal-predictions.json"
        self.json_command(
            "export",
            identifiers[1],
            "--workspace",
            workspace,
            "--output",
            seasonal_path,
        )
        seasonal = canonical_rows(strict_json(seasonal_path.read_text()))
        seasonal_expected = [
            (row[0], "seasonal_naive", *row[2:6], row[5] - 1, row[7])
            for row in expected
        ]
        require(
            seasonal == seasonal_expected,
            "Seasonal forecasts do not follow the constructed seasonal baseline",
        )
        before = exports["csv"].read_bytes()
        refusal = self.command(
            "export",
            identifiers[0],
            "--workspace",
            workspace,
            "--output",
            exports["csv"],
            expected=1,
        )
        require(
            not refusal.stdout
            and "already exists" in refusal.stderr
            and exports["csv"].read_bytes() == before,
            "Export overwrite refusal did not preserve the existing file",
        )
        self.done("csv_json_parquet_export_reload_and_overwrite_refusal")

    def invalid_configuration(self, dataset):
        self.check = "invalid_configuration"
        invalid_workspace = self.root / "invalid-workspace"
        invalid = self.config("last_value", dataset, invalid_workspace)
        invalid["evaluation"]["horizon"] = 0
        failure = self.command("run", self.save_config("invalid", invalid), expected=2)
        require(
            not failure.stdout
            and "horizon" in failure.stderr
            and not invalid_workspace.exists(),
            "Invalid configuration created a workspace or failed without an actionable error",
        )
        self.done("invalid_configuration_fails_before_artifacts")

    def optional_boundaries(self):
        self.check = "optional_dependency_boundaries"
        probe = _BLOCK_OPTIONAL + """
import tsfm_lab.repl.session
print(json.dumps({'optional_loaded': sorted(blocked.intersection(sys.modules))}))
"""
        require(
            strict_json(self.command("-c", probe, python=True).stdout)
            == {"optional_loaded": []},
            "Interactive prompt metadata required optional model or Textual runtimes",
        )
        runtime_probe = _BLOCK_OPTIONAL + """
from importlib import import_module
for module, extra in [('chronos_bolt', 'chronos'), ('timesfm', 'timesfm')]:
    try:
        import_module('tsfm_lab.models.' + module)._runtime()
    except ImportError as exc:
        assert '[' + extra + ']' in str(exc)
    else:
        raise AssertionError('Optional runtime unexpectedly loaded')
print(json.dumps({'missing_runtime': 'actionable_error'}))
"""
        require(
            strict_json(self.command("-c", runtime_probe, python=True).stdout)
            == {"missing_runtime": "actionable_error"},
            "Missing optional model runtime did not fail clearly",
        )
        self.done("prompt_without_textual_and_missing_model_runtime")


def main():
    started, verifier, evidence = time.monotonic(), None, {}
    try:
        launcher, evidence = installed_environment()
        with tempfile.TemporaryDirectory(prefix="tsfm-distribution-") as directory:
            verifier = Verifier(launcher, Path(directory))
            details = verifier.run(evidence)
        result = {"status": "passed", "installed": evidence, **details}
        code = 0
    except Exception as exc:
        result = {
            "status": "failed",
            "check": verifier.check if verifier else "installed_environment",
            "message": str(exc),
            "details": getattr(exc, "details", {}),
            "passed_checks": verifier.passed if verifier else [],
        }
        code = 1
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

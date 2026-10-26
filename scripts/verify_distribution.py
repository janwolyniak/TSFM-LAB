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
    "tirex2",
    "flashrnn",
    "mlstm_kernels",
    "tsfm_public",
    "httpx",
    "jwt",
    "cryptography",
    "keyring",
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
_FINANCE_STRATEGY_FIELDS = (
    "equity",
    "net_return",
    "traded_notional",
    "entry_fee",
    "exit_fee",
    "entry_slippage",
    "exit_slippage",
    "short_financing",
)
_FINANCE_COLUMNS = {
    "origin",
    "open_timestamp",
    "close_timestamp",
    "signal_position",
    "cash_equity",
    "buy_and_hold_equity",
} | {f"strategy_{name}" for name in _FINANCE_STRATEGY_FIELDS}
_BLOCK_OPTIONAL = """import importlib.abc
import json
import sys

blocked = {'torch', 'chronos', 'timesfm', 'transformers', 'huggingface_hub', 'textual', 'tirex2', 'flashrnn', 'mlstm_kernels', 'tsfm_public', 'httpx', 'jwt', 'cryptography', 'keyring'}

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
        self.launcher, self.root = launcher, root.resolve()
        root = self.root
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
        self.finance()
        self.native_point_fixture()
        self.usage_profiles()
        self.connections()
        self.managed_worker(dataset)
        self.optional_boundaries()
        return {
            "baseline_run_ids": identifiers,
            "forecast_cases_per_run": 16,
            "finance_cases": 4,
            "native_point_fixture_cases": 16,
            "usage_profiles": 5,
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
            == {
                "last_value",
                "seasonal_naive",
                "chronos_bolt",
                "timesfm_2p5",
                "chronos_2",
                "tirex_2",
                "ttm_r2",
            },
            "Installed model catalog differs from supported adapters",
        )
        for entry in catalog:
            adapter = entry["adapter_id"]
            expected = (
                "deterministic"
                if adapter in {"last_value", "seasonal_naive"}
                else "native_point" if adapter == "ttm_r2" else "median"
            )
            require(
                entry["point_statistic"] == expected
                and entry["supported_tasks"]
                == (
                    ["native_point"]
                    if adapter == "ttm_r2"
                    else ["median", "native_point"]
                ),
                "Installed catalog changed task or actual output semantics",
                adapter=adapter,
            )
        for identifier, adapter in (
            ("last_value", "last_value"),
            ("amazon/chronos-bolt-tiny", "chronos_bolt"),
            ("amazon/chronos-bolt-base", "chronos_bolt"),
            ("google/timesfm-2.5-200m-pytorch", "timesfm_2p5"),
            ("autogluon/chronos-2-small", "chronos_2"),
            ("amazon/chronos-2", "chronos_2"),
            ("NX-AI/TiRex-2", "tirex_2"),
            ("NX-AI/TiRex-2-fevbench", "tirex_2"),
            ("NX-AI/TiRex-2-gifteval-zs", "tirex_2"),
            ("NX-AI/TiRex-2-gifteval-pretrain", "tirex_2"),
        ):
            require(
                self.json_command("models", "info", identifier)["adapter_id"]
                == adapter,
                "Model information lookup returned a different adapter",
            )
        base = self.json_command("models", "info", "amazon/chronos-bolt-base")[
            "selected_checkpoint"
        ]
        require(
            base["reference_revision"] == "5d9f166d69f47aef3401367a7b842e78fe97b121"
            and base["weight_bytes"] == 821203576
            and base["project_inference_status"] == "not_recorded"
            and len(base["representative_checks"]) == 2,
            "Installed Base metadata or representative verification is incorrect",
        )
        small = self.json_command("models", "info", "autogluon/chronos-2-small")[
            "selected_checkpoint"
        ]
        require(
            small["reference_revision"] == "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a"
            and small["weight_bytes"] == 111749048
            and small["project_inference_status"] == "recorded"
            and len(small["representative_checks"]) == 1,
            "Installed Chronos-2 Small verification evidence is incorrect",
        )
        standard = self.json_command("models", "info", "amazon/chronos-2")
        require(
            standard["capabilities"]["max_context"] == 8192
            and standard["capabilities"]["max_horizon"] == 1024
            and standard["selected_checkpoint"]["reference_revision"]
            == "29ec3766d36d6f73f0696f85560a422f50e8498c"
            and standard["selected_checkpoint"]["weight_bytes"] == 477930472
            and standard["selected_checkpoint"]["project_inference_status"]
            == "not_recorded"
            and len(standard["selected_checkpoint"]["representative_checks"]) == 1,
            "Installed Chronos-2 metadata or checkpoint status is incorrect",
        )
        tirex_profiles = (
            ("NX-AI/TiRex-2", "30788a311699acba7e6393ad9700ea7b7cb9bb2d", 2048, 320),
            (
                "NX-AI/TiRex-2-fevbench",
                "acd6153b8ee2179f38cfd3cc4fd64a3c045a424f",
                2048,
                320,
            ),
            (
                "NX-AI/TiRex-2-gifteval-zs",
                "c94872d1c8caacbe7755cbfa01bc5c4c5c5ddb7f",
                8192,
                928,
            ),
            (
                "NX-AI/TiRex-2-gifteval-pretrain",
                "54f88ac06ccfba9fe3e727aa8856b39775481f0f",
                8192,
                928,
            ),
        )
        for repository, revision, context, horizon in tirex_profiles:
            value = self.json_command("models", "info", repository)
            variant = value["selected_checkpoint"]
            notices = "\n".join(value["runtime_notices"])
            require(
                value["capabilities"]["max_context"] == context
                and value["capabilities"]["max_horizon"] == horizon
                and variant["reference_revision"] == revision
                and variant["project_inference_status"]
                == ("recorded" if repository == "NX-AI/TiRex-2" else "not_recorded")
                and value["license"] == "apache-2.0"
                and "Built with technology from NXAI" in notices
                and "NXAI Community License" in notices
                and value["runtime_requirements"]["huggingface-hub"] == "==0.35.3",
                "Installed TiRex-2 profile bounds, verification, or runtime notices are incorrect",
                repository=repository,
            )
        for variant, revision, context, weight_bytes in (
            ("512-96-r2", "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f", 512, 3240592),
            ("1024-96-r2", "1a382618f3c5b63836281ad632e54724e55e85a5", 1024, 11879480),
            ("1536-96-r2", "f6ed09d67cf42e4877f7fa33d98326911385693d", 1536, 12344144),
        ):
            value = self.json_command(
                "models",
                "info",
                "ibm-granite/granite-timeseries-ttm-r2",
                "--variant",
                variant,
            )
            selected = value["selected_checkpoint"]
            require(
                value["adapter_id"] == "ttm_r2"
                and value["capabilities"]["min_context"]
                == value["capabilities"]["max_context"]
                == context
                and value["capabilities"]["max_horizon"] == 96
                and value["point_statistic"] == "native_point"
                and value["supported_tasks"] == ["native_point"]
                and value["frequency_policy"] == "minutely_hourly"
                and selected["variant"] == variant
                and selected["reference_revision"] == revision
                and selected["weight_bytes"] == weight_bytes
                and value["runtime_requirements"]["granite-tsfm"] == "==0.3.4",
                "Installed TTM variant, native semantics, or pinned metadata is incorrect",
                variant=variant,
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

    def native_point_fixture(self):
        self.check = "native_point_artifact_fixture"
        # Inert weights and a last-value stand-in exercise packaging and records only.
        probe = _BLOCK_OPTIONAL + """
from importlib.resources import files
from pathlib import Path
import numpy as np
from tsfm_lab.contracts import EvaluationSpec, ForecastOutput, fixed_interval
from tsfm_lab.data import load_csv
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.models.baselines import LastValue
from tsfm_lab.models.checkpoints import inspect_local_checkpoint
from tsfm_lab.models.registry import adapter_profile
from tsfm_lab.models.ttm import TTM
from tsfm_lab.models.ttm_profile import _configuration

root = Path(sys.argv[1])
root.mkdir()
source = root / 'ttm-seasonal.csv'
source.write_bytes(files('tsfm_lab.resources').joinpath('ttm-seasonal.csv').read_bytes())
dataset = load_csv(source, timestamp_column='timestamp', target_column='demand',
                   frequency='1h', units='synthetic_units')
assert len(dataset.values) == 1600 and dataset.values[0] == 52.0
checkpoint_path = root / 'inert-checkpoint'
checkpoint_path.mkdir()
(checkpoint_path / 'config.json').write_text(json.dumps(_configuration(512)))
(checkpoint_path / 'model.safetensors').write_bytes(b'inert packaging fixture only')
checkpoint = inspect_local_checkpoint(checkpoint_path)

class FixtureTTM(TTM):
    def load(self):
        pass

    def predict(self, requests):
        return tuple(ForecastOutput(
            request_id=item.request_id, series_id=item.series_id, origin=item.origin,
            target_timestamps=item.target_timestamps,
            point=np.full(len(item.target_timestamps), item.context_values[-1]),
            adapter_id=self.adapter_id, point_statistic='native_point',
        ) for item in requests)

    def environment_metadata(self):
        return {name: 'packaging_fixture' for name, _ in
                adapter_profile('ttm_r2').runtime_requirements}

spec = EvaluationSpec(512, 4, dataset.timestamps[-16],
                      dataset.timestamps[-1] + fixed_interval(dataset.frequency),
                      stride=4, seasonal_period=24, point_statistic='native_point')
store = ExperimentStore(root / 'workspace')
identifiers = []
for name, model in [('TTM metadata fixture only', FixtureTTM(checkpoint)),
                    ('Native deterministic fixture', LastValue())]:
    result = run_experiment(dataset, model, spec)
    identifier = store.create(name, result.configuration, '# No model inference')
    store.complete(identifier, result.to_dict(), result.predictions)
    reloaded, frame = store.load_result(identifier)
    assert reloaded['configuration']['evaluation']['point_statistic'] == 'native_point'
    assert len(frame) == 16
    identifiers.append(identifier)
record = store.load_result(identifiers[0])[0]['configuration']['model']['options']['checkpoint']
assert record['schema_version'] == 2 and record['quantiles'] == []
assert record['point_forecast']['statistic'] == 'native_point'
assert record['point_forecast']['training_loss'] == 'mse'
checkpoint_path.rename(root / 'checkpoint-unavailable')
source.rename(root / 'source-unavailable.csv')
print(json.dumps({'run_ids': identifiers, 'optional_loaded': sorted(blocked.intersection(sys.modules))}))
"""
        directory = self.root / "native-point-fixture"
        identity = strict_json(
            self.command("-c", probe, str(directory), python=True).stdout
        )
        identifiers = identity["run_ids"]
        require(
            len(identifiers) == 2 and identity["optional_loaded"] == [],
            "Native-point fixture required optional model runtimes",
        )
        workspace = directory / "workspace"
        saved = self.json_command(
            "experiments", "show", identifiers[0], "--workspace", workspace
        )["result"]
        checkpoint = saved["configuration"]["model"]["options"]["checkpoint"]
        require(
            checkpoint["schema_version"] == 2
            and checkpoint["point_forecast"]["statistic"] == "native_point"
            and saved["configuration"]["evaluation"]["point_statistic"]
            == "native_point",
            "Saved TTM metadata fixture lost its schema or point semantics",
        )
        exported = directory / "native-predictions.json"
        response = self.json_command(
            "export", identifiers[0], "--workspace", workspace, "--output", exported
        )
        rows = canonical_rows(strict_json(exported.read_text()))
        require(
            response["rows"] == len(rows) == 16
            and all(row[1] == "ttm_r2" and row[7] == "native_point" for row in rows),
            "Native-point export changed case counts or output semantics",
        )
        comparison = self.json_command(
            "compare", *identifiers, "--workspace", workspace
        )
        require(
            comparison["comparison_policy"] == "controlled_native_point_v1"
            and comparison["mixed_point_statistics"] is True
            and [item["output_statistics"] for item in comparison["runs"]]
            == [["native_point"], ["deterministic"]],
            "Installed generic comparison lost explicit mixed output meanings",
        )
        self.done("schema2_native_point_fixture_reload_export_without_model_runtime")

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

    def finance_inputs(self):
        directory = self.root / "finance-inputs"
        directory.mkdir()
        prices = (100, 102, 101, 103, 100, 105, 104, 106)
        start = datetime(2026, 1, 1, tzinfo=UTC)
        dataset, bars = directory / "closes.csv", directory / "ohlc.csv"
        with dataset.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(("timestamp", "close"))
            writer.writerows(
                ((start + timedelta(hours=index)).isoformat(), price)
                for index, price in enumerate(prices)
            )
        with bars.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(
                ("open_timestamp", "close_timestamp", "open", "high", "low", "close")
            )
            for index, close in enumerate(prices):
                closing = start + timedelta(hours=index)
                opening = prices[index - 1] if index else prices[0]
                writer.writerow(
                    (
                        (closing - timedelta(minutes=45)).isoformat(),
                        closing.isoformat(),
                        opening,
                        max(opening, close) + 1,
                        min(opening, close) - 1,
                        close,
                    )
                )
        configuration = self.config("seasonal_naive", dataset, directory / "workspace")
        configuration["model"]["options"]["seasonal_period"] = 2
        configuration["dataset"].update(
            target_column="close", units="synthetic_quote_per_asset"
        )
        configuration["evaluation"].update(
            context_length=2,
            horizon=1,
            stride=1,
            seasonal_period=2,
            evaluation_start=(start + timedelta(hours=4)).isoformat(),
            evaluation_end=(start + timedelta(hours=8)).isoformat(),
        )
        path = directory / "source.yaml"
        path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
        return directory, path, bars, directory / "workspace"

    def finance_arithmetic(self, report, identifier):
        evaluation = report["evaluation"]
        strategy = evaluation["strategy"]
        # Independently calculated from the fixture's four explicit forecast cases.
        expected = (
            1027.4274027598157,
            1077.2895110990087,
            1085.719300113609,
            1085.719300113609,
        )
        require(
            report["parent"]["run_id"] == identifier
            and evaluation["cases"] == 4
            and evaluation["signal_positions"] == [-1, 1, -1, 0]
            and [row["actual_close"] for row in report["cases"]]
            == [100, 105, 104, 106],
            "Finance report changed the saved parent, cases, or causal signals",
        )
        require(
            len(strategy["curve"]) == 4
            and all(
                math.isclose(row["equity"], value, rel_tol=1e-12)
                for row, value in zip(strategy["curve"], expected, strict=True)
            ),
            "Costed finance equity differs from the independent fixture arithmetic",
        )
        for name, value in (
            ("fees", 3.1085335685808443),
            ("slippage", 1.2434134274323377),
            ("short_financing", 0.6227509276803264),
        ):
            require(
                math.isclose(strategy["metrics"][name], value, rel_tol=1e-12),
                f"Finance {name} differs from explicit-cost arithmetic",
            )
        benchmarks = evaluation["benchmarks"]
        require(
            benchmarks["cash"]["metrics"]["final_capital"] == 1000
            and benchmarks["cash"]["metrics"]["total_costs"] == 0
            and math.isclose(
                benchmarks["buy_and_hold"]["metrics"]["final_capital"],
                1027.6864447314064,
                rel_tol=1e-12,
            ),
            "Costed buy-and-hold or zero-interest cash benchmark is incorrect",
        )

    def finance_export(self, report_path, report):
        path = self.root / "finance-equity.csv"
        identity = self.json_command("finance", "export", report_path, "--output", path)
        raw = path.read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
        rows = list(reader)
        require(
            identity["rows"] == len(rows) == 4
            and identity["sha256"] == hashlib.sha256(raw).hexdigest()
            and len(reader.fieldnames) == len(_FINANCE_COLUMNS)
            and set(reader.fieldnames) == _FINANCE_COLUMNS,
            "Finance export schema, count, or checksum is incorrect",
        )
        evaluation = report["evaluation"]
        for index, row in enumerate(rows):
            case = report["cases"][index]
            require(
                all(
                    row[name] == case[name]
                    for name in ("origin", "open_timestamp", "close_timestamp")
                )
                and int(row["signal_position"])
                == evaluation["signal_positions"][index],
                "Finance export lost case timing or signal alignment",
            )
            for name in _FINANCE_STRATEGY_FIELDS:
                require(
                    float(row[f"strategy_{name}"])
                    == evaluation["strategy"]["curve"][index][name],
                    "Finance CSV disagrees with checked strategy costs or equity",
                )
            for path_name, column in (
                ("cash", "cash_equity"),
                ("buy_and_hold", "buy_and_hold_equity"),
            ):
                series = evaluation["benchmarks"][path_name]
                require(
                    float(row[column]) == series["curve"][index]["equity"],
                    "Finance CSV disagrees with the checked equity report",
                )
        refused = self.command(
            "finance", "export", report_path, "--output", path, expected=1
        )
        require(
            not refused.stdout
            and "already exists" in refused.stderr
            and path.read_bytes() == raw,
            "Finance export overwrote or damaged an existing destination",
        )

    def finance(self):
        self.check = "finance_workflow"
        directory, configuration, bars, workspace = self.finance_inputs()
        parent = self.json_command("run", configuration)
        require(
            parent["status"] == "completed" and parent["counts"]["forecast_cases"] == 4,
            "Finance fixture baseline did not retain four lead-1 cases",
        )
        report_path = self.root / "finance-report.json"
        arguments = (
            "finance",
            "backtest",
            parent["run_id"],
            "--workspace",
            workspace,
            "--source-config",
            configuration,
            "--ohlc",
            bars,
            "--price-target",
            "--initial-capital",
            "1000",
            "--threshold-bps",
            "100",
            "--fee-bps",
            "5",
            "--slippage-bps",
            "2",
            "--short-financing-bps",
            "3",
            "--signal-latency-seconds",
            "60",
            "--output",
            report_path,
        )
        report = self.json_command(*arguments)
        raw = report_path.read_bytes()
        require(
            strict_json(raw.decode("utf-8")) == report,
            "Saved finance report differs from backtest JSON output",
        )
        self.finance_arithmetic(report, parent["run_id"])
        refused = self.command(*arguments, expected=1)
        require(
            not refused.stdout
            and "already exists" in refused.stderr
            and report_path.read_bytes() == raw,
            "Finance backtest overwrote or damaged an existing report",
        )
        directory.rename(self.root / "finance-sources-unavailable")
        require(
            not directory.exists() and not workspace.exists(),
            "Finance fixture source paths unexpectedly remain available",
        )
        require(
            self.json_command("finance", "show", report_path) == report,
            "Finance reload required unavailable source data or changed the report",
        )
        self.finance_export(report_path, report)
        self.done("finance_backtest_self_contained_reload_export_and_overwrite_refusal")

    def usage_profiles(self):
        self.check = "usage_profiles"
        directory = self.root / "usage-profiles"
        probe = _BLOCK_OPTIONAL + """
from importlib.resources import files
from pathlib import Path
root = Path(sys.argv[1])
root.mkdir()
for profile in ('generic', 'finance', 'energy', 'temperature', 'weather'):
    name = 'seasonal.csv' if profile == 'generic' else profile + '-demo.csv'
    raw = files('tsfm_lab.resources').joinpath(name).read_bytes()
    (root / (profile + '.csv')).write_bytes(raw)
print(json.dumps({'copied': 5, 'optional_loaded': sorted(blocked.intersection(sys.modules))}))
"""
        identity = strict_json(
            self.command("-c", probe, str(directory), python=True).stdout
        )
        require(
            identity == {"copied": 5, "optional_loaded": []},
            "Usage examples required optional runtimes",
        )
        choices = {
            "generic": ("demand", "synthetic_units", {}),
            "finance": ("close", "toy_quote_per_asset", {"target_kind": "close_price"}),
            "energy": ("power", "kW", {"measurement": "power"}),
            "temperature": ("temperature", "degC", {}),
            "weather": (
                "wind_speed",
                "m/s",
                {
                    "variable": "wind_speed",
                    "station": "synthetic-station",
                    "measurement_semantics": "instantaneous",
                },
            ),
        }
        workspace = directory / "workspace"
        identifiers = {}
        temperature_config = None
        for profile, (column, units, options) in choices.items():
            configuration = self.config(
                "last_value", directory / f"{profile}.csv", workspace
            )
            configuration["workflow"] = {
                "schema_version": 1,
                "profile": profile,
                "options": options,
            }
            configuration["dataset"].update(target_column=column, units=units)
            path = self.save_config(f"usage-{profile}", configuration)
            summary = self.json_command("run", path)
            identifiers[profile] = summary["run_id"]
            result = self.json_command(
                "experiments", "show", summary["run_id"], "--workspace", workspace
            )["result"]
            require(
                result["configuration"]["workflow"] == configuration["workflow"],
                "Saved usage metadata changed",
            )
            require(
                result["counts"]["forecast_cases"] == 16,
                "Usage example dropped forecast cases",
            )
            if profile == "temperature":
                temperature_config = configuration
                output = directory / "temperature-predictions.json"
                self.json_command(
                    "export",
                    summary["run_id"],
                    "--workspace",
                    workspace,
                    "--output",
                    output,
                )
                rows = strict_json(output.read_text())
                temperature_rows = rows
                with (directory / "temperature.csv").open(newline="") as source:
                    source_actuals = [
                        float(row["temperature"]) for row in csv.DictReader(source)
                    ][16:32]
                require(
                    [row["actual"] for row in rows] == source_actuals
                    and any(row["actual"] < 0 for row in rows),
                    "Temperature values were clipped or transformed",
                )
        # Relabel the exact same input as Generic: point values and eligibility stay equal.
        temperature_config["workflow"] = {
            "schema_version": 1,
            "profile": "generic",
            "options": {},
        }
        generic = self.json_command(
            "run", self.save_config("usage-relabelled", temperature_config)
        )
        comparison = self.json_command(
            "compare",
            identifiers["temperature"],
            generic["run_id"],
            "--workspace",
            workspace,
        )
        require(
            comparison["compatible"]
            and comparison["runs"][0]["metrics"] == comparison["runs"][1]["metrics"],
            "Usage metadata changed numerical comparison",
        )
        output = directory / "generic-temperature-predictions.json"
        self.json_command(
            "export", generic["run_id"], "--workspace", workspace, "--output", output
        )
        relabelled_rows = strict_json(output.read_text())
        require(
            [(row["actual"], row["prediction"]) for row in relabelled_rows]
            == [(row["actual"], row["prediction"]) for row in temperature_rows],
            "Relabelling a profile changed individual forecast values",
        )
        for profile in choices:
            (directory / f"{profile}.csv").rename(
                directory / f"{profile}-unavailable.csv"
            )
            self.json_command(
                "experiments", "show", identifiers[profile], "--workspace", workspace
            )
        self.done("five_usage_profiles_saved_reload_and_metadata_only_comparison")

    def connections(self):
        self.check = "fake_connections_and_assistance_without_provider_dependencies"
        directory = self.root / "connection-metadata"
        require(
            self.json_command("connections", "list", "--state-directory", directory)
            == [],
            "Fresh connection metadata is not empty",
        )
        require(
            not directory.exists(), "Reading empty metadata created connection state"
        )
        probe = _BLOCK_OPTIONAL + """
import time
from pathlib import Path
from types import SimpleNamespace
from importlib.resources import files
from tsfm_lab.assistant.drafts import resolve_draft
from tsfm_lab.assistant.evidence import model_evidence
from tsfm_lab.assistant.service import AssistantService
from tsfm_lab.repl.wizard import build_experiment
from tsfm_lab.connections.contracts import KINDS
from tsfm_lab.connections.credentials import Credentials
from tsfm_lab.connections.manager import ConnectionManager
from tsfm_lab.connections.state import StateStore
from tsfm_lab.connections.transport import HTTPResponse

fake_secret = 'fake-installed-connection-secret-12345678'
class Fake:
    kind = None
    calls = []
    answer = 'confirmed'
    def send(self, request, **kwargs):
        self.calls.append(request)
        if request.method == 'GET':
            payload = {
                'chatgpt_plan': {'models':[{'slug':'fake-model','visibility':'list'}]},
                'openai_api': {'data':[{'id':'fake-model'}]},
                'anthropic_api': {'data':[{'id':'fake-model'}],'has_more':False},
                'gemini_api': {'models':[{'name':'models/fake-model','supportedGenerationMethods':['generateContent']}]},
            }[self.kind]
        elif self.kind == 'anthropic_api':
            payload = {'type':'message','role':'assistant','stop_reason':'end_turn','content':[{'type':'text','text':self.answer}]}
        elif self.kind == 'gemini_api':
            payload = {'candidates':[{'finishReason':'STOP','content':{'role':'model','parts':[{'text':self.answer}]}}]}
        else:
            payload = {'status':'completed','output':[{'type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':self.answer}]}]}
        if request.method == 'POST' and self.kind == 'chatgpt_plan':
            frame = {'type':'response.completed','response':payload}
            return HTTPResponse(200, ('data: '+json.dumps(frame)+'\\n\\n').encode(),headers={'content-type':'text/event-stream'})
        return HTTPResponse(200,json.dumps(payload).encode())

transport = Fake()
manager = ConnectionManager(StateStore(Path('connection-metadata')),transport=transport,oauth_factory=lambda _:SimpleNamespace(revoke=lambda *a,**kw:True))
for kind in KINDS:
    transport.kind = kind
    transport.answer = 'confirmed'
    if kind == 'chatgpt_plan':
        item = manager.add_plan('Installed fake plan',storage='session')
        account = {'issuer':'https://auth.openai.com','subject':'fake-account','client_id':'fake-issued-client'}
        credentials = Credentials({**account,'access_token':fake_secret,'refresh_token':fake_secret+'-refresh','id_token':fake_secret+'-identity','token_type':'Bearer','scopes':['chatgpt.tokens.use.direct'],'access_expires_at':time.time()+3600,'identity_pending':False})
        manager.finish_sign_in(item['id'],SimpleNamespace(account=account,credentials=credentials,identity_pending=False),manager.state.read()['generation'])
    else:
        item = manager.add_key(kind,'Installed fake '+kind,fake_secret,storage='session')
    manager.select(item['id'])
    assert [model.identifier for model in manager.models()] == ['fake-model']
    manager.set_model('fake-model')
    preview = manager.preview_test()
    assert fake_secret not in json.dumps(preview)
    assert manager.submit_test(preview,confirmed=True) == 'confirmed'
    assistant = AssistantService(manager)
    proposal = {'schema_version':1,'state':'ready','intent':'historical_evaluation','dataset_ref':'selected_dataset','fields':{'usage_profile':'generic','adapter':'last_value','timestamp_column':'timestamp','target_column':'demand','frequency':'1h','units':'synthetic_units','context_length':'8','horizon':'4','stride':'4','mase_period':'4','evaluation_start':'2026-01-01T16:00:00Z','evaluation_end':'2026-01-02T08:00:00Z','point_statistic':'median'},'explicit_fields':[],'assumptions':[],'missing_fields':[]}
    transport.answer = json.dumps(proposal)
    preview = assistant.preview_draft('Evaluate historical demand with context 8, horizon 4 and stride 4.')
    draft = assistant.submit_draft(preview,confirmed=True)
    dataset = Path(str(files('tsfm_lab.resources').joinpath('seasonal.csv')))
    resolved = resolve_draft(draft,dataset_refs={'selected_dataset':dataset})
    assert resolved.state == 'ready'
    questions = []
    def ask(label,default='',choices=()):
        questions.append(label)
        return default
    assert build_experiment(ask,lambda _:None,workspace=Path('must-not-create'),demo_path=dataset,seed_fields=resolved.seed_fields,explicit_fields=resolved.explicit_fields) is None
    assert questions.count('Run experiment?') == 1
    assert not Path('must-not-create').exists()
    transport.answer = json.dumps({'schema_version':1,'annotations':[{'text':'This is the supported baseline model.','evidence_ids':['/data/adapter_id']}]})
    preview = assistant.preview_explanation('Explain this model.',model_evidence('last_value'))
    assert str(dataset) not in preview['text'] and fake_secret not in json.dumps(preview)
    assert assistant.submit_explanation(preview,confirmed=True)[0].text == 'This is the supported baseline model.'
    manager.disconnect(item['id'])
assert len(transport.calls) == 16
assert not manager.session._secrets
assert fake_secret not in manager.state.path.read_text()
assert not blocked.intersection(sys.modules)
print(json.dumps({'fake_routes':4,'inference_requests':12,'optional_loaded':[]}))
"""
        require(
            strict_json(self.command("-c", probe, python=True).stdout)
            == {"fake_routes": 4, "inference_requests": 12, "optional_loaded": []},
            "Installed fake connection and language workflow failed",
        )
        self.done(
            "four_fake_connections_drafts_and_explanations_without_provider_dependencies"
        )

    def managed_worker(self, dataset):
        self.check = "installed_clean_worker"
        self.save_config(
            "managed", self.config("last_value", dataset, self.root / "managed-results")
        )
        probe = _BLOCK_OPTIONAL + """
import os
import time
from pathlib import Path
from tsfm_lab import worker_entry
from tsfm_lab.config import load_config
from tsfm_lab.execution import ManagedRun

names = ('OPENAI_API_KEY','ANTHROPIC_API_KEY','GEMINI_API_KEY','TSFM_PRIVATE_SENTINEL')
for name in names:
    os.environ[name] = 'fake-installed-worker-secret'
prefix = "import json,os\\nfrom pathlib import Path\\nPath('worker-startup.json').write_text(json.dumps({name:name in os.environ for name in " + repr(names) + "}))\\n"
worker_entry.BOOTSTRAP = prefix + worker_entry.BOOTSTRAP
configuration = load_config(Path('managed.yaml'))
worker = ManagedRun(configuration)
events = []
try:
    worker.start()
    end = time.monotonic()+10
    while worker.is_running and time.monotonic()<end:
        events.extend(worker.poll())
        time.sleep(.005)
    assert not worker.is_running
    assert events[-1]['type'] == 'completed'
    assert worker._process.exitcode == 0
    assert not any(json.loads(Path('worker-startup.json').read_text()).values())
    assert all(os.environ[name] == 'fake-installed-worker-secret' for name in names)
    assert 'fake-installed-worker-secret' not in worker.log_path.read_text()
    assert not blocked.intersection(sys.modules)
finally:
    log = worker.log_path
    worker.close()
assert not log.exists()
for path in configuration.output_directory.rglob('*'):
    if path.is_file():
        assert b'fake-installed-worker-secret' not in path.read_bytes()
print(json.dumps({'completed':True,'clean_environment':True,'exitcode':0}))
"""
        require(
            strict_json(self.command("-c", probe, python=True).stdout)
            == {"completed": True, "clean_environment": True, "exitcode": 0},
            "Installed managed worker failed environment isolation or execution",
        )
        self.done(
            "installed_clean_worker_completes_without_credentials_or_optional_imports"
        )

    def optional_boundaries(self):
        self.check = "optional_dependency_boundaries"
        probe = _BLOCK_OPTIONAL + """
import tsfm_lab.repl.session
import tsfm_lab.finance_commands
print(json.dumps({'optional_loaded': sorted(blocked.intersection(sys.modules))}))
"""
        require(
            strict_json(self.command("-c", probe, python=True).stdout)
            == {"optional_loaded": []},
            "Interactive prompt metadata required optional model or Textual runtimes",
        )
        runtime_probe = _BLOCK_OPTIONAL + """
from importlib import import_module
for module, extra in [('chronos_bolt', 'chronos'), ('timesfm', 'timesfm'), ('chronos2', 'chronos'), ('tirex2', 'tirex2'), ('ttm', 'ttm')]:
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

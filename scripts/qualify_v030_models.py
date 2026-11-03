"""Qualify explicitly acquired v0.3 profiles on CPU; never install or download.

One invocation selects one profile and existing checkpoint/cache. All inference
runs in managed, reaped children. Output is new, disposable evidence, not a
forecast-accuracy or platform-support certification.
"""

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
# Only trusted controller helpers enter sys.path; package imports must use the
# ordinary wheel installation rather than this checkout's src directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
PROFILES = {
    "52-16-ft-r2.1": "5a017bde4dd785b7d366ac200c46bceea356cbf68f7c2b719f73d5b7370d7168",
    "90-30-ft-r2.1": "17cdd864fa419e5aad8665789a92b6347e97f137ac2ff3c605887225f503f491",
    "512-192-r2": "748437cbdbe19030679445aa12737bd14a983ee8d8aa9e8610e317275e0547d5",
    "512-336-r2": "05e55d45980ad904de5fd2ff5c41b401234073e450acf09fc6152bb99016ced8",
    "512-720-r2": "db3c51b1321c8f87cd0cfb10a48f896741a314349519e1a14714160bbabe574b",
    "toto-4m": "316660d5afb47943e531f39242e0b02ca0b8bb73be5709dfe07ca80dfce9805e",
}
CONFIG_HASHES = {
    "52-16-ft-r2.1": "2276979b72771bf1c460356b9211fa24479a0546d5c03ddf52cccdb0f86c2b44",
    "90-30-ft-r2.1": "abf1b5a26903f1d6b8eb62e66a72bf85bb0ced591b5ffaa655ec031b4d1ec326",
    "512-192-r2": "012025d3a082d44c593f5f23a0b623be427e628d19094e58e22121941d42f1c8",
    "512-336-r2": "1c25e5da9d2151047511fd6062ab93ee323a05774a879bbdda856bda406058f6",
    "512-720-r2": "2b3f031ae0c7a33fb32621feea6381ae63c9c26d170621c3a765fc9d483835c2",
    "toto-4m": "7a926d130e401ab0c5fdb3564f46c8d917bd05c7b3ae26b9c22d2da2ef01d2d8",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selection(profile):
    from tsfm_lab.models.catalog import catalog_info

    toto = profile == "toto-4m"
    repository = (
        "Datadog/Toto-2.0-4m" if toto else "ibm-granite/granite-timeseries-ttm-r2"
    )
    entry = catalog_info(repository, "4m" if toto else profile)
    return {
        "profile": profile,
        "adapter": entry["adapter_id"],
        "repository": repository,
        "revision": entry["reference_revision"],
        "context": 64 if toto else entry["capabilities"]["min_context"],
        "horizon": 65 if toto else entry["capabilities"]["max_horizon"],
        "frequency": (
            "7D"
            if profile.startswith("52-")
            else "1D" if profile.startswith("90-") else "1h"
        ),
        "statistic": "median" if toto else "native_point",
        "weight_sha256": PROFILES[profile],
        "config_sha256": CONFIG_HASHES[profile],
    }


def fixture(directory, chosen, model):
    import numpy as np
    import pandas as pd

    context, horizon = chosen["context"], chosen["horizon"]
    positions = np.arange(context + horizon)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-01", periods=len(positions), freq=chosen["frequency"], tz="UTC"
            ),
            "value": 50 + 5 * np.sin(positions / 4) + positions / 200,
        }
    )
    original, causal, history = (
        directory / name for name in ("input.csv", "causal.csv", "history.csv")
    )
    frame.to_csv(original, index=False)
    frame.iloc[:context].to_csv(history, index=False)
    perturbed = frame.copy()
    perturbed.loc[context:, "value"] += 1000
    perturbed.to_csv(causal, index=False)
    raw = {
        "schema_version": 1,
        "name": "primary",
        "model": model,
        "dataset": {
            "path": str(original),
            "timestamp_column": "timestamp",
            "target_column": "value",
            "frequency": chosen["frequency"],
            "units": "synthetic_original_units",
        },
        "evaluation": {
            "context_length": context,
            "horizon": horizon,
            "stride": horizon,
            "evaluation_start": frame.timestamp.iloc[context].isoformat(),
            "evaluation_end": (
                frame.timestamp.iloc[-1] + pd.Timedelta(chosen["frequency"])
            ).isoformat(),
            "point_statistic": chosen["statistic"],
        },
        "runtime": {"device": "cpu", "dtype": "float32", "batch_size": 1, "seed": 0},
        "output": {"directory": str(directory / "workspace")},
    }
    future = json.loads(json.dumps(raw))
    future.pop("evaluation")
    future.update(
        schema_version=2,
        mode="future_forecast",
        name="future",
        forecast={
            "context_length": context,
            "horizon": horizon,
            "point_statistic": chosen["statistic"],
            "origin": "latest",
        },
    )
    future["dataset"]["path"] = str(history)
    return raw, future


def native_projection(model, request):
    """Explicit pinned native call/inverse; no assumed affine invariance."""
    import numpy as np

    torch = model._torch
    values, horizon = request.context_values, len(request.target_timestamps)
    with torch.device("cpu"), torch.inference_mode():
        if model.adapter_id == "ttm_r2":
            location = values.mean()
            scale = 1.0 if np.all(values == values[0]) else values.std(ddof=0)
            arguments = {
                "past_values": torch.as_tensor(
                    ((values - location) / scale).astype(np.float32)[None, :, None],
                    dtype=torch.float32,
                    device="cpu",
                ),
                "return_dict": True,
            }
            token = model.configuration()["checkpoint"]["point_forecast"].get(
                "frequency_token"
            )
            if token is not None:
                arguments["freq_token"] = torch.as_tensor(
                    [token], dtype=torch.long, device="cpu"
                )
            native = model._model(**arguments).prediction_outputs.detach().cpu().numpy()
            return native[0, :horizon, 0].astype(np.float64) * scale + location
        target = torch.as_tensor(
            values.astype(np.float32)[None, None, :], dtype=torch.float32, device="cpu"
        )
        native = (
            model._model.forecast(
                {
                    "target": target,
                    "target_mask": torch.ones_like(target, dtype=torch.bool),
                    "series_ids": torch.zeros((1, 1), dtype=torch.long, device="cpu"),
                },
                horizon=horizon,
                decode_block_size=32,
                has_missing_values=False,
                scaler_fallback_min_obs=0,
                quantile_real_cap_k=0.0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        require(
            native.shape == (9, 1, 1, horizon), "Unexpected native TOTO quantile shape"
        )
        return native[4, 0, 0].astype(np.float64)


def qualification_worker(config, events, cancellation):
    """Trusted qualification-only hooks around the ordinary application worker."""
    import numpy as np
    import torch

    from tsfm_lab import run_modes
    from tsfm_lab.execution import _worker_main
    from tsfm_lab.models.registry import adapter_profile

    torch.set_num_threads(1)
    network_attempts = []
    compilation_attempts = []

    def no_network(*args, **kwargs):
        network_attempts.append(True)
        raise AssertionError("Qualification inference must remain offline")

    def no_compilation(*args, **kwargs):
        compilation_attempts.append(True)
        raise AssertionError("Qualification inference must remain eager")

    dynamo = importlib.import_module("torch._dynamo.convert_frame")

    profile = adapter_profile(config.model["adapter_id"])
    adapter = getattr(importlib.import_module(profile.module), profile.class_name)
    original_predict, original_run = adapter.predict, run_modes.run_validated

    def predict(model, requests):
        result = original_predict(model, requests)
        if config.name == "primary":
            for request, output in zip(requests, result, strict=True):
                reference = native_projection(model, request)
                np.testing.assert_array_equal(output.point, reference)
            events.send(
                {"type": "qualification_original_units", "native_projection": "exact"}
            )
        return result

    def run(configuration, **options):
        progress = options.get("window_progress")

        def window(completed, total):
            if progress:
                progress(completed, total)
            if completed and configuration.name in {"cancel-probe", "failure-probe"}:
                events.send(
                    {
                        "type": "qualification_after_native_forecast",
                        "completed": completed,
                    }
                )
                if configuration.name == "failure-probe":
                    raise RuntimeError(
                        "Injected qualification failure after native forecast"
                    )
                while not cancellation.wait(0.01):
                    pass
                raise KeyboardInterrupt(
                    "Qualification cancellation after native forecast"
                )

        return original_run(configuration, **{**options, "window_progress": window})

    with (
        patch("socket.getaddrinfo", no_network),
        patch("socket.create_connection", no_network),
        patch("socket.socket.connect", no_network),
        patch.object(dynamo, "_compile", no_compilation),
        patch.object(adapter, "predict", predict),
        patch.object(run_modes, "run_validated", run),
    ):
        _worker_main(config, events, cancellation)
    require(not network_attempts, "A native runtime attempted network access")
    require(not compilation_attempts, "A native runtime attempted compilation")
    events.send(
        {
            "type": "qualification_runtime_guards",
            "network_attempts": 0,
            "compilation_attempts": 0,
        }
    )


def managed(config, deadline, *, cancel=False):
    from tsfm_lab.execution import ManagedRun

    # A stable import name makes the trusted target picklable for clean children.
    target = importlib.import_module("qualify_v030_models").qualification_worker
    worker = ManagedRun(config, grace_seconds=2, _target=target)
    events = []
    try:
        worker.start()
        while worker.is_running:
            require(time.monotonic() < deadline, "Qualification deadline exceeded")
            batch = worker.poll()
            events.extend(batch)
            if cancel and any(
                item["type"] == "qualification_after_native_forecast" for item in batch
            ):
                worker.request_cancel()
            time.sleep(0.01)
        events.extend(worker.poll())
        terminal = [
            item
            for item in events
            if item["type"] in {"completed", "failed", "cancelled"}
        ]
        require(len(terminal) == 1, "Managed worker did not return one terminal event")
        require(not worker._process.is_alive(), "Managed child was not reaped")
        return terminal[0], events
    finally:
        try:
            evidence = config.output_directory.parent
            (evidence / f"{config.name}-events.json").write_text(
                json.dumps(events, indent=2) + "\n", encoding="utf-8"
            )
            if worker.log_path is not None and worker.log_path.is_file():
                (evidence / f"{config.name}-worker.log").write_bytes(
                    worker.log_path.read_bytes()
                )
        finally:
            worker.close()


def detached(workspace, identifiers, directory):
    from tsfm_lab.experiments import ExperimentStore
    from tsfm_lab.exports import export_predictions

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Detached reads cannot access source data, weights or inference"
        )

    store = ExperimentStore(workspace)
    formats = ["csv", "json"]
    if importlib.util.find_spec("pyarrow") is not None:
        formats.append("parquet")
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for identifier in identifiers
        for path in (store.root / identifier).iterdir()
        if path.is_file()
    }
    with (
        patch("tsfm_lab.application.load_dataset", forbidden),
        patch("tsfm_lab.future_application.load_dataset", forbidden),
        patch("tsfm_lab.application.build_model", forbidden),
        patch("tsfm_lab.future_application.build_model", forbidden),
        patch("tsfm_lab.models.checkpoints._safe_file", forbidden),
    ):
        for identifier in identifiers:
            for extension in formats:
                export_predictions(
                    identifier, workspace, directory / f"{identifier}.{extension}"
                )
    require(
        all(
            hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
            for path, digest in before.items()
        ),
        "Detached reading changed saved artifacts",
    )
    return {
        "records": len(identifiers),
        "exports": len(formats) * len(identifiers),
        "formats": formats,
        "saved_bytes_unchanged": True,
    }


def qualify(chosen, *, checkpoint=None, cache=None, directory, timeout=300):
    import numpy as np

    from tsfm_lab.config import config_from_mapping
    from tsfm_lab.experiments import ExperimentStore
    from tsfm_lab.future_config import future_config_from_mapping
    from tsfm_lab.models.checkpoints import inspect_local_checkpoint
    from tsfm_lab.models.preparation import plan_checkpoint, prepare_checkpoint

    deadline = time.monotonic() + timeout
    stages = []
    if cache is not None:
        cache = Path(cache).expanduser().resolve()
        plan = plan_checkpoint(
            chosen["repository"], chosen["revision"], cache_dir=cache, offline=True
        )
        require(
            plan.cached and plan.download_bytes == 0,
            "Only fully cached, explicit checkpoints may qualify",
        )
        checked = prepare_checkpoint(plan, progress=stages.append)
        model = {
            "source": "huggingface",
            "adapter_id": chosen["adapter"],
            "repository": chosen["repository"],
            "revision": chosen["revision"],
            "cache_dir": str(cache),
            "offline": True,
        }
    else:
        checked = inspect_local_checkpoint(checkpoint)
        model = {
            "source": "local",
            "adapter_id": chosen["adapter"],
            "path": str(checked.path),
        }
    require(
        checked.files["model.safetensors"]["sha256"] == chosen["weight_sha256"],
        "Weights do not match the exact selected curated profile",
    )
    require(
        checked.files["config.json"]["sha256"] == chosen["config_sha256"],
        "Configuration does not match the exact selected curated profile",
    )
    raw, future = fixture(directory, chosen, model)
    workspace = Path(raw["output"]["directory"])
    store = ExperimentStore(workspace)
    completed, arrays, results, events = [], {}, {}, {}
    for name in (
        "primary",
        "replay",
        "causal",
        "future",
        "cancel-probe",
        "failure-probe",
        "recovery",
    ):
        fields = json.loads(json.dumps(future if name == "future" else raw))
        fields["name"] = name
        if name == "causal":
            fields["dataset"]["path"] = str(directory / "causal.csv")
        config = (
            future_config_from_mapping if name == "future" else config_from_mapping
        )(fields, base_directory=directory)
        terminal, emitted = managed(config, deadline, cancel=name == "cancel-probe")
        events[name] = emitted
        expected = (
            "cancelled"
            if name == "cancel-probe"
            else "failed" if name == "failure-probe" else "completed"
        )
        require(terminal["type"] == expected, f"{name}: {terminal}")
        identifier = terminal["run_id"]
        require(
            identifier is not None
            and store.get_manifest(identifier)["status"] == expected,
            "Terminal event and persisted manifest disagree",
        )
        if expected != "completed":
            require(
                any(
                    item["type"] == "qualification_after_native_forecast"
                    for item in emitted
                ),
                "Fault/cancellation probe never completed native inference",
            )
            continue
        completed.append(identifier)
        if name == "future":
            payload, predictions, _ = store.load_future_result(identifier)
            require(
                payload["evaluation"]["reason"] == "future_actuals_absent"
                and "actual" not in predictions,
                "Future actuals/accuracy semantics changed",
            )
        else:
            payload, predictions = store.load_result(identifier)
        require(len(predictions) == chosen["horizon"], "Incorrect qualified horizon")
        require(
            set(predictions.point_statistic) == {chosen["statistic"]},
            "Incorrect point-output statistic",
        )
        require(
            payload["configuration"]["dataset"]["units"] == "synthetic_original_units",
            "Original units missing from saved configuration",
        )
        arrays[name] = predictions.prediction.to_numpy()
        results[name] = {
            "run_id": identifier,
            "provenance": payload["provenance"],
            "rows": len(predictions),
        }
    require(
        any(
            item["type"] == "qualification_original_units" for item in events["primary"]
        ),
        "No native original-unit comparison ran",
    )
    for name in ("replay", "causal", "future", "recovery"):
        np.testing.assert_array_equal(arrays["primary"], arrays[name])
    for name in ("input.csv", "causal.csv", "history.csv"):
        (directory / name).rename(directory / f"unavailable-{name}")
    return {
        "checkpoint": checked.record(),
        "cache_preparation": stages,
        "runs": results,
        "events": events,
        "comparisons": "exact replay, causal target perturbation, same-context historical/future, and recovery",
        "detached": detached(workspace, completed, directory),
    }


def source_identity():
    import tsfm_lab

    package = Path(tsfm_lab.__file__).resolve().parent
    files = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def installed_identity(wheel, expected_version="0.3.0"):
    from verify_release_identity import verify_installed

    import tsfm_lab

    package = Path(tsfm_lab.__file__).resolve().parent
    require(sys.flags.isolated == 1, "Run qualification using python -I")
    require(sys.prefix != sys.base_prefix, "Use an isolated virtual environment")
    require(
        package.is_relative_to(Path(sys.prefix).resolve()),
        "Package is outside this environment",
    )
    require(
        not package.is_relative_to(ROOT),
        "Source checkouts cannot qualify a published wheel",
    )
    identity = verify_installed(wheel, package)
    require(
        identity["version"] == expected_version,
        "Installed wheel version disagrees with the explicit qualification target",
    )
    return identity


def installed_versions(adapter):
    from tsfm_lab.models.registry import adapter_profile

    result = {}
    for package in {"numpy", "pandas", "PyYAML"} | {
        name for name, _ in adapter_profile(adapter).runtime_requirements
    }:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument(
        "--expected-version", choices=("0.3.0", "0.3.1"), default="0.3.0"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    require(
        1 <= args.timeout_seconds <= 540, "Timeout must be between 1 and 540 seconds"
    )
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
    ):
        os.environ[key] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    report = {
        "profile": selection(args.profile),
        "python": sys.version,
        "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "real_inference_requested": True,
        "cloud_compute": False,
        "executed_package_tree_sha256": source_identity(),
    }
    report["installed_versions"] = installed_versions(report["profile"]["adapter"])
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    report["harness_repository_revision"] = (
        revision.stdout.strip() if revision.returncode == 0 else None
    )
    started = time.monotonic()
    try:
        report["installed_wheel"] = installed_identity(
            args.wheel, args.expected_version
        )
        report["evidence"] = qualify(
            report["profile"],
            checkpoint=args.checkpoint,
            cache=args.cache_dir,
            directory=directory,
            timeout=args.timeout_seconds,
        )
        require(
            report["executed_package_tree_sha256"] == source_identity(),
            "Installed package files changed during qualification",
        )
        report["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(
            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
        )
    report["elapsed_seconds"] = time.monotonic() - started
    (directory / "qualification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "profile": args.profile,
                "report": str(directory / "qualification.json"),
            }
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

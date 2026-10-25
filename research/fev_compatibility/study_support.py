"""Isolated synthetic conversion helpers; not part of TSFM Lab's package."""

import hashlib
import importlib.metadata
import json
import math
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import datasets
import fev
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(os.environ["TSFM_FEV_STUDY_OUTPUT"])
SOURCE = Path(__file__).resolve().parents[2]
datasets.disable_progress_bars()
(ROOT / "inputs").mkdir(exist_ok=True)


def no_network(*args, **kwargs):
    raise RuntimeError("Network access is forbidden during this synthetic study")


_connect = socket.socket.connect


def local_ipc_only(connection, address):
    if connection.family in {socket.AF_INET, socket.AF_INET6}:
        no_network()
    return _connect(connection, address)


socket.socket.connect = local_ipc_only
socket.create_connection = no_network


def clean_json(value):
    if isinstance(value, str):
        return value.replace(str(ROOT), "{output}").replace(str(SOURCE), "{repository}")
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return value


def write_json(name, value):
    (ROOT / name).write_text(
        json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n"
    )


def row(identifier, values, **covariates):
    return {
        "id": identifier,
        "timestamp": pd.date_range("2026-01-01", periods=len(values), freq="h")
        .to_pydatetime()
        .tolist(),
        "target": list(map(float, values)),
        **{name: list(values) for name, values in covariates.items()},
    }


def task(name, rows, **options):
    path = ROOT / "inputs" / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    write_json(f"inputs/{name}.json", rows)
    result = fev.Task(dataset_path=str(path), eval_metric="MAE", **options)
    result.load_full_dataset(num_proc=1)
    return result


def timestamp(value):
    return pd.Timestamp(value).tz_localize("UTC").isoformat()


def last_value(past, horizon):
    return np.repeat(past["target"][-1], horizon).astype(float)


def seasonal(past, horizon):
    return np.resize(np.asarray(past["target"][-4:], dtype=float), horizon)


def convert(task, forecast=last_value, require_all_items=True):
    predictions, cases, histories, item_ids = [], [], [], []
    expected = sorted(str(x) for x in task.load_full_dataset()["id"])
    for window in task.iter_windows(num_proc=1):
        past, future = window.get_input_data()
        truth = window.get_ground_truth()
        ids = [str(x) for x in past["id"]]
        if require_all_items and ids != expected:
            raise ValueError(f"Incomplete item coverage: {ids}; expected {expected}")
        item_ids.append(ids)
        rows, history_rows = [], []
        for known, observed in zip(past, truth, strict=True):
            assert str(known["id"]) == str(observed["id"])
            values = np.asarray(forecast(known, task.horizon), dtype=float)
            assert values.shape == (task.horizon,) and np.isfinite(values).all()
            rows.append({"predictions": values.tolist()})
            history_rows.append(np.asarray(known["target"], dtype=float).tolist())
            for lead, (time, actual, predicted) in enumerate(
                zip(observed["timestamp"], observed["target"], values, strict=True),
                start=1,
            ):
                cases.append(
                    {
                        "series_id": str(known["id"]),
                        "origin": timestamp(known["timestamp"][-1]),
                        "target_timestamp": timestamp(time),
                        "lead": lead,
                        "actual": float(actual),
                        "prediction": float(predicted),
                    }
                )
        predictions.append(rows)
        histories.append(history_rows)
        assert "target" not in future.column_names
    keys = ["series_id", "origin", "target_timestamp", "lead"]
    frame = pd.DataFrame(cases)
    assert not frame.duplicated(keys).any()
    summary = task.evaluation_summary(predictions, model_name="synthetic_fixture")
    return {
        "summary": summary,
        "predictions": predictions,
        "cases": cases,
        "histories": histories,
        "item_ids": item_ids,
    }


def expect_error(operation, kind=ValueError):
    try:
        operation()
    except kind as exc:
        return {"type": type(exc).__name__, "message": str(exc)}
    raise AssertionError("Expected validation to reject this input")


def dependency_closure():
    requested, checked, result = {"fev": set()}, {}, {}
    pending = ["fev"]
    while pending:
        name = canonicalize_name(pending.pop())
        extras = requested[name]
        if checked.get(name) == extras:
            continue
        checked[name] = set(extras)
        dist = importlib.metadata.distribution(name)
        requirements = []
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker is not None and not any(
                req.marker.evaluate({"extra": extra}) for extra in {"", *extras}
            ):
                continue
            version = importlib.metadata.version(req.name)
            assert (
                not req.specifier or version in req.specifier
            ), f"{name}: {text}, got {version}"
            dependency = canonicalize_name(req.name)
            requested.setdefault(dependency, set()).update(req.extras)
            pending.append(dependency)
            requirements.append({"requirement": text, "installed": version})
        result[name] = {
            "name": name,
            "version": dist.version,
            "extras": sorted(extras),
            "installed_in_study_environment": Path(dist.locate_file(""))
            .resolve()
            .is_relative_to(Path(sys.prefix).resolve()),
            "requirements": requirements,
        }
    return [result[name] for name in sorted(result)]


def identities():
    files = []
    for path in sorted((ROOT / "inputs").glob("*")):
        data = path.read_bytes()
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return files

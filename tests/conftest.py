"""Small disposable configurations for the complete shell workflow."""

import shutil
from pathlib import Path
from uuid import uuid4

import pytest
import yaml


@pytest.fixture
def config_factory(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples/datasets/seasonal.csv"
    shutil.copyfile(source, tmp_path / "seasonal.csv")
    directory = tmp_path / "configs"
    directory.mkdir()

    def create(adapter="last_value", **changes):
        fields = {
            "schema_version": 1,
            "name": f"demo-{adapter}",
            "model": {"source": "builtin", "adapter_id": adapter},
            "dataset": {
                "path": "../seasonal.csv",
                "timestamp_column": "timestamp",
                "target_column": "demand",
                "frequency": "1h",
                "units": "synthetic_units",
            },
            "evaluation": {
                "context_length": 8,
                "horizon": 4,
                "stride": 4,
                "evaluation_start": "2026-01-01T16:00:00Z",
                "evaluation_end": "2026-01-02T08:00:00Z",
                "seasonal_period": 4,
            },
            "output": {"directory": "../workspace"},
        }
        if adapter == "seasonal_naive":
            fields["model"]["options"] = {"seasonal_period": 4}
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(fields.get(key), dict):
                fields[key].update(value)
            else:
                fields[key] = value
        path = directory / f"{uuid4().hex}.yaml"
        path.write_text(yaml.safe_dump(fields, sort_keys=False), encoding="utf-8")
        return path

    return create

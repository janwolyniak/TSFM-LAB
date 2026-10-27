"""Check frozen old-code artifacts using the currently importable package.

Run with an ordinary wheel environment's ``python -I path/to/verify.py`` to
exercise installed code. This helper never adds a source checkout to sys.path.
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

SOURCE_REVISION = "0ef0064ee80c83e90c6d9c927e0f5c5af4ede1a2"


def _hashes(root, names):
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }


def verify(root=None):
    from tsfm_lab.application import load_dataset
    from tsfm_lab.comparison import compare_runs
    from tsfm_lab.config import load_config
    from tsfm_lab.data import canonical_fingerprint
    from tsfm_lab.experiments import ExperimentStore
    from tsfm_lab.exports import export_predictions
    from tsfm_lab.finance.reports import load_report

    root = Path(root) if root is not None else Path(__file__).resolve().parent
    identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
    assert identity["source_revision"] == SOURCE_REVISION
    assert _hashes(root, identity["files"]) == identity["files"]
    for adapter in ("last_value", "seasonal_naive"):
        config = load_config(root / f"{adapter}.yaml")
        assert config.workflow is None and "workflow" not in config.resolved_dict()
        assert (
            canonical_fingerprint(load_dataset(config.dataset))
            == identity["dataset_fingerprint"]
        )
    expected_comparison = json.loads((root / "comparison.json").read_text())
    expected_report = json.loads((root / "finance-report.json").read_text())
    with tempfile.TemporaryDirectory(prefix="tsfm-legacy-read-") as directory:
        detached = Path(directory)
        shutil.copytree(root / "workspace", detached / "workspace")
        shutil.copyfile(root / "finance-report.json", detached / "finance-report.json")
        saved_names = [
            name for name in identity["files"] if name.startswith("workspace/")
        ] + ["finance-report.json"]
        before = _hashes(detached, saved_names)
        store = ExperimentStore(detached / "workspace")
        for identifier in identity["run_ids"]:
            original = root / "workspace" / "experiments" / identifier
            expected_config = json.loads((original / "configuration.json").read_text())
            assert not Path(expected_config["dataset"]["path"]).exists()
            assert store.load_configuration(identifier) == expected_config
            saved, predictions = store.load_result(identifier)
            assert saved == json.loads((original / "result.json").read_text())
            assert "workflow" not in saved["configuration"]
            assert (
                saved["provenance"]["dataset"]["canonical_sha256"]
                == identity["dataset_fingerprint"]
            )
            assert len(predictions) == 6
            destination = detached / f"{identifier}.json"
            export_predictions(identifier, store.workspace, destination)
            assert len(json.loads(destination.read_text())) == 6
        assert compare_runs(identity["run_ids"], store.workspace) == expected_comparison
        report = load_report(detached / "finance-report.json")
        assert report == expected_report
        assert report["fingerprint"] == identity["finance_fingerprint"]
        assert _hashes(detached, saved_names) == before
    assert _hashes(root, identity["files"]) == identity["files"]
    return {
        "source_revision": identity["source_revision"],
        "dataset_fingerprint": identity["dataset_fingerprint"],
        "runs": len(identity["run_ids"]),
        "unchanged_artifacts": len(identity["files"]),
        "comparison_policy": expected_comparison["comparison_policy"],
        "finance_fingerprint": identity["finance_fingerprint"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))

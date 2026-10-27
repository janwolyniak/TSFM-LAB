"""Generate a small compatibility fixture with the genuine pre-profile code.

Run from a clone containing SOURCE_REVISION with already installed core/CLI
dependencies. This reads Git, creates disposable source files, and writes a fresh
destination; it never installs dependencies or uses the network.
"""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

SOURCE_REVISION = "0ef0064ee80c83e90c6d9c927e0f5c5af4ede1a2"
ROOT = Path(__file__).resolve().parents[1]
GENERATE = """
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

source, target = (Path(value).resolve() for value in sys.argv[1:])
sys.path.insert(0, str(source / 'src'))
import tsfm_lab
assert Path(tsfm_lab.__file__).resolve().is_relative_to(source)
import pandas as pd
import yaml
from tsfm_lab.application import load_dataset, run_from_config
from tsfm_lab.comparison import compare_runs
from tsfm_lab.config import load_config
from tsfm_lab.data import canonical_fingerprint
from tsfm_lab.finance.backtest import backtest_run
from tsfm_lab.finance.contracts import BacktestRules
from tsfm_lab.finance.ohlc import load_ohlc_csv
from tsfm_lab.finance.reports import save_report

target.mkdir(parents=True)
times = pd.date_range('2026-01-01', periods=8, freq='h', tz='UTC')
closes = [100., 102., 101., 103., 100., 105., 104., 106.]
pd.DataFrame({
    'open_timestamp': times - pd.Timedelta(minutes=45),
    'close_timestamp': times,
    'open': closes,
    'high': [value + 2 for value in closes],
    'low': [value - 2 for value in closes],
    'close': closes,
}).to_csv(target / 'ohlc.csv', index=False)
identifiers = []
for adapter in ('last_value', 'seasonal_naive'):
    config = {
        'schema_version': 1,
        'name': 'Historical ' + adapter,
        'model': {'source': 'builtin', 'adapter_id': adapter},
        'dataset': {
            'path': 'ohlc.csv',
            'timestamp_column': 'close_timestamp',
            'target_column': 'close',
            'frequency': '1h',
            'units': 'synthetic_quote',
            'series_id': 'synthetic_asset',
        },
        'evaluation': {
            'context_length': 2,
            'horizon': 1,
            'stride': 1,
            'evaluation_start': times[2].isoformat(),
            'evaluation_end': (times[-1] + pd.Timedelta('1h')).isoformat(),
            'seasonal_period': 2,
        },
        'output': {'directory': 'workspace'},
    }
    if adapter == 'seasonal_naive':
        config['model']['options'] = {'seasonal_period': 2}
    path = target / (adapter + '.yaml')
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    identifiers.append(run_from_config(path)['run_id'])
config = load_config(target / 'last_value.yaml')
dataset = load_dataset(config.dataset)
bars = load_ohlc_csv(target / 'ohlc.csv', frequency='1h',
                     units='synthetic_quote', series_id='synthetic_asset')
report = backtest_run(identifiers[1], target / 'workspace', dataset, bars,
                      BacktestRules(1000, 0, 5, 2, 1, 0), target_is_price=True)
save_report(report, target / 'finance-report.json')
comparison = compare_runs(identifiers, target / 'workspace')
(target / 'comparison.json').write_text(
    json.dumps(comparison, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
print(json.dumps({
    'run_ids': identifiers,
    'dataset_fingerprint': canonical_fingerprint(dataset),
    'finance_fingerprint': json.loads((target / 'finance-report.json').read_text())['fingerprint'],
    'environment': {'python': platform.python_version(),
                    **{name: importlib.metadata.version(name)
                       for name in ('numpy', 'pandas', 'PyYAML')}},
}))
"""


def generate(destination):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Use a fresh destination: {destination}")
    revision = subprocess.check_output(
        ["git", "rev-parse", SOURCE_REVISION], cwd=ROOT, text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("Historical source revision disagrees")
    archive = subprocess.check_output(
        ["git", "archive", revision, "src/tsfm_lab"], cwd=ROOT
    )
    with tempfile.TemporaryDirectory(
        prefix="tsfm-legacy-source-", dir="/tmp"
    ) as folder:
        source = Path(folder) / "source"
        source.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(source, filter="data")
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                GENERATE,
                str(source),
                str(Path(folder) / "fixture"),
            ],
            cwd=source,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        shutil.copytree(Path(folder) / "fixture", destination)
    identity = json.loads(result.stdout)
    identity["source_revision"] = revision
    identity["source_archive_sha256"] = hashlib.sha256(archive).hexdigest()
    identity["files"] = {
        path.relative_to(destination)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    (destination / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(json.dumps(generate(parser.parse_args().destination), indent=2))

"""Explicit dispatch between historical evaluation and future point forecasts."""


def is_future(config):
    from tsfm_lab.future_config import FutureForecastConfig

    return isinstance(config, FutureForecastConfig)


def forecast_spec(config):
    from tsfm_lab.config import ExperimentConfig

    if is_future(config):
        return config.forecast
    if isinstance(config, ExperimentConfig):
        return config.evaluation
    raise ValueError("Expected a validated historical or future configuration")


def run_validated(config, **options):
    from tsfm_lab.config import ExperimentConfig

    if is_future(config):
        from tsfm_lab.future_application import run_future_config

        return run_future_config(config, **options)
    if isinstance(config, ExperimentConfig):
        from tsfm_lab.application import run_config

        return run_config(config, **options)
    raise ValueError("Expected a validated historical or future configuration")


def run_from_file(path, **options):
    from tsfm_lab.config import load_run_config

    return run_validated(load_run_config(path), **options)


def read_completed(store, run_id):
    """Keep historical readers strict while sharing read-only command views."""
    if store.get_manifest(run_id).get("kind") == "future_forecast":
        result, predictions, _ = store.load_future_result(run_id)
        return result, predictions
    return store.load_result(run_id)

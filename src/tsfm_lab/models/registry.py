"""Implemented adapter and curated variant definitions without runtime imports."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointVariant:
    repository: str
    variant: str
    display_name: str
    reference_revision: str
    weight_bytes: int
    parameter_count: int | None = None


@dataclass(frozen=True)
class AdapterProfile:
    adapter_id: str
    family: str
    display_name: str
    module: str
    class_name: str
    architecture: str
    pipeline_class: str
    model_type: str
    max_context: int
    max_horizon: int
    context_reserved: int
    runtime_requirements: tuple[tuple[str, str], ...]
    variants: tuple[CheckpointVariant, ...]


ADAPTER_PROFILES = (
    AdapterProfile(
        "chronos_bolt",
        "chronos",
        "Chronos-Bolt",
        "tsfm_lab.models.chronos_bolt",
        "ChronosBolt",
        "ChronosBoltModelForForecasting",
        "ChronosBoltPipeline",
        "t5",
        2048,
        64,
        0,
        (
            ("chronos-forecasting", "==2.2.0"),
            ("torch", ">=2.8,<2.9"),
            ("transformers", ">=4.55,<4.56"),
            ("huggingface-hub", "==0.34.4"),
            ("safetensors", ">=0.6,<0.7"),
            ("accelerate", ">=1.12,<1.13"),
        ),
        (
            CheckpointVariant(
                "amazon/chronos-bolt-tiny",
                "tiny",
                "Tiny",
                "a0e552de83495b5c28c14c71c374f3e33280b340",
                34622352,
                8652672,
            ),
            CheckpointVariant(
                "amazon/chronos-bolt-small",
                "small",
                "Small",
                "772f3d25d38aec6d914c8949dab4462e2d46f5d8",
                190888824,
                47718016,
            ),
            CheckpointVariant(
                "amazon/chronos-bolt-base",
                "base",
                "Base",
                "5d9f166d69f47aef3401367a7b842e78fe97b121",
                821203576,
                205292928,
            ),
        ),
    ),
    AdapterProfile(
        "timesfm_2p5",
        "timesfm",
        "TimesFM 2.5",
        "tsfm_lab.models.timesfm",
        "TimesFM",
        "TimesFmModelForPrediction",
        "TimesFM_2p5_200M_torch",
        "timesfm",
        16256,
        128,
        128,
        (
            ("timesfm", "==3.0.2"),
            ("torch", ">=2.8,<2.9"),
            ("huggingface-hub", "==0.34.4"),
            ("safetensors", ">=0.6,<0.7"),
        ),
        (
            CheckpointVariant(
                "google/timesfm-2.5-200m-pytorch",
                "200m",
                "2.5 200M",
                "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
                925181104,
                231289280,
            ),
        ),
    ),
)


def adapter_profile(adapter_id: str) -> AdapterProfile:
    for profile in ADAPTER_PROFILES:
        if profile.adapter_id == adapter_id:
            return profile
    raise ValueError(f"Unsupported checkpoint adapter: {adapter_id!r}")


def checkpoint_variant(repository: str) -> tuple[AdapterProfile, CheckpointVariant]:
    for profile in ADAPTER_PROFILES:
        for variant in profile.variants:
            if variant.repository == repository:
                return profile, variant
    raise ValueError("Only curated checkpoint repositories are supported")

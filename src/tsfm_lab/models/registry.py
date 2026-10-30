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
    max_context: int | None = None
    max_horizon: int | None = None
    min_context: int | None = None


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
    runtime_notices: tuple[str, ...] = ()
    min_context: int = 1
    point_statistic: str = "median"
    supported_tasks: tuple[str, ...] = ("median", "native_point")
    frequency_policy: str = "any_fixed"


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
            ("transformers", ">=4.56,<4.57"),
            ("huggingface-hub", "==0.35.3"),
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
            ("huggingface-hub", "==0.35.3"),
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
    AdapterProfile(
        "chronos_2",
        "chronos",
        "Chronos-2",
        "tsfm_lab.models.chronos2",
        "Chronos2",
        "Chronos2Model",
        "Chronos2Pipeline",
        "t5",
        8192,
        1024,
        0,
        (
            ("chronos-forecasting", "==2.2.0"),
            ("torch", ">=2.8,<2.9"),
            ("transformers", ">=4.56,<4.57"),
            ("huggingface-hub", "==0.35.3"),
            ("safetensors", ">=0.6,<0.7"),
            ("accelerate", ">=1.12,<1.13"),
        ),
        (
            CheckpointVariant(
                "autogluon/chronos-2-small",
                "small",
                "Small",
                "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a",
                111749048,
                27934624,
            ),
            CheckpointVariant(
                "amazon/chronos-2",
                "standard",
                "Standard",
                "29ec3766d36d6f73f0696f85560a422f50e8498c",
                477930472,
                119477664,
            ),
        ),
    ),
    AdapterProfile(
        "tirex_2",
        "tirex",
        "TiRex-2",
        "tsfm_lab.models.tirex2",
        "TiRex2",
        "TiRex2",
        "TiRex2",
        "tirex2",
        8192,
        928,
        0,
        (
            ("tirex-2", "==0.4.0"),
            ("torch", ">=2.8,<2.9"),
            ("huggingface-hub", "==0.35.3"),
            ("safetensors", ">=0.6,<0.7"),
            ("numpy", ">=2.1.3,<3"),
            ("flashrnn", "==1.0.8"),
            ("mlstm-kernels", "==2.0.6"),
            ("ninja", "==1.13.2"),
            ("narwhals", "==2.13.0"),
        ),
        (
            CheckpointVariant(
                "NX-AI/TiRex-2",
                "main",
                "Default training profile",
                "30788a311699acba7e6393ad9700ea7b7cb9bb2d",
                330038664,
                82501744,
                2048,
                320,
            ),
            CheckpointVariant(
                "NX-AI/TiRex-2-fevbench",
                "fevbench",
                "FEV benchmark training profile",
                "acd6153b8ee2179f38cfd3cc4fd64a3c045a424f",
                330038640,
                82501744,
                2048,
                320,
            ),
            CheckpointVariant(
                "NX-AI/TiRex-2-gifteval-zs",
                "gifteval-zs",
                "GiftEval zero-shot training profile",
                "c94872d1c8caacbe7755cbfa01bc5c4c5c5ddb7f",
                330038640,
                82501744,
                8192,
                928,
            ),
            CheckpointVariant(
                "NX-AI/TiRex-2-gifteval-pretrain",
                "gifteval-pretrain",
                "GiftEval pretrain training profile",
                "54f88ac06ccfba9fe3e727aa8856b39775481f0f",
                330038640,
                82501744,
                8192,
                928,
            ),
        ),
        (
            "Built with technology from NXAI",
            "The optional mlstm-kernels 2.0.6 dependency uses the NXAI Community License, including additional commercial terms.",
            "https://github.com/NX-AI/mlstm_kernels/blob/v2.0.6/LICENSE",
        ),
    ),
    AdapterProfile(
        "ttm_r2",
        "ttm",
        "IBM Granite TTM R2 / R2.1",
        "tsfm_lab.models.ttm",
        "TTM",
        "TinyTimeMixerForPrediction",
        "TinyTimeMixerForPrediction",
        "tinytimemixer",
        1536,
        720,
        0,
        (
            ("granite-tsfm", "==0.3.4"),
            ("torch", ">=2.8,<2.9"),
            ("transformers", "==4.56.0"),
            ("tokenizers", "==0.22.0"),
            ("scikit-learn", "==1.7.2"),
            ("filelock", "==3.20.3"),
            ("huggingface-hub", "==0.35.3"),
            ("safetensors", ">=0.6,<0.7"),
            ("accelerate", ">=1.12,<1.13"),
        ),
        (
            CheckpointVariant(
                "ibm-granite/granite-timeseries-ttm-r2",
                "512-96-r2",
                "R2 512 context / 96 horizon",
                "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f",
                3240592,
                805280,
                512,
                96,
                512,
            ),
            CheckpointVariant(
                "ibm-granite/granite-timeseries-ttm-r2",
                "1024-96-r2",
                "R2 1024 context / 96 horizon",
                "1a382618f3c5b63836281ad632e54724e55e85a5",
                11879480,
                None,
                1024,
                96,
                1024,
            ),
            CheckpointVariant(
                "ibm-granite/granite-timeseries-ttm-r2",
                "1536-96-r2",
                "R2 1536 context / 96 horizon",
                "f6ed09d67cf42e4877f7fa33d98326911385693d",
                12344144,
                None,
                1536,
                96,
                1536,
            ),
            CheckpointVariant(
                "ibm-granite/granite-timeseries-ttm-r2",
                "512-192-r2",
                "R2 512 context / 192 horizon",
                "25f4a00a25e1d0b67c40a10b06645d72e2266929",
                3634200,
                None,
                512,
                192,
                512,
            ),
            CheckpointVariant(
                "ibm-granite/granite-timeseries-ttm-r2",
                "512-336-r2",
                "R2 512 context / 336 horizon",
                "0009f51f52c87473e60ff3a03cd8633ee85d61e8",
                4224600,
                None,
                512,
                336,
                512,
            ),
            CheckpointVariant(
                "ibm-granite/granite-timeseries-ttm-r2",
                "512-720-r2",
                "R2 512 context / 720 horizon",
                "bcfc4a497430a228afcbaf378f18f316642ece2a",
                5799000,
                None,
                512,
                720,
                512,
            ),
            CheckpointVariant(
                "ibm-granite/granite-timeseries-ttm-r2",
                "52-16-ft-r2.1",
                "R2.1 weekly: 52 context / 16 horizon",
                "8c9c1d742ec5893d6419e3aa2ab21ed23e28df85",
                319440,
                None,
                52,
                16,
                52,
            ),
            CheckpointVariant(
                "ibm-granite/granite-timeseries-ttm-r2",
                "90-30-ft-r2.1",
                "R2.1 daily: 90 context / 30 horizon",
                "6e5cb8ee51e0634a45637490f5db43148b2fa6be",
                3496760,
                None,
                90,
                30,
                90,
            ),
        ),
        min_context=52,
        point_statistic="native_point",
        supported_tasks=("native_point",),
        frequency_policy="profile_specific",
    ),
    AdapterProfile(
        "toto_2",
        "toto",
        "Datadog TOTO 2",
        "tsfm_lab.models.toto2",
        "Toto2",
        "Toto2Model",
        "Toto2Model",
        "toto2",
        2048,
        128,
        0,
        (
            ("toto-2", "==2.0.0"),
            ("torch", ">=2.8,<2.9"),
            ("huggingface-hub", "==0.35.3"),
            ("safetensors", ">=0.6,<0.7"),
            ("numpy", "==2.1.3"),
            ("gluonts", "==0.16.2"),
            ("dd-unit-scaling", "==0.1.0"),
            ("unit-scaling", "==0.3.5"),
            ("einops", "==0.8.2"),
            ("jaxtyping", "==0.3.11"),
            ("lightning", "==2.4.0"),
            ("pytorch-lightning", "==2.4.0"),
            ("scipy", ">=1.14,<1.16"),
            ("matplotlib", "==3.10.8"),
        ),
        (
            CheckpointVariant(
                "Datadog/Toto-2.0-4m",
                "4m",
                "2.0 4M",
                "8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9",
                16582848,
                min_context=32,
            ),
        ),
        (
            "Requires Python 3.12 or 3.13; Python 3.11 is unsupported for this optional family.",
            "Project integration bounds: context 32–2048 in multiples of 32, horizon 1–128; these are not upstream architectural limits.",
            "Runtime metadata compatibility and offline fixtures checked; installed imports, real inference and native platform qualification pending.",
        ),
        min_context=32,
    ),
)


def adapter_profile(adapter_id: str) -> AdapterProfile:
    for profile in ADAPTER_PROFILES:
        if profile.adapter_id == adapter_id:
            return profile
    raise ValueError(f"Unsupported checkpoint adapter: {adapter_id!r}")


def checkpoint_variant(
    repository: str, variant: str | None = None
) -> tuple[AdapterProfile, CheckpointVariant]:
    for profile in ADAPTER_PROFILES:
        for choice in profile.variants:
            if choice.repository == repository and (
                variant is None or choice.variant == variant
            ):
                return profile, choice
    if variant is not None:
        raise ValueError("Select a curated checkpoint profile for this repository")
    raise ValueError("Only curated checkpoint repositories are supported")

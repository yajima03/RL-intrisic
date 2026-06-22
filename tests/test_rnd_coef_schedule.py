from __future__ import annotations

import pytest

from src.intrinsic.rnd import RNDConfig, RNDIntrinsicReward
from src.training.train import build_intrinsic_module


def test_rnd_current_coef_linearly_decays() -> None:
    rnd = RNDIntrinsicReward(
        RNDConfig(
            initial_coef=0.05,
            final_coef=0.005,
            coef_decay_steps=100,
        )
    )

    assert rnd.current_coef() == pytest.approx(0.05)

    rnd._env_step_count = 50
    assert rnd.current_coef() == pytest.approx(0.0275)

    rnd._env_step_count = 200
    assert rnd.current_coef() == pytest.approx(0.005)


def test_rnd_build_intrinsic_module_reads_coef_schedule() -> None:
    module, coef, _ = build_intrinsic_module(
        {
            "name": "rnd",
            "initial_coef": 0.1,
            "final_coef": 0.01,
        },
        {"policy": "MlpPolicy", "total_timesteps": 1000},
    )

    assert isinstance(module, RNDIntrinsicReward)
    assert coef == pytest.approx(1.0)
    assert module.config.initial_coef == pytest.approx(0.1)
    assert module.config.final_coef == pytest.approx(0.01)
    assert module.config.coef_decay_steps == 1000


def test_rnd_build_intrinsic_module_reads_coef_decay_fraction() -> None:
    module, _, _ = build_intrinsic_module(
        {
            "name": "rnd",
            "initial_coef": 0.1,
            "final_coef": 0.01,
            "coef_decay_fraction": 0.25,
        },
        {"policy": "MlpPolicy", "total_timesteps": 1000},
    )

    assert isinstance(module, RNDIntrinsicReward)
    assert module.config.coef_decay_steps == 250


def test_rnd_build_intrinsic_module_prefers_explicit_decay_steps() -> None:
    module, _, _ = build_intrinsic_module(
        {
            "name": "rnd",
            "initial_coef": 0.1,
            "final_coef": 0.01,
            "coef_decay_fraction": 0.25,
            "coef_decay_steps": 400,
        },
        {"policy": "MlpPolicy", "total_timesteps": 1000},
    )

    assert isinstance(module, RNDIntrinsicReward)
    assert module.config.coef_decay_steps == 400


def test_rnd_build_intrinsic_module_keeps_legacy_coef_behavior() -> None:
    module, coef, _ = build_intrinsic_module(
        {
            "name": "rnd",
            "coef": 0.05,
        },
        {"policy": "MlpPolicy", "total_timesteps": 1000},
    )

    assert isinstance(module, RNDIntrinsicReward)
    assert coef == pytest.approx(1.0)
    assert module.current_coef() == pytest.approx(0.05)


def test_rnd_build_intrinsic_module_accepts_inital_coef_alias() -> None:
    module, _, _ = build_intrinsic_module(
        {
            "name": "rnd",
            "inital_coef": 0.2,
            "final_coef": 0.02,
        },
        {"policy": "MlpPolicy", "total_timesteps": 1000},
    )

    assert isinstance(module, RNDIntrinsicReward)
    assert module.config.initial_coef == pytest.approx(0.2)

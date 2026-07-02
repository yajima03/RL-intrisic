from __future__ import annotations

import numpy as np
import pytest
import torch as th

from src.intrinsic.ama import AMAConfig, AMAIntrinsicReward
from src.training.train import build_intrinsic_module


def test_ama_mlp_compute_and_update_are_finite() -> None:
    ama = AMAIntrinsicReward(
        AMAConfig(
            encoder_type="mlp",
            feature_dim=8,
            predictor_hidden_layers=(16,),
            clamp_min=None,
            device="cpu",
            seed=3,
        )
    )

    reward = ama.compute(
        previous_observation=np.zeros(4, dtype=np.float32),
        observation=np.ones(4, dtype=np.float32),
        action=1,
        info={"action_space_n": 2},
    )
    assert np.isfinite(reward)

    loss = ama.update_from_batch(
        th.tensor([[0.0] * 4, [1.0] * 4]),
        th.tensor([[1.0] * 4, [0.5] * 4]),
        th.tensor([[1], [0]]),
    )
    assert np.isfinite(loss)


def test_ama_cnn_accepts_sp_observation_shape() -> None:
    ama = AMAIntrinsicReward(
        AMAConfig(
            encoder_type="cnn",
            feature_dim=8,
            predictor_hidden_layers=(16,),
            device="cpu",
            seed=0,
        )
    )
    reward = ama.compute(
        previous_observation=np.zeros((1, 32, 32), dtype=np.float32),
        observation=np.ones((1, 32, 32), dtype=np.float32),
        action=2,
        info={"action_space_n": 4},
    )
    assert np.isfinite(reward)
    assert reward >= 0.0


def test_build_intrinsic_module_constructs_ama_and_schedule() -> None:
    module, coef, _ = build_intrinsic_module(
        {"name": "ama", "initial_coef": 0.2, "final_coef": 0.02},
        {"policy": "MlpPolicy", "total_timesteps": 1000},
    )
    assert isinstance(module, AMAIntrinsicReward)
    assert coef == pytest.approx(1.0)
    assert module.config.encoder_type == "mlp"
    assert module.config.coef_decay_steps == 1000
    assert module.current_coef() == pytest.approx(0.2)

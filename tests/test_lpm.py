from __future__ import annotations

import numpy as np
import pytest
import torch as th

from src.intrinsic.lpm import LPMConfig, LPMIntrinsicReward
from src.training.train import build_intrinsic_module


def test_lpm_mlp_compute_and_update_are_finite() -> None:
    lpm = LPMIntrinsicReward(
        LPMConfig(
            encoder_type="mlp",
            feature_dim=8,
            dynamics_hidden_layers=(16,),
            error_hidden_layers=(16,),
            error_buffer_size=16,
            error_batch_size=2,
            reward_warmup_size=2,
            device="cpu",
            seed=3,
        )
    )
    first = lpm.compute(
        previous_observation=np.zeros(4, dtype=np.float32),
        observation=np.ones(4, dtype=np.float32),
        action=0,
        info={"action_space_n": 2},
    )
    second = lpm.compute(
        previous_observation=np.ones(4, dtype=np.float32),
        observation=np.full(4, 0.5, dtype=np.float32),
        action=1,
        info={"action_space_n": 2},
    )
    assert first == 0.0
    assert np.isfinite(second)

    losses = lpm.update_from_batch(
        th.tensor([[0.0] * 4, [1.0] * 4]),
        th.tensor([[1.0] * 4, [0.5] * 4]),
        th.tensor([[0], [1]]),
    )
    assert losses["dynamics_loss"] >= 0.0
    assert losses["error_loss"] >= 0.0
    assert all(np.isfinite(value) for value in losses.values())


def test_lpm_cnn_accepts_sp_observation_shape() -> None:
    lpm = LPMIntrinsicReward(
        LPMConfig(
            encoder_type="cnn",
            feature_dim=8,
            dynamics_hidden_layers=(16,),
            error_hidden_layers=(16,),
            reward_warmup_size=0,
            device="cpu",
            seed=0,
        )
    )
    reward = lpm.compute(
        previous_observation=np.zeros((1, 32, 32), dtype=np.float32),
        observation=np.ones((1, 32, 32), dtype=np.float32),
        action=2,
        info={"action_space_n": 4},
    )
    assert np.isfinite(reward)


def test_build_intrinsic_module_constructs_lpm_and_schedule() -> None:
    module, coef, _ = build_intrinsic_module(
        {"name": "lpm", "initial_coef": 0.2, "final_coef": 0.02},
        {"policy": "MlpPolicy", "total_timesteps": 1000},
    )
    assert isinstance(module, LPMIntrinsicReward)
    assert coef == pytest.approx(1.0)
    assert module.config.encoder_type == "mlp"
    assert module.config.coef_decay_steps == 1000
    assert module.current_coef() == pytest.approx(0.2)


def test_lpm_error_model_respects_update_interval() -> None:
    lpm = LPMIntrinsicReward(
        LPMConfig(
            encoder_type="mlp",
            feature_dim=8,
            dynamics_hidden_layers=(8,),
            error_hidden_layers=(8,),
            error_batch_size=1,
            reward_warmup_size=0,
            error_update_interval=2,
            device="cpu",
            seed=0,
        )
    )
    lpm.compute(
        previous_observation=np.zeros(2, dtype=np.float32),
        observation=np.ones(2, dtype=np.float32),
        action=0,
        info={"action_space_n": 1},
    )
    batch = th.zeros((1, 2))
    actions = th.zeros((1, 1), dtype=th.long)
    first = lpm.update_from_batch(batch, th.ones((1, 2)), actions)
    second = lpm.update_from_batch(batch, th.ones((1, 2)), actions)
    assert first["error_loss"] == 0.0
    assert second["error_loss"] > 0.0


def test_lpm_supports_one_sided_miniworld_reward_clip() -> None:
    lpm = LPMIntrinsicReward(LPMConfig(clamp_max=0.5, device="cpu"))
    assert lpm._postprocess_reward(2.0) == pytest.approx(0.5)
    assert lpm._postprocess_reward(-2.0) == pytest.approx(-2.0)


def test_lpm_zero_coefficient_still_collects_errors_and_updates() -> None:
    lpm = LPMIntrinsicReward(
        LPMConfig(
            encoder_type="mlp",
            feature_dim=8,
            dynamics_hidden_layers=(8,),
            error_hidden_layers=(8,),
            error_batch_size=1,
            reward_warmup_size=0,
            initial_coef=0.0,
            final_coef=0.0,
            device="cpu",
            seed=0,
        )
    )
    raw_reward = lpm.compute(
        previous_observation=np.zeros(2, dtype=np.float32),
        observation=np.ones(2, dtype=np.float32),
        action=0,
        info={"action_space_n": 1},
    )
    losses = lpm.update_from_batch(
        th.zeros((1, 2)),
        th.ones((1, 2)),
        th.zeros((1, 1), dtype=th.long),
    )

    assert np.isfinite(raw_reward)
    assert lpm.current_coef() == 0.0
    assert len(lpm.error_buffer) == 1
    assert losses["dynamics_loss"] > 0.0

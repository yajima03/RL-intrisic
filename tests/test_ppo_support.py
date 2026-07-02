from __future__ import annotations

import pytest

from src.training.make_env import make_env
from src.training.train import build_model, build_policy_kwargs, load_algo_config


def test_load_ppo_config_selects_ppo() -> None:
    cfg = load_algo_config("configs/algo/ppo.yaml")

    assert cfg["algo_name"] == "ppo"
    assert cfg["policy"] == "CnnPolicy"
    assert cfg["n_steps"] == 2048


def test_ppo_policy_kwargs_accept_actor_critic_net_arch() -> None:
    cfg = load_algo_config("configs/algo/ppo_mlp.yaml")

    policy_kwargs = build_policy_kwargs(cfg)

    assert policy_kwargs["net_arch"] == {"pi": [128, 128], "vf": [128, 128]}


def test_build_ppo_model_from_config(tmp_path) -> None:
    pytest.importorskip("stable_baselines3")
    cfg = load_algo_config("configs/algo/ppo.yaml")
    cfg.update(
        {
            "n_steps": 8,
            "batch_size": 4,
            "n_epochs": 1,
            "device": "cpu",
            "progress_bar": False,
        }
    )
    env_cfg = {
        "env_name": "ScalablePyramid-v0",
        "image_size": 16,
        "param": {
            "depth": 4,
            "hyperplane_dim": 2,
            "start_random": False,
            "general_seed": 0,
            "features": {"seed": 1},
            "rewards": {
                "depth": [4],
                "coordinates": [[1, 1]],
                "value": [1.0],
                "type": ["fix"],
                "var": [0.0],
            },
        },
    }
    env = make_env(env_cfg, seed=0, record_episode_statistics=False)
    try:
        model = build_model(env, cfg, tmp_path)
        assert model.__class__.__name__ == "PPO"
        assert model.n_steps == 8
    finally:
        env.close()

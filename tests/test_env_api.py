from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.training.make_env import DEFAULT_ENV_CONFIG, load_env_config, make_env
from src.envs.scalable_pyramid_env import ScalablePyramidEnv

try:
    from stable_baselines3.common.env_checker import check_env
except Exception:  # pragma: no cover
    check_env = None


@pytest.fixture
def env_config() -> dict:
    return {
        "env_name": "ScalablePyramid-v0",
        "image_size": 32,
        "sigma_range": [1.0, 3.0],
        "coeff_range": [-1.0, 1.0],
        "obs_noise_std": 0.0,
        "param": {
            "depth": 4,
            "hyperplane_dim": 2,
            "start_random": False,
            "general_seed": 123,
            "features": {"seed": 456},
            "rewards": {
                "depth": [4],
                "coordinates": [[2, 2]],
                "value": [1.0],
                "type": ["fix"],
                "var": [0.0],
            },
        },
    }


@pytest.fixture
def env(env_config: dict):
    environment = make_env(env_config, seed=7, record_episode_statistics=False)
    yield environment
    environment.close()



def test_load_env_config_defaults() -> None:
    cfg = load_env_config(None)
    assert cfg["env_name"] == DEFAULT_ENV_CONFIG["env_name"]
    assert "param" in cfg
    assert "depth" in cfg["param"]



def test_load_env_config_from_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "sp_base.yaml"
    yaml_config = {
        "image_size": 48,
        "param": {
            "depth": 5,
            "hyperplane_dim": 3,
        },
    }
    yaml_path.write_text(yaml.safe_dump(yaml_config, sort_keys=False), encoding="utf-8")

    cfg = load_env_config(yaml_path)
    assert cfg["image_size"] == 48
    assert cfg["param"]["depth"] == 5
    assert cfg["param"]["hyperplane_dim"] == 3
    assert cfg["env_name"] == "ScalablePyramid-v0"



def test_make_env_returns_env_instance(env_config: dict) -> None:
    environment = make_env(env_config, seed=42, record_episode_statistics=False)
    try:
        assert isinstance(environment, ScalablePyramidEnv)
    finally:
        environment.close()



def test_reset_returns_valid_observation(env: ScalablePyramidEnv) -> None:
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["depth"] == 1
    assert info["node_id"] == 0



def test_step_returns_gymnasium_five_tuple(env: ScalablePyramidEnv) -> None:
    env.reset(seed=0)
    step_out = env.step(0)
    assert len(step_out) == 5

    obs, reward, terminated, truncated, info = step_out
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert "external_reward" in info
    assert "coordinates" in info



def test_episode_terminates_after_fixed_horizon(env: ScalablePyramidEnv) -> None:
    env.reset(seed=0)
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(0)
        steps += 1
        assert steps <= env.core.depth

    assert terminated is True
    assert steps == env.core.depth - 1



def test_same_seed_produces_same_trajectory(env_config: dict) -> None:
    env1 = make_env(env_config, seed=123, record_episode_statistics=False)
    env2 = make_env(env_config, seed=123, record_episode_statistics=False)

    try:
        obs1, info1 = env1.reset(seed=123)
        obs2, info2 = env2.reset(seed=123)
        assert np.allclose(obs1, obs2)
        assert info1["node_id"] == info2["node_id"]

        for action in [0, 3, 1]:
            out1 = env1.step(action)
            out2 = env2.step(action)
            assert np.allclose(out1[0], out2[0])
            assert out1[1] == pytest.approx(out2[1])
            assert out1[2] == out2[2]
            assert out1[3] == out2[3]
            assert out1[4]["node_id"] == out2[4]["node_id"]
    finally:
        env1.close()
        env2.close()



def test_state_action_count_increments(env: ScalablePyramidEnv) -> None:
    env.reset(seed=0)
    start_node = env.current_node_idx
    assert start_node is not None
    before = int(env.state_action_counts[start_node, 2])
    env.step(2)
    after = int(env.state_action_counts[start_node, 2])
    assert after == before + 1


@pytest.mark.skipif(check_env is None, reason="stable-baselines3 is not installed")
def test_check_env_passes(env_config: dict) -> None:
    environment = make_env(env_config, seed=0, record_episode_statistics=False)
    try:
        check_env(environment, warn=True, skip_render_check=True)
    finally:
        environment.close()

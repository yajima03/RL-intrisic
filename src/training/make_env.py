from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

import yaml
from gymnasium.wrappers import RecordEpisodeStatistics

from src.envs.scalable_pyramid_env import ScalablePyramidEnv

try:
    from stable_baselines3.common.monitor import Monitor
except Exception:  # pragma: no cover
    Monitor = None

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
except Exception:  # pragma: no cover
    DummyVecEnv = None
    SubprocVecEnv = None
    VecMonitor = None


ConfigLike = Union[str, Path, Mapping[str, Any]]


DEFAULT_ENV_CONFIG: Dict[str, Any] = {
    "env_name": "ScalablePyramid-v0",
    "image_size": 64,
    "sigma_range": [1.0, 5.0],
    "coeff_range": [-1.0, 1.0],
    "obs_noise_std": 0.0,
    "param": {
        "depth": 6,
        "hyperplane_dim": 2,
        "start_random": True,
        "general_seed": 0,
        "features": {"seed": 1},
        "rewards": {
            "depth": [6],
            "coordinates": [[3, 3]],
            "value": [1.0],
            "type": ["fix"],
            "var": [0.0],
        },
    },
}


class UnsupportedEnvError(ValueError):
    """Raised when an unsupported environment name is requested."""



def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base



def load_env_config(config: Optional[ConfigLike] = None) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_ENV_CONFIG)

    if config is None:
        return merged

    if isinstance(config, Mapping):
        return _deep_update(merged, config)

    path = Path(config)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError("YAML config must decode to a mapping/dictionary.")

    return _deep_update(merged, loaded)



def _normalize_seed(config: Dict[str, Any], seed: Optional[int]) -> Dict[str, Any]:
    cfg = deepcopy(config)
    cfg.setdefault("param", {})
    cfg["param"].setdefault("features", {})

    if seed is not None:
        # general_seed may vary across train/eval/worker processes.
        cfg["param"]["general_seed"] = int(seed)
        # Keep the feature-generation seed fixed from the config so that
        # train and eval share the same SP task structure.
        cfg["param"]["features"].setdefault("seed", 1)

    return cfg



def _check_env_name(config: Mapping[str, Any]) -> None:
    env_name = str(config.get("env_name", "ScalablePyramid-v0"))
    valid_names = {"ScalablePyramid-v0", "ScalablePyramidEnv", "sp"}
    if env_name not in valid_names:
        raise UnsupportedEnvError(
            f"Unsupported env_name '{env_name}'. Supported: {sorted(valid_names)}"
        )



def assert_has_reward_nodes(env: ScalablePyramidEnv) -> None:
    base_env = env.unwrapped
    core = getattr(base_env, "core", None)
    if core is None:
        core = getattr(base_env, "env", None)
    if core is None:
        raise RuntimeError("Could not locate the underlying Scalable Pyramid core object.")

    rewarded = [node for node in core.nodes if getattr(node, "reward_type", None) is not None]
    if len(rewarded) == 0:
        raise RuntimeError(
            "No reward nodes were created. Check configs/env/sp_base.yaml and reward parsing."
        )



def make_env(
    config: Optional[ConfigLike] = None,
    *,
    seed: Optional[int] = None,
    record_episode_statistics: bool = True,
    monitor: bool = False,
    monitor_dir: Optional[Union[str, Path]] = None,
    env_kwargs: Optional[Mapping[str, Any]] = None,
) -> ScalablePyramidEnv:
    cfg = load_env_config(config)
    cfg = _normalize_seed(cfg, seed)
    if env_kwargs:
        cfg = _deep_update(cfg, env_kwargs)

    _check_env_name(cfg)
    env = ScalablePyramidEnv(**cfg)
    assert_has_reward_nodes(env)

    if record_episode_statistics:
        env = RecordEpisodeStatistics(env)

    if monitor:
        if Monitor is None:
            raise ImportError(
                "stable-baselines3 is required to use monitor=True. "
                "Install stable-baselines3 or disable the monitor wrapper."
            )
        monitor_path = str(monitor_dir) if monitor_dir is not None else None
        env = Monitor(env, filename=monitor_path)

    if seed is not None:
        env.reset(seed=int(seed))

    return env



def make_env_fn(
    config: Optional[ConfigLike] = None,
    *,
    rank: int = 0,
    base_seed: int = 0,
    record_episode_statistics: bool = True,
    monitor: bool = False,
    monitor_dir: Optional[Union[str, Path]] = None,
    env_kwargs: Optional[Mapping[str, Any]] = None,
) -> Callable[[], ScalablePyramidEnv]:
    def _thunk() -> ScalablePyramidEnv:
        seed = int(base_seed) + int(rank)
        return make_env(
            config=config,
            seed=seed,
            record_episode_statistics=record_episode_statistics,
            monitor=monitor,
            monitor_dir=monitor_dir,
            env_kwargs=env_kwargs,
        )

    return _thunk



def make_vec_env(
    config: Optional[ConfigLike] = None,
    *,
    n_envs: int = 1,
    base_seed: int = 0,
    vec_env_type: str = "dummy",
    record_episode_statistics: bool = True,
    monitor: bool = True,
    monitor_dir: Optional[Union[str, Path]] = None,
    env_kwargs: Optional[Mapping[str, Any]] = None,
):
    if DummyVecEnv is None:
        raise ImportError("stable-baselines3 is required to create vectorized environments.")

    env_fns = [
        make_env_fn(
            config=config,
            rank=rank,
            base_seed=base_seed,
            record_episode_statistics=record_episode_statistics,
            monitor=False,
            env_kwargs=env_kwargs,
        )
        for rank in range(int(n_envs))
    ]

    vec_env_type = vec_env_type.lower()
    if vec_env_type == "dummy":
        vec_env = DummyVecEnv(env_fns)
    elif vec_env_type == "subproc":
        if SubprocVecEnv is None:
            raise ImportError(
                "stable-baselines3 does not provide SubprocVecEnv in this environment."
            )
        vec_env = SubprocVecEnv(env_fns)
    else:
        raise ValueError("vec_env_type must be either 'dummy' or 'subproc'.")

    if monitor:
        if VecMonitor is None:
            raise ImportError(
                "stable-baselines3 is required to wrap vectorized envs with VecMonitor."
            )
        vec_env = VecMonitor(vec_env, filename=str(monitor_dir) if monitor_dir else None)

    return vec_env

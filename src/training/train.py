from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch as th
import torch.nn as nn
import yaml
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.intrinsic.count_bonus import CountBasedBonus, CountBonusConfig
from src.intrinsic.rnd import RNDConfig, RNDIntrinsicReward, RNDAugmentedDQN
from src.intrinsic.pglp_local import PGLPLocalConfig, PGLPLocalIntrinsicReward, PGLPAugmentedDQN
from src.training.make_env import load_env_config, make_env
from src.training.reward_wrapper import RewardWrapper


DEFAULT_ALGO_CONFIG: Dict[str, Any] = {
    "algo_name": "dqn",
    "policy": "CnnPolicy",
    "device": "auto",
    "seed": 0,
    "progress_bar": True,
    "deterministic_eval": True,
    "total_timesteps": 100_000,
    "learning_rate": 1e-4,
    "buffer_size": 100_000,
    "learning_starts": 5_000,
    "batch_size": 64,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 1_000,
    "exploration_fraction": 0.20,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "max_grad_norm": 10.0,
    "policy_kwargs": {
        "normalize_images": False,
        "features_extractor": {
            "name": "ConfigurableCNN",
            "conv_layers": [
                {"out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
                {"out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 2},
                {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
            ],
            "global_pool": "avg",
            "post_pool_norm": "layernorm",
            "linear_layers": [64],
            "activation": "relu",
        },
        "q_net_hidden_layers": [256],
    },
    "eval": {"eval_freq": 10_000, "n_eval_episodes": 10},
    "checkpoint": {"checkpoint_freq": 10_000},
    "logging": {"log_interval": 500},
}

DEFAULT_INTRINSIC_CONFIG: Dict[str, Any] = {
    "name": "none",
    "coef": 0.0,
}

MONITOR_INFO_KEYS: Sequence[str] = (
    "episode_external_return",
    "episode_intrinsic_return",
    "episode_total_return",
)


class ZeroIntrinsicReward:
    """No-op intrinsic module for keeping eval/train monitor columns aligned."""

    def reset_episode(self) -> None:
        return None

    def compute(
        self,
        *,
        previous_observation=None,
        observation=None,
        info=None,
        action=None,
    ) -> float:
        return 0.0


class EnvStatusLoggingCallback(BaseCallback):
    """
    Periodic callback for:
      1. env status csv snapshots via fprint_env_status
      2. raw intrinsic state-level snapshots via fprint_env_rnd
      3. pglp debug csv via flush_debug_stats()
    """

    def __init__(
        self,
        *,
        train_env,
        run_log_dir: Path,
        log_interval: int,
        intrinsic_module=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.train_env = train_env
        self.run_log_dir = Path(run_log_dir)
        self.log_interval = int(log_interval)
        self.intrinsic_module = intrinsic_module

        self.raw_intrinsic_logname = "raw_intrinsic_history.csv"
        self.pglp_debug_csv_path = self.run_log_dir / "pglp_debug.csv"
        self._ensure_pglp_debug_header()

    def _ensure_pglp_debug_header(self) -> None:
        if self.pglp_debug_csv_path.exists():
            return

        self.pglp_debug_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.pglp_debug_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestep",
                    "mean_gate",
                    "mean_progress",
                    "mean_raw_intrinsic",
                    "mean_current_error",
                    "mean_neighbor_count",
                    "mean_prototype_count_for_action",
                    "mean_prototype_long",
                    "mean_prototype_short",
                    "mean_prototype_index",
                    "num_samples",
                ]
            )

    def _append_pglp_debug_row(self, timestep: int, stats: Dict[str, float]) -> None:
        with self.pglp_debug_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestep,
                    stats.get("mean_gate", 0.0),
                    stats.get("mean_progress", 0.0),
                    stats.get("mean_raw_intrinsic", 0.0),
                    stats.get("mean_current_error", 0.0),
                    stats.get("mean_neighbor_count", 0.0),
                    stats.get("mean_prototype_count_for_action", 0.0),
                    stats.get("mean_prototype_long", 0.0),
                    stats.get("mean_prototype_short", 0.0),
                    stats.get("mean_prototype_index", -1.0),
                    stats.get("num_samples", 0.0),
                ]
            )

    def _on_step(self) -> bool:
        if self.log_interval <= 0:
            return True

        if self.num_timesteps <= 0 or self.num_timesteps % self.log_interval != 0:
            return True

        base_env = self.train_env.unwrapped if hasattr(self.train_env, "unwrapped") else self.train_env

        # 1) full env status snapshot
        if hasattr(base_env, "fprint_env_status"):
            try:
                base_env.fprint_env_status(
                    role="t",
                    base_dir=str(self.run_log_dir),
                    step_count=self.num_timesteps,
                )
            except Exception as e:
                if self.verbose > 0:
                    print(f"[EnvStatusLoggingCallback] fprint_env_status failed at step {self.num_timesteps}: {e}")

        # 2) raw intrinsic state-level history
        if self.intrinsic_module is not None and hasattr(self.intrinsic_module, "export_raw_intrinsic_per_state"):
            try:
                raw_values = self.intrinsic_module.export_raw_intrinsic_per_state(base_env)
                if hasattr(base_env, "fprint_env_rnd"):
                    base_env.fprint_env_rnd(
                        logname=self.raw_intrinsic_logname,
                        step=self.num_timesteps,
                        all_intrinsic=raw_values,
                        base_dir=str(self.run_log_dir),
                    )
            except Exception as e:
                if self.verbose > 0:
                    print(f"[EnvStatusLoggingCallback] raw intrinsic logging failed at step {self.num_timesteps}: {e}")

        # 3) pglp step diagnostics
        if self.intrinsic_module is not None and hasattr(self.intrinsic_module, "flush_debug_stats"):
            try:
                stats = self.intrinsic_module.flush_debug_stats()
                self._append_pglp_debug_row(self.num_timesteps, stats)
            except Exception as e:
                if self.verbose > 0:
                    print(f"[EnvStatusLoggingCallback] pglp_debug.csv logging failed at step {self.num_timesteps}: {e}")

        return True


def _deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base



def load_algo_config(config: Optional[str]) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_ALGO_CONFIG)
    if config is None:
        return merged

    path = Path(config)
    if not path.exists():
        raise FileNotFoundError(f"Algorithm config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError("Algorithm YAML config must decode to a mapping/dictionary.")

    return _deep_update(merged, loaded)


def load_intrinsic_config(config: Optional[str]) -> Dict[str, Any]:
    merged = deepcopy(DEFAULT_INTRINSIC_CONFIG)
    if config is None:
        return merged

    path = Path(config)
    if not path.exists():
        raise FileNotFoundError(f"Intrinsic config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError("Intrinsic YAML config must decode to a mapping/dictionary.")

    return _deep_update(merged, loaded)


def get_activation(name: str) -> nn.Module:
    key = str(name).lower()
    table = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "selu": nn.SELU,
        "silu": nn.SiLU,
    }
    if key not in table:
        raise ValueError(f"Unsupported activation: {name}")
    return table[key]()


class ConfigurableCNN(BaseFeaturesExtractor):
    """Config-driven CNN extractor for channel-first grayscale observations."""

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        conv_layers: Sequence[Mapping[str, Any]],
        global_pool: str = "avg",
        post_pool_norm: str = "none",
        linear_layers: Optional[Sequence[int]] = None,
        activation: str = "relu",
    ) -> None:
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("ConfigurableCNN requires a Box observation space.")
        if len(observation_space.shape) != 3:
            raise ValueError("ConfigurableCNN expects observations with shape (C, H, W).")

        super().__init__(observation_space, features_dim=1)

        in_channels = int(observation_space.shape[0])
        activation_name = str(activation)
        conv_blocks = []
        out_channels_last = in_channels

        for layer_cfg in conv_layers:
            out_channels = int(layer_cfg["out_channels"])
            kernel_size = int(layer_cfg.get("kernel_size", 3))
            stride = int(layer_cfg.get("stride", 1))
            padding = int(layer_cfg.get("padding", 0))
            conv_blocks.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    get_activation(activation_name),
                ]
            )
            in_channels = out_channels
            out_channels_last = out_channels
        self.conv = nn.Sequential(*conv_blocks)

        pool_key = str(global_pool).lower()
        if pool_key == "avg":
            self.global_pool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))
        elif pool_key == "max":
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        elif pool_key == "none":
            self.global_pool = nn.Identity()
        else:
            raise ValueError(f"Unsupported global_pool: {global_pool}")

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            pooled_sample = self.global_pool(self.conv(sample))
            flat_dim = int(th.flatten(pooled_sample, start_dim=1).shape[1])

        norm_key = str(post_pool_norm).lower()
        if norm_key == "none":
            self.post_pool_norm: nn.Module = nn.Identity()
        elif norm_key == "layernorm":
            self.post_pool_norm = nn.LayerNorm(flat_dim)
        elif norm_key == "batchnorm":
            self.post_pool_norm = nn.BatchNorm1d(flat_dim)
        else:
            raise ValueError(f"Unsupported post_pool_norm: {post_pool_norm}")

        linear_layers = list(linear_layers or [])
        mlp_blocks = []
        in_dim = flat_dim
        for idx, out_dim in enumerate(linear_layers):
            out_dim = int(out_dim)
            mlp_blocks.append(nn.Linear(in_dim, out_dim))
            if idx < len(linear_layers) - 1:
                mlp_blocks.append(get_activation(activation_name))
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_blocks)
        self._features_dim = int(in_dim)
        self._config_summary = {
            "conv_layers": [dict(layer) for layer in conv_layers],
            "global_pool": pool_key,
            "post_pool_norm": norm_key,
            "linear_layers": linear_layers,
            "activation": activation_name,
            "out_channels_last": out_channels_last,
        }

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.conv(observations)
        x = self.global_pool(x)
        x = th.flatten(x, start_dim=1)
        x = self.post_pool_norm(x)
        x = self.mlp(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train base RL algorithm on Scalable Pyramid with a selectable intrinsic module."
    )
    parser.add_argument("--env-config", type=str, default=None, help="Path to env YAML config.")
    parser.add_argument("--algo-config", type=str, default=None, help="Path to algo YAML config.")
    parser.add_argument(
        "--intrinsic-config",
        type=str,
        default=None,
        help="Path to intrinsic YAML config.",
    )
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from algo config.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from algo config.",
    )
    return parser.parse_args()



def prepare_run_dirs(output_dir: Path, run_name: Optional[str]) -> Dict[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    resolved_run_name = run_name or f"dqn_sp_{timestamp}"
    run_dir = output_dir / resolved_run_name
    dirs = {
        "run_dir": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "logs": run_dir / "logs",
        "tensorboard": run_dir / "tensorboard",
        "eval": run_dir / "eval",
        "models": run_dir / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs



def save_run_metadata(
    *,
    run_dirs: Dict[str, Path],
    env_config: Dict[str, Any],
    algo_config: Dict[str, Any],
    intrinsic_config: Dict[str, Any],
    cli_args: argparse.Namespace,
) -> None:
    with (run_dirs["run_dir"] / "env_config_resolved.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(env_config, fh, sort_keys=False, allow_unicode=True)

    with (run_dirs["run_dir"] / "algo_config_resolved.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(algo_config, fh, sort_keys=False, allow_unicode=True)

    with (run_dirs["run_dir"] / "intrinsic_config_resolved.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(intrinsic_config, fh, sort_keys=False, allow_unicode=True)

    with (run_dirs["run_dir"] / "cli_args.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(cli_args), fh, indent=2, ensure_ascii=False)



def build_policy_kwargs(algo_config: Dict[str, Any]) -> Dict[str, Any]:
    policy_cfg = dict(algo_config.get("policy_kwargs", {}))
    extractor_cfg = dict(policy_cfg.get("features_extractor", {}))
    extractor_name = extractor_cfg.pop("name", "ConfigurableCNN")
    if extractor_name != "ConfigurableCNN":
        raise ValueError(f"Unsupported features extractor: {extractor_name}")

    q_net_hidden_layers = policy_cfg.get(
        "q_net_hidden_layers",
        algo_config.get("q_net_hidden_layers", [256]),
    )
    if isinstance(q_net_hidden_layers, int):
        q_net_hidden_layers = [int(q_net_hidden_layers)]
    q_net_hidden_layers = [int(v) for v in q_net_hidden_layers]

    return {
        "features_extractor_class": ConfigurableCNN,
        "features_extractor_kwargs": extractor_cfg,
        "net_arch": q_net_hidden_layers,
        "normalize_images": bool(policy_cfg.get("normalize_images", False)),
    }



def build_intrinsic_module(intrinsic_config: Dict[str, Any]):
    intrinsic_name = str(intrinsic_config.get("name", "none")).lower()
    coef = float(intrinsic_config.get("coef", 1.0))

    if intrinsic_name in {"none", "null", ""}:
        return None, coef, intrinsic_config

    if intrinsic_name in {"count", "count_bonus", "count-based", "count_based"}:
        config = CountBonusConfig(
            power=float(intrinsic_config.get("power", intrinsic_config.get("count_power", 0.5))),
            eps=float(intrinsic_config.get("eps", 1e-8)),
            use_state_action=bool(intrinsic_config.get("use_state_action", False)),
        )
        return CountBasedBonus(config), coef, intrinsic_config

    if intrinsic_name in {"rnd", "random_network_distillation"}:
        rnd_config = RNDConfig(
            learning_rate=float(intrinsic_config.get("learning_rate", 1.0e-4)),
            embedding_dim=int(intrinsic_config.get("embedding_dim", 64)),
            conv_layers=intrinsic_config.get("conv_layers", None),
            activation=str(intrinsic_config.get("activation", "relu")),
            normalize_reward=bool(intrinsic_config.get("normalize_reward", False)),
            intrinsic_clip=(
                None
                if intrinsic_config.get("intrinsic_clip", None) is None
                else float(intrinsic_config.get("intrinsic_clip"))
            ),
            device=str(intrinsic_config.get("device", "auto")),
            seed=(
                None
                if intrinsic_config.get("seed", None) is None
                else int(intrinsic_config.get("seed"))
            ),
        )
        return RNDIntrinsicReward(rnd_config), coef, intrinsic_config

    if intrinsic_name in {"pglp_local", "pglp-local", "pglp"}:
        pglp_config = PGLPLocalConfig(
            learning_rate=float(intrinsic_config.get("learning_rate", 1.0e-4)),
            conv_layers=intrinsic_config.get("conv_layers", None),
            activation=str(intrinsic_config.get("activation", "relu")),
            embedding_dim=int(intrinsic_config.get("embedding_dim", 64)),
            encoder_fc_layers=tuple(intrinsic_config.get("encoder_fc_layers", [256])),
            predictor_fc_layers=tuple(intrinsic_config.get("predictor_fc_layers", [256, 256])),
            candidate_size=int(intrinsic_config.get("candidate_size", 512)),
            knn_k=int(intrinsic_config.get("knn_k", 16)),
            min_candidates=int(intrinsic_config.get("min_candidates", 16)),
            gate_lambda=float(intrinsic_config.get("gate_lambda", 1.0)),
            cache_capacity_per_action=int(intrinsic_config.get("cache_capacity_per_action", 20000)),
            key_encoder_tau=float(intrinsic_config.get("key_encoder_tau", 0.01)),
            num_prototypes_per_action=int(intrinsic_config.get("num_prototypes_per_action", 8)),
            prototype_center_tau=float(intrinsic_config.get("prototype_center_tau", 0.05)),
            prototype_min_count=int(intrinsic_config.get("prototype_min_count", 4)),
            progress_ema_alpha=float(intrinsic_config.get("progress_ema_alpha", 0.99)),
            progress_short_ema_alpha=float(intrinsic_config.get("progress_short_ema_alpha", 0.9)),
            device=str(intrinsic_config.get("device", "auto")),
            seed=(
                None
                if intrinsic_config.get("seed", None) is None
                else int(intrinsic_config.get("seed"))
            ),
        )
        return PGLPLocalIntrinsicReward(pglp_config), coef, intrinsic_config

    raise ValueError(f"Unsupported intrinsic reward name: {intrinsic_name}")




def build_dqn_model(train_env, algo_config: Dict[str, Any], tensorboard_log: Path, intrinsic_module=None):
    if str(algo_config.get("policy", "CnnPolicy")) != "CnnPolicy":
        raise ValueError("This train.py currently supports only policy='CnnPolicy'.")

    policy_kwargs = build_policy_kwargs(algo_config)

    common_kwargs = dict(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=float(algo_config.get("learning_rate", 1e-4)),
        buffer_size=int(algo_config.get("buffer_size", 100_000)),
        learning_starts=int(algo_config.get("learning_starts", 5_000)),
        batch_size=int(algo_config.get("batch_size", 64)),
        tau=float(algo_config.get("tau", 1.0)),
        gamma=float(algo_config.get("gamma", 0.99)),
        train_freq=int(algo_config.get("train_freq", 4)),
        gradient_steps=int(algo_config.get("gradient_steps", 1)),
        target_update_interval=int(algo_config.get("target_update_interval", 1_000)),
        exploration_fraction=float(algo_config.get("exploration_fraction", 0.20)),
        exploration_initial_eps=float(algo_config.get("exploration_initial_eps", 1.0)),
        exploration_final_eps=float(algo_config.get("exploration_final_eps", 0.05)),
        max_grad_norm=float(algo_config.get("max_grad_norm", 10.0)),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_log),
        seed=int(algo_config.get("seed", 0)),
        device=str(algo_config.get("device", "auto")),
        verbose=1,
    )

    if isinstance(intrinsic_module, RNDIntrinsicReward):
        return RNDAugmentedDQN(rnd_module=intrinsic_module, **common_kwargs)

    if isinstance(intrinsic_module, PGLPLocalIntrinsicReward):
        return PGLPAugmentedDQN(pglp_module=intrinsic_module, **common_kwargs)

    return DQN(**common_kwargs)


def build_callbacks(
    *,
    algo_config: Dict[str, Any],
    run_dirs: Dict[str, Path],
    eval_env,
    train_env,
    intrinsic_module=None,
) -> Optional[CallbackList]:
    callbacks = []

    checkpoint_freq = int(algo_config.get("checkpoint", {}).get("checkpoint_freq", 0))
    if checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(run_dirs["checkpoints"]),
                name_prefix="dqn_sp",
                save_replay_buffer=True,
                save_vecnormalize=False,
            )
        )

    eval_cfg = dict(algo_config.get("eval", {}))
    eval_freq = int(eval_cfg.get("eval_freq", 0))
    if eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(run_dirs["models"] / "best_model"),
                log_path=str(run_dirs["eval"]),
                eval_freq=eval_freq,
                n_eval_episodes=int(eval_cfg.get("n_eval_episodes", 10)),
                deterministic=bool(algo_config.get("deterministic_eval", True)),
                render=False,
                warn=False,
            )
        )

    status_log_interval = int(algo_config.get("logging", {}).get("log_interval", 0))
    if status_log_interval > 0:
        callbacks.append(
            EnvStatusLoggingCallback(
                train_env=train_env,
                run_log_dir=run_dirs["logs"],
                log_interval=status_log_interval,
                intrinsic_module=intrinsic_module,
                verbose=1,
            )
        )

    if not callbacks:
        return None
    return CallbackList(callbacks)



def make_train_env(
    *,
    env_config: Dict[str, Any],
    algo_config: Dict[str, Any],
    intrinsic_config: Dict[str, Any],
    monitor_path: Path,
):
    env = make_env(
        config=env_config,
        seed=int(algo_config.get("seed", 0)),
        record_episode_statistics=False,
        monitor=False,
    )

    intrinsic_module, intrinsic_coef, intrinsic_cfg = build_intrinsic_module(intrinsic_config)

    actual_intrinsic_module = intrinsic_module

    if intrinsic_module is not None:
        env = RewardWrapper(
            env,
            intrinsic_module=intrinsic_module,
            intrinsic_coef=intrinsic_coef,
        )
    else:
        env = RewardWrapper(
            env,
            intrinsic_module=ZeroIntrinsicReward(),
            intrinsic_coef=0.0,
        )

    env = Monitor(
        env,
        filename=str(monitor_path),
        info_keywords=MONITOR_INFO_KEYS,
    )
    return env, intrinsic_cfg, actual_intrinsic_module


def make_eval_env(
    *,
    env_config: Dict[str, Any],
    algo_config: Dict[str, Any],
    monitor_path: Path,
):
    env = make_env(
        config=env_config,
        seed=int(algo_config.get("seed", 0)) + 10_000,
        record_episode_statistics=False,
        monitor=False,
    )

    env = RewardWrapper(
        env,
        intrinsic_module=ZeroIntrinsicReward(),
        intrinsic_coef=0.0,
    )

    env = Monitor(
        env,
        filename=str(monitor_path),
        info_keywords=MONITOR_INFO_KEYS,
    )
    return env



def main() -> None:
    args = parse_args()
    env_config = load_env_config(args.env_config)
    algo_config = load_algo_config(args.algo_config)
    intrinsic_config = load_intrinsic_config(args.intrinsic_config)

    if args.seed is not None:
        algo_config["seed"] = int(args.seed)
    if args.total_timesteps is not None:
        algo_config["total_timesteps"] = int(args.total_timesteps)

    run_dirs = prepare_run_dirs(Path(args.output_dir), args.run_name)
    save_run_metadata(
        run_dirs=run_dirs,
        env_config=env_config,
        algo_config=algo_config,
        intrinsic_config=intrinsic_config,
        cli_args=args,
    )

    train_monitor_path = run_dirs["logs"] / "train_monitor.csv"
    eval_monitor_path = run_dirs["logs"] / "eval_monitor.csv"

    train_env, intrinsic_cfg, intrinsic_module = make_train_env(
        env_config=env_config,
        algo_config=algo_config,
        intrinsic_config=intrinsic_config,
        monitor_path=train_monitor_path,
    )
    eval_env = make_eval_env(
        env_config=env_config,
        algo_config=algo_config,
        monitor_path=eval_monitor_path,
    )

    model = build_dqn_model(
        train_env=train_env,
        algo_config=algo_config,
        tensorboard_log=run_dirs["tensorboard"],
        intrinsic_module=intrinsic_module,
    )
    callbacks = build_callbacks(
        algo_config=algo_config,
        run_dirs=run_dirs,
        eval_env=eval_env,
        train_env=train_env,
        intrinsic_module=intrinsic_module,
    )

    model.learn(
        total_timesteps=int(algo_config.get("total_timesteps", 100_000)),
        callback=callbacks,
        log_interval=int(algo_config.get("logging", {}).get("log_interval", 10)),
        progress_bar=bool(algo_config.get("progress_bar", True)),
        tb_log_name="dqn_sp",
    )

    final_model_path = run_dirs["models"] / "final_model"
    model.save(str(final_model_path))
    model.save_replay_buffer(str(run_dirs["models"] / "final_replay_buffer.pkl"))

    run_summary = {
        "final_model_path": str(final_model_path) + ".zip",
        "tensorboard_dir": str(run_dirs["tensorboard"]),
        "checkpoints_dir": str(run_dirs["checkpoints"]),
        "seed": int(algo_config.get("seed", 0)),
        "total_timesteps": int(algo_config.get("total_timesteps", 100_000)),
        "algo_name": algo_config.get("algo_name", "unknown"),
        "intrinsic_name": intrinsic_config.get("name", "none"),
        "intrinsic": intrinsic_cfg,
    }

    with (run_dirs["run_dir"] / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)

    train_env.close()
    eval_env.close()

    print("Training finished.")
    print(f"Run directory      : {run_dirs['run_dir']}")
    print(f"Final model        : {final_model_path}.zip")
    print(f"TensorBoard logs   : {run_dirs['tensorboard']}")
    print(f"Checkpoint dir     : {run_dirs['checkpoints']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
import csv
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Mapping, Optional, Sequence

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd:
    def __init__(self, epsilon: float = 1.0e-4) -> None:
        self.mean = np.zeros((), dtype=np.float64)
        self.var = np.ones((), dtype=np.float64)
        self.count = float(epsilon)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        batch_count = values.shape[0]
        batch_mean = np.mean(values)
        batch_var = np.var(values)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m_2 / total
        self.count = total


def _get_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "selu": nn.SELU,
        "silu": nn.SiLU,
    }
    key = str(name).lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[key]()


def _resolve_device(device: str) -> th.device:
    if str(device).lower() != "auto":
        return th.device(device)
    if th.cuda.is_available():
        return th.device("cuda")
    if hasattr(th.backends, "mps") and th.backends.mps.is_available():
        return th.device("mps")
    return th.device("cpu")


@dataclass
class LPMConfig:
    """Configuration for Learning Progress Monitoring (LPM)."""

    dynamics_learning_rate: float = 1.0e-4
    error_learning_rate: float = 1.0e-3
    encoder_type: str = "auto"
    conv_layers: Optional[Sequence[Mapping[str, Any]]] = None
    activation: str = "relu"
    feature_dim: int = 64
    dynamics_hidden_layers: Sequence[int] = (256, 256)
    error_hidden_layers: Sequence[int] = (256, 128)
    error_buffer_size: int = 10_000
    error_batch_size: int = 64
    reward_warmup_size: int = 64
    model_update_cycle: int = 64
    dynamics_updates: int = 1
    error_updates: int = 1
    error_update_interval: int = 1
    log_error_epsilon: float = 1.0e-6
    expected_error_scale: float = 1.0
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
    intrinsic_clip: Optional[float] = None
    normalize_reward: bool = False
    reward_gamma: float = 0.99
    reward_norm_eps: float = 1.0e-8
    initial_coef: float = 1.0
    final_coef: float = 1.0
    coef_decay_steps: int = 1
    max_grad_norm: Optional[float] = 10.0
    transition_log_interval: int = 1
    update_log_interval: int = 1
    probe_interval: int = 1_000
    device: str = "auto"
    seed: Optional[int] = None


@dataclass
class _ErrorQueueEntry:
    observation: np.ndarray
    action: int
    error_mse: float
    error_log: float
    dynamics_version: int


class _CSVLogger:
    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        path.parent.mkdir(parents=True, exist_ok=True)
        is_empty = not path.exists() or path.stat().st_size == 0
        self._file = path.open("a", encoding="utf-8", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if is_empty:
            self._writer.writeheader()

    def write(self, values: Mapping[str, Any]) -> None:
        row = {name: values.get(name, "") for name in self.fieldnames}
        self._writer.writerow(row)

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = int(input_dim)
        for width in hidden_layers:
            layers.extend((nn.Linear(current, int(width)), _get_activation(activation)))
            current = int(width)
        layers.append(nn.Linear(current, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class _ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_layers: Sequence[Mapping[str, Any]],
        feature_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = int(in_channels)
        for layer in conv_layers:
            width = int(layer["out_channels"])
            blocks.extend(
                (
                    nn.Conv2d(
                        current,
                        width,
                        kernel_size=int(layer.get("kernel_size", 3)),
                        stride=int(layer.get("stride", 1)),
                        padding=int(layer.get("padding", 0)),
                    ),
                    _get_activation(activation),
                )
            )
            current = width
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(current, int(feature_dim))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.projection(th.flatten(self.pool(self.conv(x)), start_dim=1))


class _DynamicsModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        action_dim: int,
        output_dim: int,
        hidden_layers: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_dim = int(action_dim)
        self.head = _MLP(
            int(feature_dim) + self.action_dim,
            hidden_layers,
            int(output_dim),
            activation,
        )

    def encode(self, observation: th.Tensor) -> th.Tensor:
        return self.encoder(observation)

    def forward(self, observation: th.Tensor, action: th.Tensor) -> th.Tensor:
        features = self.encode(observation)
        action_one_hot = F.one_hot(
            action.reshape(-1).long(), num_classes=self.action_dim
        ).float()
        return self.head(th.cat((features, action_one_hot), dim=1))


class _ErrorModel(nn.Module):
    """Predict E[log MSE] of the previous dynamics model."""

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_dim = int(action_dim)
        self.head = _MLP(
            int(feature_dim) + self.action_dim,
            hidden_layers,
            1,
            activation,
        )

    def forward(self, observation: th.Tensor, action: th.Tensor) -> th.Tensor:
        features = self.encoder(observation)
        action_one_hot = F.one_hot(
            action.reshape(-1).long(), num_classes=self.action_dim
        ).float()
        return self.head(th.cat((features, action_one_hot), dim=1)).squeeze(1)


class LPMIntrinsicReward:
    """Noise-robust intrinsic reward based on learning progress.

    The implementation follows Algorithm 1 of the LPM paper:

        r_i = g_phi^(tau)(o_t, a_t) - epsilon_theta^(tau)(o_{t+1})

    ``g_phi`` is trained on errors collected before the latest dynamics-model
    updates, so it estimates the previous model's expected error.  ``compute``
    only records transitions and computes rewards; optimization is performed by
    ``update_from_batch`` using DQN replay batches.
    """

    TRANSITION_FIELDS = (
        "env_step", "episode", "episode_step", "seed", "state_id", "depth",
        "action", "next_state_id", "edge_id", "terminated", "truncated",
        "extrinsic_reward", "dynamics_error_mse", "dynamics_error_log",
        "predicted_previous_error", "predicted_previous_error_mse", "lpm_signed",
        "lpm_positive", "lpm_raw_used", "intrinsic_reward_normalized",
        "intrinsic_coef", "intrinsic_reward_to_rl", "total_reward_to_rl",
        "dynamics_version", "error_model_version", "error_target_dynamics_version",
        "replay_size", "error_queue_size", "warmup_active",
    )
    UPDATE_FIELDS = (
        "env_step", "lpm_update_step", "dynamics_version_before",
        "dynamics_version_after", "error_model_version_before",
        "error_model_version_after", "error_target_version_min",
        "error_target_version_max", "batch_size", "replay_size", "error_queue_size",
        "dynamics_loss", "error_model_loss", "batch_current_error_mean",
        "batch_current_error_std", "batch_current_log_error_mean",
        "batch_current_log_error_std", "batch_error_prediction_mean",
        "batch_error_target_mean", "batch_error_prediction_bias",
        "batch_error_prediction_mae", "batch_error_prediction_rmse",
        "batch_lpm_signed_mean", "batch_lpm_signed_std", "batch_lpm_positive_mean",
        "batch_lpm_positive_rate", "batch_lpm_negative_rate", "dynamics_grad_norm",
        "error_model_grad_norm", "encoder_grad_norm", "dynamics_param_norm",
        "error_model_param_norm", "dynamics_learning_rate", "error_model_learning_rate",
    )
    PROBE_FIELDS = (
        "env_step", "probe_id", "state_id", "depth", "action", "next_state_id",
        "edge_id", "state_visit_count", "state_action_visit_count",
        "current_dynamics_version", "previous_dynamics_version", "current_error_mse",
        "current_error_log", "previous_error_mse", "previous_error_log",
        "predicted_previous_error", "oracle_pointwise_progress",
        "estimated_lpm_signed", "estimated_lpm_positive", "error_prediction_residual",
    )

    def __init__(self, config: LPMConfig) -> None:
        if int(config.error_buffer_size) <= 0:
            raise ValueError("error_buffer_size must be positive")
        if int(config.error_batch_size) <= 0:
            raise ValueError("error_batch_size must be positive")
        self.config = config
        self.device = _resolve_device(config.device)
        self.dynamics_model: Optional[_DynamicsModel] = None
        self.error_model: Optional[_ErrorModel] = None
        self.dynamics_optimizer: Optional[th.optim.Optimizer] = None
        self.error_optimizer: Optional[th.optim.Optimizer] = None
        self.action_dim: Optional[int] = None
        self._encoder_type_resolved: Optional[str] = None
        self._observation_shape: Optional[tuple[int, ...]] = None
        self.error_buffer: Deque[_ErrorQueueEntry] = deque(
            maxlen=int(config.error_buffer_size)
        )
        self._rng = np.random.default_rng(config.seed)
        self._env_step_count = 0
        self._model_update_count = 0
        self._last_lpm_update_env_step = 0
        self.dynamics_version = 0
        self.error_model_version = 0
        self.error_target_dynamics_version = -1
        self._last_replay_size = 0
        self._episode = -1
        self._episode_step = 0
        self._logging_seed = int(config.seed or 0)
        self._transition_logger: Optional[_CSVLogger] = None
        self._update_logger: Optional[_CSVLogger] = None
        self._probe_logger: Optional[_CSVLogger] = None
        self._probe_transitions: list[dict[str, Any]] = []
        self._probe_env: Any = None
        self._last_probe_env_step = -1
        self._pending_transition: Optional[dict[str, Any]] = None
        self._discounted_intrinsic_return = 0.0
        self.return_rms = RunningMeanStd(epsilon=1.0e-4)
        self.last_actual_error = 0.0
        self.last_expected_error = 0.0
        self.last_raw_intrinsic = 0.0
        self.last_dynamics_loss = 0.0
        self.last_error_loss = 0.0

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        # Open files and live Gym environments must not become part of SB3 model files.
        state["_transition_logger"] = None
        state["_update_logger"] = None
        state["_probe_logger"] = None
        state["_probe_env"] = None
        state["_pending_transition"] = None
        return state

    def reset_episode(self) -> None:
        self._discounted_intrinsic_return = 0.0
        self._episode += 1
        self._episode_step = 0

    def configure_logging(self, log_dir: Path | str, *, env: Any = None, seed: int = 0) -> None:
        log_path = Path(log_dir)
        self._logging_seed = int(seed)
        self._transition_logger = _CSVLogger(
            log_path / "lpm_transition.csv", self.TRANSITION_FIELDS
        )
        self._update_logger = _CSVLogger(log_path / "lpm_update.csv", self.UPDATE_FIELDS)
        self._probe_logger = _CSVLogger(log_path / "lpm_probe.csv", self.PROBE_FIELDS)
        self._probe_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self._probe_transitions = self._build_sp_probe_transitions(self._probe_env)

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def close_logging(self) -> None:
        for logger in (self._transition_logger, self._update_logger, self._probe_logger):
            if logger is not None:
                logger.close()

    @staticmethod
    def _build_sp_probe_transitions(env: Any) -> list[dict[str, Any]]:
        core = getattr(env, "env", None)
        if core is None or not hasattr(core, "nodes") or not hasattr(core, "node_images"):
            return []
        probes: list[dict[str, Any]] = []
        action_dim = int(getattr(core, "action_space_size", 0))
        for state_id, node in enumerate(core.nodes):
            if int(node.depth) >= int(core.depth) - 1:
                continue
            candidates = [
                idx
                for idx, child in enumerate(core.nodes)
                if child.depth == node.depth + 1
                and np.sum(
                    np.abs(
                        np.asarray(child.coordinates[: core.hyperplane_dim])
                        - np.asarray(node.coordinates[: core.hyperplane_dim])
                    )
                )
                == 0.5 * core.hyperplane_dim
            ]
            for action in range(action_dim):
                next_state_id = int(candidates[action % len(candidates)])
                probes.append(
                    {
                        "probe_id": len(probes),
                        "state_id": state_id,
                        "depth": int(node.depth) + 1,
                        "action": action,
                        "next_state_id": next_state_id,
                        "edge_id": f"{state_id}:{action}:{next_state_id}",
                        "observation": np.asarray(core.node_images[state_id], dtype=np.float32),
                        "next_observation": np.asarray(
                            core.node_images[next_state_id], dtype=np.float32
                        ),
                    }
                )
        return probes

    def current_coef(self) -> float:
        steps = max(int(self.config.coef_decay_steps), 1)
        fraction = min(float(self._env_step_count) / float(steps), 1.0)
        return float(
            self.config.initial_coef
            + fraction * (self.config.final_coef - self.config.initial_coef)
        )

    @staticmethod
    def _default_conv_layers() -> Sequence[Mapping[str, Any]]:
        return (
            {"out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
            {"out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        )

    @staticmethod
    def _canonicalize_single(observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 0:
            raise ValueError("LPM does not support scalar observations")
        if obs.ndim == 3 and obs.shape[0] not in (1, 3, 4) and obs.shape[-1] in (1, 3, 4):
            obs = np.transpose(obs, (2, 0, 1))
        return np.ascontiguousarray(obs)

    def _resolve_encoder_type(self, observation: np.ndarray) -> str:
        kind = str(self.config.encoder_type).lower()
        if kind in {"mlp", "cnn"}:
            return kind
        if kind in {"auto", "infer"}:
            return "mlp" if observation.ndim == 1 else "cnn"
        raise ValueError(f"Unsupported LPM encoder_type: {self.config.encoder_type}")

    def _make_encoder(self, observation: np.ndarray, encoder_type: str) -> nn.Module:
        if encoder_type == "mlp":
            return _MLP(
                int(np.prod(observation.shape)),
                (int(self.config.feature_dim),),
                int(self.config.feature_dim),
                self.config.activation,
            )
        channels = int(observation.shape[0]) if observation.ndim == 3 else 1
        return _ConvEncoder(
            channels,
            self.config.conv_layers or self._default_conv_layers(),
            self.config.feature_dim,
            self.config.activation,
        )

    def _ensure_networks(self, observation: np.ndarray, action_dim: int) -> None:
        observation = self._canonicalize_single(observation)
        if self.dynamics_model is not None:
            if int(action_dim) != self.action_dim:
                raise ValueError(
                    f"LPM action dimension changed from {self.action_dim} to {action_dim}"
                )
            if tuple(observation.shape) != self._observation_shape:
                raise ValueError(
                    f"LPM observation shape changed from {self._observation_shape} "
                    f"to {tuple(observation.shape)}"
                )
            return

        if self.config.seed is not None:
            th.manual_seed(int(self.config.seed))
        encoder_type = self._resolve_encoder_type(observation)
        if encoder_type == "cnn" and observation.ndim == 2:
            observation = observation[None, :, :]
        if encoder_type == "cnn" and observation.ndim != 3:
            raise ValueError(
                f"CNN LPM expects (C,H,W) or (H,W), got {tuple(observation.shape)}"
            )
        self.action_dim = int(action_dim)
        self._encoder_type_resolved = encoder_type
        self._observation_shape = tuple(observation.shape)
        output_dim = int(np.prod(observation.shape))
        self.dynamics_model = _DynamicsModel(
            self._make_encoder(observation, encoder_type),
            self.config.feature_dim,
            self.action_dim,
            output_dim,
            self.config.dynamics_hidden_layers,
            self.config.activation,
        ).to(self.device)
        self.error_model = _ErrorModel(
            self._make_encoder(observation, encoder_type),
            self.config.feature_dim,
            self.action_dim,
            self.config.error_hidden_layers,
            self.config.activation,
        ).to(self.device)
        self.dynamics_optimizer = th.optim.Adam(
            self.dynamics_model.parameters(), lr=float(self.config.dynamics_learning_rate)
        )
        self.error_optimizer = th.optim.Adam(
            self.error_model.parameters(), lr=float(self.config.error_learning_rate)
        )

    def _prepare_single(self, observation: np.ndarray) -> th.Tensor:
        obs = self._canonicalize_single(observation)
        if self._encoder_type_resolved == "cnn" and obs.ndim == 2:
            obs = obs[None, :, :]
        return th.as_tensor(obs[None, ...], dtype=th.float32, device=self.device)

    def _prepare_batch(self, observations: th.Tensor | np.ndarray) -> th.Tensor:
        obs = th.as_tensor(observations, dtype=th.float32, device=self.device)
        if self._encoder_type_resolved == "mlp":
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
            return obs
        if obs.ndim == 3:
            obs = obs.unsqueeze(1)
        elif obs.ndim == 4 and obs.shape[1] not in (1, 3, 4) and obs.shape[-1] in (1, 3, 4):
            obs = obs.permute(0, 3, 1, 2)
        if obs.ndim != 4:
            raise ValueError(f"CNN LPM expects a 4-D batch, got {tuple(obs.shape)}")
        return obs

    def _prediction_error(
        self, observations: th.Tensor, next_observations: th.Tensor, actions: th.Tensor
    ) -> th.Tensor:
        assert self.dynamics_model is not None
        prediction = self.dynamics_model(observations, actions)
        target = next_observations.reshape(next_observations.shape[0], -1)
        return th.mean((prediction - target) ** 2, dim=1)

    def _postprocess_reward(self, raw_reward: float) -> tuple[float, float]:
        reward = float(raw_reward)
        if self.config.clamp_min is not None:
            reward = max(reward, float(self.config.clamp_min))
        if self.config.clamp_max is not None:
            reward = min(reward, float(self.config.clamp_max))
        raw_used = reward
        if self.config.normalize_reward:
            self._discounted_intrinsic_return = (
                float(self.config.reward_gamma) * self._discounted_intrinsic_return + reward
            )
            self.return_rms.update(np.asarray([self._discounted_intrinsic_return]))
            reward /= np.sqrt(float(self.return_rms.var) + float(self.config.reward_norm_eps))
        if self.config.intrinsic_clip is not None:
            limit = float(self.config.intrinsic_clip)
            reward = float(np.clip(reward, -limit, limit))
        return float(raw_used), float(reward)

    def compute(
        self,
        *,
        previous_observation: Optional[np.ndarray] = None,
        observation: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
        action: Optional[int] = None,
    ) -> float:
        if previous_observation is None or observation is None or action is None:
            return 0.0
        action_dim = int((info or {}).get("action_space_n", int(action) + 1))
        previous = self._canonicalize_single(previous_observation)
        current = self._canonicalize_single(observation)
        self._ensure_networks(previous, action_dim)
        previous_tensor = self._prepare_single(previous)
        current_tensor = self._prepare_single(current)
        action_tensor = th.as_tensor([int(action)], dtype=th.long, device=self.device)

        assert self.error_model is not None
        with th.no_grad():
            actual_error = float(
                self._prediction_error(previous_tensor, current_tensor, action_tensor).item()
            )
            log_expected = self.error_model(previous_tensor, action_tensor)
            predicted_log_error = float(log_expected.item())
            expected_error = float(
                th.exp(
                    th.clamp(log_expected, min=-20.0, max=20.0)
                ).item()
            )

        # Queue entries are generated by the pre-update dynamics model.  The
        # error model consumes these entries after collection, approximating
        # epsilon^(tau-1) as required by LPM.
        current_log_error = float(
            np.log(max(actual_error, float(self.config.log_error_epsilon)))
        )
        self.error_buffer.append(
            _ErrorQueueEntry(
                observation=previous.copy(),
                action=int(action),
                error_mse=actual_error,
                error_log=current_log_error,
                dynamics_version=self.dynamics_version,
            )
        )
        # Paper Eq. (3): intrinsic reward is the difference in LOG-MSE space.
        lpm_signed = predicted_log_error - current_log_error

        # Do not emit LPM reward until:
        #   1) the fixed-size error queue is full,
        #   2) both models have completed at least one joint update cycle, and
        #   3) g_phi predicts the immediately previous dynamics version.
        warmup = max(int(self.config.reward_warmup_size), 0)
        queue_ready = len(self.error_buffer) >= warmup
        models_ready = self.dynamics_version > 0 and self.error_model_version > 0
        version_ready = (
            self.error_target_dynamics_version == self.dynamics_version - 1
        )
        warmup_active = not (queue_ready and models_ready and version_ready)

        paper_lpm_reward = 0.0 if warmup_active else lpm_signed
        raw_used, normalized_reward = self._postprocess_reward(paper_lpm_reward)

        self.last_actual_error = actual_error
        self.last_expected_error = expected_error
        self.last_raw_intrinsic = paper_lpm_reward
        info_values = info or {}
        self._pending_transition = {
            "env_step": self._env_step_count + 1,
            "episode": self._episode,
            "episode_step": self._episode_step,
            "seed": self._logging_seed,
            "state_id": info_values.get("state_id", ""),
            "depth": info_values.get("state_depth", info_values.get("depth", "")),
            "action": int(action),
            "next_state_id": info_values.get("next_state_id", ""),
            "edge_id": info_values.get("edge_id", ""),
            "dynamics_error_mse": actual_error,
            "dynamics_error_log": current_log_error,
            "predicted_previous_error": predicted_log_error,
            "predicted_previous_error_mse": expected_error,
            "lpm_signed": lpm_signed,
            "lpm_positive": max(lpm_signed, 0.0),
            "lpm_raw_used": raw_used,
            "intrinsic_reward_normalized": normalized_reward,
            "dynamics_version": self.dynamics_version,
            "error_model_version": self.error_model_version,
            "error_target_dynamics_version": self.error_target_dynamics_version,
            "replay_size": self._last_replay_size,
            "error_queue_size": len(self.error_buffer),
            "warmup_active": int(warmup_active),
        }
        self._env_step_count += 1
        self._episode_step += 1
        return normalized_reward

    def record_rl_transition(
        self,
        *,
        extrinsic_reward: float,
        intrinsic_reward_normalized: float,
        intrinsic_coef: float,
        intrinsic_reward_to_rl: float,
        total_reward_to_rl: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self._pending_transition is None:
            return
        row = dict(self._pending_transition)
        row.update(
            {
                "extrinsic_reward": float(extrinsic_reward),
                "intrinsic_reward_normalized": float(intrinsic_reward_normalized),
                "intrinsic_coef": float(intrinsic_coef),
                "intrinsic_reward_to_rl": float(intrinsic_reward_to_rl),
                "total_reward_to_rl": float(total_reward_to_rl),
                "terminated": int(terminated),
                "truncated": int(truncated),
            }
        )
        interval = max(int(self.config.transition_log_interval), 1)
        if self._transition_logger is not None and self._env_step_count % interval == 0:
            self._transition_logger.write(row)
        self._pending_transition = None

    @staticmethod
    def _parameter_norm(module: nn.Module) -> float:
        with th.no_grad():
            total = sum(
                float(th.sum(parameter.detach() ** 2).item())
                for parameter in module.parameters()
            )
        return float(np.sqrt(total))

    @staticmethod
    def _gradient_norm(module: nn.Module) -> float:
        total = 0.0
        for parameter in module.parameters():
            if parameter.grad is not None:
                total += float(th.sum(parameter.grad.detach() ** 2).item())
        return float(np.sqrt(total))

    def _update_error_model(
        self, target_dynamics_version: int
    ) -> Optional[dict[str, float]]:
        # Algorithm 1 requires g_phi^(tau+1) to learn errors produced by
        # exactly f_theta^(tau), not by a mixture of older model versions.
        version_samples = [
            sample
            for sample in self.error_buffer
            if sample.dynamics_version == int(target_dynamics_version)
        ]
        batch_size = min(int(self.config.error_batch_size), len(version_samples))
        if batch_size <= 0:
            return None
        indices = self._rng.choice(len(version_samples), size=batch_size, replace=False)
        samples = [version_samples[int(index)] for index in indices]
        observations = np.stack([sample.observation for sample in samples])
        actions = th.as_tensor(
            [sample.action for sample in samples], dtype=th.long, device=self.device
        )
        target_log_error = th.as_tensor(
            [sample.error_log for sample in samples], dtype=th.float32, device=self.device
        )
        target_versions = np.asarray([sample.dynamics_version for sample in samples])
        x = self._prepare_batch(observations)
        assert self.error_model is not None
        assert self.error_optimizer is not None
        predicted_log_error = self.error_model(x, actions)
        loss = F.mse_loss(predicted_log_error, target_log_error)
        self.error_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._gradient_norm(self.error_model)
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.error_model.parameters(), self.config.max_grad_norm)
        self.error_optimizer.step()
        self.error_model_version += 1
        self.error_target_dynamics_version = int(np.max(target_versions))
        residual = predicted_log_error.detach() - target_log_error
        return {
            "loss": float(loss.detach().cpu().item()),
            "prediction_mean": float(predicted_log_error.detach().mean().cpu().item()),
            "target_mean": float(target_log_error.mean().cpu().item()),
            "bias": float(residual.mean().cpu().item()),
            "mae": float(residual.abs().mean().cpu().item()),
            "rmse": float(th.sqrt(th.mean(residual**2)).cpu().item()),
            "grad_norm": grad_norm,
            "target_version_min": int(np.min(target_versions)),
            "target_version_max": int(np.max(target_versions)),
        }

    def _write_probe(
        self, previous_model: _DynamicsModel, previous_dynamics_version: int
    ) -> None:
        if self._probe_logger is None or not self._probe_transitions:
            return
        assert self.dynamics_model is not None
        assert self.error_model is not None
        observations = self._prepare_batch(
            np.stack([probe["observation"] for probe in self._probe_transitions])
        )
        next_observations = self._prepare_batch(
            np.stack([probe["next_observation"] for probe in self._probe_transitions])
        )
        actions = th.as_tensor(
            [probe["action"] for probe in self._probe_transitions],
            dtype=th.long,
            device=self.device,
        )
        targets = next_observations.reshape(next_observations.shape[0], -1)
        epsilon = float(self.config.log_error_epsilon)
        with th.no_grad():
            current_prediction = self.dynamics_model(observations, actions)
            previous_prediction = previous_model(observations, actions)
            current_mse = th.mean((current_prediction - targets) ** 2, dim=1)
            previous_mse = th.mean((previous_prediction - targets) ** 2, dim=1)
            current_log = th.log(current_mse.clamp_min(epsilon))
            previous_log = th.log(previous_mse.clamp_min(epsilon))
            predicted_previous_log = self.error_model(observations, actions)

        base_env = self._probe_env
        core = getattr(base_env, "env", None)
        state_action_counts = getattr(base_env, "state_action_counts", None)
        for index, probe in enumerate(self._probe_transitions):
            state_id = int(probe["state_id"])
            action = int(probe["action"])
            state_visits = (
                int(core.nodes[state_id].visit_count)
                if core is not None and hasattr(core, "nodes")
                else ""
            )
            state_action_visits = (
                int(state_action_counts[state_id, action])
                if state_action_counts is not None
                else ""
            )
            oracle_progress = float((previous_log[index] - current_log[index]).cpu().item())
            estimated_progress = float(
                (predicted_previous_log[index] - current_log[index]).cpu().item()
            )
            self._probe_logger.write(
                {
                    "env_step": self._env_step_count,
                    "probe_id": probe["probe_id"],
                    "state_id": state_id,
                    "depth": probe["depth"],
                    "action": action,
                    "next_state_id": probe["next_state_id"],
                    "edge_id": probe["edge_id"],
                    "state_visit_count": state_visits,
                    "state_action_visit_count": state_action_visits,
                    "current_dynamics_version": self.dynamics_version,
                    "previous_dynamics_version": previous_dynamics_version,
                    "current_error_mse": float(current_mse[index].cpu().item()),
                    "current_error_log": float(current_log[index].cpu().item()),
                    "previous_error_mse": float(previous_mse[index].cpu().item()),
                    "previous_error_log": float(previous_log[index].cpu().item()),
                    "predicted_previous_error": float(
                        predicted_previous_log[index].cpu().item()
                    ),
                    "oracle_pointwise_progress": oracle_progress,
                    "estimated_lpm_signed": estimated_progress,
                    "estimated_lpm_positive": max(estimated_progress, 0.0),
                    "error_prediction_residual": float(
                        (predicted_previous_log[index] - previous_log[index]).cpu().item()
                    ),
                }
            )

    def update_from_batch(
        self,
        observations: th.Tensor,
        next_observations: th.Tensor,
        actions: th.Tensor,
        *,
        env_step: Optional[int] = None,
        replay_size: Optional[int] = None,
    ) -> dict[str, float]:
        if self.dynamics_model is None:
            sample = observations[0].detach().cpu().numpy()
            inferred_action_dim = int(actions.max().item()) + 1
            self._ensure_networks(sample, inferred_action_dim)
        previous = self._prepare_batch(observations)
        current = self._prepare_batch(next_observations)
        action = actions.detach().to(self.device).reshape(-1).long()
        assert self.dynamics_optimizer is not None

        self._model_update_count += 1
        if replay_size is not None:
            self._last_replay_size = int(replay_size)
        dynamics_version_before = self.dynamics_version
        error_version_before = self.error_model_version
        with th.no_grad():
            batch_errors = self._prediction_error(previous, current, action)
            batch_log_errors = th.log(
                batch_errors.clamp_min(float(self.config.log_error_epsilon))
            )
            assert self.error_model is not None
            batch_predicted_logs = self.error_model(previous, action)
            batch_lpm = batch_predicted_logs - batch_log_errors

        # Joint update cycle: first fit g_phi to errors generated by the
        # current frozen dynamics version, then update f_theta once.
        error_metrics: list[dict[str, float]] = []
        for _ in range(max(int(self.config.error_updates), 0)):
            metrics = self._update_error_model(dynamics_version_before)
            if metrics is not None:
                error_metrics.append(metrics)

        # Keep f_theta fixed if g_phi could not be updated for this version.
        # This preserves the one-cycle lag required by LPM.
        if max(int(self.config.error_updates), 0) > 0 and not error_metrics:
            return {
                "dynamics_loss": self.last_dynamics_loss,
                "error_loss": self.last_error_loss,
            }

        dynamics_losses: list[float] = []
        dynamics_grad_norms: list[float] = []
        encoder_grad_norms: list[float] = []
        probe_interval = max(int(self.config.probe_interval), 1)
        last_probe_step = max(self._last_probe_env_step, 0)
        probe_due = (
            self._probe_logger is not None
            and bool(self._probe_transitions)
            and self._env_step_count >= probe_interval
            and self._env_step_count - last_probe_step >= probe_interval
        )
        previous_model: Optional[_DynamicsModel] = None
        for _ in range(max(int(self.config.dynamics_updates), 0)):
            if probe_due and previous_model is None:
                assert self.dynamics_model is not None
                previous_model = copy.deepcopy(self.dynamics_model).to(self.device).eval()
            errors = self._prediction_error(previous, current, action)
            loss = errors.mean()
            self.dynamics_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert self.dynamics_model is not None
            dynamics_grad_norms.append(self._gradient_norm(self.dynamics_model))
            encoder_grad_norms.append(self._gradient_norm(self.dynamics_model.encoder))
            if self.config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.dynamics_model.parameters(), self.config.max_grad_norm
                )
            self.dynamics_optimizer.step()
            self.dynamics_version += 1
            dynamics_losses.append(float(loss.detach().cpu().item()))

        self.last_dynamics_loss = float(np.mean(dynamics_losses)) if dynamics_losses else 0.0
        self.last_error_loss = (
            float(np.mean([metrics["loss"] for metrics in error_metrics]))
            if error_metrics
            else 0.0
        )
        error_average = {
            key: float(np.mean([metrics[key] for metrics in error_metrics]))
            for key in ("prediction_mean", "target_mean", "bias", "mae", "rmse", "grad_norm")
        } if error_metrics else {}
        update_values = {
            "env_step": int(env_step if env_step is not None else self._env_step_count),
            "lpm_update_step": self._model_update_count,
            "dynamics_version_before": dynamics_version_before,
            "dynamics_version_after": self.dynamics_version,
            "error_model_version_before": error_version_before,
            "error_model_version_after": self.error_model_version,
            "error_target_version_min": (
                min(int(metrics["target_version_min"]) for metrics in error_metrics)
                if error_metrics else ""
            ),
            "error_target_version_max": (
                max(int(metrics["target_version_max"]) for metrics in error_metrics)
                if error_metrics else ""
            ),
            "batch_size": int(action.shape[0]),
            "replay_size": self._last_replay_size,
            "error_queue_size": len(self.error_buffer),
            "dynamics_loss": self.last_dynamics_loss,
            "error_model_loss": self.last_error_loss if error_metrics else "",
            "batch_current_error_mean": float(batch_errors.mean().cpu().item()),
            "batch_current_error_std": float(batch_errors.std(unbiased=False).cpu().item()),
            "batch_current_log_error_mean": float(batch_log_errors.mean().cpu().item()),
            "batch_current_log_error_std": float(batch_log_errors.std(unbiased=False).cpu().item()),
            "batch_error_prediction_mean": error_average.get("prediction_mean", ""),
            "batch_error_target_mean": error_average.get("target_mean", ""),
            "batch_error_prediction_bias": error_average.get("bias", ""),
            "batch_error_prediction_mae": error_average.get("mae", ""),
            "batch_error_prediction_rmse": error_average.get("rmse", ""),
            "batch_lpm_signed_mean": float(batch_lpm.mean().cpu().item()),
            "batch_lpm_signed_std": float(batch_lpm.std(unbiased=False).cpu().item()),
            "batch_lpm_positive_mean": float(batch_lpm.clamp_min(0).mean().cpu().item()),
            "batch_lpm_positive_rate": float((batch_lpm > 0).float().mean().cpu().item()),
            "batch_lpm_negative_rate": float((batch_lpm < 0).float().mean().cpu().item()),
            "dynamics_grad_norm": (
                float(np.mean(dynamics_grad_norms)) if dynamics_grad_norms else 0.0
            ),
            "error_model_grad_norm": error_average.get("grad_norm", ""),
            "encoder_grad_norm": float(np.mean(encoder_grad_norms)) if encoder_grad_norms else 0.0,
            "dynamics_param_norm": self._parameter_norm(self.dynamics_model),
            "error_model_param_norm": self._parameter_norm(self.error_model),
            "dynamics_learning_rate": self.dynamics_optimizer.param_groups[0]["lr"],
            "error_model_learning_rate": (
                self.error_optimizer.param_groups[0]["lr"] if self.error_optimizer else ""
            ),
        }
        log_interval = max(int(self.config.update_log_interval), 1)
        if self._update_logger is not None and self._model_update_count % log_interval == 0:
            self._update_logger.write(update_values)
        if probe_due and previous_model is not None:
            self._write_probe(previous_model, dynamics_version_before)
            self._last_probe_env_step = self._env_step_count

        return {
            "dynamics_loss": self.last_dynamics_loss,
            "error_loss": self.last_error_loss,
        }


def get_lpm_augmented_dqn_class():
    from stable_baselines3 import DQN

    class LPMAugmentedDQN(DQN):
        """DQN with paper-style periodic LPM model updates.

        LPM is updated independently of DQN ``learning_starts``.  The dynamics
        model and error model remain frozen during each collection cycle and
        are updated jointly every ``model_update_cycle`` environment steps.
        """

        def __init__(
            self,
            *args,
            lpm_module: Optional[LPMIntrinsicReward] = None,
            **kwargs,
        ) -> None:
            self.lpm_module = lpm_module
            super().__init__(*args, **kwargs)

        def _on_step(self) -> None:
            super()._on_step()
            self._maybe_update_lpm()

        def _maybe_update_lpm(self) -> None:
            if self.lpm_module is None:
                return

            cycle = max(int(self.lpm_module.config.model_update_cycle), 1)
            elapsed = self.num_timesteps - self.lpm_module._last_lpm_update_env_step
            if elapsed < cycle:
                return
            if self.replay_buffer.size() < int(self.batch_size):
                return

            replay_data = self.replay_buffer.sample(
                int(self.batch_size), env=self._vec_normalize_env
            )
            losses = self.lpm_module.update_from_batch(
                replay_data.observations,
                replay_data.next_observations,
                replay_data.actions,
                env_step=self.num_timesteps,
                replay_size=self.replay_buffer.size(),
            )
            self.lpm_module._last_lpm_update_env_step = int(self.num_timesteps)

            self.logger.record("train/lpm_dynamics_loss", losses["dynamics_loss"])
            self.logger.record("train/lpm_error_loss", losses["error_loss"])
            self.logger.record("train/lpm_actual_error", self.lpm_module.last_actual_error)
            self.logger.record("train/lpm_expected_error", self.lpm_module.last_expected_error)
            self.logger.record("train/lpm_raw_intrinsic", self.lpm_module.last_raw_intrinsic)
            self.logger.record(
                "train/lpm_error_buffer_size", len(self.lpm_module.error_buffer)
            )
            self.logger.record(
                "train/lpm_model_updates", self.lpm_module._model_update_count
            )
            self.logger.record("train/lpm_reward_coef", self.lpm_module.current_coef())

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            # DQN updates remain unchanged. LPM updates happen in _on_step()
            # according to its own paper-style collection/update cycle.
            super().train(gradient_steps=gradient_steps, batch_size=batch_size)

    return LPMAugmentedDQN

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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
    device: str = "auto"
    seed: Optional[int] = None


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
    """Predict log E[epsilon] from the current observation and action."""

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
        self.error_buffer: Deque[tuple[np.ndarray, int, float]] = deque(
            maxlen=int(config.error_buffer_size)
        )
        self._rng = np.random.default_rng(config.seed)
        self._env_step_count = 0
        self._model_update_count = 0
        self._discounted_intrinsic_return = 0.0
        self.return_rms = RunningMeanStd(epsilon=1.0e-4)
        self.last_actual_error = 0.0
        self.last_expected_error = 0.0
        self.last_raw_intrinsic = 0.0
        self.last_dynamics_loss = 0.0
        self.last_error_loss = 0.0

    def reset_episode(self) -> None:
        self._discounted_intrinsic_return = 0.0

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

    def _postprocess_reward(self, raw_reward: float) -> float:
        reward = float(raw_reward)
        if self.config.clamp_min is not None:
            reward = max(reward, float(self.config.clamp_min))
        if self.config.clamp_max is not None:
            reward = min(reward, float(self.config.clamp_max))
        if self.config.normalize_reward:
            self._discounted_intrinsic_return = (
                float(self.config.reward_gamma) * self._discounted_intrinsic_return + reward
            )
            self.return_rms.update(np.asarray([self._discounted_intrinsic_return]))
            reward /= np.sqrt(float(self.return_rms.var) + float(self.config.reward_norm_eps))
        if self.config.intrinsic_clip is not None:
            limit = float(self.config.intrinsic_clip)
            reward = float(np.clip(reward, -limit, limit))
        return reward

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
            expected_error = float(
                th.exp(
                    th.clamp(log_expected, min=-20.0, max=20.0)
                ).item()
            )

        # Queue entries are generated by the pre-update dynamics model.  The
        # error model consumes these entries after collection, approximating
        # epsilon^(tau-1) as required by LPM.
        self.error_buffer.append((previous.copy(), int(action), actual_error))
        warmup = max(int(self.config.reward_warmup_size), 0)
        raw = 0.0
        if len(self.error_buffer) >= warmup:
            raw = float(self.config.expected_error_scale) * expected_error - actual_error

        self.last_actual_error = actual_error
        self.last_expected_error = expected_error
        self.last_raw_intrinsic = raw
        self._env_step_count += 1
        return self._postprocess_reward(raw)

    def _update_error_model(self) -> Optional[float]:
        batch_size = min(int(self.config.error_batch_size), len(self.error_buffer))
        if batch_size <= 0:
            return None
        indices = self._rng.choice(len(self.error_buffer), size=batch_size, replace=False)
        samples = [self.error_buffer[int(index)] for index in indices]
        observations = np.stack([sample[0] for sample in samples])
        actions = th.as_tensor(
            [sample[1] for sample in samples], dtype=th.long, device=self.device
        )
        errors = th.as_tensor(
            [sample[2] for sample in samples], dtype=th.float32, device=self.device
        )
        x = self._prepare_batch(observations)
        assert self.error_model is not None
        assert self.error_optimizer is not None
        predicted_log_error = self.error_model(x, actions)
        target_log_error = th.log(errors.clamp_min(float(self.config.log_error_epsilon)))
        loss = F.mse_loss(predicted_log_error, target_log_error)
        self.error_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.error_model.parameters(), self.config.max_grad_norm)
        self.error_optimizer.step()
        return float(loss.detach().cpu().item())

    def update_from_batch(
        self,
        observations: th.Tensor,
        next_observations: th.Tensor,
        actions: th.Tensor,
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
        error_losses: list[float] = []
        error_update_interval = max(int(self.config.error_update_interval), 1)
        if self._model_update_count % error_update_interval == 0:
            for _ in range(max(int(self.config.error_updates), 0)):
                error_loss = self._update_error_model()
                if error_loss is not None:
                    error_losses.append(error_loss)

        dynamics_losses: list[float] = []
        for _ in range(max(int(self.config.dynamics_updates), 0)):
            errors = self._prediction_error(previous, current, action)
            loss = errors.mean()
            self.dynamics_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.max_grad_norm is not None:
                assert self.dynamics_model is not None
                nn.utils.clip_grad_norm_(
                    self.dynamics_model.parameters(), self.config.max_grad_norm
                )
            self.dynamics_optimizer.step()
            dynamics_losses.append(float(loss.detach().cpu().item()))

        self.last_dynamics_loss = float(np.mean(dynamics_losses)) if dynamics_losses else 0.0
        self.last_error_loss = float(np.mean(error_losses)) if error_losses else 0.0
        return {
            "dynamics_loss": self.last_dynamics_loss,
            "error_loss": self.last_error_loss,
        }


def get_lpm_augmented_dqn_class():
    from stable_baselines3 import DQN

    class LPMAugmentedDQN(DQN):
        """DQN that trains LPM from the same replay buffer."""

        def __init__(
            self,
            *args,
            lpm_module: Optional[LPMIntrinsicReward] = None,
            **kwargs,
        ) -> None:
            self.lpm_module = lpm_module
            super().__init__(*args, **kwargs)

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            super().train(gradient_steps=gradient_steps, batch_size=batch_size)
            if self.lpm_module is None:
                return
            dynamics_losses: list[float] = []
            error_losses: list[float] = []
            for _ in range(gradient_steps):
                replay_data = self.replay_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )
                losses = self.lpm_module.update_from_batch(
                    replay_data.observations,
                    replay_data.next_observations,
                    replay_data.actions,
                )
                dynamics_losses.append(losses["dynamics_loss"])
                error_losses.append(losses["error_loss"])
            self.logger.record("train/lpm_dynamics_loss", float(np.mean(dynamics_losses)))
            self.logger.record("train/lpm_error_loss", float(np.mean(error_losses)))
            self.logger.record("train/lpm_actual_error", self.lpm_module.last_actual_error)
            self.logger.record("train/lpm_expected_error", self.lpm_module.last_expected_error)
            self.logger.record("train/lpm_raw_intrinsic", self.lpm_module.last_raw_intrinsic)
            self.logger.record("train/lpm_error_buffer_size", len(self.lpm_module.error_buffer))
            self.logger.record("train/lpm_model_updates", self.lpm_module._model_update_count)
            self.logger.record("train/lpm_reward_coef", self.lpm_module.current_coef())

    return LPMAugmentedDQN

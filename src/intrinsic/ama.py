from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd:
    """Running mean and variance with Welford-style parallel updates."""

    def __init__(self, epsilon: float = 1.0e-4, shape: Sequence[int] | tuple = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            x = x.reshape(1)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count


def _get_activation(name: str) -> nn.Module:
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


def _resolve_device(device: str) -> th.device:
    key = str(device).lower()
    if key == "auto":
        if th.cuda.is_available():
            return th.device("cuda")
        if hasattr(th.backends, "mps") and th.backends.mps.is_available():
            return th.device("mps")
        return th.device("cpu")
    return th.device(device)


@dataclass
class AMAConfig:
    learning_rate: float = 1.0e-4
    encoder_type: str = "auto"
    conv_layers: Optional[Sequence[Mapping[str, Any]]] = None
    activation: str = "relu"
    feature_dim: int = 64
    predictor_hidden_layers: Sequence[int] = (256, 256)
    uncertainty_lambda: float = 0.1
    uncertainty_eta: float = 1.0
    logvar_min: float = -10.0
    logvar_max: float = 6.0
    clamp_min: Optional[float] = 0.0
    clamp_max: Optional[float] = None
    normalize_reward: bool = False
    reward_gamma: float = 0.99
    reward_norm_eps: float = 1.0e-8
    intrinsic_clip: Optional[float] = None
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
        in_dim = int(input_dim)
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(_get_activation(activation))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class _ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_layers: Sequence[Mapping[str, Any]],
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = int(in_channels)
        for layer_cfg in conv_layers:
            out_channels = int(layer_cfg["out_channels"])
            blocks.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=int(layer_cfg.get("kernel_size", 3)),
                    stride=int(layer_cfg.get("stride", 1)),
                    padding=int(layer_cfg.get("padding", 0)),
                )
            )
            blocks.append(_get_activation(activation))
            current_channels = out_channels
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = _MLP(current_channels, (), output_dim, activation)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = th.flatten(x, start_dim=1)
        return self.proj(x)


class _AMAForwardModel(nn.Module):
    def __init__(
        self,
        *,
        obs_shape: tuple[int, ...],
        action_dim: int,
        encoder_type: str,
        conv_layers: Sequence[Mapping[str, Any]],
        activation: str,
        feature_dim: int,
        predictor_hidden_layers: Sequence[int],
    ) -> None:
        super().__init__()
        self.obs_dim = int(np.prod(obs_shape))
        self.action_dim = int(action_dim)

        if encoder_type == "mlp":
            self.encoder = _MLP(self.obs_dim, predictor_hidden_layers[:1], feature_dim, activation)
        else:
            in_channels = int(obs_shape[0])
            self.encoder = _ConvEncoder(in_channels, conv_layers, feature_dim, activation)

        self.head = _MLP(
            int(feature_dim) + self.action_dim,
            predictor_hidden_layers,
            2 * self.obs_dim,
            activation,
        )

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        features = self.encoder(obs)
        actions = actions.reshape(-1).long().clamp(0, self.action_dim - 1)
        action_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        out = self.head(th.cat([features, action_onehot], dim=1))
        mean, logvar = th.chunk(out, chunks=2, dim=1)
        return mean, logvar


class AMAIntrinsicReward:
    """Aleatoric Mapping Agent intrinsic reward.

    The forward model predicts both the mean and log variance of the next
    observation. Intrinsic reward follows the paper's Eq. 6:
    mean squared prediction error minus ``eta`` times predicted variance.
    """

    def __init__(self, config: AMAConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.model: Optional[_AMAForwardModel] = None
        self.optimizer: Optional[th.optim.Optimizer] = None
        self._encoder_type_resolved: Optional[str] = None
        self._obs_shape: Optional[tuple[int, ...]] = None
        self._action_dim: Optional[int] = None
        self._discounted_intrinsic_return = 0.0
        self._env_step_count = 0
        self.return_rms = RunningMeanStd(epsilon=1.0e-4)
        self.last_prediction_error = 0.0
        self.last_predicted_variance = 0.0
        self.last_raw_intrinsic = 0.0

    def reset_episode(self) -> None:
        self._discounted_intrinsic_return = 0.0

    def current_coef(self) -> float:
        decay_steps = max(int(self.config.coef_decay_steps), 1)
        t = min(float(self._env_step_count) / float(decay_steps), 1.0)
        return float(self.config.initial_coef + t * (self.config.final_coef - self.config.initial_coef))

    def _default_conv_layers(self) -> Sequence[Mapping[str, Any]]:
        return [
            {"out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
            {"out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        ]

    def _canonicalize_single_observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 0:
            raise ValueError(f"AMA expects non-scalar observation, got shape {obs.shape}")
        if obs.ndim == 3 and obs.shape[0] not in (1, 3, 4) and obs.shape[-1] in (1, 3, 4):
            obs = np.transpose(obs, (2, 0, 1))
        return obs

    def _resolve_encoder_type(self, sample_obs: np.ndarray) -> str:
        key = str(self.config.encoder_type).lower()
        if key in {"mlp", "cnn"}:
            return key
        if key in {"auto", "infer"}:
            return "mlp" if sample_obs.ndim == 1 else "cnn"
        raise ValueError(f"Unsupported AMA encoder_type: {self.config.encoder_type}")

    def _ensure_networks(self, sample_obs: np.ndarray, action_dim: int) -> None:
        sample_obs = self._canonicalize_single_observation(sample_obs)
        if sample_obs.ndim == 2:
            sample_obs = sample_obs[None, :, :]
        action_dim = max(int(action_dim), 1)

        if self.model is not None:
            if self._action_dim is not None and action_dim > self._action_dim:
                raise ValueError(
                    f"AMA was initialized with action_dim={self._action_dim}, got {action_dim}."
                )
            return

        if self.config.seed is not None:
            th.manual_seed(int(self.config.seed))
            np.random.seed(int(self.config.seed))

        encoder_type = self._resolve_encoder_type(sample_obs)
        if encoder_type == "cnn" and sample_obs.ndim != 3:
            raise ValueError(f"CNN AMA expects observation shape (C,H,W) or (H,W), got {sample_obs.shape}")

        conv_layers = self.config.conv_layers or self._default_conv_layers()
        self.model = _AMAForwardModel(
            obs_shape=tuple(sample_obs.shape),
            action_dim=action_dim,
            encoder_type=encoder_type,
            conv_layers=conv_layers,
            activation=self.config.activation,
            feature_dim=int(self.config.feature_dim),
            predictor_hidden_layers=tuple(self.config.predictor_hidden_layers),
        ).to(self.device)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=float(self.config.learning_rate))
        self._encoder_type_resolved = encoder_type
        self._obs_shape = tuple(sample_obs.shape)
        self._action_dim = action_dim

    def _action_dim_from_info(self, info: Optional[dict[str, Any]], action: Optional[int]) -> int:
        if info is not None and "action_space_n" in info:
            return int(info["action_space_n"])
        if action is None:
            return 1
        return int(action) + 1

    def _prepare_single_state(self, observation: np.ndarray, action_dim: int) -> th.Tensor:
        obs = self._canonicalize_single_observation(observation)
        self._ensure_networks(obs, action_dim)
        assert self._encoder_type_resolved is not None
        if self._encoder_type_resolved == "mlp":
            x = obs.reshape(1, -1)
        else:
            if obs.ndim == 2:
                obs = obs[None, :, :]
            x = obs[None, ...]
        return th.as_tensor(x, dtype=th.float32, device=self.device)

    def _prepare_batch_states(self, observations: th.Tensor | np.ndarray, action_dim: int) -> th.Tensor:
        if isinstance(observations, np.ndarray):
            obs = th.as_tensor(observations, dtype=th.float32, device=self.device)
        else:
            obs = observations.detach().to(self.device).float()

        if self.model is None:
            self._ensure_networks(obs[0].detach().cpu().numpy(), action_dim)
        assert self._encoder_type_resolved is not None

        if self._encoder_type_resolved == "mlp":
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
            return obs.reshape(obs.shape[0], -1)

        if obs.ndim == 3:
            if obs.shape[-1] in (1, 3, 4) and obs.shape[0] not in (1, 3, 4):
                obs = obs.permute(2, 0, 1).unsqueeze(0)
            else:
                obs = obs.unsqueeze(1)
        elif obs.ndim == 4:
            if obs.shape[1] not in (1, 3, 4) and obs.shape[-1] in (1, 3, 4):
                obs = obs.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"CNN AMA expects batched image observations with ndim 3 or 4, got {tuple(obs.shape)}")
        return obs

    def _prepare_targets(self, observations: th.Tensor | np.ndarray, action_dim: int) -> th.Tensor:
        obs = self._prepare_batch_states(observations, action_dim)
        return obs.reshape(obs.shape[0], -1)

    def _raw_terms(self, previous_obs: th.Tensor, actions: th.Tensor, next_obs_flat: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        assert self.model is not None
        mean, logvar = self.model(previous_obs, actions)
        logvar = th.clamp(logvar, float(self.config.logvar_min), float(self.config.logvar_max))
        squared_error = (next_obs_flat - mean) ** 2
        variance = th.exp(logvar)
        raw = th.mean(squared_error - float(self.config.uncertainty_eta) * variance, dim=1)
        return raw, squared_error, variance

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
            self.return_rms.update(np.asarray([self._discounted_intrinsic_return], dtype=np.float64))
            reward /= np.sqrt(float(self.return_rms.var) + float(self.config.reward_norm_eps))
        if self.config.intrinsic_clip is not None:
            limit = float(self.config.intrinsic_clip)
            reward = float(np.clip(reward, -limit, limit))
        return float(reward)

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
        action_dim = self._action_dim_from_info(info, action)
        prev = self._prepare_single_state(previous_observation, action_dim)
        next_flat = self._prepare_targets(np.asarray(observation, dtype=np.float32)[None, ...], action_dim)
        act = th.as_tensor([int(action)], dtype=th.long, device=self.device)
        with th.no_grad():
            raw, squared_error, variance = self._raw_terms(prev, act, next_flat)
        raw_value = float(raw.detach().cpu().item())
        self.last_prediction_error = float(th.mean(squared_error).detach().cpu().item())
        self.last_predicted_variance = float(th.mean(variance).detach().cpu().item())
        self.last_raw_intrinsic = raw_value
        self._env_step_count += 1
        return self._postprocess_reward(raw_value)

    def update_from_batch(
        self,
        observations: th.Tensor,
        next_observations: th.Tensor,
        actions: th.Tensor,
    ) -> float:
        action_dim = self._action_dim
        if action_dim is None:
            action_dim = int(actions.detach().max().cpu().item()) + 1 if actions.numel() else 1
        prev = self._prepare_batch_states(observations, action_dim)
        next_flat = self._prepare_targets(next_observations, action_dim)
        act = actions.detach().to(self.device).reshape(-1).long()
        assert self.model is not None
        assert self.optimizer is not None

        mean, logvar = self.model(prev, act)
        logvar = th.clamp(logvar, float(self.config.logvar_min), float(self.config.logvar_max))
        squared_error = (next_flat - mean) ** 2
        loss = th.mean(
            0.5 * th.exp(-logvar) * squared_error
            + 0.5 * float(self.config.uncertainty_lambda) * logvar
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm is not None and float(self.config.max_grad_norm) > 0:
            th.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.max_grad_norm))
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def export_raw_intrinsic_per_state(self, env) -> np.ndarray:
        if self.model is None or self._action_dim is None:
            raise RuntimeError("AMA networks are not initialized yet.")
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if hasattr(base_env, "export_all_observations"):
            all_obs = np.asarray(base_env.export_all_observations(), dtype=np.float32)
        else:
            core = getattr(base_env, "core", None)
            if core is None:
                core = getattr(base_env, "env", None)
            if core is None or not hasattr(core, "node_images"):
                raise RuntimeError("Could not export all observations for AMA logging.")
            all_obs = np.asarray(core.node_images, dtype=np.float32)

        rewards = []
        for action in range(self._action_dim):
            prev = self._prepare_batch_states(all_obs, self._action_dim)
            next_flat = self._prepare_targets(all_obs, self._action_dim)
            act = th.full((prev.shape[0],), action, dtype=th.long, device=self.device)
            with th.no_grad():
                raw, _, _ = self._raw_terms(prev, act, next_flat)
            rewards.append(raw.detach().cpu().numpy())
        return np.mean(np.stack(rewards, axis=0), axis=0).astype(np.float32)


def get_ama_augmented_dqn_class():
    from stable_baselines3 import DQN

    class AMAAugmentedDQN(DQN):
        """DQN that additionally updates an AMA forward model from replay batches."""

        def __init__(self, *args, ama_module: Optional[AMAIntrinsicReward] = None, **kwargs) -> None:
            self.ama_module = ama_module
            super().__init__(*args, **kwargs)

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)

            losses = []
            ama_losses = []
            for _ in range(gradient_steps):
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

                with th.no_grad():
                    next_q_values = self.q_net_target(replay_data.next_observations)
                    next_q_values, _ = next_q_values.max(dim=1)
                    next_q_values = next_q_values.reshape(-1, 1)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                current_q_values = self.q_net(replay_data.observations)
                current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                losses.append(float(loss.detach().cpu().item()))

                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(self.max_grad_norm) > 0:
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                if self.ama_module is not None:
                    ama_loss = self.ama_module.update_from_batch(
                        replay_data.observations,
                        replay_data.next_observations,
                        replay_data.actions,
                    )
                    ama_losses.append(float(ama_loss))

            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            if losses:
                self.logger.record("train/loss", float(np.mean(losses)))
            if ama_losses:
                self.logger.record("train/ama_loss", float(np.mean(ama_losses)))
                assert self.ama_module is not None
                self.logger.record("train/ama_prediction_error", self.ama_module.last_prediction_error)
                self.logger.record("train/ama_predicted_variance", self.ama_module.last_predicted_variance)
                self.logger.record("train/ama_raw_intrinsic", self.ama_module.last_raw_intrinsic)
                self.logger.record("train/ama_reward_coef", self.ama_module.current_coef())

    return AMAAugmentedDQN

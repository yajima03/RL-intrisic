from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update


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
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = new_var
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
class RNDConfig:
    learning_rate: float = 1.0e-4
    encoder_type: str = "auto"  # "auto", "mlp", "cnn"
    conv_layers: Optional[Sequence[Mapping[str, Any]]] = None
    activation: str = "relu"
    embedding_dim: int = 64
    target_hidden_layers: Sequence[int] = (256,)
    predictor_hidden_layers: Sequence[int] = (256, 256, 256)
    normalize_reward: bool = True
    reward_gamma: float = 0.99
    reward_norm_eps: float = 1.0e-8
    intrinsic_clip: Optional[float] = None
    device: str = "auto"
    seed: Optional[int] = None


class _MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        embedding_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for hidden_dim in hidden_layers:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_get_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, int(embedding_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = x.reshape(x.shape[0], -1)
        return self.net(x)


class _SharedConvBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_layers: Sequence[Mapping[str, Any]],
        activation: str,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        current_channels = int(in_channels)

        for layer_cfg in conv_layers:
            out_channels = int(layer_cfg["out_channels"])
            kernel_size = int(layer_cfg.get("kernel_size", 3))
            stride = int(layer_cfg.get("stride", 1))
            padding = int(layer_cfg.get("padding", 0))

            blocks.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            blocks.append(_get_activation(activation))
            current_channels = out_channels

        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = int(current_channels)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = th.flatten(x, start_dim=1)
        return x


class _RNDConvModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        conv_layers: Sequence[Mapping[str, Any]],
        activation: str,
        hidden_layers: Sequence[int],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = _SharedConvBackbone(
            in_channels=in_channels,
            conv_layers=conv_layers,
            activation=activation,
        )
        self.head = _MLPHead(
            input_dim=self.backbone.output_dim,
            hidden_layers=hidden_layers,
            embedding_dim=embedding_dim,
            activation=activation,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


class RNDIntrinsicReward:
    """
    RND intrinsic reward module.

    Supports both:
    - explicit MLP backend for flat observations [D]
    - CNN backend for image observations [C,H,W]

    - compute(): returns current prediction error without updating the predictor
    - update_from_batch(): updates predictor using replay-buffer batch samples
    - reward normalization: divides raw prediction error by std of discounted intrinsic returns
    """

    def __init__(self, config: RNDConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)

        self.target: Optional[nn.Module] = None
        self.predictor: Optional[nn.Module] = None
        self.optimizer: Optional[th.optim.Optimizer] = None
        self._encoder_type_resolved: Optional[str] = None

        self._discounted_intrinsic_return = 0.0
        self.return_rms = RunningMeanStd(epsilon=1.0e-4)

    def reset_episode(self) -> None:
        self._discounted_intrinsic_return = 0.0

    def _default_conv_layers(self) -> Sequence[Mapping[str, Any]]:
        return [
            {"out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
            {"out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 2},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        ]

    def _canonicalize_single_observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)

        if obs.ndim == 0:
            raise ValueError(f"RND expects non-scalar observation, got shape {obs.shape}")

        # If HWC image is accidentally passed, convert to CHW.
        if obs.ndim == 3 and obs.shape[0] not in (1, 3, 4) and obs.shape[-1] in (1, 3, 4):
            obs = np.transpose(obs, (2, 0, 1))

        return obs

    def _resolve_encoder_type(self, sample_obs: np.ndarray) -> str:
        key = str(self.config.encoder_type).lower()
        if key in {"mlp", "cnn"}:
            return key
        if key in {"auto", "infer"}:
            return "mlp" if sample_obs.ndim == 1 else "cnn"
        raise ValueError(f"Unsupported RND encoder_type: {self.config.encoder_type}")

    def _ensure_networks(self, sample_obs: np.ndarray) -> None:
        if self.target is not None:
            return

        sample_obs = self._canonicalize_single_observation(sample_obs)

        if self.config.seed is not None:
            th.manual_seed(int(self.config.seed))
            np.random.seed(int(self.config.seed))

        encoder_type = self._resolve_encoder_type(sample_obs)
        self._encoder_type_resolved = encoder_type

        if encoder_type == "mlp":
            input_dim = int(np.prod(sample_obs.shape))
            self.target = _MLPHead(
                input_dim=input_dim,
                hidden_layers=self.config.target_hidden_layers,
                embedding_dim=self.config.embedding_dim,
                activation=self.config.activation,
            ).to(self.device)

            self.predictor = _MLPHead(
                input_dim=input_dim,
                hidden_layers=self.config.predictor_hidden_layers,
                embedding_dim=self.config.embedding_dim,
                activation=self.config.activation,
            ).to(self.device)

        else:
            if sample_obs.ndim == 2:
                sample_obs = sample_obs[None, :, :]
            if sample_obs.ndim != 3:
                raise ValueError(
                    f"CNN RND expects observation with shape (C,H,W) or (H,W), got {sample_obs.shape}"
                )

            conv_layers = self.config.conv_layers or self._default_conv_layers()
            in_channels = int(sample_obs.shape[0])

            self.target = _RNDConvModel(
                in_channels=in_channels,
                conv_layers=conv_layers,
                activation=self.config.activation,
                hidden_layers=self.config.target_hidden_layers,
                embedding_dim=self.config.embedding_dim,
            ).to(self.device)

            self.predictor = _RNDConvModel(
                in_channels=in_channels,
                conv_layers=conv_layers,
                activation=self.config.activation,
                hidden_layers=self.config.predictor_hidden_layers,
                embedding_dim=self.config.embedding_dim,
            ).to(self.device)

        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

        self.optimizer = th.optim.Adam(
            self.predictor.parameters(),
            lr=float(self.config.learning_rate),
        )

    def _prepare_single_input(self, observation: np.ndarray) -> th.Tensor:
        obs = self._canonicalize_single_observation(observation)
        self._ensure_networks(obs)
        assert self._encoder_type_resolved is not None

        if self._encoder_type_resolved == "mlp":
            x = obs.reshape(1, -1)
        else:
            if obs.ndim == 2:
                obs = obs[None, :, :]
            x = obs[None, ...]
        return th.as_tensor(x, dtype=th.float32, device=self.device)

    def _prepare_batch_input(self, observations: th.Tensor | np.ndarray) -> th.Tensor:
        if isinstance(observations, np.ndarray):
            obs = th.as_tensor(observations, dtype=th.float32, device=self.device)
        else:
            obs = observations.detach().to(self.device).float()

        if self.target is None:
            sample_obs = obs[0].detach().cpu().numpy()
            self._ensure_networks(sample_obs)
        assert self._encoder_type_resolved is not None

        if self._encoder_type_resolved == "mlp":
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
            obs = obs.reshape(obs.shape[0], -1)
            return obs

        # CNN path
        if obs.ndim == 3:
            # [B,H,W] -> [B,1,H,W] or [H,W,C] (single obs batchless)
            if obs.shape[-1] in (1, 3, 4) and obs.shape[0] not in (1, 3, 4):
                obs = obs.permute(2, 0, 1).unsqueeze(0)
            else:
                obs = obs.unsqueeze(1)
        elif obs.ndim == 4:
            # [B,H,W,C] -> [B,C,H,W]
            if obs.shape[1] not in (1, 3, 4) and obs.shape[-1] in (1, 3, 4):
                obs = obs.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                f"CNN RND expects batched image observations with ndim 3 or 4, got {tuple(obs.shape)}"
            )

        return obs

    def _raw_prediction_error(self, x: th.Tensor) -> th.Tensor:
        assert self.target is not None
        assert self.predictor is not None
        with th.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        per_sample = th.mean((pred_feat - target_feat) ** 2, dim=1)
        return per_sample

    def _normalize_reward(self, raw_reward: float) -> float:
        if not self.config.normalize_reward:
            reward = raw_reward
        else:
            self._discounted_intrinsic_return = (
                float(self.config.reward_gamma) * self._discounted_intrinsic_return + raw_reward
            )
            self.return_rms.update(np.array([self._discounted_intrinsic_return], dtype=np.float64))
            reward = raw_reward / np.sqrt(float(self.return_rms.var) + float(self.config.reward_norm_eps))

        if self.config.intrinsic_clip is not None:
            reward = float(np.clip(reward, 0.0, float(self.config.intrinsic_clip)))
        return float(reward)

    def compute(
        self,
        *,
        observation: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
        action: Optional[int] = None,
    ) -> float:
        if observation is None:
            return 0.0

        x = self._prepare_single_input(observation)
        raw_reward = float(self._raw_prediction_error(x).detach().cpu().item())
        return self._normalize_reward(raw_reward)

    def update_from_batch(self, observations: th.Tensor) -> float:
        x = self._prepare_batch_input(observations)
        assert self.target is not None
        assert self.predictor is not None
        assert self.optimizer is not None


        with th.no_grad():
            target_feat = self.target(x)

        pred_feat = self.predictor(x)
        loss = F.mse_loss(pred_feat, target_feat, reduction="mean")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return float(loss.detach().cpu().item())

    def export_raw_intrinsic_per_state(self, env) -> np.ndarray:
        """Return raw, pre-normalization, pre-coefficient intrinsic reward for all states."""
        if self.target is None or self.predictor is None:
            raise RuntimeError("RND networks are not initialized yet.")

        base_env = env.unwrapped if hasattr(env, "unwrapped") else env

        if hasattr(base_env, "export_all_observations"):
            all_obs = base_env.export_all_observations()
        else:
            core = getattr(base_env, "core", None)
            if core is None:
                core = getattr(base_env, "env", None)
            if core is None or not hasattr(core, "node_images"):
                raise RuntimeError("Could not export all observations for RND logging.")
            all_obs = np.asarray(core.node_images, dtype=np.float32)
            

        x = self._prepare_batch_input(all_obs)

        with th.no_grad():
            raw = self._raw_prediction_error(x)

        return raw.detach().cpu().numpy().astype(np.float32)


class RNDAugmentedDQN(DQN):
    """DQN that additionally updates an RND predictor from the same replay buffer batch."""

    def __init__(self, *args, rnd_module: Optional[RNDIntrinsicReward] = None, **kwargs) -> None:
        self.rnd_module = rnd_module
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        rnd_losses = []

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

            if self.rnd_module is not None:
                rnd_loss = self.rnd_module.update_from_batch(replay_data.next_observations)
                rnd_losses.append(float(rnd_loss))

        self._n_updates += gradient_steps

        if self._n_updates % max(self.target_update_interval, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            if hasattr(self, "batch_norm_stats") and hasattr(self, "batch_norm_stats_target"):
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if losses:
            self.logger.record("train/loss", float(np.mean(losses)))
        if rnd_losses:
            self.logger.record("train/rnd_loss", float(np.mean(rnd_losses)))

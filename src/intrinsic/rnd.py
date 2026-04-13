from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


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
    embedding_dim: int = 64
    hidden_dim: int = 256
    conv_layers: Optional[Sequence[Mapping[str, Any]]] = None
    activation: str = "relu"
    normalize_reward: bool = False
    reward_rms_alpha: float = 0.99
    intrinsic_clip: Optional[float] = None
    device: str = "auto"
    seed: Optional[int] = None


class _RNDNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        conv_layers: Sequence[Mapping[str, Any]],
        hidden_dim: int,
        embedding_dim: int,
        activation: str,
    ) -> None:
        super().__init__()

        blocks = []
        current_channels = int(in_channels)
        act_name = str(activation)

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
            blocks.append(_get_activation(act_name))
            current_channels = out_channels

        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels, int(hidden_dim)),
            _get_activation(act_name),
            nn.Linear(int(hidden_dim), int(embedding_dim)),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.head(x)
        return x


class RNDIntrinsicReward:
    """
    Simple online RND module.

    - target network: fixed random network
    - predictor network: updated every compute() call
    - intrinsic reward: prediction error ||pred - target||^2

    This is a lightweight baseline implementation intended to run easily
    with the current RewardWrapper API.
    """

    def __init__(self, config: RNDConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)

        self.target: Optional[_RNDNetwork] = None
        self.predictor: Optional[_RNDNetwork] = None
        self.optimizer: Optional[th.optim.Optimizer] = None

        self.reward_second_moment = 1.0

    def reset_episode(self) -> None:
        # No episode-specific state is required for plain RND.
        return None

    def _default_conv_layers(self) -> Sequence[Mapping[str, Any]]:
        return [
            {"out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
            {"out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 2},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        ]

    def _preprocess_observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)

        if obs.ndim == 2:
            obs = obs[None, :, :]
        elif obs.ndim != 3:
            raise ValueError(f"RND expects observation with shape (C,H,W) or (H,W), got {obs.shape}")

        return obs

    def _lazy_build(self, obs_chw: np.ndarray) -> None:
        if self.target is not None:
            return

        if self.config.seed is not None:
            th.manual_seed(int(self.config.seed))

        conv_layers = self.config.conv_layers or self._default_conv_layers()
        in_channels = int(obs_chw.shape[0])

        self.target = _RNDNetwork(
            in_channels=in_channels,
            conv_layers=conv_layers,
            hidden_dim=self.config.hidden_dim,
            embedding_dim=self.config.embedding_dim,
            activation=self.config.activation,
        ).to(self.device)

        self.predictor = _RNDNetwork(
            in_channels=in_channels,
            conv_layers=conv_layers,
            hidden_dim=self.config.hidden_dim,
            embedding_dim=self.config.embedding_dim,
            activation=self.config.activation,
        ).to(self.device)

        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

        self.optimizer = th.optim.Adam(
            self.predictor.parameters(),
            lr=float(self.config.learning_rate),
        )

    def compute(
        self,
        *,
        observation: Optional[np.ndarray] = None,
        info: Optional[dict[str, Any]] = None,
        action: Optional[int] = None,
    ) -> float:
        if observation is None:
            return 0.0

        obs_chw = self._preprocess_observation(observation)
        self._lazy_build(obs_chw)

        assert self.target is not None
        assert self.predictor is not None
        assert self.optimizer is not None

        x = th.as_tensor(obs_chw[None], dtype=th.float32, device=self.device)

        with th.no_grad():
            target_feat = self.target(x)

        pred_feat = self.predictor(x)
        loss = F.mse_loss(pred_feat, target_feat, reduction="mean")

        raw_reward = float(loss.detach().cpu().item())

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        reward = raw_reward

        if self.config.normalize_reward:
            alpha = float(self.config.reward_rms_alpha)
            self.reward_second_moment = alpha * self.reward_second_moment + (1.0 - alpha) * (
                raw_reward**2
            )
            reward = raw_reward / np.sqrt(self.reward_second_moment + 1.0e-8)

        if self.config.intrinsic_clip is not None:
            reward = min(float(reward), float(self.config.intrinsic_clip))

        return float(reward)
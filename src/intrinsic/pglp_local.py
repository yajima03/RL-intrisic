from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update


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
class _TransitionCacheEntry:
    state_key: np.ndarray
    next_state_key: np.ndarray
    error: float


@dataclass
class _PrototypeEntry:
    center_state_key: np.ndarray
    center_next_state_key: np.ndarray
    ema_long: float
    ema_short: float
    count: int = 1


@dataclass
class PGLPLocalConfig:
    learning_rate: float = 1.0e-4
    conv_layers: Optional[Sequence[Mapping[str, Any]]] = None
    activation: str = "relu"
    embedding_dim: int = 64
    encoder_fc_layers: Sequence[int] = (256,)
    predictor_fc_layers: Sequence[int] = (256, 256)

    # neighbor search for predictability gate
    candidate_size: int = 512
    knn_k: int = 16
    min_candidates: int = 16

    # local gate
    gate_lambda: float = 1.0

    # cache
    cache_capacity_per_action: int = 20000
    key_encoder_tau: float = 0.01

    # online prototype-based progress
    num_prototypes_per_action: int = 8
    prototype_center_tau: float = 0.05
    prototype_min_count: int = 4
    progress_ema_alpha: float = 0.99
    progress_short_ema_alpha: float = 0.90

    device: str = "auto"
    seed: Optional[int] = None


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


class _EncoderModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        conv_layers: Sequence[Mapping[str, Any]],
        activation: str,
        head_layers: Sequence[int],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.backbone = _SharedConvBackbone(
            in_channels=in_channels,
            conv_layers=conv_layers,
            activation=activation,
        )
        layers: list[nn.Module] = []
        in_dim = self.backbone.output_dim
        for hidden_dim in head_layers:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_get_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, int(embedding_dim)))
        self.head = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.head(self.backbone(x))


class _ActionConditionedPredictor(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(embedding_dim) + int(action_dim)
        for hidden_dim in hidden_layers:
            hidden_dim = int(hidden_dim)
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_get_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, int(embedding_dim)))
        self.net = nn.Sequential(*layers)
        self.action_dim = int(action_dim)

    def forward(self, z: th.Tensor, action: th.Tensor) -> th.Tensor:
        action = action.view(-1).long()
        a_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        return self.net(th.cat([z, a_onehot], dim=1))


class PGLPLocalIntrinsicReward:
    """
    PGLP-local with:
      - local predictability gate from same-action + latent k-NN
      - online prototype-based progress

    Raw intrinsic reward:
        gate_t * progress_t

    where
        gate_t     = exp(-lambda * u_t)
        progress_t = max(proto.ema_long - proto.ema_short, 0)
        proto      = nearest prototype for current (state_key, action)
    """

    def __init__(self, config: PGLPLocalConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)

        self.encoder_online: Optional[_EncoderModel] = None
        self.encoder_key: Optional[_EncoderModel] = None
        self.predictor: Optional[_ActionConditionedPredictor] = None
        self.optimizer: Optional[th.optim.Optimizer] = None

        self.action_dim: Optional[int] = None
        self.cache_by_action: Dict[int, Deque[_TransitionCacheEntry]] = {}
        self.prototypes_by_action: Dict[int, List[_PrototypeEntry]] = {}

        # debug accumulators
        self._debug_sums: Dict[str, float] = {
            "gate": 0.0,
            "progress": 0.0,
            "raw_intrinsic": 0.0,
            "current_error": 0.0,
            "neighbor_count": 0.0,
            "prototype_count_for_action": 0.0,
            "prototype_long": 0.0,
            "prototype_short": 0.0,
            "prototype_index": 0.0,
        }
        self._debug_count: int = 0

    def reset_episode(self) -> None:
        return None

    def _default_conv_layers(self) -> Sequence[Mapping[str, Any]]:
        return [
            {"out_channels": 32, "kernel_size": 5, "stride": 2, "padding": 2},
            {"out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 2},
            {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
        ]

    def _preprocess_single_observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 2:
            obs = obs[None, :, :]
        elif obs.ndim != 3:
            raise ValueError(f"PGLP expects observation with shape (C,H,W) or (H,W), got {obs.shape}")
        return obs

    def _infer_action_dim(self, info: Optional[dict[str, Any]]) -> int:
        if self.action_dim is not None:
            return self.action_dim

        if info is None:
            raise RuntimeError("PGLP could not infer action_dim before first transition.")

        action_dim = info.get("action_space_n")
        if action_dim is None:
            action_dim = info.get("action_dim")
        if action_dim is None:
            raise RuntimeError("PGLP needs info['action_space_n'] or info['action_dim'] for initialization.")

        self.action_dim = int(action_dim)
        self.cache_by_action = {
            a: deque(maxlen=int(self.config.cache_capacity_per_action))
            for a in range(self.action_dim)
        }
        self.prototypes_by_action = {a: [] for a in range(self.action_dim)}
        return self.action_dim

    def _ensure_networks(self, sample_obs_chw: np.ndarray, *, action_dim: int) -> None:
        if self.encoder_online is not None:
            return

        if self.config.seed is not None:
            th.manual_seed(int(self.config.seed))
            np.random.seed(int(self.config.seed))

        conv_layers = self.config.conv_layers or self._default_conv_layers()
        in_channels = int(sample_obs_chw.shape[0])

        self.encoder_online = _EncoderModel(
            in_channels=in_channels,
            conv_layers=conv_layers,
            activation=self.config.activation,
            head_layers=self.config.encoder_fc_layers,
            embedding_dim=self.config.embedding_dim,
        ).to(self.device)

        self.encoder_key = _EncoderModel(
            in_channels=in_channels,
            conv_layers=conv_layers,
            activation=self.config.activation,
            head_layers=self.config.encoder_fc_layers,
            embedding_dim=self.config.embedding_dim,
        ).to(self.device)
        self.encoder_key.load_state_dict(self.encoder_online.state_dict())
        for p in self.encoder_key.parameters():
            p.requires_grad_(False)
        self.encoder_key.eval()

        self.predictor = _ActionConditionedPredictor(
            embedding_dim=self.config.embedding_dim,
            action_dim=action_dim,
            hidden_layers=self.config.predictor_fc_layers,
            activation=self.config.activation,
        ).to(self.device)

        params = list(self.encoder_online.parameters()) + list(self.predictor.parameters())
        self.optimizer = th.optim.Adam(params, lr=float(self.config.learning_rate))

    def _encode_online(self, x: th.Tensor) -> th.Tensor:
        assert self.encoder_online is not None
        return self.encoder_online(x)

    def _encode_key(self, x: th.Tensor) -> th.Tensor:
        assert self.encoder_key is not None
        with th.no_grad():
            return self.encoder_key(x)

    def _prediction_error_tensor(
        self,
        prev_obs: th.Tensor,
        next_obs: th.Tensor,
        action: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        assert self.predictor is not None
        z_t = self._encode_online(prev_obs)
        z_tp1 = self._encode_online(next_obs)
        z_hat = self.predictor(z_t, action)
        e = th.mean((z_hat - z_tp1.detach()) ** 2, dim=1)
        return z_t, z_tp1, z_hat, e

    def _query_neighbors(self, action: int, state_key: np.ndarray) -> List[_TransitionCacheEntry]:
        entries = list(self.cache_by_action.get(int(action), []))
        if len(entries) < int(self.config.min_candidates):
            return []

        if len(entries) > int(self.config.candidate_size):
            idx = np.random.choice(len(entries), size=int(self.config.candidate_size), replace=False)
            entries = [entries[i] for i in idx]

        dists = [float(np.linalg.norm(state_key - entry.state_key)) for entry in entries]
        order = np.argsort(dists)
        k = min(int(self.config.knn_k), len(order))
        return [entries[i] for i in order[:k]]

    def _local_gate_from_neighbors(
        self,
        *,
        next_state_key: np.ndarray,
        neighbors: List[_TransitionCacheEntry],
    ) -> float:
        if len(neighbors) < int(self.config.min_candidates):
            return 0.0
        next_dists = [float(np.linalg.norm(next_state_key - entry.next_state_key)) for entry in neighbors]
        u_t = float(np.median(next_dists))
        gate = float(np.exp(-float(self.config.gate_lambda) * u_t))
        return gate

    def _append_cache(
        self,
        *,
        action: int,
        state_key: np.ndarray,
        next_state_key: np.ndarray,
        error: float,
    ) -> None:
        if self.action_dim is None:
            return
        entry = _TransitionCacheEntry(
            state_key=state_key.astype(np.float32, copy=True),
            next_state_key=next_state_key.astype(np.float32, copy=True),
            error=float(error),
        )
        self.cache_by_action[int(action)].append(entry)

    def _nearest_prototype_index(self, action: int, state_key: np.ndarray) -> Optional[int]:
        prototypes = self.prototypes_by_action.get(int(action), [])
        if len(prototypes) == 0:
            return None

        dists = [float(np.linalg.norm(state_key - proto.center_state_key)) for proto in prototypes]
        return int(np.argmin(dists))

    def _get_prototype_progress(
        self, action: int, state_key: np.ndarray
    ) -> tuple[Optional[float], Optional[int], Optional[float], Optional[float]]:
        idx = self._nearest_prototype_index(action, state_key)
        if idx is None:
            return None, None, None, None

        proto = self.prototypes_by_action[int(action)][idx]
        long_v = float(proto.ema_long)
        short_v = float(proto.ema_short)

        if proto.count < int(self.config.prototype_min_count):
            return None, idx, long_v, short_v

        return max(long_v - short_v, 0.0), idx, long_v, short_v

    def _update_or_create_prototype(
        self,
        *,
        action: int,
        state_key: np.ndarray,
        next_state_key: np.ndarray,
        error: float,
    ) -> None:
        action = int(action)
        prototypes = self.prototypes_by_action[action]

        alpha_long = float(self.config.progress_ema_alpha)
        alpha_short = float(self.config.progress_short_ema_alpha)
        center_tau = float(self.config.prototype_center_tau)

        if len(prototypes) < int(self.config.num_prototypes_per_action):
            prototypes.append(
                _PrototypeEntry(
                    center_state_key=state_key.astype(np.float32, copy=True),
                    center_next_state_key=next_state_key.astype(np.float32, copy=True),
                    ema_long=float(error),
                    ema_short=float(error),
                    count=1,
                )
            )
            return

        idx = self._nearest_prototype_index(action, state_key)
        if idx is None:
            return

        proto = prototypes[idx]
        proto.center_state_key = (
            (1.0 - center_tau) * proto.center_state_key + center_tau * state_key
        ).astype(np.float32)
        proto.center_next_state_key = (
            (1.0 - center_tau) * proto.center_next_state_key + center_tau * next_state_key
        ).astype(np.float32)

        proto.ema_long = alpha_long * float(proto.ema_long) + (1.0 - alpha_long) * float(error)
        proto.ema_short = alpha_short * float(proto.ema_short) + (1.0 - alpha_short) * float(error)
        proto.count += 1

    def _accumulate_debug(
        self,
        *,
        gate: float,
        progress: float,
        raw_intrinsic: float,
        current_error: float,
        neighbor_count: int,
        prototype_count_for_action: int,
        prototype_long: float,
        prototype_short: float,
        prototype_index: float,
    ) -> None:
        self._debug_sums["gate"] += float(gate)
        self._debug_sums["progress"] += float(progress)
        self._debug_sums["raw_intrinsic"] += float(raw_intrinsic)
        self._debug_sums["current_error"] += float(current_error)
        self._debug_sums["neighbor_count"] += float(neighbor_count)
        self._debug_sums["prototype_count_for_action"] += float(prototype_count_for_action)
        self._debug_sums["prototype_long"] += float(prototype_long)
        self._debug_sums["prototype_short"] += float(prototype_short)
        self._debug_sums["prototype_index"] += float(prototype_index)
        self._debug_count += 1

    def flush_debug_stats(self) -> Dict[str, float]:
        if self._debug_count == 0:
            return {
                "mean_gate": 0.0,
                "mean_progress": 0.0,
                "mean_raw_intrinsic": 0.0,
                "mean_current_error": 0.0,
                "mean_neighbor_count": 0.0,
                "mean_prototype_count_for_action": 0.0,
                "mean_prototype_long": 0.0,
                "mean_prototype_short": 0.0,
                "mean_prototype_index": -1.0,
                "num_samples": 0.0,
            }

        n = float(self._debug_count)
        out = {
            "mean_gate": self._debug_sums["gate"] / n,
            "mean_progress": self._debug_sums["progress"] / n,
            "mean_raw_intrinsic": self._debug_sums["raw_intrinsic"] / n,
            "mean_current_error": self._debug_sums["current_error"] / n,
            "mean_neighbor_count": self._debug_sums["neighbor_count"] / n,
            "mean_prototype_count_for_action": self._debug_sums["prototype_count_for_action"] / n,
            "mean_prototype_long": self._debug_sums["prototype_long"] / n,
            "mean_prototype_short": self._debug_sums["prototype_short"] / n,
            "mean_prototype_index": self._debug_sums["prototype_index"] / n,
            "num_samples": n,
        }

        for key in self._debug_sums:
            self._debug_sums[key] = 0.0
        self._debug_count = 0
        return out

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

        prev_chw = self._preprocess_single_observation(previous_observation)
        obs_chw = self._preprocess_single_observation(observation)
        action_dim = self._infer_action_dim(info)
        self._ensure_networks(prev_chw, action_dim=action_dim)

        prev_x = th.as_tensor(prev_chw[None], dtype=th.float32, device=self.device)
        next_x = th.as_tensor(obs_chw[None], dtype=th.float32, device=self.device)
        action_t = th.as_tensor([int(action)], dtype=th.long, device=self.device)

        _, _, _, e_t = self._prediction_error_tensor(prev_x, next_x, action_t)
        current_error = float(e_t.detach().cpu().item())

        state_key = self._encode_key(prev_x).detach().cpu().numpy()[0].astype(np.float32)
        next_state_key = self._encode_key(next_x).detach().cpu().numpy()[0].astype(np.float32)

        neighbors = self._query_neighbors(int(action), state_key)
        gate = self._local_gate_from_neighbors(
            next_state_key=next_state_key,
            neighbors=neighbors,
        )

        progress, proto_idx, proto_long, proto_short = self._get_prototype_progress(int(action), state_key)
        if progress is None:
            progress = 0.0
        if proto_idx is None:
            proto_idx = -1
        if proto_long is None:
            proto_long = 0.0
        if proto_short is None:
            proto_short = 0.0

        raw = gate * float(progress)

        # debug accumulation
        self._accumulate_debug(
            gate=gate,
            progress=float(progress),
            raw_intrinsic=float(raw),
            current_error=current_error,
            neighbor_count=len(neighbors),
            prototype_count_for_action=len(self.prototypes_by_action.get(int(action), [])),
            prototype_long=float(proto_long),
            prototype_short=float(proto_short),
            prototype_index=float(proto_idx),
        )

        # cache update
        self._append_cache(
            action=int(action),
            state_key=state_key,
            next_state_key=next_state_key,
            error=current_error,
        )

        # online prototype update
        self._update_or_create_prototype(
            action=int(action),
            state_key=state_key,
            next_state_key=next_state_key,
            error=current_error,
        )

        return float(raw)

    def update_from_batch(
        self,
        observations: th.Tensor,
        next_observations: th.Tensor,
        actions: th.Tensor,
    ) -> float:
        prev_obs = observations.detach().to(self.device).float()
        next_obs = next_observations.detach().to(self.device).float()
        act = actions.detach().to(self.device).view(-1).long()

        if prev_obs.ndim != 4 or next_obs.ndim != 4:
            raise ValueError(
                f"PGLP update_from_batch expects [B,C,H,W], got {tuple(prev_obs.shape)} and {tuple(next_obs.shape)}"
            )

        if self.action_dim is None:
            inferred_action_dim = int(act.max().item()) + 1
            sample_obs = prev_obs[0].detach().cpu().numpy()
            self.action_dim = inferred_action_dim
            self.cache_by_action = {
                a: deque(maxlen=int(self.config.cache_capacity_per_action))
                for a in range(self.action_dim)
            }
            self.prototypes_by_action = {a: [] for a in range(self.action_dim)}
            self._ensure_networks(sample_obs, action_dim=inferred_action_dim)

        assert self.optimizer is not None
        assert self.encoder_online is not None
        assert self.encoder_key is not None
        assert self.predictor is not None

        _, _, _, e_t = self._prediction_error_tensor(prev_obs, next_obs, act)
        loss = e_t.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        polyak_update(
            self.encoder_online.parameters(),
            self.encoder_key.parameters(),
            float(self.config.key_encoder_tau),
        )

        return float(loss.detach().cpu().item())

    def export_raw_intrinsic_per_state(self, env) -> np.ndarray:
        """
        Diagnostic proxy for state-level logging.

        Since true PGLP reward depends on an actual transition (s, a, s'),
        we export a proxy:
            mean_a [ gate_proxy(s, a) * prototype_progress_proxy(s, a) ]
        """
        if self.encoder_online is None or self.encoder_key is None or self.predictor is None:
            raise RuntimeError("PGLP networks are not initialized yet.")

        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if not hasattr(base_env, "export_all_observations"):
            raise RuntimeError("Environment does not provide export_all_observations().")

        all_obs = base_env.export_all_observations()
        if all_obs is None:
            raise RuntimeError("export_all_observations() returned None.")

        x = all_obs.to(self.device).float()
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected all observations with shape [N,C,H,W], got {tuple(x.shape)}")

        z_key = self._encode_key(x).detach().cpu().numpy().astype(np.float32)

        action_space = getattr(base_env, "action_space", None)
        if action_space is not None and hasattr(action_space, "n"):
            num_actions = int(action_space.n)
        elif self.action_dim is not None:
            num_actions = int(self.action_dim)
        else:
            raise RuntimeError("Could not infer action dimension for PGLP export.")

        outputs: List[float] = []
        for i in range(x.shape[0]):
            state_key = z_key[i]
            per_action: List[float] = []

            next_state_key_proxy = state_key

            for action in range(num_actions):
                neighbors = self._query_neighbors(action, state_key)
                if len(neighbors) < int(self.config.min_candidates):
                    per_action.append(0.0)
                    continue

                gate = self._local_gate_from_neighbors(
                    next_state_key=next_state_key_proxy,
                    neighbors=neighbors,
                )

                progress, _, _, _ = self._get_prototype_progress(action, state_key)
                if progress is None:
                    per_action.append(0.0)
                    continue

                per_action.append(float(gate * progress))

            outputs.append(float(np.mean(per_action)) if per_action else 0.0)

        return np.asarray(outputs, dtype=np.float32)


class PGLPAugmentedDQN(DQN):
    """DQN that additionally updates a PGLP predictor from the same replay buffer batch."""

    def __init__(self, *args, pglp_module: Optional[PGLPLocalIntrinsicReward] = None, **kwargs) -> None:
        self.pglp_module = pglp_module
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        pglp_losses = []

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
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            if self.pglp_module is not None:
                pglp_loss = self.pglp_module.update_from_batch(
                    replay_data.observations,
                    replay_data.next_observations,
                    replay_data.actions,
                )
                pglp_losses.append(float(pglp_loss))

        self._n_updates += gradient_steps

        if self._n_updates % max(self.target_update_interval, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))
        if pglp_losses:
            self.logger.record("train/pglp_pred_loss", float(np.mean(pglp_losses)))
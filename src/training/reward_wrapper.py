from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np
import gymnasium as gym


@runtime_checkable
class IntrinsicRewardModule(Protocol):
    def reset_episode(self) -> None: ...

    def compute(
        self,
        *,
        observation: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        action: Optional[int] = None,
    ) -> float: ...


class RewardWrapper(gym.Wrapper):
    """Combine external reward with intrinsic reward.

    This wrapper also tracks per-episode external/intrinsic/total returns and
    stores them in ``info`` at episode end so Monitor can write them to CSV.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        intrinsic_module: IntrinsicRewardModule,
        intrinsic_coef: float = 1.0,
        store_transitions: bool = False,
    ) -> None:
        super().__init__(env)
        self.intrinsic_module = intrinsic_module
        self.intrinsic_coef = float(intrinsic_coef)
        self.store_transitions = bool(store_transitions)
        self._transition_buffer: list[tuple[np.ndarray, np.ndarray, Any]] = []

        self.episode_external_return = 0.0
        self.episode_intrinsic_return = 0.0
        self.episode_total_return = 0.0
        self._last_observation: Optional[np.ndarray] = None
        self._compute_accepts_previous = self._check_accepts_previous_observation()
        self._has_dynamic_coef = self._check_has_dynamic_coef()

    def _check_accepts_previous_observation(self) -> bool:
        try:
            sig = inspect.signature(self.intrinsic_module.compute)
            return "previous_observation" in sig.parameters
        except (TypeError, ValueError):
            return False

    def _check_has_dynamic_coef(self) -> bool:
        return hasattr(self.intrinsic_module, "current_coef") and callable(getattr(self.intrinsic_module, "current_coef"))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_external_return = 0.0
        self.episode_intrinsic_return = 0.0
        self.episode_total_return = 0.0
        self._last_observation = np.array(obs, copy=True)
        self.intrinsic_module.reset_episode()
        return obs, info

    def pop_transition_batch(self):
        transitions = self._transition_buffer
        self._transition_buffer = []
        return transitions

    def step(self, action):
        obs, ext_reward, terminated, truncated, info = self.env.step(action)
        if info is None:
            info = {}
        else:
            info = dict(info)

        if hasattr(self.env, "action_space") and hasattr(self.env.action_space, "n"):
            info.setdefault("action_space_n", int(self.env.action_space.n))

        coef = float(self.intrinsic_coef)
        if self._has_dynamic_coef:
            coef = float(self.intrinsic_module.current_coef())

        previous_observation = self._last_observation
        if self._compute_accepts_previous:
            raw_intrinsic = self.intrinsic_module.compute(
                previous_observation=previous_observation,
                observation=obs,
                info=info,
                action=action,
            )
        else:
            raw_intrinsic = self.intrinsic_module.compute(
                observation=obs,
                info=info,
                action=action,
            )
        intrinsic_for_rl = float(raw_intrinsic)
        if self.intrinsic_module is not None and hasattr(self.intrinsic_module, "normalize_raw_intrinsic_for_rl"):
            intrinsic_for_rl = float(self.intrinsic_module.normalize_raw_intrinsic_for_rl(intrinsic_for_rl))

        intrinsic_reward = coef * intrinsic_for_rl
        total_reward = float(ext_reward) + intrinsic_reward

        if self.store_transitions and previous_observation is not None:
            self._transition_buffer.append(
                (
                    np.array(previous_observation, copy=True),
                    np.array(obs, copy=True),
                    np.array(action, copy=True),
                )
            )

        if hasattr(self.intrinsic_module, "record_rl_transition"):
            self.intrinsic_module.record_rl_transition(
                extrinsic_reward=float(ext_reward),
                intrinsic_reward_normalized=intrinsic_for_rl,
                intrinsic_coef=coef,
                intrinsic_reward_to_rl=intrinsic_reward,
                total_reward_to_rl=total_reward,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

        self.episode_external_return += float(ext_reward)
        self.episode_intrinsic_return += intrinsic_reward
        self.episode_total_return += total_reward

        done = bool(terminated or truncated)
        if done:
            info["episode_external_return"] = float(self.episode_external_return)
            info["episode_intrinsic_return"] = float(self.episode_intrinsic_return)
            info["episode_total_return"] = float(self.episode_total_return)

        self._last_observation = np.array(obs, copy=True)
        return obs, total_reward, terminated, truncated, info

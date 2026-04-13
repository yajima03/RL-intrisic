from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
from gymnasium import Wrapper


class IntrinsicRewardModule(Protocol):
    def reset_episode(self) -> None: ...

    def compute(
        self,
        *,
        observation: Optional[np.ndarray] = None,
        info: Optional[Dict[str, Any]] = None,
        action: Optional[int] = None,
    ) -> float: ...


class RewardWrapper(Wrapper):
    """Add intrinsic reward on top of the environment's external reward.

    The wrapped environment is expected to expose the external reward in the
    standard Gymnasium ``step`` return. The wrapper returns

    ``r_total = r_ext + intrinsic_coef * r_int``.

    It also records the following values into ``info``:

    - ``external_reward``
    - ``intrinsic_reward``
    - ``total_reward``
    - ``episode_external_return``
    - ``episode_intrinsic_return``
    - ``episode_total_return``
    """

    def __init__(self, env, intrinsic_module: IntrinsicRewardModule, intrinsic_coef: float = 1.0):
        super().__init__(env)
        self.intrinsic_module = intrinsic_module
        self.intrinsic_coef = float(intrinsic_coef)
        self.episode_external_return = 0.0
        self.episode_intrinsic_return = 0.0
        self.episode_total_return = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_external_return = 0.0
        self.episode_intrinsic_return = 0.0
        self.episode_total_return = 0.0
        self.intrinsic_module.reset_episode()
        info = dict(info)
        info.setdefault("external_reward", 0.0)
        info.setdefault("intrinsic_reward", 0.0)
        info.setdefault("total_reward", 0.0)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, ext_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        intrinsic_raw = float(
            self.intrinsic_module.compute(
                observation=np.asarray(obs),
                info=info,
                action=int(action),
            )
        )
        intrinsic_reward = self.intrinsic_coef * intrinsic_raw
        total_reward = float(ext_reward) + float(intrinsic_reward)

        self.episode_external_return += float(ext_reward)
        self.episode_intrinsic_return += float(intrinsic_reward)
        self.episode_total_return += float(total_reward)

        info["external_reward"] = float(ext_reward)
        info["intrinsic_reward_raw"] = intrinsic_raw
        info["intrinsic_reward"] = float(intrinsic_reward)
        info["total_reward"] = float(total_reward)
        info["intrinsic_coef"] = float(self.intrinsic_coef)

        if terminated or truncated:
            info["episode_external_return"] = float(self.episode_external_return)
            info["episode_intrinsic_return"] = float(self.episode_intrinsic_return)
            info["episode_total_return"] = float(self.episode_total_return)

        return obs, total_reward, terminated, truncated, info

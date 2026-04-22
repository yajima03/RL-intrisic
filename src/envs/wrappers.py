from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class HWCToCHWObservation(gym.ObservationWrapper):
    """
    Convert Box observations from (H, W, C) to (C, H, W).

    MiniGrid's ImgObsWrapper returns image observations in HWC format,
    while the current CNN pipeline expects channel-first CHW.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Box):
            raise TypeError("HWCToCHWObservation requires a Box observation space.")
        if len(obs_space.shape) != 3:
            raise ValueError(
                f"HWCToCHWObservation expects 3D observations, got shape={obs_space.shape}"
            )

        h, w, c = obs_space.shape
        low = np.transpose(obs_space.low, (2, 0, 1))
        high = np.transpose(obs_space.high, (2, 0, 1))
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(c, h, w),
            dtype=obs_space.dtype,
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))
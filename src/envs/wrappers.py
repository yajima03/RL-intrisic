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


class MiniGridThreeActionWrapper(gym.ActionWrapper):
    """
    Restrict MiniGrid actions to:
      0 -> left
      1 -> right
      2 -> forward

    Useful for FourRooms, where other actions are unnecessary.
    Do NOT use this wrapper for DoorKey.
    """

    ACTION_MAP = [0, 1, 2]  # left, right, forward

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        action = int(action)
        if action < 0 or action >= len(self.ACTION_MAP):
            raise ValueError(f"Invalid 3-action wrapper action: {action}")
        return self.ACTION_MAP[action]
    
    
    
class MiniGridDoorKeyFiveActionWrapper(gym.ActionWrapper):
    """
    Restrict DoorKey actions to:
      0 -> left
      1 -> right
      2 -> forward
      3 -> pickup
      4 -> toggle

    Original MiniGrid DoorKey action space is Discrete(7),
    but drop(4) and done(6) are unused.
    """

    ACTION_MAP = [0, 1, 2, 3, 5]

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(5)

    def action(self, action):
        action = int(action)
        if action < 0 or action >= len(self.ACTION_MAP):
            raise ValueError(f"Invalid 5-action wrapper action: {action}")
        return self.ACTION_MAP[action]
    
    
    
class MiniGridImageFlatObsWrapper(gym.ObservationWrapper):
    """
    Use only obs["image"], flatten it, and convert to float32.

    - Input : Dict observation from MiniGrid
    - Output: Box(shape=(H*W*C,), dtype=float32)

    By default, values are scaled to [0, 1] by dividing by 255.0.
    """

    def __init__(self, env: gym.Env, *, scale_to_unit: bool = True):
        super().__init__(env)

        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Dict):
            raise TypeError(
                "MiniGridImageFlatObsWrapper requires a Dict observation space."
            )
        if "image" not in obs_space.spaces:
            raise KeyError(
                "MiniGridImageFlatObsWrapper requires observation key 'image'."
            )

        image_space = obs_space.spaces["image"]
        if not isinstance(image_space, spaces.Box):
            raise TypeError("obs['image'] must be a Box space.")
        if len(image_space.shape) != 3:
            raise ValueError(
                f"obs['image'] must have shape (H, W, C), got {image_space.shape}"
            )

        self.scale_to_unit = bool(scale_to_unit)
        flat_dim = int(np.prod(image_space.shape))

        low = np.zeros((flat_dim,), dtype=np.float32)
        if self.scale_to_unit:
            high = np.ones((flat_dim,), dtype=np.float32)
        else:
            high = np.full((flat_dim,), float(np.max(image_space.high)), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(flat_dim,),
            dtype=np.float32,
        )

    def observation(self, obs):
        image = np.asarray(obs["image"], dtype=np.float32).reshape(-1)
        if self.scale_to_unit:
            image = image / 255.0
        return image
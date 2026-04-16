from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Optional

import numpy as np


@dataclass(slots=True)
class CountBonusConfig:
    """Configuration for count-based intrinsic rewards.

    Parameters
    ----------
    power:
        Exponent ``alpha`` in the bonus ``1 / N(s)^alpha``.
        The common count-based choice is 0.5.
    eps:
        Small constant used for numerical safety when needed.
    use_state_action:
        If True, count visits to state-action pairs instead of states.
    """

    power: float = 0.5
    eps: float = 1e-8
    use_state_action: bool = False


class CountBasedBonus:
    """Simple count-based intrinsic reward module.

    This module tracks counts over the full training run. The bonus for a state
    (or state-action pair) after the new visit count becomes ``n`` is:

    ``bonus = 1 / max(n, 1) ** power``

    The caller is expected to apply any global coefficient outside this class.
    """

    def __init__(self, config: Optional[CountBonusConfig] = None) -> None:
        self.config = config or CountBonusConfig()
        self._counts: Dict[Hashable, int] = {}

    @property
    def counts(self) -> Dict[Hashable, int]:
        return self._counts

    def reset_all(self) -> None:
        self._counts.clear()

    def reset_episode(self) -> None:
        """Episode hook kept for API symmetry with future intrinsic modules."""
        return None

    def _state_key(
        self,
        *,
        observation: Optional[np.ndarray],
        info: Optional[Mapping[str, Any]],
        action: Optional[int],
    ) -> Hashable:
        if info is not None and "node_id" in info:
            base_key: Hashable = int(info["node_id"])
        elif observation is not None:
            arr = np.asarray(observation)
            base_key = arr.tobytes()
        else:
            raise ValueError("CountBasedBonus requires either info['node_id'] or an observation.")

        if self.config.use_state_action:
            if action is None:
                raise ValueError("action must be provided when use_state_action=True")
            return (base_key, int(action))
        return base_key

    def get_count(self, key: Hashable) -> int:
        return int(self._counts.get(key, 0))

    def compute(
        self,
        *,
        observation: Optional[np.ndarray] = None,
        info: Optional[Mapping[str, Any]] = None,
        action: Optional[int] = None,
    ) -> float:
        """Update the count and return the intrinsic bonus for the new visit."""
        key = self._state_key(observation=observation, info=info, action=action)
        new_count = self._counts.get(key, 0) + 1
        self._counts[key] = new_count

        denom = max(float(new_count), 1.0)
        power = float(self.config.power)
        bonus = 1.0 / (denom**power + float(self.config.eps))
        return float(bonus)

    def export_raw_intrinsic_per_state(self, env) -> np.ndarray:
        """Return raw count-based intrinsic reward for all states.

        Notes
        -----
        - Returned values are **before** applying any global coefficient.
        - When ``use_state_action=False``, this is simply state bonus.
        - When ``use_state_action=True``, a single scalar per state is needed
          for logging, so we use the **mean** over action-wise raw bonuses.
        """
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        core = getattr(base_env, "core", None)
        if core is None:
            core = getattr(base_env, "env", None)
        if core is None or not hasattr(core, "nodes"):
            raise RuntimeError("Could not access core environment for count bonus logging.")

        power = float(self.config.power)
        eps = float(self.config.eps)

        def _bonus_from_count(count_value: int) -> float:
            if count_value <= 0:
                return 1.0 / (1.0**power + eps)
            return 1.0 / ((float(count_value) ** power) + eps)

        values = []

        if not self.config.use_state_action:
            for node_id, _node in enumerate(core.nodes):
                count_value = int(self._counts.get(int(node_id), 0))
                values.append(_bonus_from_count(count_value))
            return np.asarray(values, dtype=np.float32)

        action_space_size = None
        if hasattr(core, "action_space_size"):
            action_space_size = int(core.action_space_size)
        elif hasattr(base_env, "action_space") and hasattr(base_env.action_space, "n"):
            action_space_size = int(base_env.action_space.n)

        if action_space_size is None:
            raise RuntimeError("Could not infer action_space_size for state-action count logging.")

        for node_id, _node in enumerate(core.nodes):
            action_vals = []
            for action_id in range(action_space_size):
                count_value = int(self._counts.get((int(node_id), int(action_id)), 0))
                action_vals.append(_bonus_from_count(count_value))
            values.append(float(np.mean(action_vals)))

        return np.asarray(values, dtype=np.float32)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "num_entries": len(self._counts),
            "power": float(self.config.power),
            "eps": float(self.config.eps),
            "use_state_action": bool(self.config.use_state_action),
        }
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Optional, Tuple

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
        """Update the count and return the intrinsic bonus for the new visit.

        Parameters
        ----------
        observation:
            Usually the *next* observation after the environment transition.
            Used as a fallback key when ``info['node_id']`` is unavailable.
        info:
            Environment info dictionary. If it contains ``node_id``, that is used
            as the canonical state identifier.
        action:
            Optional action that produced the new state. Only used when
            ``use_state_action=True``.
        """
        key = self._state_key(observation=observation, info=info, action=action)
        new_count = self._counts.get(key, 0) + 1
        self._counts[key] = new_count

        denom = max(float(new_count), 1.0)
        power = float(self.config.power)
        bonus = 1.0 / (denom**power + float(self.config.eps))
        return float(bonus)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "num_entries": len(self._counts),
            "power": float(self.config.power),
            "eps": float(self.config.eps),
            "use_state_action": bool(self.config.use_state_action),
        }

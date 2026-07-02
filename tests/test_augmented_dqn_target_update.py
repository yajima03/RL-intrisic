from __future__ import annotations

import inspect

from src.intrinsic.ama import get_ama_augmented_dqn_class
from src.intrinsic.pglp_local import get_pglp_augmented_dqn_class
from src.intrinsic.rnd import get_rnd_augmented_dqn_class


def test_rnd_augmented_dqn_does_not_manually_update_target_network() -> None:
    source = inspect.getsource(get_rnd_augmented_dqn_class)

    assert "polyak_update" not in source
    assert "self.q_net.parameters(), self.q_net_target.parameters()" not in source
    assert "batch_norm_stats_target" not in source


def test_pglp_augmented_dqn_does_not_manually_update_target_network() -> None:
    source = inspect.getsource(get_pglp_augmented_dqn_class)

    assert "polyak_update" not in source
    assert "self.q_net.parameters(), self.q_net_target.parameters()" not in source
    assert "batch_norm_stats_target" not in source


def test_ama_augmented_dqn_does_not_manually_update_target_network() -> None:
    source = inspect.getsource(get_ama_augmented_dqn_class)

    assert "polyak_update" not in source
    assert "self.q_net.parameters(), self.q_net_target.parameters()" not in source
    assert "batch_norm_stats_target" not in source

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.training.train import init_wandb_run, prepare_run_dirs


class _FakeWandb:
    def __init__(self) -> None:
        self.init_kwargs = None

    @staticmethod
    def Settings(**kwargs):
        return SimpleNamespace(**kwargs)

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return SimpleNamespace(id="fake-run", url="https://example.invalid/fake-run")


def test_wandb_disabled_does_not_initialize(tmp_path) -> None:
    run_dirs = prepare_run_dirs(tmp_path, "disabled")

    run = init_wandb_run(
        algo_config={"seed": 7, "wandb": {"enabled": False}},
        env_config={"name": "test-env"},
        intrinsic_config={"name": "none"},
        run_dirs=run_dirs,
    )

    assert run is None


def test_wandb_config_is_resolved_and_forwarded(monkeypatch, tmp_path) -> None:
    fake_wandb = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    run_dirs = prepare_run_dirs(tmp_path, "trial")
    algo_config = {
        "seed": 100,
        "learning_rate": 1.0e-4,
        "wandb": {
            "enabled": True,
            "project": "RND-on-ScalablePyramid",
            "group": "door-rsrs",
            "name": "seed-{seed}-{run_name}",
            "mode": "offline",
            "init_timeout": 45,
            "sync_tensorboard": True,
            "save_code": False,
            "tags": ["smoke"],
        },
    }

    run = init_wandb_run(
        algo_config=algo_config,
        env_config={"name": "sp"},
        intrinsic_config={"name": "lpm"},
        run_dirs=run_dirs,
    )

    assert run.id == "fake-run"
    kwargs = fake_wandb.init_kwargs
    assert kwargs is not None
    assert kwargs["project"] == "RND-on-ScalablePyramid"
    assert kwargs["group"] == "door-rsrs"
    assert kwargs["name"] == "seed-100-trial"
    assert kwargs["mode"] == "offline"
    assert kwargs["dir"] == str(run_dirs["run_dir"])
    assert kwargs["sync_tensorboard"] is True
    assert kwargs["settings"].init_timeout == pytest.approx(45.0)
    assert kwargs["config"]["seed"] == 100
    assert kwargs["config"]["environment"]["name"] == "sp"
    assert kwargs["config"]["intrinsic"]["name"] == "lpm"
    assert "wandb" not in kwargs["config"]["algorithm"]


def test_wandb_mode_is_validated(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb())
    run_dirs = prepare_run_dirs(tmp_path, "invalid-mode")

    with pytest.raises(ValueError, match="wandb.mode"):
        init_wandb_run(
            algo_config={
                "seed": 0,
                "wandb": {"enabled": True, "mode": "somewhere"},
            },
            env_config={},
            intrinsic_config={},
            run_dirs=run_dirs,
        )

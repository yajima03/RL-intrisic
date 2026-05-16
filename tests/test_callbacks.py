from __future__ import annotations

from src.training.train import EnvStatusLoggingCallback


class _EnvWithoutRndLogging:
    @property
    def unwrapped(self):
        return self


class _FailingIntrinsicExport:
    def export_raw_intrinsic_per_state(self, env):
        raise AssertionError("raw intrinsic export should be skipped")


def test_env_status_callback_skips_raw_intrinsic_export_without_rnd_logger(tmp_path):
    callback = EnvStatusLoggingCallback(
        train_env=_EnvWithoutRndLogging(),
        run_log_dir=tmp_path,
        log_interval=100,
        intrinsic_module=_FailingIntrinsicExport(),
        verbose=1,
    )
    callback.num_timesteps = 100

    assert callback._on_step() is True

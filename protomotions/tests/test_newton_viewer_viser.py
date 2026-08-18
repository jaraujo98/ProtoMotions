# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Newton simulator's viser viewer backend support.

These exercise the viewer_backend config option and the NewtonSimulator methods
that had to be made viewer-agnostic (close, _write_viewport_to_file,
_update_camera) so ViewerViser can be used in place of ViewerGL. Unlike
ViewerGL, ViewerViser only starts a local web server, so these tests need no
display and no Xvfb.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def _newton_simulator_cls():
    pytest.importorskip("warp")
    pytest.importorskip("newton")
    from protomotions.simulator.newton.simulator import NewtonSimulator

    return NewtonSimulator


def _newton_config_classes():
    from protomotions.simulator.newton.config import NewtonSimParams, NewtonSimulatorConfig

    return NewtonSimulatorConfig, NewtonSimParams


def test_viewer_backend_config_defaults_to_gl():
    NewtonSimulatorConfig, NewtonSimParams = _newton_config_classes()

    cfg = NewtonSimulatorConfig(
        num_envs=1, headless=True, sim=NewtonSimParams(), experiment_name="t"
    )

    assert cfg.viewer_backend == "gl"
    assert cfg.viewer_port == 8080
    assert cfg.viewer_share is False


def test_viewer_backend_accepts_viser():
    NewtonSimulatorConfig, NewtonSimParams = _newton_config_classes()

    cfg = NewtonSimulatorConfig(
        num_envs=1,
        headless=False,
        sim=NewtonSimParams(),
        experiment_name="t",
        viewer_backend="viser",
        viewer_port=8123,
        viewer_share=True,
    )

    assert cfg.viewer_backend == "viser"
    assert cfg.viewer_port == 8123
    assert cfg.viewer_share is True


def test_viewer_backend_rejects_invalid_value():
    NewtonSimulatorConfig, NewtonSimParams = _newton_config_classes()

    with pytest.raises(AssertionError):
        NewtonSimulatorConfig(
            num_envs=1,
            headless=True,
            sim=NewtonSimParams(),
            experiment_name="t",
            viewer_backend="not_a_backend",
        )


def test_close_closes_viewer_when_present():
    NewtonSimulator = _newton_simulator_cls()
    mock_self = MagicMock()
    mock_self.viewer = MagicMock()

    NewtonSimulator.close(mock_self)

    mock_self.viewer.close.assert_called_once()


def test_close_is_noop_when_viewer_is_none():
    NewtonSimulator = _newton_simulator_cls()
    mock_self = MagicMock()
    mock_self.viewer = None

    NewtonSimulator.close(mock_self)  # must not raise


def test_write_viewport_to_file_skips_backends_without_get_frame():
    NewtonSimulator = _newton_simulator_cls()
    mock_self = MagicMock()
    mock_self.viewer = MagicMock(spec=[])  # no get_frame attribute, like ViewerViser

    NewtonSimulator._write_viewport_to_file(mock_self, "unused.png")  # must not raise


def test_write_viewport_to_file_uses_get_frame_when_available(tmp_path):
    NewtonSimulator = _newton_simulator_cls()
    mock_self = MagicMock()
    frame = MagicMock()
    frame.numpy.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    mock_self.viewer = MagicMock(spec=["get_frame"])
    mock_self.viewer.get_frame.return_value = frame

    out_file = tmp_path / "frame.png"
    NewtonSimulator._write_viewport_to_file(mock_self, str(out_file))

    mock_self.viewer.get_frame.assert_called_once()
    assert out_file.exists()


def test_update_camera_falls_back_to_last_commanded_pos_without_camera_attr():
    NewtonSimulator = _newton_simulator_cls()
    mock_self = MagicMock()
    mock_self._camera_target = {"element": 0, "env": 0}
    mock_self._get_simulator_root_state.return_value = SimpleNamespace(
        root_pos=torch.tensor([[1.0, 2.0, 0.5]])
    )
    mock_self._cam_prev_char_pos = np.array([0.0, 0.0, 0.0])
    mock_self._last_cam_pos = np.array([0.0, -5.0, 1.0])
    mock_self.viewer = MagicMock(spec=["set_camera"])  # no `.camera` attribute, like ViewerViser

    NewtonSimulator._update_camera(mock_self)

    mock_self.viewer.set_camera.assert_called_once()
    assert np.allclose(mock_self._last_cam_pos, [1.0, -3.0, 1.5])


def test_viewer_viser_matches_expected_upstream_api():
    """Guard against upstream newton/viser API drift this integration relies on."""
    pytest.importorskip("viser")
    import newton

    viewer = newton.viewer.ViewerViser(port=0, verbose=False)
    try:
        assert viewer.is_running() is True
        assert not hasattr(viewer, "camera")
        assert not hasattr(viewer, "get_frame")
    finally:
        viewer.close()
    assert viewer.is_running() is False

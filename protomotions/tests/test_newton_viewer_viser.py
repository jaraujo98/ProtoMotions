# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Newton simulator's viser viewer backend support.

These exercise the viewer_backend config option and the NewtonSimulator methods
that had to be made viewer-agnostic (close, _write_viewport_to_file, and
render()'s camera-update dispatch) so ViewerViser can be used in place of
ViewerGL. ViewerViser has no live, locally-readable `.camera` like ViewerGL
does, so render() only positions the camera once and otherwise leaves it
under the browser's own orbit/zoom/pan controls rather than fighting them.
Unlike ViewerGL, ViewerViser only starts a local web server, so these tests
need no display and no Xvfb.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def test_add_viser_recenter_button_registers_click_handler():
    NewtonSimulator = _newton_simulator_cls()
    mock_self = MagicMock()
    mock_self._viser_recenter_requested = False
    mock_button = MagicMock()
    mock_server = MagicMock()
    mock_server.gui.add_button.return_value = mock_button
    mock_self.viewer = MagicMock(spec=["_server"])
    mock_self.viewer._server = mock_server

    NewtonSimulator._add_viser_recenter_button(mock_self)

    mock_server.gui.add_button.assert_called_once()
    assert mock_server.gui.add_button.call_args[0][0] == "Focus Camera"
    on_click_callback = mock_button.on_click.call_args[0][0]

    on_click_callback(MagicMock())  # simulate the button being clicked

    assert mock_self._viser_recenter_requested is True


def test_add_viser_recenter_button_noop_without_server():
    NewtonSimulator = _newton_simulator_cls()
    mock_self = MagicMock()
    mock_self.viewer = MagicMock(spec=[])  # no `_server` attribute, like ViewerGL

    NewtonSimulator._add_viser_recenter_button(mock_self)  # must not raise


def _make_render_mock_self(NewtonSimulator, viewer_spec, recenter_requested=False):
    """Build a MagicMock `self` that real, unmocked NewtonSimulator.render()
    (and its super().render() call) can run against safely. Spoofing
    `__class__` makes `isinstance`/`super()` treat it as a real
    NewtonSimulator, while attribute access still auto-mocks like a normal
    MagicMock (so e.g. `self._update_camera()` calls are simply recorded).
    """
    mock_self = MagicMock()
    mock_self.__class__ = NewtonSimulator
    mock_self.headless = False
    mock_self._camera_initialized = True
    mock_self._viser_recenter_requested = recenter_requested
    mock_self.viewer = MagicMock(spec=viewer_spec)
    mock_self.user_interface.registered_key_names.return_value = []
    return mock_self


def test_render_skips_update_camera_for_viewers_without_camera_attr():
    """ViewerViser has no `.camera`, so render() must not re-command the
    camera every frame and fight the user's in-browser drag/zoom/pan."""
    NewtonSimulator = _newton_simulator_cls()
    from protomotions.simulator.base_simulator.record import RecordingMixin

    mock_self = _make_render_mock_self(
        NewtonSimulator, viewer_spec=["begin_frame", "log_state", "end_frame", "is_key_down"]
    )

    with patch.object(RecordingMixin, "render"):
        NewtonSimulator.render(mock_self)

    mock_self._update_camera.assert_not_called()
    mock_self._init_camera.assert_not_called()


def test_render_recenters_on_viser_when_button_clicked():
    """Clicking the "Focus Camera" button sets _viser_recenter_requested;
    render() should consume it by re-running _init_camera() once and
    clearing the flag, without touching _update_camera (still GL-only)."""
    NewtonSimulator = _newton_simulator_cls()
    from protomotions.simulator.base_simulator.record import RecordingMixin

    mock_self = _make_render_mock_self(
        NewtonSimulator,
        viewer_spec=["begin_frame", "log_state", "end_frame", "is_key_down"],
        recenter_requested=True,
    )

    with patch.object(RecordingMixin, "render"):
        NewtonSimulator.render(mock_self)

    mock_self._init_camera.assert_called_once_with()
    mock_self._update_camera.assert_not_called()
    assert mock_self._viser_recenter_requested is False


def test_render_calls_update_camera_for_viewers_with_camera_attr():
    """ViewerGL exposes a live `.camera`, so render() should keep following
    the target via _update_camera() as before."""
    NewtonSimulator = _newton_simulator_cls()
    from protomotions.simulator.base_simulator.record import RecordingMixin

    mock_self = _make_render_mock_self(
        NewtonSimulator,
        viewer_spec=["camera", "begin_frame", "log_state", "end_frame", "is_key_down"],
    )

    with patch.object(RecordingMixin, "render"):
        NewtonSimulator.render(mock_self)

    mock_self._update_camera.assert_called_once_with()


def test_update_camera_reads_live_camera_pos():
    NewtonSimulator = _newton_simulator_cls()
    mock_self = MagicMock()
    mock_self._camera_target = {"element": 0, "env": 0}
    mock_self._get_simulator_root_state.return_value = SimpleNamespace(
        root_pos=torch.tensor([[1.0, 2.0, 0.5]])
    )
    mock_self._cam_prev_char_pos = np.array([0.0, 0.0, 0.0])
    mock_self.viewer = MagicMock(spec=["camera", "set_camera"])
    mock_self.viewer.camera.pos = np.array([0.0, -5.0, 1.0])

    NewtonSimulator._update_camera(mock_self)

    mock_self.viewer.set_camera.assert_called_once()
    # cam_delta (from viewer.camera.pos) is preserved onto the new char pos.
    new_cam_pos = np.asarray(mock_self.viewer.set_camera.call_args[0][0])
    assert np.allclose(new_cam_pos, [1.0, -3.0, 1.5])


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

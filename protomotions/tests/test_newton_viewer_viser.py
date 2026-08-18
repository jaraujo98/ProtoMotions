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

import threading
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


def _build_real_simulator(viewer_backend, num_envs=1):
    """Construct a real (unmocked) NewtonSimulator, headless for speed --
    _create_simulation() builds the ground plane regardless of headless, so
    this doesn't need an actual display/viewer to exercise that logic."""
    NewtonSimulator = _newton_simulator_cls()
    NewtonSimulatorConfig, NewtonSimParams = _newton_config_classes()
    from protomotions.components.scene_lib import SceneLib
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.components.terrains.terrain import Terrain
    from protomotions.robot_configs.factory import robot_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot_cfg = robot_config("g1")
    terrain_config = TerrainConfig(
        map_length=20.0,
        map_width=20.0,
        border_size=40.0,
        num_levels=1,
        num_terrains=1,
        terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        horizontal_scale=0.1,
        vertical_scale=0.005,
    )
    terrain = Terrain(config=terrain_config, num_envs=num_envs, device=device)
    scene_lib = SceneLib.empty(num_envs=num_envs, device=str(device), terrain=terrain)

    sim_config = NewtonSimulatorConfig(
        num_envs=num_envs,
        headless=True,
        sim=NewtonSimParams(fps=30, decimation=1),
        experiment_name="ground_plane_test",
        viewer_backend=viewer_backend,
    )
    simulator = NewtonSimulator(
        config=sim_config,
        robot_config=robot_cfg,
        terrain=terrain,
        device=device,
        scene_lib=scene_lib,
    )
    simulator._initialize_with_markers({})
    return simulator


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


def _ground_plane_scale(simulator):
    idx = simulator.model.shape_label.index("ground_plane")
    return simulator.model.shape_scale.numpy()[idx]


def test_ground_plane_is_generously_sized_for_viser_backend():
    """ViewerViser auto-sizes Newton's "infinite" (zero-scale) ground plane
    from a single world's extents, which can leave the floor invisible in
    multi-env layouts -- for viser we give it an explicit large finite size
    instead."""
    simulator = _build_real_simulator(viewer_backend="viser", num_envs=2)

    width, length, _ = _ground_plane_scale(simulator)

    assert width == 1000.0
    assert length == 1000.0


def test_ground_plane_stays_infinite_for_gl_backend():
    """ViewerGL already renders the "infinite" zero-scale plane correctly,
    so the gl/default path must be untouched by the viser-only workaround."""
    simulator = _build_real_simulator(viewer_backend="gl", num_envs=2)

    width, length, _ = _ground_plane_scale(simulator)

    assert width == 0.0
    assert length == 0.0


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


def _make_reset_envs_mock_self(viewer_backend, headless, camera_target_env=0, num_envs=4):
    mock_self = MagicMock()
    mock_self.__class__ = _newton_simulator_cls()
    mock_self.headless = headless
    mock_self.config.viewer_backend = viewer_backend
    mock_self.num_envs = num_envs
    mock_self.device = torch.device("cpu")
    mock_self._camera_target = {"env": camera_target_env, "element": 0}
    mock_self._viser_recenter_requested = False
    return mock_self


def test_reset_envs_requests_recenter_when_tracked_env_is_reset():
    NewtonSimulator = _newton_simulator_cls()
    from protomotions.simulator.base_simulator.simulator import Simulator

    mock_self = _make_reset_envs_mock_self(
        "viser", headless=False, camera_target_env=2
    )
    new_states = MagicMock()
    new_object_states = MagicMock()
    env_ids = torch.tensor([1, 2, 3])

    with patch.object(Simulator, "reset_envs") as mock_super_reset:
        NewtonSimulator.reset_envs(mock_self, new_states, new_object_states, env_ids)

    mock_super_reset.assert_called_once_with(new_states, new_object_states, env_ids)
    assert mock_self._viser_recenter_requested is True


def test_reset_envs_does_not_recenter_for_untracked_env():
    NewtonSimulator = _newton_simulator_cls()
    from protomotions.simulator.base_simulator.simulator import Simulator

    mock_self = _make_reset_envs_mock_self(
        "viser", headless=False, camera_target_env=0
    )
    env_ids = torch.tensor([1, 2, 3])

    with patch.object(Simulator, "reset_envs"):
        NewtonSimulator.reset_envs(mock_self, MagicMock(), env_ids=env_ids)

    assert mock_self._viser_recenter_requested is False


def test_reset_envs_recenters_when_env_ids_is_none():
    NewtonSimulator = _newton_simulator_cls()
    from protomotions.simulator.base_simulator.simulator import Simulator

    mock_self = _make_reset_envs_mock_self(
        "viser", headless=False, camera_target_env=3, num_envs=4
    )

    with patch.object(Simulator, "reset_envs"):
        NewtonSimulator.reset_envs(mock_self, MagicMock(), env_ids=None)

    assert mock_self._viser_recenter_requested is True


def test_reset_envs_ignores_gl_backend():
    NewtonSimulator = _newton_simulator_cls()
    from protomotions.simulator.base_simulator.simulator import Simulator

    mock_self = _make_reset_envs_mock_self("gl", headless=False, camera_target_env=0)
    env_ids = torch.tensor([0])

    with patch.object(Simulator, "reset_envs"):
        NewtonSimulator.reset_envs(mock_self, MagicMock(), env_ids=env_ids)

    assert mock_self._viser_recenter_requested is False


def test_reset_envs_ignores_headless():
    NewtonSimulator = _newton_simulator_cls()
    from protomotions.simulator.base_simulator.simulator import Simulator

    mock_self = _make_reset_envs_mock_self("viser", headless=True, camera_target_env=0)
    env_ids = torch.tensor([0])

    with patch.object(Simulator, "reset_envs"):
        NewtonSimulator.reset_envs(mock_self, MagicMock(), env_ids=env_ids)

    assert mock_self._viser_recenter_requested is False


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


def test_add_viser_key_button_creates_button_with_key_and_description():
    from protomotions.simulator.base_simulator.user_interface import UserInterface

    NewtonSimulator = _newton_simulator_cls()
    mock_self = MagicMock()
    mock_self._viser_shortcuts_folder = MagicMock()  # supports `with folder:` out of the box
    button = MagicMock()
    mock_self.viewer._server.gui.add_button.return_value = button
    mock_self._viser_pending_presses = set()
    mock_self._viser_pending_lock = threading.Lock()

    ui = UserInterface()
    handle = ui.register_key("R", owner="test", description="Reset all environments")

    NewtonSimulator._add_viser_key_button(mock_self, handle)

    # GuiFolderHandle has no add_button of its own; the real button is added
    # via server.gui.add_button() while the folder is entered as a context
    # manager (which retargets where it lands).
    mock_self._viser_shortcuts_folder.__enter__.assert_called_once()
    mock_self.viewer._server.gui.add_button.assert_called_once_with(
        "R: Reset all environments"
    )
    # The on_click decorator was used to register the callback.
    button.on_click.assert_called_once()
    on_click_callback = button.on_click.call_args[0][0]

    on_click_callback(None)

    assert mock_self._viser_pending_presses == {"R"}


def test_setup_viser_key_buttons_replays_existing_and_future_keys():
    from protomotions.simulator.base_simulator.user_interface import UserInterface

    NewtonSimulator = _newton_simulator_cls()
    mock_self = MagicMock()
    ui = UserInterface()
    ui.register_key("Q", owner="simulator", description="Close simulator viewer")
    mock_self.user_interface = ui
    mock_self.viewer = MagicMock()
    folder = MagicMock()
    mock_self.viewer._server.gui.add_folder.return_value = folder
    added_labels = []

    def fake_add_viser_key_button(handle):
        added_labels.append(handle.key)

    mock_self._add_viser_key_button = fake_add_viser_key_button

    NewtonSimulator._setup_viser_key_buttons(mock_self)

    assert added_labels == ["Q"]  # replay_existing picked up the pre-registered key

    ui.register_key("M", owner="simulator", description="Toggle markers")

    assert added_labels == ["Q", "M"]  # future registrations are also mirrored
    mock_self.viewer._server.gui.add_folder.assert_called_once_with(
        "Keyboard Shortcuts"
    )


def test_is_key_pressed_reads_real_key_state():
    NewtonSimulator = _newton_simulator_cls()
    mock_self = MagicMock()
    mock_self.viewer.is_key_down.return_value = True
    mock_self._viser_pending_presses = set()
    mock_self._viser_pending_lock = threading.Lock()

    assert NewtonSimulator._is_key_pressed(mock_self, "Q") is True
    mock_self.viewer.is_key_down.assert_called_once_with("q")


def test_is_key_pressed_drains_pending_viser_click_as_one_shot():
    NewtonSimulator = _newton_simulator_cls()
    mock_self = MagicMock()
    mock_self.viewer.is_key_down.return_value = False
    mock_self._viser_pending_presses = {"R"}
    mock_self._viser_pending_lock = threading.Lock()

    assert NewtonSimulator._is_key_pressed(mock_self, "R") is True
    # One-shot: the pending press is consumed, so it doesn't stay "down".
    assert mock_self._viser_pending_presses == set()
    assert NewtonSimulator._is_key_pressed(mock_self, "R") is False


def test_viser_button_click_fires_on_press_exactly_once_via_real_user_interface():
    """End-to-end through the real UserInterface: a click is one edge-triggered
    press, not a stuck-down key, matching a real keypress's semantics."""
    from protomotions.simulator.base_simulator.user_interface import UserInterface

    NewtonSimulator = _newton_simulator_cls()
    ui = UserInterface()
    press_count = 0

    def on_press():
        nonlocal press_count
        press_count += 1

    handle = ui.register_key(
        "R", owner="test", description="Reset", on_press=on_press
    )

    mock_self = MagicMock()
    mock_self.viewer.is_key_down.return_value = False
    mock_self._viser_pending_presses = set()
    mock_self._viser_pending_lock = threading.Lock()

    # Simulate a viser button click (as done from _add_viser_key_button's
    # on_click closure) landing in the pending set from another thread.
    with mock_self._viser_pending_lock:
        mock_self._viser_pending_presses.add("R")

    ui.begin_step()
    ui.handle_key_event("R", pressed=NewtonSimulator._is_key_pressed(mock_self, "R"))
    assert press_count == 1
    assert handle.consume() is True

    # Next frame: no viewer key-down and no new click, so it releases cleanly.
    ui.begin_step()
    ui.handle_key_event("R", pressed=NewtonSimulator._is_key_pressed(mock_self, "R"))
    assert press_count == 1  # no repeated fire
    assert handle.consume() is False


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

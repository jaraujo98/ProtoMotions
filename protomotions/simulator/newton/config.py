# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig


@dataclass
class NewtonSimParams(SimParams):
    """Newton/MuJoCo solver parameters."""

    solver: str = field(
        default="newton",
        metadata={"help": "Constraint solver: 'newton', 'cg', or 'direct'."}
    )
    integrator: str = field(
        default="implicitfast",
        metadata={"help": "Integrator: 'euler', 'implicit', or 'implicitfast'."}
    )
    iterations: int = field(
        default=100,
        metadata={"help": "Max solver iterations."}
    )
    ls_iterations: int = field(
        default=50,
        metadata={"help": "Line search iterations."}
    )
    ls_parallel: bool = field(
        default=True,
        metadata={"help": "Run line search in parallel."}
    )
    impratio: float = field(
        default=10.0,
        metadata={"help": "Implicit integration ratio."}
    )
    njmax: int = field(
        default=450,
        metadata={"help": "Max constraint Jacobian rows."}
    )
    nconmax: int = field(
        default=300,
        metadata={"help": "Max contacts."}
    )
    cone: str = field(
        default="pyramidal",
        metadata={"help": "Friction cone: 'pyramidal' or 'elliptic'."}
    )
    ccd_iterations: int = field(
        default=200,
        metadata={"help": "CCD (continuous collision detection) iterations."}
    )


@dataclass
class NewtonSimulatorConfig(SimulatorConfig):
    """Configuration specific to Newton simulator."""

    _target_: str = "protomotions.simulator.newton.simulator.NewtonSimulator"
    sim: NewtonSimParams = field(default_factory=NewtonSimParams)  # Override sim type
    w_last: bool = True  # Newton uses xyzw quaternions
    viewer_backend: str = field(
        default="gl",
        metadata={"help": "Viewer backend: 'gl' (local OpenGL window) or 'viser' (browser-based, remote-friendly)."}
    )
    viewer_port: int = field(
        default=8080,
        metadata={"help": "Port for the viser web server (only used when viewer_backend='viser')."}
    )
    viewer_share: bool = field(
        default=False,
        metadata={"help": "If True, expose the viser viewer via a public share URL (only used when viewer_backend='viser')."}
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.viewer_backend in ("gl", "viser"), (
            f"NewtonSimulatorConfig.viewer_backend must be 'gl' or 'viser', got {self.viewer_backend!r}"
        )

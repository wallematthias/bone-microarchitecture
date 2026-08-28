"""Direct 3D bone microarchitecture measurements from aligned arrays."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .opencl import opencl_hildebrand_thickness_map
from .pipeline import compute_microarchitecture
from .results import MicroarchitectureResult, PARAMETER_DEFINITIONS
from .thickness import (
    default_thickness_backend,
    hildebrand_thickness_map,
    local_thickness_map,
    trabecular_number_map,
)

try:
    __version__ = version("bone-microarchitecture")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "MicroarchitectureResult",
    "PARAMETER_DEFINITIONS",
    "__version__",
    "compute_microarchitecture",
    "default_thickness_backend",
    "hildebrand_thickness_map",
    "local_thickness_map",
    "opencl_hildebrand_thickness_map",
    "trabecular_number_map",
]

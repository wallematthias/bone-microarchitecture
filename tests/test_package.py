from __future__ import annotations

from bone_microarchitecture import (
    MicroarchitectureResult,
    PARAMETER_DEFINITIONS,
    __version__,
    compute_microarchitecture,
)


def test_public_api_exports_result_and_compute_function():
    assert MicroarchitectureResult.__name__ == "MicroarchitectureResult"
    assert "Tb.Th" in PARAMETER_DEFINITIONS
    assert __version__
    assert callable(compute_microarchitecture)

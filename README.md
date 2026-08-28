# Bone Microarchitecture

Lightweight microarchitecture measurements from binary masks and optional calibrated grayscale arrays.

This package intentionally has no Slicer dependency and no image I/O dependency. Callers are responsible for loading images, calibration, and putting masks on a common grid.

## GPU Backends

Exact Hildebrand sphere fitting supports three diameter-accumulation backends:

- `cpu`: NumPy/SciPy fallback.
- `mps`: native Apple Metal backend for macOS.
- `opencl`: OpenCL backend for Windows/Linux systems with `pyopencl` and a working GPU OpenCL runtime.

Use `thickness_backend="auto"` to select Metal on macOS when available, OpenCL on Windows/Linux when available, and CPU otherwise.

Optional installs:

```bash
pip install "bone-microarchitecture[mps]"
pip install "bone-microarchitecture[opencl]"
pip install "bone-microarchitecture[gpu]"
```

## Testing

Run the standalone package tests before pushing changes:

```bash
python -m pip install -e ".[test]"
python -m py_compile src/bone_microarchitecture/*.py
python -m pytest -q
```

## Packaging

The package follows the same PyPI release pattern as the Timelapsed HR-pQCT
packages: build from `pyproject.toml`, verify distributions with Twine, and
publish tagged releases through GitHub trusted publishing.

Local build check:

```bash
python -m pip install -e ".[dev]"
python -m build
python -m twine check dist/*
```

To publish from GitHub, create a tag that matches the version in
`pyproject.toml`, for example `v0.1.0`. The `Publish To PyPI` workflow builds
the wheel and source distribution, checks them, and publishes to PyPI from the
protected `pypi` environment.

When testing through SlicerBoneImagingToolbox, also run the focused wrapper tests
from the extension checkout:

```bash
python3 -m pytest tests/test_microarchitecture_module.py tests/test_package_status.py -q
```

Some extension tests import Slicer-only modules such as `ctk` and need to be run
inside the Slicer Python environment rather than plain system Python.

## Parameter Definitions

- `Tb.BMD`: mean calibrated grayscale value inside the trabecular compartment.
- `Tb.BV/TV`: trabecular bone volume divided by trabecular total volume.
- `Tb.Th`: maximal-sphere local thickness of trabecular bone.
- `Tb.Sp`: maximal-sphere local thickness of non-bone space in the trabecular compartment.
- `Tb.N`: inverse ridge-to-ridge spacing estimate in the trabecular compartment.
- `Tb.1/N.SD`: standard deviation of ridge-to-ridge spacing.
- `Tb.BV`: trabecular bone volume.
- `Tb.TV`: trabecular compartment volume.
- `Ct.BMD`: mean calibrated grayscale value inside the cortical compartment.
- `Ct.Th`: maximal-sphere local thickness of cortical bone.
- `Ct.Po`: cortical pore volume divided by cortical total volume.
- `Ct.Po.V`: cortical pore volume.
- `Ct.Po.Dm`: maximal-sphere local diameter of cortical pore space.
- `Ct.BV`: cortical bone volume.
- `Ct.TV`: cortical compartment volume.

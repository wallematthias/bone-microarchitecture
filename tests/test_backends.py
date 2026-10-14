from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import pytest


def _opencl_devices(*devices):
    return SimpleNamespace(
        device_type=SimpleNamespace(GPU=4, CPU=2),
        get_platforms=lambda: [SimpleNamespace(get_devices=lambda: list(devices))],
    )


def test_opencl_gpu_selection_rejects_cpu_only_runtime():
    from bone_microarchitecture.opencl import _select_device

    cpu = SimpleNamespace(type=2, name="OpenCL CPU")
    with pytest.raises(RuntimeError, match="GPU"):
        _select_device(_opencl_devices(cpu), prefer_gpu=True)


def test_opencl_gpu_selection_skips_cpu_and_returns_gpu():
    from bone_microarchitecture.opencl import _select_device

    cpu = SimpleNamespace(type=2, name="OpenCL CPU")
    gpu = SimpleNamespace(type=4, name="Windows GPU")
    assert _select_device(_opencl_devices(cpu, gpu), prefer_gpu=True) is gpu
    assert _select_device(_opencl_devices(cpu), prefer_gpu=False) is cpu


def test_auto_backend_explains_missing_opencl_gpu(monkeypatch):
    from bone_microarchitecture import thickness

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "pyopencl", _opencl_devices(SimpleNamespace(type=2)))
    status = thickness.thickness_backend_status()
    assert status["thickness_backend"] == "cpu"
    assert status["thickness_gpu"] is False
    assert "GPU" in status["thickness_backend_reason"]
    assert thickness.default_thickness_backend() == "cpu"


def test_auto_backend_reports_gpu_device_and_compiles_before_selection(monkeypatch):
    from bone_microarchitecture import opencl, thickness

    monkeypatch.setattr(sys, "platform", "win32")
    gpu = SimpleNamespace(type=4, name="Windows GPU")
    monkeypatch.setattr(opencl, "_get_opencl_program", lambda cl: (None, SimpleNamespace(device=gpu), None))
    monkeypatch.setitem(sys.modules, "pyopencl", _opencl_devices(gpu))
    status = thickness.thickness_backend_status()
    assert status["thickness_backend"] == "opencl"
    assert status["thickness_device"] == "Windows GPU"
    assert status["thickness_gpu"] is True

    def broken_program(cl):
        raise RuntimeError("driver cannot compile kernel")

    monkeypatch.setattr(opencl, "_get_opencl_program", broken_program)
    status = thickness.thickness_backend_status()
    assert status["thickness_backend"] == "cpu"
    assert "driver cannot compile" in status["thickness_backend_reason"]
    with pytest.raises(RuntimeError, match="driver cannot compile"):
        thickness.thickness_backend_status("opencl")


def test_metal_backend_accumulates_local_diameters():
    from bone_microarchitecture import metal

    assert "kernel void accumulate_local_diameters" in metal.METAL_SOURCE
    assert "atomic_fetch_max_explicit" in metal.METAL_SOURCE


def test_opencl_backend_copies_from_output_buffer():
    from bone_microarchitecture import opencl

    source = inspect.getsource(opencl.opencl_hildebrand_thickness_map)

    assert "accumulate_local_diameters_kernel" in opencl.OPENCL_SOURCE
    assert "atomic_max" in opencl.OPENCL_SOURCE
    assert "cl.enqueue_copy(queue, output, buffers[5])" in source
    assert "cl.enqueue_copy(queue, output, buffers[4])" not in source

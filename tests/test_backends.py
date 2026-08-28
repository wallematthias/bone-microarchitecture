from __future__ import annotations

import inspect


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

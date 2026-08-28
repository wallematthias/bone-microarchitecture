from __future__ import annotations

import numpy as np


OPENCL_SOURCE = r"""
__kernel void accumulate_local_diameters_kernel(
    __global const unsigned int *seed_z,
    __global const unsigned int *seed_y,
    __global const unsigned int *seed_x,
    __global const float *seed_radius,
    const unsigned int seed_count,
    __global volatile unsigned int *diameter_map,
    __global const unsigned int *volume_shape,
    __global const float *spacing,
    const float diameter_margin,
    const float output_scale,
    const float inclusion_tolerance
) {
    const int gid = get_global_id(0);
    if ((unsigned int)gid >= seed_count) {
        return;
    }
    const float radius = seed_radius[gid];
    const float value_mm = fmax(2.0f * (radius - diameter_margin), 0.0f);
    if (value_mm <= 0.0f) {
        return;
    }
    const unsigned int value = (unsigned int)rint(value_mm * output_scale);

    const int zc = (int)seed_z[gid];
    const int yc = (int)seed_y[gid];
    const int xc = (int)seed_x[gid];
    const int z_extent = (int)ceil(radius / spacing[0]);
    const int y_extent = (int)ceil(radius / spacing[1]);
    const int x_extent = (int)ceil(radius / spacing[2]);

    const int z0 = max(0, zc - z_extent);
    const int y0 = max(0, yc - y_extent);
    const int x0 = max(0, xc - x_extent);
    const int z1 = min((int)volume_shape[0], zc + z_extent + 1);
    const int y1 = min((int)volume_shape[1], yc + y_extent + 1);
    const int x1 = min((int)volume_shape[2], xc + x_extent + 1);
    const float r2 = radius * radius;

    for (int z = z0; z < z1; ++z) {
        const float dz = (float)(z - zc) * spacing[0];
        for (int y = y0; y < y1; ++y) {
            const float dy = (float)(y - yc) * spacing[1];
            for (int x = x0; x < x1; ++x) {
                const float dx = (float)(x - xc) * spacing[2];
                if ((dx * dx + dy * dy + dz * dz) <= (r2 + inclusion_tolerance)) {
                    const unsigned int index = ((unsigned int)z * volume_shape[1] + (unsigned int)y) * volume_shape[2] + (unsigned int)x;
                    atomic_max(&diameter_map[index], value);
                }
            }
        }
    }
}
"""

_OPENCL_CACHE = {}


def is_opencl_available(prefer_gpu: bool = True) -> bool:
    """Return whether an OpenCL device is available through PyOpenCL."""
    try:
        import pyopencl as cl
    except Exception:
        return False
    try:
        _select_device(cl, prefer_gpu=prefer_gpu)
        return True
    except Exception:
        return False


def opencl_hildebrand_thickness_map(
    *,
    shape: tuple[int, int, int],
    seed_z,
    seed_y,
    seed_x,
    seed_radius,
    spacing: tuple[float, float, float],
    diameter_margin: float,
    inclusion_tolerance: float,
    output_scale: float = 1_000_000.0,
) -> np.ndarray:
    """Accumulate maximal-sphere diameter values with an OpenCL kernel.

    Seed locations and inscribed radii are selected by :mod:`bone_microarchitecture.thickness`.
    This backend only applies the corresponding diameters to the output volume
    using atomic maximum updates.
    """
    try:
        import pyopencl as cl
    except Exception as exc:
        raise RuntimeError("OpenCL backend requires pyopencl and an OpenCL runtime.") from exc

    seed_z = np.asarray(seed_z, dtype=np.uint32)
    seed_y = np.asarray(seed_y, dtype=np.uint32)
    seed_x = np.asarray(seed_x, dtype=np.uint32)
    seed_radius = np.asarray(seed_radius, dtype=np.float32)
    if not (seed_z.size == seed_y.size == seed_x.size == seed_radius.size):
        raise ValueError("Seed coordinate and radius arrays must have the same length.")
    if seed_radius.size == 0:
        return np.zeros(tuple(shape), dtype=np.float32)

    ctx, queue, program = _get_opencl_program(cl)
    flags = cl.mem_flags
    output = np.zeros(int(np.prod(shape)), dtype=np.uint32)
    buffers = [
        cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(seed_z)),
        cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(seed_y)),
        cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(seed_x)),
        cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(seed_radius)),
        np.uint32(seed_radius.size),
        cl.Buffer(ctx, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=output),
        cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=np.asarray(shape, dtype=np.uint32)),
        cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=np.asarray(spacing, dtype=np.float32)),
    ]

    local_size = 256
    global_size = int(((int(seed_radius.size) + local_size - 1) // local_size) * local_size)
    program.accumulate_local_diameters_kernel(
        queue,
        (global_size,),
        (local_size,),
        *buffers,
        np.float32(diameter_margin),
        np.float32(output_scale),
        np.float32(inclusion_tolerance),
    )
    cl.enqueue_copy(queue, output, buffers[5])
    queue.finish()
    return (output.reshape(tuple(shape)).astype(np.float32) / float(output_scale)).copy()


def _get_opencl_program(cl):
    key = "default"
    if key in _OPENCL_CACHE:
        return _OPENCL_CACHE[key]
    device = _select_device(cl, prefer_gpu=True)
    ctx = cl.Context([device])
    command_queue = cl.CommandQueue(ctx)
    program = cl.Program(ctx, OPENCL_SOURCE).build()
    _OPENCL_CACHE[key] = (ctx, command_queue, program)
    return _OPENCL_CACHE[key]


def _select_device(cl, *, prefer_gpu: bool):
    fallback = None
    for platform in cl.get_platforms():
        devices = platform.get_devices()
        for device in devices:
            if fallback is None:
                fallback = device
            if prefer_gpu and device.type & cl.device_type.GPU:
                return device
    if fallback is not None and not prefer_gpu:
        return fallback
    if fallback is not None:
        return fallback
    raise RuntimeError("No OpenCL devices are available.")

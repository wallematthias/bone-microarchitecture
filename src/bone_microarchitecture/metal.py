from __future__ import annotations

import numpy as np


METAL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void accumulate_local_diameters(
    device const uint* seed_z [[buffer(0)]],
    device const uint* seed_y [[buffer(1)]],
    device const uint* seed_x [[buffer(2)]],
    device const float* seed_radius [[buffer(3)]],
    device atomic_uint* diameter_map [[buffer(4)]],
    device const uint* volume_shape [[buffer(5)]],
    device const float* spacing [[buffer(6)]],
    constant float& diameter_margin [[buffer(7)]],
    constant float& output_scale [[buffer(8)]],
    constant float& inclusion_tolerance [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    const float radius = seed_radius[gid];
    const float value_mm = max(2.0f * (radius - diameter_margin), 0.0f);
    if (value_mm <= 0.0f) {
        return;
    }
    const uint value = uint(round(value_mm * output_scale));

    const int zc = int(seed_z[gid]);
    const int yc = int(seed_y[gid]);
    const int xc = int(seed_x[gid]);
    const int z_extent = int(ceil(radius / spacing[0]));
    const int y_extent = int(ceil(radius / spacing[1]));
    const int x_extent = int(ceil(radius / spacing[2]));

    const int z0 = max(0, zc - z_extent);
    const int y0 = max(0, yc - y_extent);
    const int x0 = max(0, xc - x_extent);
    const int z1 = min(int(volume_shape[0]), zc + z_extent + 1);
    const int y1 = min(int(volume_shape[1]), yc + y_extent + 1);
    const int x1 = min(int(volume_shape[2]), xc + x_extent + 1);
    const float r2 = radius * radius;

    for (int z = z0; z < z1; ++z) {
        const float dz = float(z - zc) * spacing[0];
        for (int y = y0; y < y1; ++y) {
            const float dy = float(y - yc) * spacing[1];
            for (int x = x0; x < x1; ++x) {
                const float dx = float(x - xc) * spacing[2];
                if ((dx * dx + dy * dy + dz * dz) <= (r2 + inclusion_tolerance)) {
                    const uint index = (uint(z) * volume_shape[1] + uint(y)) * volume_shape[2] + uint(x);
                    atomic_fetch_max_explicit(&diameter_map[index], value, memory_order_relaxed);
                }
            }
        }
    }
}
"""


def is_metal_available() -> bool:
    """Return whether Apple Metal compute is available through PyObjC."""
    try:
        import Metal
    except Exception:
        return False
    try:
        return Metal.MTLCreateSystemDefaultDevice() is not None
    except Exception:
        return False


def metal_hildebrand_thickness_map(
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
    """Accumulate maximal-sphere diameter values with a native Metal kernel.

    Seed locations and inscribed radii are selected by :mod:`bone_microarchitecture.thickness`.
    This backend only applies the corresponding diameters to the output volume
    using atomic maximum updates.
    """
    try:
        import Metal
    except Exception as exc:
        raise RuntimeError("Apple Metal backend requires PyObjC Metal bindings.") from exc

    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("Apple Metal backend is not available on this machine.")

    seed_z = np.asarray(seed_z, dtype=np.uint32)
    seed_y = np.asarray(seed_y, dtype=np.uint32)
    seed_x = np.asarray(seed_x, dtype=np.uint32)
    seed_radius = np.asarray(seed_radius, dtype=np.float32)
    if not (seed_z.size == seed_y.size == seed_x.size == seed_radius.size):
        raise ValueError("Seed coordinate and radius arrays must have the same length.")
    if seed_radius.size == 0:
        return np.zeros(tuple(shape), dtype=np.float32)

    library, error = device.newLibraryWithSource_options_error_(METAL_SOURCE, None, None)
    if library is None:
        raise RuntimeError(f"Could not compile Metal sphere fitting kernel: {error}")
    function = library.newFunctionWithName_("accumulate_local_diameters")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    if pipeline is None:
        raise RuntimeError(f"Could not create Metal sphere fitting pipeline: {error}")

    queue = device.newCommandQueue()
    if queue is None:
        raise RuntimeError("Could not create Metal command queue.")

    output = np.zeros(int(np.prod(shape)), dtype=np.uint32)
    buffers = [
        _readonly_buffer(device, seed_z),
        _readonly_buffer(device, seed_y),
        _readonly_buffer(device, seed_x),
        _readonly_buffer(device, seed_radius),
        _shared_buffer(device, output),
        _readonly_buffer(device, np.asarray(shape, dtype=np.uint32)),
        _readonly_buffer(device, np.asarray(spacing, dtype=np.float32)),
        _readonly_buffer(device, np.asarray([diameter_margin], dtype=np.float32)),
        _readonly_buffer(device, np.asarray([output_scale], dtype=np.float32)),
        _readonly_buffer(device, np.asarray([inclusion_tolerance], dtype=np.float32)),
    ]

    command_buffer = queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for index, buffer in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(buffer, 0, index)

    threads_per_group = Metal.MTLSizeMake(min(int(pipeline.maxTotalThreadsPerThreadgroup()), 256), 1, 1)
    grid = Metal.MTLSizeMake(int(seed_radius.size), 1, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid, threads_per_group)
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    if command_buffer.error() is not None:
        raise RuntimeError(f"Metal sphere fitting command failed: {command_buffer.error()}")

    output_view = np.frombuffer(buffers[4].contents().as_buffer(output.nbytes), dtype=np.uint32)
    return (output_view.reshape(tuple(shape)).astype(np.float32) / float(output_scale)).copy()


def _readonly_buffer(device, array: np.ndarray):
    array = np.ascontiguousarray(array)
    return device.newBufferWithBytes_length_options_(array.tobytes(), array.nbytes, 0)


def _shared_buffer(device, array: np.ndarray):
    array = np.ascontiguousarray(array)
    return device.newBufferWithBytes_length_options_(array.tobytes(), array.nbytes, 0)

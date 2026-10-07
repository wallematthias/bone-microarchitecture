"""Fixed cortical pore-selection stages translated from the supplied IPL recipe.

The stages follow Burghardt et al., Bone 2010, Figure 2: enclosed XY seeds,
longitudinal hysteresis growth, a second XY pass, and >=5-voxel components.
IPL's native connectivity/percentage conventions are not bitwise validated.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def cortical_pore_mask(bone_mask, cortical_mask) -> np.ndarray:
    """Select intracortical pores without treating every cortical void as a pore.

    Arrays are Z,Y,X. Do not close the scan ends: longitudinal canals may remain
    open at the acquired Z faces. Connectivity is 8 in XY, 26 for final volume
    cleanup, and axial-only during hysteresis growth. Those choices are explicit
    approximations of the native IPL operators, not inferred API guarantees.
    """
    cort = np.asarray(cortical_mask, dtype=bool)
    bone = np.asarray(bone_mask, dtype=bool)
    if cort.ndim != 3 or bone.shape != cort.shape:
        raise ValueError("Bone and cortical masks must be aligned 3D arrays")
    output = np.zeros(cort.shape, dtype=bool)
    if not cort.any():
        return output
    # Match the compact mask-AIM context rather than letting unrelated image
    # padding decide which background component ranks first. Never crop Z.
    _, yy, xx = np.nonzero(cort)
    region = (slice(None), slice(max(0, int(yy.min()) - 1), min(cort.shape[1], int(yy.max()) + 2)),
              slice(max(0, int(xx.min()) - 1), min(cort.shape[2], int(xx.max()) + 2)))
    compartment = cort[region]
    material = bone[region] & compartment
    seeds = _slice_pores(material) & compartment
    # The script explicitly removes the largest 3D background component before
    # Z growth (pores_M). It can represent marrow or extra-osseal background.
    labels, count = ndimage.label(~material, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    excluded = labels == int(sizes.argmax())
    candidates = compartment & ~material & ~excluded
    axial = np.zeros((3, 3, 3), dtype=bool)
    axial[:, 1, 1] = True
    grown = ndimage.binary_propagation(seeds & candidates, structure=axial, mask=candidates)
    residual = _slice_pores(material | grown) & compartment
    combined = (grown | residual) & ~material
    labels, count = ndimage.label(combined, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= 5
    keep[0] = False
    output[region] = keep[labels]
    return output


def _slice_pores(material: np.ndarray) -> np.ndarray:
    """Enclosed XY background components no larger than 5% of background area."""
    result = np.zeros(material.shape, dtype=bool)
    for z, image in enumerate(material):
        background = ~image
        labels, count = ndimage.label(background, structure=np.ones((3, 3), dtype=bool))
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        keep = (sizes > 0) & (sizes <= 0.05 * np.count_nonzero(background))
        keep[0] = False
        edges = np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
        keep[np.unique(edges)] = False
        result[z] = keep[labels]
    return result

<p align="center">
  <img src="resources/bone-microarchitecture.png" alt="bone-microarchitecture icon" width="240">
</p>

# Bone Microarchitecture

Lightweight microarchitecture measurements from binary masks and optional calibrated grayscale arrays.

Author: Matthias Walle.

The array measurement API has no Slicer dependency and expects calibrated arrays
on a common grid. File-based batch workflows use SimpleITK (and aimio-py for AIM)
to load inputs and place cropped masks on the grayscale grid with nearest-neighbor
resampling using origin, spacing, and direction. This is grid reconciliation, not
registration; geometry-free arrays must already have the grayscale dimensions.
The grid policy participates in batch cache compatibility so older incompatible
maps are recomputed.

Batch discovery accepts canonical segmentation, periosteal, trabecular, and cortical roles from shared contour derivatives and their sidecars. See [CHANGELOG.md](CHANGELOG.md) for release notes.

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

## Parameter Definitions

- `Tb.BMD`: mean calibrated grayscale value inside the trabecular compartment.
- `Tb.BV/TV`: trabecular bone volume divided by trabecular total volume.
- `Tb.Th`: maximal-sphere local thickness of trabecular bone.
- `Tb.Sp`: maximal-sphere local thickness of non-bone space in the trabecular compartment.
- `Tb.N`: reciprocal of mean ridge-to-ridge spacing in the trabecular compartment (not mean reciprocal spacing).
- `Tb.1/N.SD`: standard deviation of ridge-to-ridge spacing.
- `Tb.BV`: trabecular bone volume.
- `Tb.TV`: trabecular compartment volume.
- `Ct.BMD`: mean calibrated grayscale value inside the cortical compartment.
- `Ct.Th`: maximal-sphere local thickness of the cortical compartment, including internal pores.
- `Ct.Po`: cortical pore volume divided by cortical total volume.
- `Ct.Po.V`: selected intracortical pore volume after connectivity and size cleanup.
- `Ct.Po.Dm`: maximal-sphere local diameter of the selected intracortical pores.
- `Ct.BV`: cortical bone volume.
- `Ct.TV`: cortical compartment volume.

## IPL-aligned measurement definitions

The standard method is `ipl-aligned-v1`, replacing the previous definitions.
It follows the supplied `IPL_UPAT_CALGARY_EVAL_XT2_NOREG.COM` evaluation recipe:

- Pore selection uses enclosed XY seeds, Z-only hysteresis growth, a second XY
  pass and removal of components smaller than five voxels. It does not classify
  every non-bone cortical voxel as a pore or close longitudinal canals at scan
  ends. The stages are described by [Burghardt et al., Bone 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2926164/).
- `Ct.Th` uses the supplied cortical compartment, not only its mineralized phase.
  `Ct.Po = selected pore volume / cortical compartment volume`, matching the
  supplied script's denominator. `Ct.Po` is therefore not necessarily `1 - Ct.BV/Ct.TV`.
- `Tb.Sp` and ridge spacing retain the full bone-phase context before restricting
  reporting to the trabecular compartment. `Tb.N` is inverse mean spacing.
- Hildebrand fitting excludes sphere centres whose radius reaches the image's
  physical boundary. Complete spheres may still contribute to end slices; no
  fixed slice strip is blanked. CPU, Metal and OpenCL use the same centre filter.
  The optional `edt` preview remains a distance-transform approximation.

Native maps are computed before common-region or Functional Bone restrictions.
The additional `Ct.Po.Mask` map preserves pore membership independently of
diameter fitting. Cropping a reporting region does not repeat pore cleanup or
introduce new anatomical walls. `restrict_measurements` in the batch module
applies this same reporting rule to array/scene results.

Results and CSV JSON sidecars record the scientific method separately from the
package version. Batch hashes invalidate pre-alignment maps and measurements;
legacy results must be recalculated, not mixed into a new comparison. Ratios
remain fractions, not percentages. For `Tb.N`, the CSV Mean is inverse mean
spacing; its other distribution columns describe the local inverse-spacing map.

**Exact IPL parity is not validated.** The pore implementation explicitly uses
8-connectivity in XY, 26-connectivity in 3D, a 5%-of-background-area XY cutoff,
and a compact XY crop with one-voxel margin. Native IPL percentage/rank and GOBJ
boundary conventions may differ. Ridge dominance and the existing half-voxel
radius correction remain approximations, not verified translations of IPL's
`assign_epsilon`. Validation requires identical input AIM masks and native IPL
pore/ridge/diameter maps, not comparison of final scalar tables alone.

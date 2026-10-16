# Changelog

## Unreleased

### Fixed

- Constrain sphere centres to the requested common/functional region before fitting, retaining native distance/ridge context and cortical pore selection.
- Follow documented IPL `suppress_boundary=2` centre eligibility without the extra whole-sphere image-edge rejection, consistently on CPU, Metal and OpenCL. Keep ridge and radius-correction epsilon settings unchanged.
- Store common-region and Functional Bone maps in their own analysis folders. Reuse only compatible final outputs, fingerprint input contents/geometry and detect missing or rewritten maps; do not cache distance fields.
- Record scientific method `ipl-aligned-region-v3`, invalidating earlier post-hoc native-map summaries and maps made with whole-sphere image-edge rejection.

## 0.3.1 - 2026-10-14

### Fixed

- Prefer actual OpenCL GPU devices and compile the kernel before automatic selection. Fall back to CPU with an explicit reason if bindings, GPU drivers or kernel compilation are unavailable; explicit GPU requests still fail clearly.
- Log the diameter backend and device before batch calculations and retain device provenance in CSV sidecars and native-map manifests, including reused maps.

## 0.3.0 - 2026-10-07

### Changed

- Replace cortical non-bone porosity with slice-seeded, longitudinally grown, five-voxel-cleaned intracortical pores; persist `Ct.Po.Mask` independently of pore diameter.
- Measure cortical compartment thickness rather than mineralized cortical thickness, and report Tb.N as reciprocal mean ridge spacing in both summaries and CSVs.
- Retain full-phase context for spacing/ridge extraction and suppress incomplete image-boundary spheres consistently across diameter backends.
- Restrict native maps to reporting regions without rerunning pore cleanup or thickness fitting. Record `ipl-aligned-v1` scientific provenance and invalidate pre-alignment caches.
- Document unvalidated native IPL operator conventions; these changes do not establish bitwise IPL equivalence.

## 0.2.5 - 2026-10-02

### Fixed

- Place imported/cropped compartment and common-region masks on the grayscale physical grid with nearest-neighbor resampling, including same-size images with different origins, spacings, or directions.
- Include the mask-grid policy in cache compatibility to invalidate maps calculated under the old equal-array assumption.

## 0.2.4 - 2026-10-02

### Fixed

- Recognize canonical `bone_segmentation`, `periosteal_mask`, `trabecular_mask`, and `cortical_mask` roles in shared contour batch inputs, including BoneContours sidecars.
- Record the current package version in batch derivative manifests.

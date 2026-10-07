# Changelog

## Unreleased

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

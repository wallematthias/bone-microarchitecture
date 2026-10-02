# Changelog

## 0.2.5 - 2026-10-02

### Fixed

- Place imported/cropped compartment and common-region masks on the grayscale physical grid with nearest-neighbor resampling, including same-size images with different origins, spacings, or directions.
- Include the mask-grid policy in cache compatibility to invalidate maps calculated under the old equal-array assumption.

## 0.2.4 - 2026-10-02

### Fixed

- Recognize canonical `bone_segmentation`, `periosteal_mask`, `trabecular_mask`, and `cortical_mask` roles in shared contour batch inputs, including BoneContours sidecars.
- Record the current package version in batch derivative manifests.

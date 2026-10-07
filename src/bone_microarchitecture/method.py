"""Scientific method identity, independent of packaging/backend versions."""

METHOD_ID = "ipl-aligned-v1"

METHOD_METADATA = {
    "measurement_method": METHOD_ID,
    "cortical_porosity_method": "slice-seeds-z-growth-five-voxel-cleanup",
    "cortical_thickness_input": "cortical-compartment",
    "trabecular_number_statistic": "inverse-mean-ridge-spacing",
    "boundary_policy": "full-phase-context-complete-image-spheres",
    "ipl_parity": "not-validated",
}

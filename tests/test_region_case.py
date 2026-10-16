"""File-level functional-bone calculation and cache contract."""

import json
import numpy as np
import pytest


def test_functional_case_saves_its_own_maps_and_invalidates_changed_regions(
    tmp_path, monkeypatch
):
    sitk = pytest.importorskip("SimpleITK")
    from bone_microarchitecture import case

    paths = {}
    full = np.ones((5, 5, 5), dtype=np.uint8)
    void = np.zeros_like(full)
    void[2, 2, 2] = 1
    common = full.copy()
    common[-1] = 0
    for name, array in dict(
        image=full * 100,
        bone=full,
        peri=full,
        trab=full,
        cort=full * 0,
        common=common,
        void=void,
    ).items():
        path = tmp_path / f"{name}.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(array), str(path))
        paths[name] = path
    kwargs = dict(
        image_path=paths["image"],
        bone_mask_path=paths["bone"],
        periosteal_mask_path=paths["peri"],
        trabecular_mask_path=paths["trab"],
        cortical_mask_path=paths["cort"],
        common_region_path=paths["common"],
        voidspace_mask_path=paths["void"],
        output_dir=tmp_path / "functional",
        prefix="sub-001_ses-001_voi-radiusleft",
        thickness_method="edt",
        thickness_backend="cpu",
    )
    output = case.run_microarchitecture_case(**kwargs)
    maps = list((output.parent / "maps").glob("*"))
    assert len(maps) == 9
    assert not (tmp_path / "maps").exists()
    meta = json.loads(output.with_suffix(".json").read_text())
    assert meta["analysis_region"] == "functional-common"
    mask_path = (
        output.parent
        / "sub-001_ses-001_voi-radiusleft_desc-functional-bone-analysis_mask.nii.gz"
    )
    assert (
        np.count_nonzero(sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))) == 99
    )
    original = case.compute_microarchitecture
    calls = []

    def counted(**kw):
        calls.append(kw)
        return original(**kw)

    monkeypatch.setattr(case, "compute_microarchitecture", counted)
    assert case.run_microarchitecture_case(**kwargs) == output
    assert calls == []
    void[2, 2, 3] = 1
    sitk.WriteImage(sitk.GetImageFromArray(void), str(paths["void"]))
    case.run_microarchitecture_case(**kwargs)
    assert len(calls) == 1
    assert not calls[-1]["analysis_mask"][2, 2, 3]
    assert (
        np.count_nonzero(sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))) == 98
    )
    calls.clear()
    maps[0].unlink()
    case.run_microarchitecture_case(**kwargs)
    assert len(calls) == 1
    assert maps[0].exists()
    calls.clear()
    case.run_microarchitecture_case(**kwargs, force=True)
    assert len(calls) == 1
    calls.clear()
    # Same set of files but different roles must not hit the prior cache.
    case.run_microarchitecture_case(
        **{
            **kwargs,
            "common_region_path": paths["void"],
            "voidspace_mask_path": paths["common"],
        }
    )
    assert len(calls) == 1

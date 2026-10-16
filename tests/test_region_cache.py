import json

import numpy as np
import pytest

from bone_microarchitecture.cache import (
    input_fingerprints,
    output_file_states,
    cache_is_current,
)


def test_cache_rejects_changes_to_detached_image_data(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    header = tmp_path / "mask.mhd"
    sitk.WriteImage(
        sitk.GetImageFromArray(np.zeros((3, 3, 3), dtype=np.uint8)), str(header)
    )
    output = tmp_path / "result.csv"
    output.write_text("completed")
    metadata = dict(
        input_fingerprints=input_fingerprints([header]),
        output_file_states=output_file_states([output]),
    )
    assert cache_is_current(metadata)
    payload = tmp_path / "mask.raw"
    payload.write_bytes(bytes([1]) * 27)
    assert not cache_is_current(metadata)


@pytest.mark.parametrize("change", ["missing", "modified", "invalid_metadata"])
def test_final_cache_rejects_incomplete_or_modified_outputs(tmp_path, change):
    image = tmp_path / "input.npy"
    np.save(image, np.ones((3, 3, 3)))
    output = tmp_path / "map.npy"
    np.save(output, np.ones((3, 3, 3)))
    metadata = dict(
        input_fingerprints=input_fingerprints([image]),
        output_file_states=output_file_states([output]),
    )
    assert cache_is_current(json.loads(json.dumps(metadata)))
    if change == "missing":
        output.unlink()
    elif change == "modified":
        np.save(output, np.zeros((3, 3, 3)))
    else:
        metadata = {}
    assert not cache_is_current(metadata)

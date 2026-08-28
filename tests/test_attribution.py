from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ormir_xct_attribution_is_only_in_thickness_module():
    matches = []
    for path in (ROOT / "src" / "bone_microarchitecture").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "ORMiR" in text or "ormir" in text:
            matches.append(path.name)

    assert matches == ["thickness.py"]

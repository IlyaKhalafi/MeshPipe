import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from meshpipe.accessories import AccessoryAnchor, AccessoryRig
from meshpipe.geometry import canonical_vertices


@pytest.mark.parametrize("name", ["Glasses", "Crown", "Brows"])
def test_accessories_follow_rigid_head_motion(name):
    reference = canonical_vertices().copy()
    rig = AccessoryRig(AccessoryAnchor(name=name))
    original = rig.deform(reference)
    rotation = Rotation.from_euler("xyz", [0.1, -0.3, 0.2]).as_matrix()
    moved = rig.deform(reference @ rotation.T + [2, 3, 1])
    np.testing.assert_allclose(moved, original @ rotation.T + [2, 3, 1], atol=1e-9)
    assert rig.triangles.min() >= 0 and rig.triangles.max() < len(original)


def test_anchor_roundtrip_and_validation(tmp_path):
    anchor = AccessoryAnchor(name="Crown", landmark=10, offset=(0, 0.3, 0))
    path = tmp_path / "anchor.json"
    anchor.save(path)
    restored = AccessoryAnchor.load(path)
    assert restored.landmark == 10
    np.testing.assert_allclose(restored.offset, anchor.offset)
    with pytest.raises(ValueError):
        AccessoryRig(AccessoryAnchor(scale=float("nan")))

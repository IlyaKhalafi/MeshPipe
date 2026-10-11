import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from meshpipe.geometry import canonical_vertices
from meshpipe.heatmap import DisplacementHeatmap, align_to_reference


def test_rigid_head_motion_removed():
    reference = canonical_vertices().copy()
    rotation = Rotation.from_euler("xyz", [0.3, -0.5, 0.1]).as_matrix()
    moving = reference @ rotation.T + [2, 1, -4]
    aligned = align_to_reference(moving, reference)
    np.testing.assert_allclose(aligned, reference, atol=1e-10)


def test_expression_displacement_and_color_bounds():
    reference = canonical_vertices().copy()
    heatmap = DisplacementHeatmap()
    heatmap.capture(reference)
    moving = reference.copy()
    moving[152, 1] -= 1
    colors = heatmap.colors(moving)
    assert heatmap.distances[152] > 0
    assert heatmap.distances[168] == pytest.approx(0, abs=1e-10)
    assert colors.shape == (468, 3)
    assert colors.min() >= 0 and colors.max() <= 1
    reference[:] = 0
    assert heatmap.reference.any()


def test_neutral_required():
    heatmap = DisplacementHeatmap()
    with pytest.raises(ValueError):
        heatmap.colors(canonical_vertices())
    with pytest.raises(ValueError):
        heatmap.capture(None)

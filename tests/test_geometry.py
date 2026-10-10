import numpy as np
import pytest

from meshpipe.geometry import canonical_vertices, normals, sample_colors, to_vertices, topology


def test_canonical_topology_cached_and_valid():
    vertices = canonical_vertices()
    faces = topology()
    assert vertices.shape == (468, 3)
    assert faces is topology()
    assert 700 < len(faces) < 950
    assert faces.min() >= 0
    assert faces.max() < 468
    assert not faces.flags.writeable
    assert np.all(np.linalg.norm(normals(vertices, faces), axis=1) <= 1.000001)


def test_normalization_center_and_eye_distance():
    vertices = to_vertices(canonical_vertices(), 4 / 3)
    np.testing.assert_allclose(vertices.mean(axis=0), 0, atol=1e-12)
    assert np.linalg.norm(vertices[33] - vertices[263]) == pytest.approx(2)


def test_color_sampling_clamps_outside_frame():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    points = np.array([[-1, -1, 0], [2, 2, 0]])
    np.testing.assert_allclose(sample_colors(image, points), image[[0, 1], [0, 1]] / 255)


def test_invalid_geometry():
    with pytest.raises(ValueError):
        to_vertices(np.zeros((2, 3)), 1)
    with pytest.raises(ValueError):
        to_vertices(canonical_vertices(), 0)

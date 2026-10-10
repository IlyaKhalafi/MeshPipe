from functools import lru_cache
from importlib.resources import files

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay

LANDMARK_COUNT = 468


@lru_cache(maxsize=1)
def canonical_vertices() -> NDArray:
    path = files("meshpipe").joinpath("assets/canonical.npy")
    with path.open("rb") as stream:
        vertices = np.load(stream, allow_pickle=False)
    vertices.setflags(write=False)
    return vertices


def inside_polygon(points: NDArray, polygon: NDArray) -> NDArray:
    inside = np.zeros(len(points), dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        crosses = (current[1] > points[:, 1]) != (previous[1] > points[:, 1])
        divisor = previous[1] - current[1]
        if abs(divisor) > 1e-12:
            boundary = (previous[0] - current[0]) * (points[:, 1] - current[1]) / divisor + current[
                0
            ]
            inside ^= crosses & (points[:, 0] < boundary)
        previous = current
    return inside


@lru_cache(maxsize=1)
def topology() -> NDArray:
    """Cache canonical-space Delaunay faces, excluding eyes and inner mouth."""
    planar = canonical_vertices()[:, :2]
    faces = Delaunay(planar).simplices
    centers = planar[faces].mean(axis=1)
    holes = (
        [33, 160, 158, 133, 153, 144],
        [362, 385, 387, 263, 373, 380],
        [78, 81, 13, 311, 308, 402, 14, 178],
    )
    keep = np.ones(len(faces), dtype=bool)
    for indices in holes:
        keep &= ~inside_polygon(centers, planar[indices])
    faces = np.ascontiguousarray(faces[keep], dtype=np.int32)
    faces.setflags(write=False)
    return faces


def to_vertices(landmarks: NDArray, aspect: float) -> NDArray:
    if landmarks.shape != (LANDMARK_COUNT, 3) or not np.isfinite(landmarks).all():
        raise ValueError("Expected 468 finite facial landmarks")
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError("Aspect ratio must be finite and positive")
    vertices = landmarks * np.array([aspect, -1.0, -aspect])
    vertices -= vertices.mean(axis=0)
    eye_distance = np.linalg.norm(vertices[33] - vertices[263])
    return vertices / max(eye_distance, 1e-6) * 2.0


def sample_colors(rgb: NDArray, landmarks: NDArray) -> NDArray:
    height, width = rgb.shape[:2]
    pixels = np.rint(landmarks[:, :2] * [width - 1, height - 1]).astype(np.int32)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    return rgb[pixels[:, 1], pixels[:, 0]].astype(np.float64) / 255.0


def normals(vertices: NDArray, faces: NDArray) -> NDArray:
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    result = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(result, faces[:, corner], face_normals)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result

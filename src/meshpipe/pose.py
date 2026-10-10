import numpy as np
from numpy.typing import NDArray


def head_rotation(vertices: NDArray) -> NDArray:
    horizontal = vertices[263] - vertices[33]
    horizontal = horizontal / max(np.linalg.norm(horizontal), 1e-9)
    vertical = vertices[10] - vertices[152]
    vertical -= horizontal * np.dot(vertical, horizontal)
    vertical = vertical / max(np.linalg.norm(vertical), 1e-9)
    forward = np.cross(horizontal, vertical)
    if np.linalg.norm(forward) < 0.5:
        return np.eye(3)
    return np.column_stack([horizontal, vertical, forward])


def head_angles(vertices: NDArray) -> tuple[float, float, float]:
    from scipy.spatial.transform import Rotation

    return tuple(
        float(angle) for angle in Rotation.from_matrix(head_rotation(vertices)).as_euler("xyz")
    )

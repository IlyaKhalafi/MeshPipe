import numpy as np

ANCHORS = [33, 133, 362, 263, 168, 6, 197, 195, 5]


def align_to_reference(vertices, reference):
    for points in (vertices, reference):
        if np.shape(points) != (468, 3) or not np.isfinite(points).all():
            raise ValueError("Heatmap requires 468 finite vertices")
    source, target = vertices[ANCHORS], reference[ANCHORS]
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    left, _, right = np.linalg.svd((source - source_center).T @ (target - target_center))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    rotation = left @ correction @ right
    return (vertices - source_center) @ rotation + target_center


class DisplacementHeatmap:
    def __init__(self):
        self.reference = None
        self.limit = 0.08
        self.distances = np.zeros(468)

    def capture(self, vertices):
        if vertices is None:
            raise ValueError("Track a face before capturing neutral geometry")
        align_to_reference(vertices, vertices)
        self.reference = vertices.copy()

    def colors(self, vertices):
        if self.reference is None:
            raise ValueError("Capture neutral geometry before opening Heatmap")
        if not np.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("Heatmap color limit must be positive and finite")
        aligned = align_to_reference(vertices, self.reference)
        scale = max(np.linalg.norm(self.reference[33] - self.reference[263]), 1e-6)
        self.distances = np.linalg.norm(aligned - self.reference, axis=1) / scale
        normalized = np.clip(self.distances / self.limit, 0, 1)
        stops = np.array([[0.08, 0.23, 0.30], [0.24, 0.86, 0.72], [1, 0.72, 0.25], [1, 0.29, 0.23]])
        return np.column_stack(
            [np.interp(normalized, [0, 0.3, 0.65, 1], stops[:, channel]) for channel in range(3)]
        )

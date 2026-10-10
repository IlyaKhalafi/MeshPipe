import numpy as np
import open3d as o3d
from open3d.visualization import rendering

from meshpipe.geometry import topology
from meshpipe.pipeline import Frame

BACKGROUND = [0.035, 0.065, 0.085, 1.0]
PALETTES = {"Mint": [0.25, 0.86, 0.69], "Amber": [1.0, 0.66, 0.27], "Cloud": [0.78, 0.86, 0.91]}


def material(color: list[float], shader: str = "defaultLit") -> rendering.MaterialRecord:
    record = rendering.MaterialRecord()
    record.shader = shader
    record.base_color = [*color, 1.0]
    record.base_roughness = 0.55
    record.base_metallic = 0.05
    record.line_width = 1.2
    record.point_size = 4.0
    return record


class FaceScene:
    """Shared scene presentation for desktop viewing and reproducible renders."""

    def __init__(self, scene: rendering.Open3DScene):
        self.scene = scene
        self.mode = "Surface"
        self.palette = "Mint"
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.triangles = o3d.utility.Vector3iVector(topology().copy())
        faces = topology()
        self.edges = np.unique(
            np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1),
            axis=0,
        )
        self.dynamic: set[str] = set()
        scene.set_background(BACKGROUND)
        scene.show_skybox(False)
        scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, [0.3, -0.4, -1])
        self._add_grid()
        self._add_avatar()
        self._show_avatar(False)

    def _add_grid(self) -> None:
        points, lines = [], []
        for offset in np.linspace(-4, 4, 17):
            start = len(points)
            points.extend(
                [[offset, -2.15, -4], [offset, -2.15, 4], [-4, -2.15, offset], [4, -2.15, offset]]
            )
            lines.extend([[start, start + 1], [start + 2, start + 3]])
        grid = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(lines)
        )
        self.scene.add_geometry("grid", grid, material([0.11, 0.19, 0.21], "unlitLine"))

    def _add_avatar(self) -> None:
        self.avatar_names = ["head", "eye-left", "eye-right", "cheek-left", "cheek-right"]
        self.scene.add_geometry(
            "head",
            o3d.geometry.TriangleMesh.create_sphere(1.4, 40).compute_vertex_normals(),
            material([1.0, 0.65, 0.14]),
        )
        for side in ["left", "right"]:
            self.scene.add_geometry(
                f"eye-{side}",
                o3d.geometry.TriangleMesh.create_sphere(1, 16).compute_vertex_normals(),
                material([0.06, 0.07, 0.07]),
            )
            self.scene.add_geometry(
                f"cheek-{side}",
                o3d.geometry.TriangleMesh.create_sphere(1, 12).compute_vertex_normals(),
                material([1.0, 0.32, 0.22]),
            )

    def _show_avatar(self, visible: bool) -> None:
        for name in self.avatar_names:
            self.scene.show_geometry(name, visible)

    def _replace(self, name: str, geometry, record: rendering.MaterialRecord) -> None:
        if name in self.dynamic:
            self.scene.remove_geometry(name)
        self.scene.add_geometry(
            name, geometry, record, add_downsampled_copy_for_fast_rendering=False
        )
        self.dynamic.add(name)

    def clear(self) -> None:
        for name in self.dynamic:
            self.scene.remove_geometry(name)
        self.dynamic.clear()
        self._show_avatar(False)

    def update(self, frame: Frame) -> None:
        if frame.vertices is None:
            self.clear()
            return
        avatar = self.mode == "Emoji"
        self._show_avatar(avatar)
        desired = "mouth" if avatar else "face"
        for name in self.dynamic - {desired}:
            self.scene.remove_geometry(name)
        self.dynamic.intersection_update({desired})
        if avatar:
            self._update_avatar(frame)
            return
        color = PALETTES[self.palette]
        if self.mode == "Wireframe":
            geometry = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(frame.vertices), o3d.utility.Vector2iVector(self.edges)
            )
            record = material(color, "unlitLine")
        elif self.mode == "Points":
            geometry = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame.vertices))
            geometry.paint_uniform_color(color)
            record = material([1, 1, 1], "defaultUnlit")
        else:
            self.mesh.vertices = o3d.utility.Vector3dVector(frame.vertices)
            self.mesh.vertex_normals = o3d.utility.Vector3dVector(frame.normals)
            colors = frame.colors if self.mode == "Source Color" else np.tile(color, (468, 1))
            self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            geometry, record = self.mesh, material([1, 1, 1])
        self._replace("face", geometry, record)

    def _update_avatar(self, frame: Frame) -> None:
        vertices = frame.vertices
        horizontal = vertices[263] - vertices[33]
        horizontal /= max(np.linalg.norm(horizontal), 1e-6)
        vertical = vertices[10] - vertices[152]
        vertical -= horizontal * np.dot(vertical, horizontal)
        vertical /= max(np.linalg.norm(vertical), 1e-6)
        forward = np.cross(horizontal, vertical)
        rotation = np.column_stack([horizontal, vertical, forward])
        root = np.eye(4)
        root[:3, :3] = rotation
        self.scene.set_geometry_transform("head", root)
        expressions = frame.expressions
        for side, sign in [("left", -1), ("right", 1)]:
            blink = np.clip(expressions.get(f"eyeBlink{side.title()}", 0), 0, 1)
            eye = np.eye(4)
            eye[:3, :3] = np.diag([0.15, max(0.025, 0.23 * (1 - blink)), 0.075])
            eye[:3, 3] = [sign * 0.48, 0.38, 1.26]
            self.scene.set_geometry_transform(f"eye-{side}", root @ eye)
            cheek = np.eye(4)
            cheek[:3, :3] *= 0.12
            cheek[:3, 3] = [sign * 0.82, -0.16, 1.11]
            self.scene.set_geometry_transform(f"cheek-{side}", root @ cheek)
        smile = np.clip(
            (expressions.get("mouthSmileLeft", 0) + expressions.get("mouthSmileRight", 0)) / 2, 0, 1
        )
        jaw = np.clip(expressions.get("jawOpen", 0), 0, 1)
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        horizontal = np.cos(angles) * (0.43 + smile * 0.16)
        vertical = -0.45 + smile * 0.24 * np.cos(angles) ** 2 + (0.03 + jaw * 0.25) * np.sin(angles)
        depth = np.sqrt(np.maximum(1.4**2 - horizontal**2 - vertical**2, 0)) + 0.035
        centers = np.column_stack([horizontal, vertical, depth])
        tube_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        offsets = np.column_stack([np.zeros(8), np.cos(tube_angles), np.sin(tube_angles)]) * 0.035
        mouth_vertices = (centers[:, None, :] + offsets).reshape(-1, 3) @ rotation.T
        faces = []
        for ring in range(48):
            for corner in range(8):
                first = ring * 8 + corner
                second = ring * 8 + (corner + 1) % 8
                third = ((ring + 1) % 48) * 8 + corner
                fourth = ((ring + 1) % 48) * 8 + (corner + 1) % 8
                faces.extend([[first, second, third], [second, fourth, third]])
        mouth = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mouth_vertices), o3d.utility.Vector3iVector(faces)
        )
        mouth.compute_vertex_normals()
        self._replace("mouth", mouth, material([0.10, 0.065, 0.035]))

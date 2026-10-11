from functools import lru_cache

import numpy as np
import open3d as o3d
from open3d.visualization import rendering

from meshpipe.geometry import to_vertices, topology
from meshpipe.pipeline import Frame
from meshpipe.pose import head_angles, head_rotation

BACKGROUND = [0.035, 0.065, 0.085, 1.0]
PALETTES = {"Mint": [0.25, 0.86, 0.69], "Amber": [1.0, 0.66, 0.27], "Cloud": [0.78, 0.86, 0.91]}


@lru_cache(maxsize=1)
def mouth_topology() -> np.ndarray:
    faces = []
    for ring in range(48):
        for corner in range(8):
            first = ring * 8 + corner
            second = ring * 8 + (corner + 1) % 8
            third = ((ring + 1) % 48) * 8 + corner
            fourth = ((ring + 1) % 48) * 8 + (corner + 1) % 8
            faces.extend([[first, second, third], [second, fourth, third]])
    return np.asarray(faces, dtype=np.int32)


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
        self.opacity = 1.0
        self.highlight: tuple[int, ...] = ()
        self.selected: int | None = None
        self.cartoon = None
        self.lit = True
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

    def set_lighting(self, name: str) -> None:
        self.lit = name != "Flat"
        profile = (
            rendering.Open3DScene.LightingProfile.MED_SHADOWS
            if name == "Dramatic"
            else rendering.Open3DScene.LightingProfile.NO_SHADOWS
        )
        self.scene.set_lighting(
            profile, [-0.8, -0.4, -1] if name == "Dramatic" else [0.3, -0.4, -1]
        )

    def _fade_color(self, color) -> list[float]:
        return (
            np.asarray(color) * self.opacity + np.array(BACKGROUND[:3]) * (1 - self.opacity)
        ).tolist()

    def update(self, frame: Frame) -> None:
        if frame.vertices is None:
            self.clear()
            return
        if not frame.source_available and self.mode in {"Textured", "Source Color"}:
            self.clear()
            return
        avatar = self.mode == "Emoji"
        self._show_avatar(avatar)
        desired = "mouth" if avatar else "face"
        keep = {desired, "highlight", "selected"}
        for name in self.dynamic - keep:
            self.scene.remove_geometry(name)
        self.dynamic.intersection_update(keep)
        if avatar:
            for name in self.avatar_names:
                color = (
                    [1.0, 0.65, 0.14]
                    if name == "head"
                    else ([0.06, 0.07, 0.07] if name.startswith("eye") else [1.0, 0.32, 0.22])
                )
                self.scene.modify_geometry_material(name, material(self._fade_color(color)))
            self._update_avatar(frame)
            self._remove_selection()
            return
        if self.mode == "Cartoon":
            self._update_cartoon(frame)
            self._remove_selection()
            return
        color = self._fade_color(PALETTES[self.palette])
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
            if self.mode == "Source Color":
                colors = colors * self.opacity + np.array(BACKGROUND[:3]) * (1 - self.opacity)
            if self.mode == "Textured":
                colors = np.ones((468, 3))
                points = frame.raw_landmarks if frame.raw_landmarks is not None else frame.landmarks
                self.mesh.triangle_uvs = o3d.utility.Vector2dVector(
                    (points[topology().ravel(), :2] * [1, -1] + [0, 1]).copy()
                )
            self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            geometry, record = (
                self.mesh,
                material([1, 1, 1], "defaultLit" if self.lit else "defaultUnlit"),
            )
            if self.mode == "Textured":
                texture = (
                    frame.rgb
                    if self.opacity == 1
                    else (
                        frame.rgb * self.opacity
                        + np.array(BACKGROUND[:3]) * 255 * (1 - self.opacity)
                    ).astype(np.uint8)
                )
                record.albedo_img = o3d.geometry.Image(np.ascontiguousarray(texture))
        self._replace("face", geometry, record)
        self._update_selection(frame)

    def _remove_selection(self) -> None:
        for name in ("highlight", "selected"):
            if name in self.dynamic:
                self.scene.remove_geometry(name)
                self.dynamic.remove(name)

    def _update_selection(self, frame: Frame) -> None:
        self._remove_selection()
        for name, indices, color, size in [
            ("highlight", self.highlight, [1, 0.67, 0.22], 6),
            ("selected", () if self.selected is None else (self.selected,), [1, 0.94, 0.72], 10),
        ]:
            if not indices:
                continue
            points = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(frame.vertices[list(indices)])
            )
            record = material(self._fade_color(color), "defaultUnlit")
            record.point_size = size
            self._replace(name, points, record)

    def _update_cartoon(self, frame: Frame) -> None:
        from meshpipe.cartoon import CartoonRig, builtin_template

        if self.cartoon is None:
            self.cartoon = CartoonRig(builtin_template())
        image = self.cartoon.render(
            frame.expressions, head_angles(frame.vertices)[2], self.opacity, rgba=True
        )
        aspect = image.shape[1] / image.shape[0]
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(
                [
                    [-1.8 * aspect, -1.8, 0],
                    [1.8 * aspect, -1.8, 0],
                    [1.8 * aspect, 1.8, 0],
                    [-1.8 * aspect, 1.8, 0],
                ]
            ),
            o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]),
        )
        mesh.triangle_uvs = o3d.utility.Vector2dVector(
            [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]]
        )
        mesh.compute_vertex_normals()
        record = material([1, 1, 1], "defaultLitTransparency")
        record.has_alpha = True
        record.base_reflectance = 0.0
        record.base_metallic = 0.0
        record.transmission = 0.0
        record.albedo_img = o3d.geometry.Image(np.ascontiguousarray(image))
        self._replace("face", mesh, record)

    def update_lab(self, result, aspect: float) -> None:
        self.clear()
        if result is None:
            return
        for index, (points, color) in enumerate(
            zip(
                (result.raw, result.kalman, result.ema),
                ([0.33, 0.86, 0.73], [0.96, 0.72, 0.29], [0.5, 0.67, 0.9]),
                strict=True,
            )
        ):
            vertices = to_vertices(points, aspect) * 0.62 + [3 * (index - 1), 0, 0]
            geometry = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(vertices), o3d.utility.Vector2iVector(self.edges)
            )
            self._replace(f"lab-{index}", geometry, material(color, "unlitLine"))

    def _update_avatar(self, frame: Frame) -> None:
        rotation = head_rotation(frame.vertices)
        root = np.eye(4)
        root[:3, :3] = rotation
        self.scene.set_geometry_transform("head", root)
        expressions = frame.expressions
        for side, sign in [("left", 1), ("right", -1)]:
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
        mouth = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mouth_vertices), o3d.utility.Vector3iVector(mouth_topology())
        )
        mouth.compute_vertex_normals()
        self._replace("mouth", mouth, material([0.10, 0.065, 0.035]))

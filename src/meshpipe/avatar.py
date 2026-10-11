import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


@dataclass
class AvatarPart:
    vertices: np.ndarray
    triangles: np.ndarray
    targets: dict[str, np.ndarray]
    color: list[float]
    uv: np.ndarray | None = None
    image: np.ndarray | None = None


class Avatar:
    def __init__(self, parts: list[AvatarPart]):
        if not parts:
            raise ValueError("GLB scene contains no triangle meshes")
        self.parts = parts
        points = np.concatenate([part.vertices for part in parts])
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        scale = 3 / max(float(np.ptp(points, axis=0).max()), 1e-6)
        for part in parts:
            part.vertices = (part.vertices - center) * scale
            part.targets = {name: delta * scale for name, delta in part.targets.items()}
        self.names = sorted({name for part in parts for name in part.targets})
        self.mapping = {name: {"channel": name, "strength": 1.0} for name in self.names}

    def deform(self, part: AvatarPart, scores: dict[str, float]):
        vertices = part.vertices.copy()
        for name, delta in part.targets.items():
            rule = self.mapping.get(name)
            if rule:
                value = scores.get(rule["channel"], 0) * rule["strength"]
                vertices += delta * np.clip(value, 0, 1)
        return vertices

    def set_mapping(self, mapping):
        if not isinstance(mapping, dict) or len(mapping) > 128:
            raise ValueError("Invalid avatar mapping")
        validated = {}
        for name, rule in mapping.items():
            if name not in self.names or not isinstance(rule, dict):
                raise ValueError("Preset references an unknown morph target")
            channel = rule.get("channel")
            strength = rule.get("strength", 1)
            if not isinstance(channel, str) or not 0 < len(channel) <= 128:
                raise ValueError("Invalid expression channel")
            if not isinstance(strength, (int, float)) or not np.isfinite(strength):
                raise ValueError("Mapping strength must be finite")
            if not 0 <= strength <= 3:
                raise ValueError("Mapping strength must be between 0 and 3")
            validated[name] = {"channel": channel, "strength": float(strength)}
        self.mapping = validated

    def save_preset(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": 1, "mapping": self.mapping}, indent=2), encoding="utf8"
        )

    def load_preset(self, path: Path):
        if path.stat().st_size > 65536:
            raise ValueError("Rig preset exceeds 64 KiB")
        try:
            data = json.loads(path.read_text(encoding="utf8"))
            if data["version"] != 1:
                raise ValueError("Unsupported rig preset")
            self.set_mapping(data["mapping"])
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid rig preset") from error


def load_glb(path: Path) -> Avatar:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("GLB exceeds 32 MiB")
    try:
        return _load_glb(path.read_bytes())
    except (KeyError, IndexError, TypeError, struct.error, UnicodeError) as error:
        raise ValueError("Invalid or unsupported GLB structure") from error


def _load_glb(data: bytes) -> Avatar:
    magic, version, length = struct.unpack_from("<4sII", data)
    if magic != b"glTF" or version != 2 or length != len(data):
        raise ValueError("Expected a complete glTF 2.0 binary file")
    chunks = []
    position = 12
    while position < length:
        size, kind = struct.unpack_from("<II", data, position)
        position += 8
        if size % 4 or position + size > length:
            raise ValueError("Invalid GLB chunk bounds")
        chunks.append((kind, data[position : position + size]))
        position += size
    if len(chunks) != 2 or [item[0] for item in chunks] != [0x4E4F534A, 0x004E4942]:
        raise ValueError("GLB requires one JSON chunk and one binary chunk")
    document = json.loads(chunks[0][1])
    binary = chunks[1][1]
    if document.get("asset", {}).get("version") != "2.0":
        raise ValueError("Unsupported glTF asset version")
    if document.get("extensionsRequired"):
        raise ValueError("Required GLB extensions and compression are unsupported")
    buffers = document["buffers"]
    if len(buffers) != 1 or "uri" in buffers[0]:
        raise ValueError("External GLB resources are unsupported")
    declared = buffers[0]["byteLength"]
    if not isinstance(declared, int) or not 0 <= len(binary) - declared <= 3:
        raise ValueError("Invalid binary buffer length")

    def view(index):
        if not isinstance(index, int) or index < 0:
            raise ValueError("Invalid buffer view index")
        record = document["bufferViews"][index]
        start, size = record.get("byteOffset", 0), record["byteLength"]
        if record.get("buffer", 0) != 0 or start < 0 or size < 0 or start + size > declared:
            raise ValueError("Buffer view exceeds embedded buffer")
        return record, memoryview(binary)[start : start + size]

    def accessor(index, expected):
        if not isinstance(index, int) or index < 0:
            raise ValueError("Invalid accessor index")
        record = document["accessors"][index]
        if "sparse" in record or record.get("normalized") or record["type"] != expected:
            raise ValueError("Sparse, normalized or mismatched accessors are unsupported")
        dtype = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}.get(record["componentType"])
        if dtype is None:
            raise ValueError("Unsupported accessor component type")
        count = record["count"]
        columns = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[expected]
        if not isinstance(count, int) or not 1 <= count <= 300000:
            raise ValueError("Accessor count outside supported bounds")
        descriptor, payload = view(record["bufferView"])
        size = np.dtype(dtype).itemsize
        stride = descriptor.get("byteStride", size * columns)
        offset = record.get("byteOffset", 0)
        if stride < size * columns or stride > 252 or stride % size or offset < 0:
            raise ValueError("Invalid accessor stride or offset")
        if offset + (count - 1) * stride + size * columns > len(payload):
            raise ValueError("Accessor exceeds buffer view")
        result = np.ndarray(
            (count, columns), dtype=dtype, buffer=payload, offset=offset, strides=(stride, size)
        ).copy()
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite GLB values")
        return result

    parts = []
    total_vertices = total_targets = 0
    active = set()

    def visit(index, parent, depth=0):
        nonlocal total_vertices, total_targets
        if not isinstance(index, int) or index < 0 or depth > 64 or index in active:
            raise ValueError("Invalid or cyclic GLB node hierarchy")
        active.add(index)
        node = document["nodes"][index]
        if "skin" in node:
            raise ValueError("Skinned avatars unsupported. Export a morph-target head mesh.")
        if "matrix" in node:
            transform = np.asarray(node["matrix"], dtype=float).reshape(4, 4).T
        else:
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(node.get("rotation", [0, 0, 0, 1])).as_matrix()
            transform[:3, :3] *= np.asarray(node.get("scale", [1, 1, 1]))
            transform[:3, 3] = node.get("translation", [0, 0, 0])
        transform = parent @ transform
        if not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("Invalid node transform")
        if "mesh" in node:
            mesh = document["meshes"][node["mesh"]]
            names = mesh.get("extras", {}).get("targetNames", [])
            for primitive in mesh["primitives"]:
                if len(parts) >= 32 or primitive.get("mode", 4) != 4 or primitive.get("extensions"):
                    raise ValueError("Only up to 32 uncompressed triangle primitives supported")
                points = accessor(primitive["attributes"]["POSITION"], "VEC3").astype(float)
                if len(points) > 50000:
                    raise ValueError("Avatar primitive exceeds 50000 vertices")
                triangles = (
                    accessor(primitive["indices"], "SCALAR").ravel()
                    if "indices" in primitive
                    else np.arange(len(points))
                )
                if (
                    len(triangles) % 3
                    or triangles.dtype.kind not in "ui"
                    or triangles.max() >= len(points)
                ):
                    raise ValueError("Invalid triangle indices")
                triangles = triangles.reshape(-1, 3).astype(np.int32)
                if np.linalg.det(transform[:3, :3]) < 0:
                    triangles = triangles[:, ::-1].copy()
                targets = {}
                descriptors = primitive.get("targets", [])
                if len(descriptors) > 64 or descriptors and len(names) != len(descriptors):
                    raise ValueError(
                        "Morph targets require extras.targetNames and at most 64 targets"
                    )
                for name, descriptor in zip(names, descriptors, strict=True):
                    if not isinstance(name, str) or not 0 < len(name) <= 128 or name in targets:
                        raise ValueError("Invalid or duplicate morph target name")
                    delta = accessor(descriptor["POSITION"], "VEC3").astype(float)
                    if delta.shape != points.shape:
                        raise ValueError("Morph target vertex count mismatch")
                    targets[name] = delta @ transform[:3, :3].T
                total_vertices += len(points)
                total_targets += len(points) * len(targets) * 3 * 8
                if total_vertices > 100000 or total_targets > 64 * 1024 * 1024:
                    raise ValueError("Avatar exceeds total geometry budget")
                color, uv, image = [0.33, 0.86, 0.73], None, None
                if "material" in primitive:
                    material = document["materials"][primitive["material"]]
                    if material.get("alphaMode", "OPAQUE") != "OPAQUE":
                        raise ValueError("Avatar materials must be opaque")
                    pbr = material.get("pbrMetallicRoughness", {})
                    color = pbr.get("baseColorFactor", [*color, 1])[:3]
                    if len(color) != 3 or not np.isfinite(color).all():
                        raise ValueError("Invalid material color")
                    if "baseColorTexture" in pbr:
                        texture = pbr["baseColorTexture"]
                        if texture.get("texCoord", 0) != 0:
                            raise ValueError("Only TEXCOORD_0 is supported")
                        uv = accessor(primitive["attributes"]["TEXCOORD_0"], "VEC2")
                        if len(uv) != len(points):
                            raise ValueError("UV vertex count mismatch")
                        source = document["textures"][texture["index"]]["source"]
                        descriptor = document["images"][source]
                        _, payload = view(descriptor["bufferView"])
                        with Image.open(io.BytesIO(payload)) as bitmap:
                            if max(bitmap.size) > 2048:
                                raise ValueError("Avatar texture exceeds 2048 pixels")
                            image = np.asarray(bitmap.convert("RGB")).copy()
                parts.append(
                    AvatarPart(
                        points @ transform[:3, :3].T + transform[:3, 3],
                        triangles,
                        targets,
                        color,
                        uv,
                        image,
                    )
                )
        for child in node.get("children", []):
            visit(child, transform, depth + 1)
        active.remove(index)

    for root in document["scenes"][document.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4))
    return Avatar(parts)

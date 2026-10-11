import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from meshpipe.geometry import normals, to_vertices, topology
from meshpipe.session import SessionReader


@dataclass(frozen=True)
class MotionEdit:
    start_ms: float = 0
    end_ms: float | None = None
    strength: float = 1
    fade_ms: float = 0
    despike: bool = False
    blend_weight: float = 0
    curves: dict[str, list[list[float]]] = field(default_factory=dict)

    def validate(self, duration: float) -> None:
        if not isinstance(self.curves, dict) or not isinstance(self.despike, bool):
            raise ValueError("Invalid motion curves or spike removal flag")
        end = duration if self.end_ms is None else self.end_ms
        values = [self.start_ms, end, self.strength, self.fade_ms, self.blend_weight]
        if not np.isfinite(values).all():
            raise ValueError("Edit values must be finite")
        if not 0 <= self.start_ms <= end <= duration:
            raise ValueError("Trim must lie within the performance")
        if not 0 <= self.strength <= 3 or not 0 <= self.fade_ms <= 60000:
            raise ValueError("Invalid strength or fade duration")
        if not 0 <= self.blend_weight <= 1 or len(self.curves) > 64:
            raise ValueError("Invalid blend weight or curve count")
        for name, points in self.curves.items():
            array = np.asarray(points, dtype=float)
            if not isinstance(name, str) or len(name) > 128:
                raise ValueError("Invalid expression channel")
            if array.ndim != 2 or array.shape[1] != 2 or not 2 <= len(array) <= 256:
                raise ValueError("Curves require 2 to 256 time and gain pairs")
            if not np.isfinite(array).all() or (np.diff(array[:, 0]) <= 0).any():
                raise ValueError("Curve times must be finite and strictly increasing")
            if (array[:, 0] < 0).any() or (array[:, 0] > duration).any():
                raise ValueError("Curve times outside performance")
            if (array[:, 1] < 0).any() or (array[:, 1] > 3).any():
                raise ValueError("Curve gains must be between 0 and 3")

    def save(self, path: Path, duration: float) -> None:
        self.validate(duration)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "edit": asdict(self)}, indent=2), encoding="utf8")

    @classmethod
    def load(cls, path: Path, duration: float):
        if path.stat().st_size > 262144:
            raise ValueError("Edit document exceeds 256 KiB")
        try:
            data = json.loads(path.read_text(encoding="utf8"))
            if data["version"] != 1:
                raise ValueError("Unsupported edit version")
            edit = cls(**data["edit"])
            edit.validate(duration)
            return edit
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("Invalid motion edit document") from error


class EditedReader:
    """Lazy non-destructive session view with a bounded neighboring-frame cache."""

    def __init__(self, base: SessionReader):
        self.base = base
        self.blend: SessionReader | None = None
        self.cache = {}
        self.undo_stack = []
        self.redo_stack = []
        self.edit = MotionEdit()
        self.set_edit(self.edit, remember=False)

    @property
    def has_images(self):
        return self.base.has_images

    @property
    def source_kind(self):
        return self.base.source_kind

    @property
    def duration_ms(self):
        return int(self.timestamps[-1] - self.timestamps[0])

    def set_edit(self, edit: MotionEdit, remember: bool = True) -> None:
        edit.validate(self.base.duration_ms)
        end = self.base.duration_ms if edit.end_ms is None else edit.end_ms
        relative = self.base.timestamps - self.base.timestamps[0]
        indices = np.flatnonzero((relative >= edit.start_ms) & (relative <= end))
        if len(indices) == 0:
            raise ValueError("Trim contains no recorded frames")
        if remember:
            self.undo_stack.append(self.edit)
            self.undo_stack = self.undo_stack[-50:]
            self.redo_stack.clear()
        self.edit = edit
        self.indices = indices
        self.timestamps = self.base.timestamps[indices]

    def undo(self, redo: bool = False) -> None:
        source, target = (
            (self.redo_stack, self.undo_stack) if redo else (self.undo_stack, self.redo_stack)
        )
        if source:
            target.append(self.edit)
            self.set_edit(source.pop(), remember=False)

    def set_blend(self, path: Path) -> None:
        reader = SessionReader(path)
        if self.blend:
            self.blend.close()
        self.blend = reader

    def _base_frame(self, index: int):
        if index not in self.cache:
            self.cache[index] = self.base.frame(index)
        if len(self.cache) > 6:
            self.cache.pop(next(iter(self.cache)))
        return self.cache[index]

    def frame(self, index: int):
        if not 0 <= index < len(self.indices):
            raise ValueError("Replay frame index out of range")
        original = int(self.indices[index])
        frame = self._base_frame(original)
        scores = frame.expressions.copy()
        landmarks = frame.landmarks
        if self.edit.despike and 0 < original < len(self.base.timestamps) - 1:
            neighbors = [self._base_frame(original - 1), frame, self._base_frame(original + 1)]
            scores = {
                name: float(np.median([item.expressions.get(name, 0) for item in neighbors]))
                for name in scores
            }
            if all(item.landmarks is not None for item in neighbors):
                landmarks = np.median([item.landmarks for item in neighbors], axis=0)
        if self.blend is not None and self.edit.blend_weight:
            fraction = (frame.timestamp_ms - self.timestamps[0]) / max(1, self.duration_ms)
            target = self.blend.timestamps[0] + fraction * self.blend.duration_ms
            other_index = int(
                np.clip(
                    np.searchsorted(self.blend.timestamps, target),
                    0,
                    len(self.blend.timestamps) - 1,
                )
            )
            other = self.blend.frame(other_index)
            weight = self.edit.blend_weight
            scores = {
                name: scores.get(name, 0) * (1 - weight) + other.expressions.get(name, 0) * weight
                for name in scores.keys() | other.expressions.keys()
            }
            if landmarks is not None and other.landmarks is not None:
                landmarks = landmarks * (1 - weight) + other.landmarks * weight
        elapsed = frame.timestamp_ms - self.timestamps[0]
        fade = (
            min(1, elapsed / self.edit.fade_ms, (self.duration_ms - elapsed) / self.edit.fade_ms)
            if self.edit.fade_ms
            else 1
        )
        source_time = frame.timestamp_ms - self.base.timestamps[0]
        for name, score in scores.items():
            curve = self.edit.curves.get(name)
            gain = np.interp(source_time, *np.asarray(curve).T) if curve else 1
            scores[name] = float(np.clip(score * self.edit.strength * fade * gain, 0, 1))
        vertices = vertex_normals = None
        if landmarks is not None:
            vertices = to_vertices(landmarks, frame.rgb.shape[1] / frame.rgb.shape[0])
            vertex_normals = normals(vertices, topology())
        return replace(
            frame,
            expressions=scores,
            landmarks=landmarks,
            vertices=vertices,
            normals=vertex_normals,
        )

    def close(self):
        self.base.close()
        if self.blend:
            self.blend.close()

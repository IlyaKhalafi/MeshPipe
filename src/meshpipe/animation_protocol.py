import json
import math
from pathlib import Path


def reduce_keys(times, values, tolerance=0.002):
    if len(times) != len(values) or not times:
        raise ValueError("Keyframe times and values must have equal nonzero length")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Key reduction tolerance must be finite and nonnegative")
    if not all(math.isfinite(value) for value in [*times, *values]):
        raise ValueError("Keyframes must be finite")
    if any(right <= left for left, right in zip(times, times[1:], strict=False)):
        raise ValueError("Keyframe times must increase")
    keep = {0, len(times) - 1}
    pending = [(0, len(times) - 1)]
    while pending:
        start, end = pending.pop()
        if end - start < 2:
            continue
        span = times[end] - times[start]
        index, error = max(
            (
                (
                    index,
                    abs(
                        values[index]
                        - values[start]
                        - (values[end] - values[start]) * (times[index] - times[start]) / span
                    ),
                )
                for index in range(start + 1, end)
            ),
            key=lambda item: item[1],
        )
        if error > tolerance:
            keep.add(index)
            pending.extend([(start, index), (index, end)])
    return [[float(times[index]), float(values[index])] for index in sorted(keep)]


def validate_animation(data):
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Unsupported animation payload")
    fps = data.get("fps")
    if not isinstance(fps, (float, int)) or not math.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError("Animation FPS outside 1 to 120")
    channels = data.get("channels")
    if not isinstance(channels, dict) or not 1 <= len(channels) <= 64:
        raise ValueError("Animation requires 1 to 64 expression channels")
    tracks = list(channels.items())
    rotation = data.get("rotation", {})
    if not isinstance(rotation, dict) or set(rotation) - {"x", "y", "z"}:
        raise ValueError("Invalid rotation tracks")
    tracks.extend(rotation.items())
    count = 0
    for name, keys in tracks:
        if not isinstance(name, str) or not 0 < len(name) <= 128:
            raise ValueError("Invalid animation channel")
        if not isinstance(keys, list) or not 1 <= len(keys) <= 36000:
            raise ValueError("Invalid animation key count")
        previous = -1
        for pair in keys:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Expected frame and value pairs")
            frame, value = pair
            if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in pair):
                raise ValueError("Animation keys must be finite")
            if not previous < frame <= 1000000 or abs(value) > 100:
                raise ValueError("Invalid animation frame or value")
            previous = frame
        count += len(keys)
    if count > 250000:
        raise ValueError("Animation exceeds 250000 keys")
    return data


def load_animation(path):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Animation payload exceeds 8 MiB")
    return validate_animation(json.loads(path.read_text(encoding="utf8")))

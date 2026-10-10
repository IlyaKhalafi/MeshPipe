"""Render genuine face video through the mesh and expression-avatar pipelines."""

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import cv2
import numpy as np
from open3d.visualization import rendering
from PIL import Image, ImageDraw, ImageFont

from meshpipe.capture import VideoSource
from meshpipe.config import Config
from meshpipe.pipeline import Frame, Processor
from meshpipe.scene import FaceScene

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 960, 560


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def compose(frame: Frame, rendered: Image.Image, emoji: bool) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#091016")
    draw = ImageDraw.Draw(image)
    accent = "#f5b94a" if emoji else "#55ddbb"
    draw.text((28, 18), "MESHPIPE", font=font(16, True), fill=accent)
    draw.text((158, 20), "FACE MESH STUDIO", font=font(11), fill="#94a4ac")
    draw.text(
        (28, 51),
        "Expressions, reimagined." if emoji else "Your face. Another dimension.",
        font=font(30, True),
        fill="#e7f0f2",
    )
    draw.text(
        (28, 100),
        "02 / EMOJI AVATAR" if emoji else "01 / FACE RECONSTRUCTION",
        font=font(12, True),
        fill=accent,
    )
    draw.text(
        (575, 100),
        "REAL EXPRESSION TRACKING" if emoji else "REAL MEDIAPIPE INFERENCE",
        font=font(11),
        fill="#94a4ac",
    )
    draw.rounded_rectangle((24, 134, 330, 510), radius=12, fill="#132129")
    preview = frame.rgb.copy()
    if frame.landmarks is not None:
        pixels = frame.landmarks[:, :2] * [preview.shape[1], preview.shape[0]]
        for point in pixels:
            cv2.circle(preview, tuple(np.rint(point).astype(int)), 1, (85, 221, 187), -1)
        minimum, maximum = pixels.min(axis=0), pixels.max(axis=0)
        padding = (maximum - minimum) * [0.5, 0.2]
        minimum = np.maximum(np.floor(minimum - padding).astype(int), 0)
        maximum = np.minimum(
            np.ceil(maximum + padding).astype(int), [preview.shape[1], preview.shape[0]]
        )
        preview = preview[minimum[1] : maximum[1], minimum[0] : maximum[0]]
    preview_image = Image.fromarray(preview)
    scale = min(282 / preview_image.width, 270 / preview_image.height)
    preview_image = preview_image.resize(
        (round(preview_image.width * scale), round(preview_image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    image.paste(preview_image, (36 + (282 - preview_image.width) // 2, 146))
    draw.text(
        (40, 432),
        "NASA INTERVIEW FOOTAGE",
        font=font(11, True),
        fill=accent,
    )
    if emoji:
        for row, (name, score) in enumerate(
            [
                ("Smile", frame.expressions.get("mouthSmileLeft", 0)),
                ("Jaw", frame.expressions.get("jawOpen", 0)),
                ("Blink", frame.expressions.get("eyeBlinkLeft", 0)),
            ]
        ):
            vertical = 454 + row * 15
            draw.text((40, vertical), name, font=font(10), fill="#c5d1d4")
            draw.rounded_rectangle((100, vertical + 3, 300, vertical + 8), radius=2, fill="#293d45")
            if score > 0:
                draw.rounded_rectangle(
                    (100, vertical + 3, 100 + max(2, round(score * 200)), vertical + 8),
                    radius=2,
                    fill=accent,
                )
    else:
        draw.text((40, 455), "468 tracked landmarks", font=font(13), fill="#e7f0f2")
        draw.text((40, 480), "Kalman smoothing enabled", font=font(11), fill="#94a4ac")
    image.paste(rendered, (346, 134))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((366, 150, 525, 176), radius=6, fill="#172930")
    draw.text(
        (378, 156),
        "EXPRESSION RETARGET" if emoji else "CACHED DELAUNAY",
        font=font(10, True),
        fill=accent,
    )
    draw.text(
        (28, 531),
        "CPU INFERENCE   /   KALMAN   /   SCIPY   /   OPEN3D",
        font=font(10),
        fill="#8c9fa8",
    )
    draw.text((761, 531), "LOCAL. PRIVATE. OPEN.", font=font(10, True), fill=accent)
    return image


def save_gif(frames: list[Image.Image], destination: Path, fps: float = 15) -> None:
    output_size = (800, 467)
    frames = [frame.resize(output_size, Image.Resampling.LANCZOS) for frame in frames]
    samples = Image.new("RGB", (output_size[0], output_size[1] * 6))
    for index in range(6):
        samples.paste(frames[index * (len(frames) - 1) // 5], (0, index * output_size[1]))
    palette = samples.quantize(colors=128, dither=Image.Dither.NONE)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    durations = [
        (round((index + 1) * 100 / fps) - round(index * 100 / fps)) * 10
        for index in range(len(frames))
    ]
    indexed[0].save(
        destination,
        save_all=True,
        append_images=indexed[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )


def tracked_frames(video: Path, model: Path, count: int, start: float, fps: float) -> list[Frame]:
    config = Config(source=str(video), model=model)
    processor = Processor(config)
    source = None
    frames = []
    try:
        source = VideoSource(str(video), config.width)
        if fps > source.fps + 0.01:
            raise ValueError("Showcase FPS cannot exceed the source frame rate")
        while len(frames) < count:
            captured = source.read()
            if captured is None:
                raise ValueError("Video ended before requested showcase duration")
            frame = processor.process(*captured)
            target_ms = start * 1000 + len(frames) * 1000 / fps
            if frame.timestamp_ms < target_ms:
                continue
            if frame.vertices is None:
                raise RuntimeError(f"No face detected at {frame.timestamp_ms} ms")
            frames.append(frame)
        return frames
    finally:
        if source is not None:
            source.close()
        processor.close()


def generate(
    video: Path, model: Path, output: Path, count: int = 60, start: float = 2, fps: float = 15
) -> None:
    tracked = tracked_frames(video, model, count, start, fps)
    output.mkdir(parents=True, exist_ok=True)
    renderer = rendering.OffscreenRenderer(590, 376)
    content = FaceScene(renderer.scene)
    for emoji, name in [(False, "face-mesh.gif"), (True, "emoji-avatar.gif")]:
        frames = []
        content.mode = "Emoji" if emoji else "Wireframe"
        for index, frame in enumerate(tracked):
            content.update(frame)
            angle = math.sin(index / count * 2 * math.pi) * 0.4
            renderer.setup_camera(
                42,
                [0, -0.05, 0],
                [6.8 * math.sin(angle), 0.1, 6.8 * math.cos(angle)],
                [0, 1, 0],
            )
            rendered = Image.fromarray(np.asarray(renderer.render_to_image()))
            frames.append(compose(frame, rendered, emoji))
        destination = output / name
        save_gif(frames, destination, fps)
        print(f"Saved {destination} ({len(frames)} real video frames)")
    del content
    del renderer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "media")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--start", type=float, default=2)
    parser.add_argument("--fps", type=float, default=15)
    options = parser.parse_args()
    if options.frames < 2:
        parser.error("--frames must be at least 2")
    if not math.isfinite(options.start) or options.start < 0:
        parser.error("--start must be finite and nonnegative")
    if not 1 <= options.fps <= 60:
        parser.error("--fps must be between 1 and 60")
    generate(
        options.video, options.model, options.output, options.frames, options.start, options.fps
    )


if __name__ == "__main__":
    main()

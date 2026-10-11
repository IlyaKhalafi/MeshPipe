import argparse
import json
import platform
import sys
import time
from pathlib import Path

from meshpipe import __version__
from meshpipe.config import Config
from meshpipe.model import default_model_path, download_model


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="MeshPipe CPU facial mesh studio")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command")
    for name, help_text in [
        ("run", "Open interactive studio"),
        ("benchmark", "Measure headless CPU pipeline"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "source", nargs="?", default="0", help="Webcam index or local video path"
        )
        command.add_argument(
            "--demo", action="store_true", help="Use synthetic landmarks without a camera or model"
        )
        command.add_argument("--model", type=Path, help="Path to official face_landmarker.task")
        command.add_argument(
            "--width", type=int, default=640, help="Maximum processing width (default 640)"
        )
        command.add_argument(
            "--max-fps", type=float, default=30, help="Studio rate cap (default 30)"
        )
        command.add_argument("--confidence", type=float, default=0.5)
        command.add_argument("--no-smoothing", action="store_true")
        command.add_argument("--loop", action="store_true", help="Loop video input")
        if name == "run":
            command.add_argument("--adaptive-cpu", action="store_true")
            command.add_argument("--budget-ms", type=float, default=33)
            command.add_argument(
                "--landmark-only", action="store_true", help="Record motion without source images"
            )
            command.add_argument(
                "--replay",
                type=Path,
                help="Open a recorded .meshpipe performance without inference",
            )
            command.add_argument("--cartoon", type=Path, help="Open PNG anchor editor at startup")
            command.add_argument(
                "--osc-port", type=int, help="Enable local OSC expression streaming"
            )
        if name == "benchmark":
            command.add_argument("--frames", type=int, default=300)
            command.add_argument("--warmup", type=int, default=10)
            command.add_argument("--output", type=Path, help="Save benchmark JSON")
    download = commands.add_parser("download-model", help="Download official model to local cache")
    download.add_argument("--output", type=Path, default=default_model_path())
    return root


def benchmark(config: Config, frames: int, warmup: int) -> dict:
    import numpy as np

    from meshpipe.capture import VideoSource
    from meshpipe.demo import demo_frame
    from meshpipe.pipeline import Processor

    if frames < 1 or warmup < 0:
        raise ValueError("Frames must be positive and warmup must be nonnegative")
    processor = Processor(config)
    source = None
    durations, processing, detected = [], [], 0
    try:
        if not config.demo:
            source = VideoSource(config.source, config.width, config.loop)
        for index in range(frames + warmup):
            started = time.perf_counter()
            if config.demo:
                image, landmarks, expressions = demo_frame(index / 30, config.width)
                frame = processor.process(
                    image,
                    round(index * 1000 / 30),
                    demo_landmarks=landmarks,
                    demo_expressions=expressions,
                )
            else:
                captured = source.read()
                if captured is None:
                    break
                frame = processor.process(*captured)
            if index >= warmup:
                durations.append((time.perf_counter() - started) * 1000)
                processing.append(frame.processing_ms)
                detected += frame.vertices is not None
    finally:
        if source is not None:
            source.close()
        processor.close()
    if not durations:
        raise ValueError("No measured frames. Use a longer video or reduce --warmup")
    return {
        "version": __version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "source": "synthetic-demo" if config.demo else config.source,
        "width": config.width,
        "processed_size": [frame.rgb.shape[1], frame.rgb.shape[0]],
        "smoothing": config.smoothing,
        "frames": len(durations),
        "faces_detected": detected,
        "warmup_frames": warmup,
        "throughput_fps": round(1000 / float(np.mean(durations)), 2),
        "processing_mean_ms": round(float(np.mean(processing)), 3),
        "processing_p95_ms": round(float(np.percentile(processing, 95)), 3),
        "end_to_end_mean_ms": round(float(np.mean(durations)), 3),
        "scope": "Capture/decode + CPU processing, excludes GUI and rendering; no pacing",
    }


def main(arguments: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if not arguments or arguments[0] not in {
        "run",
        "benchmark",
        "download-model",
        "--help",
        "-h",
        "--version",
    }:
        arguments.insert(0, "run")
    options = parser().parse_args(arguments)
    try:
        if options.command == "download-model":
            print(f"Model saved to {download_model(options.output)}")
            return 0
        config = Config(
            source=options.source,
            model=options.model,
            width=options.width,
            max_fps=options.max_fps,
            detection_confidence=options.confidence,
            smoothing=not options.no_smoothing,
            demo=options.demo,
            loop=options.loop,
            replay=getattr(options, "replay", None),
            cartoon=getattr(options, "cartoon", None),
            osc_port=getattr(options, "osc_port", None),
            landmark_only=getattr(options, "landmark_only", False),
            adaptive_cpu=getattr(options, "adaptive_cpu", False),
            budget_ms=getattr(options, "budget_ms", 33),
        )
        if options.command == "benchmark":
            result = json.dumps(benchmark(config, options.frames, options.warmup), indent=2)
            if options.output:
                options.output.parent.mkdir(parents=True, exist_ok=True)
                options.output.write_text(result + "\n", encoding="utf-8")
            print(result)
        else:
            from meshpipe.app import run_studio

            run_studio(config)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"MeshPipe error: {error}", file=sys.stderr)
        return 1

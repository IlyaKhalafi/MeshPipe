"""Build static browser assets from genuine tracking on a local video."""

import argparse
import tempfile
from pathlib import Path

from meshpipe.capture import VideoSource
from meshpipe.config import Config
from meshpipe.pipeline import Processor
from meshpipe.session import SessionReader, SessionRecorder
from meshpipe.web_export import export_showcase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/demo"))
    parser.add_argument("--frames", type=int, default=180)
    options = parser.parse_args()
    if not 2 <= options.frames <= 900:
        parser.error("frames must be between 2 and 900")
    config = Config(source=str(options.source), model=options.model, width=384)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tracked.meshpipe"
        processor = Processor(config)
        source = None
        recorder = SessionRecorder(path, save_images=False, source_kind="tracked")
        try:
            source = VideoSource(config.source, config.width)
            for _ in range(options.frames):
                captured = source.read()
                if captured is None:
                    break
                recorder.submit(processor.process(*captured))
        finally:
            if source:
                source.close()
            processor.close()
            recorder.close()
        reader = SessionReader(path)
        try:
            export_showcase(reader, options.output, options.frames)
            print(f"Saved {len(reader.timestamps)} tracked frames to {options.output}")
        finally:
            reader.close()


if __name__ == "__main__":
    main()

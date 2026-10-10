"""Package the Blender receiver with the shared dependency-free OSC decoder."""

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / "integrations/blender/__init__.py", "meshpipe_receiver/__init__.py")
        archive.write(ROOT / "src/meshpipe/osc_protocol.py", "meshpipe_receiver/osc_protocol.py")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("exports/meshpipe-blender.zip"))
    print(build(parser.parse_args().output))


if __name__ == "__main__":
    main()

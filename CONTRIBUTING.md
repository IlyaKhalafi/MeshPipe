# Contributing

Useful contributions include reproducible CPU benchmarks, better avatar rigs,
camera backend fixes, accessible studio controls and small focused bug fixes.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests tools integrations
python -m ruff format --check src tests tools integrations
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell.

Tests use synthetic data and generated temporary videos. They never request
camera access, open a window, or download a model.
OSC transport tests open local loopback UDP sockets. Restricted sandboxes must
allow those sockets to run the full suite.

## Changes worth merging

- Keep capture, inference and rendering separate.
- Keep GUI access on the main thread.
- Keep webcam buffers bounded and video timestamps monotonic.
- Add a regression test for behavior changes.
- Record hardware, input dimensions and measurement scope with performance claims.
- Do not include private face recordings, model binaries or generated exports.

## Desktop validation

Desktop validation requires a working display and OpenGL support.

```bash
python tools/smoke_studio.py
python tools/smoke_studio.py --video --source sample.mp4 --model face_landmarker.task
python tools/smoke_workbench.py
python tools/smoke_workbench.py --source sample.mp4 --model face_landmarker.task
```

The smoke tool exercises every view, orbit, pause and export. It saves a
screenshot to the system temporary directory. Use `--output` to choose a path.
Actual mouse interactions and webcam hardware still need manual testing.

Workbench smoke exercises session recording and replay, GIF and MP4 encoding,
calibration, cartoon loading and anchor persistence, OSC streaming, the lab,
landmark export and expression targets. Its temporary recordings are removed
after validation. Use genuine video input when generating public screenshots.
The Blender archive test validates syntax, shared codec content and shape-key
preservation with a stubbed API. It does not replace testing inside Blender.

## Pull requests

Describe the problem, the behavior change and the checks performed. Keep each
pull request focused on one feature or fix. For camera bugs, include OS, Python,
OpenCV version and whether the same device works outside MeshPipe.

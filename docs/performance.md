# Performance

## Initial studio baseline

Measured on October 10, 2026 with an Intel Core i7-9750H, Linux x86_64 and Python
3.11.15. MediaPipe 0.10.21 uses the explicitly selected CPU delegate.
The input is genuine NASA interview footage at 1280 by 720, resized to 640 by
360 before CPU detection. It is not an animated photograph.

| Measurement | Result |
| --- | --- |
| Input | Recorded interview with natural face motion |
| Warmup frames excluded | 30 |
| Measured frames | 300 |
| Frames with detected face | 300 |
| Headless throughput | 48.07 FPS |
| Mean CPU processing time | 19.071 ms |
| 95th percentile CPU processing time | 26.967 ms |
| Mean decoding and processing time | 20.803 ms |
| Kalman smoothing | Enabled |

Raw benchmark results are saved in `docs/benchmarks/cpu-video.json`.
This is one local run, not a cross-machine performance guarantee. Results will
vary with power limits, camera drivers, codecs, lighting and input dimensions.
No claim is made about webcam FPS or rendering speed.

## Workbench release validation

Version 0.2.0 was measured on October 11, 2026 on the same Intel Core i7-9750H,
Linux and Python environment, using the same genuine interview input and
640 by 360 processing dimensions. CPU inference and Kalman smoothing remained
enabled. GUI tools and rendering are outside this headless measurement.

| Measurement | Result |
| --- | --- |
| Warmup frames excluded | 30 |
| Measured frames | 300 |
| Frames with detected face | 300 |
| Headless throughput | 45.44 FPS |
| Mean CPU processing time | 20.283 ms |
| 95th percentile CPU processing time | 50.719 ms |
| Mean decoding and processing time | 22.005 ms |

Raw results are in `docs/benchmarks/workbench-cpu-video.json`. These are single
runs, not a controlled comparison or evidence of a measured optimization.
Scheduling, thermals and background workloads can affect both throughput and
tail latency. The README reports the newer release measurement.

Native Linux validation exercised all seven views, orbit, pause, exports,
genuine-video recording and replay, GIF and MP4 encoding, calibration, cartoon
anchors, local OSC, smoothing comparisons, landmark history and expression
targets. A separate smoke run started directly from an inference-free session.
Physical webcam operation and native Blender rig behavior remain field-test
requirements. Unit tests and mock Blender API tests do not cover those devices.

## Benchmark scope

`meshpipe benchmark` constructs the model before measuring and excludes the
requested warmup frames. Throughput includes capture or video decoding,
RGB conversion, CPU detection, expression output, filtering, vertex conversion,
color sampling and normals. It excludes the GUI and rendering.

With a webcam, capture waiting contributes to throughput. With a file, the
decoder runs as fast as the processing loop permits. Neither result is a
measurement of motion-to-display latency. Mean and 95th percentile processing
times exclude capture but include detection and all geometry updates.

```bash
meshpipe benchmark interview.ogv --model face_landmarker.task --frames 300 --warmup 30
meshpipe benchmark interview.ogv --width 320 --frames 300 --warmup 30
meshpipe benchmark --demo --frames 300 --warmup 10
```

The demo omits MediaPipe inference. Never present its throughput as face
detection performance. JSON reports the configured width cap and actual
processed dimensions separately.

## Tuning

Start with a width cap of 640 and a 30 FPS studio cap. For older CPUs, compare
480 and 320 widths on the same input. Use the lowest width that still tracks
your face and expressions reliably. Downscaling cannot repair poor lighting
or severe motion blur.

Webcam capture overwrites its single pending frame, preventing processing
backlog. The result mailbox also holds only one frame. Video decoding remains
sequential for reproducibility and timestamp correctness, so slow processing
makes playback slower rather than dropping frames.

Cached triangulation is independent of inference width. Geometry cost remains
bounded by 468 vertices. Rendering reconstructs tiny Open3D geometry uploads
without rebuilding SciPy topology or resetting the view camera.

## Reproducibility

Record CPU model, OS, Python, dependency versions, video codec, actual processed
dimensions, smoothing setting and warmup count with any published result.
Use multiple runs and compare medians before drawing conclusions from small
differences. The test suite deliberately has no hardware-specific timing gates.

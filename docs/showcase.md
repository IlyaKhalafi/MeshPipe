# Reproducing the showcases

Both README GIFs use genuine recorded NASA interview footage of Antti Pulkkinen.
The subject speaks, blinks and moves naturally. No still photograph is animated,
and no detected landmarks or expression scores are pre-baked.

The surrounding presentation is composed for documentation rather than captured
from the desktop UI. It shares the studio's `FaceScene` geometry, materials and
avatar code. The source video is credited in `THIRD_PARTY.md`.

## Processing

The generator sequentially decodes the video and runs the actual MediaPipe CPU
detector on every frame, followed by Kalman filtering and geometry updates.
It starts its four-second excerpt at two seconds and samples 60 output frames
at 15 FPS. Decoder timestamps drive both inference and sampling.

The mesh GIF renders the tracked landmarks with cached SciPy triangulation.
The emoji GIF uses the same tracked frames and actual smile, jaw-open and blink
scores. Neither GIF uses the application's synthetic demo mode.

Only the input preview is cropped around the face for readability. Detection
uses the complete resized video frame. The viewer camera orbits independently
to show depth. Audio is omitted because GIF cannot contain sound.

## Commands

Install development dependencies and download the model first. Obtain the
credited `Interview_with_Antti_Pulkkinen.ogv` file from `THIRD_PARTY.md`.

```bash
meshpipe download-model --output face_landmarker.task
python tools/generate_showcase.py --video Interview_with_Antti_Pulkkinen.ogv --model face_landmarker.task --start 2 --frames 60
meshpipe Interview_with_Antti_Pulkkinen.ogv --model face_landmarker.task --loop
```

The generator writes two 800 by 467 GIFs with a shared palette per animation
and delta-frame compression. Open3D offscreen rendering requires EGL on Linux,
even though detection runs on CPU. For Mesa software rendering, set
`LIBGL_ALWAYS_SOFTWARE=1`. Other platforms may need their native graphics
configuration instead of EGL.

To validate the desktop rather than the documentation presentation, run
`python tools/smoke_studio.py --video --source video.ogv --model face_landmarker.task`
on a working display.

`meshpipe --demo` remains available for trying the studio without camera access
or a model download. It is synthetic and is not used for these showcases or
face-detection benchmarks.

## Workbench screenshot

`docs/media/workbench.png` captures the native 1480 by 900 studio window running
the same genuine NASA interview. Textured mode maps source pixels onto filtered
geometry while Inspect shows landmark traces and previous lab measurements.
The screenshot is not a mockup or a synthetic face input.

```bash
python tools/smoke_workbench.py --source Interview_with_Antti_Pulkkinen.ogv --model face_landmarker.task --output docs/media/workbench.png
```

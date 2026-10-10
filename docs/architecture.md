# Architecture

## Data flow

```text
Webcam newest-frame mailbox or sequential video decoder
                       |
                RGB frame + timestamp
                       |
       MediaPipe Face Landmarker with CPU delegate
                       |
          468 landmarks + expression scores
                       |
         Vectorized constant-velocity Kalman filter
                       |
      Centered vertices + sampled colors + normals
                       |
             One-slot result mailbox
                       |
       Open3D studio on the application thread
```

The detector returns the first 468 face points. Extra iris points are not part
of the mesh. One tracked face keeps processing cost predictable and enables
MediaPipe's single-face temporal behavior.

## Ownership

`VideoSource` owns decoding. For webcams, a capture thread overwrites a single
pending frame so slow inference does not create an ever-growing queue.
Files decode sequentially and preserve their source frame timestamps.
Looping advances a timestamp offset rather than returning time to zero.

`Processor` owns the MediaPipe task, filter state and cached topology. It has no
GUI dependency and supports both benchmarking and headless integrations.
`Worker` owns its processor and publishes the newest completed result through
a lock-protected mailbox. Pause and smoothing settings cross the same lock.

`Studio` reads that mailbox from Open3D's tick callback. No background thread
changes a GUI widget or renderer. Open3D geometry uploads are confined to the
application thread. Changing view modes never resets the user's camera.

## Filtering

Each coordinate has position and velocity state with a 2 by 2 covariance.
NumPy evaluates all landmark coordinates together, without a Python object per
landmark. Measurement timestamps drive the transition matrix. Acceleration
noise models changes in motion, and Joseph-form covariance updates preserve
numerical stability.

Face loss, long detection gaps and smoothing toggles reset state. New tracks
start from the current measurement rather than inheriting a previous face.
The filter is intentionally configurable through its Python constructor.

## Geometry

SciPy triangulates the canonical model's XY coordinates once. Landmark IDs
then deform the same triangle indices on every frame. This avoids the flicker
and repeated Qhull work of triangulating each noisy camera projection.
Triangles with centroids inside the canonical eyes and inner mouth are removed.
This is a planar topology heuristic, not the official MediaPipe triangle set
or a watertight anatomical reconstruction.

Coordinates compensate for image aspect ratio, invert image Y, reverse depth,
center the face and normalize outer-eye distance to two units. Depth remains
monocular model output. Exported meshes do not represent measured millimeters.
Source Color samples one RGB color per landmark rather than UV texture mapping.
Textured mode uses raw landmarks as per-triangle UVs and the current RGB image
as its texture. Filtered landmarks deform the surface without shifting UVs.

## Avatar

Emoji mode transforms a built-in sphere rig using the tracked eye and forehead
axes. Smile, jaw-open and eye-blink blendshapes drive the mouth and eyes.
Most geometry is static. The mouth deforms on each processed frame.
It is a stylized expression retarget, not an arbitrary image-to-3D generator.

## Failure handling

Inputs and configuration are validated before processing. Missing model errors
include a recovery command. Model downloads use bounded reads, SHA-256
verification and atomic replacement. Failed downloads leave existing files
untouched. Model downloading is explicit, never triggered by starting a stream.

Worker failures become visible studio dialogs. No-face frames fade the previous
view for 350 milliseconds, then clear it. Video EOF leaves the final frame
available for inspection and export. Worker shutdown releases its detector and
source in a `finally` block. Capture uses a daemon thread because some camera
drivers can block inside a native read beyond the shutdown timeout.

## Optional tools

`StudioTools` owns recording, replay, calibration, streaming, inspection and
playground lifecycle. `studio_panels` builds its views. The inference pipeline
does not import these GUI modules. `Frame` carries both raw and filtered
landmarks so UV mapping, comparisons and performance recording share the same
measurements instead of running duplicate inference.

`SessionRecorder` writes through an eight-frame queue on its own thread.
`ViewRecorder` encodes through a four-frame queue. Both finalize temporary
outputs only after successful writing. Writer errors and dropped-frame counts
are surfaced in the workbench. Shutdown joins active writers with timeouts.
Open3D render callbacks submit image data, never mutate GUI widgets.

`SessionReader` validates versioned SQLite sessions and limits decoded payloads.
`Playback` advances source time from a monotonic clock and supports seek, speed,
pause and looping. It replaces the worker in replay mode. Geometry is rebuilt
from stored coordinates without detection. Source image and landmark data remain
local, sensitive recording contents, not anonymized telemetry.

`SmoothingLab` uses independent Kalman and time-adjusted EMA state with bounded
nose-motion histories. `LandmarkHistory` bounds selected-point history. Neither
tool changes the pipeline filter. Their charts refresh at five Hz, while the
playground refreshes at ten Hz. Image-quality checks run at two Hz on a reduced
face crop. Disabled lab and game tools do not calculate their optional metrics.

`CartoonRig` precomputes displacement fields around explicit image anchors.
Expressions deform a PNG puppet locally while the emoji remains a 3D sphere
rig. These are separate presentations, not interchangeable reconstruction tools.

`LocalStream` sends bounded, float-only OSC bundles to loopback with nonblocking
UDP and a rate cap. `osc_protocol` is dependency-free and packaged unchanged
with the Blender receiver. The receiver drives matching keys only, restores
original state on cancellation and explicitly closes sockets and modal timers
before unregistering.

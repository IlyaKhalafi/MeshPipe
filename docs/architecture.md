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

Worker failures become visible studio dialogs. No-face frames clear old
geometry rather than showing stale tracking. Video EOF leaves the final frame
available for inspection and export. Worker shutdown releases its detector and
source in a `finally` block. Capture uses a daemon thread because some camera
drivers can block inside a native read beyond the shutdown timeout.

# Workbench

Create, Inspect and Play organize optional tools beside the 3D viewer. The dock
hides on windows narrower than 960 pixels. Use Show / hide workbench to reclaim
space. Camera orbit, zoom and pan remain available in every view.

## Record and replay a performance

In Create, choose Record performance while webcam, video or demo input runs.
Stop performance saves a versioned SQLite `.meshpipe` file into `exports/`.
Source RGB images are compressed losslessly. Raw and filtered coordinates use
float32 storage. Expressions, source timestamps and processing durations are
stored alongside them.

The background writer has an eight-frame queue. Overload drops new recording
frames instead of delaying tracking. Saving reports dropped frames. A recording
has a 100,000-frame limit. Disk failures appear in the workbench. An interrupted
save may leave a `.meshpipe.partial` file rather than a playable session.

Open performance replaces the active input and releases its capture worker.
Replay reconstructs geometry without MediaPipe inference. Use the timeline,
Pause, Loop replay and the 0.25x through 2x speed menu. At the end, the final
frame remains available. Restart the studio to return to webcam or video input.

```bash
meshpipe run --replay exports/performance.meshpipe
meshpipe run --replay exports/performance.meshpipe --loop
```

Sessions contain full camera images and biometric-like facial motion data.
They are not anonymized or encrypted. Store them securely and obtain consent
before sharing recordings. Git ignores `.meshpipe` files and `exports/`.
Load recordings only from sources you trust.

## Capture the rendered view

Choose GIF or MP4 and Record view. Captures combine the source image with the
actual rendered scene, including camera orbit and the selected presentation.
This is not a full desktop recording. Demo captures are labeled synthetic.

MP4 uses 960 by 540 pixels at 15 FPS, with a ten-minute limit. GIF uses 800 by
450 pixels at 15 FPS, with a ten-second limit in the studio. The Python recorder
supports GIF durations up to twenty seconds. Slow rendering may repeat frames
to preserve timing. A bounded four-frame encoding queue protects the UI from
encoder backlog. GIF finalization can take time. Keep the window open until
saving completes.

An unavailable MP4 codec produces a visible error. Try GIF instead. Changing
the view during recording changes subsequent captured frames. PLY export still
saves reconstructed face geometry, not emoji or cartoon geometry.

## Personal expression response

Hold a relaxed face and choose Neutral 3 s. Then choose Range 6 s and perform
comfortable blinks, smiles and jaw opening. Each stage needs at least ten
tracked frames. Enable Apply personal calibration to use the result.

Sensitivity scales response. Dead zone removes small movement near neutral.
Calibration affects emoji, cartoon, playground and streamed expression scores,
not detected landmarks. Save and Load preserve calibration as JSON. Performance
files retain original scores so replay can use a different calibration.

Only blink, jaw and smile channels are calibrated. These scores describe facial
motion, not emotions. Comfortable expressions matter more than extreme ranges.

## Animate your PNG

Choose Load PNG and mark anchors or start with Try paper robot. The editor asks
for the subject's anatomical left eye, right eye and mouth center. Markers stay
inside a five-percent margin. Eyes must be at least ten percent of image width
apart. Choose Use this cartoon when placement looks right.

Use a front-facing character with separated eyes and mouth. PNG transparency
is accepted. Images are reduced to at most 512 pixels on their longest side.
Files larger than 20 MB are rejected. Save anchors and Load anchors reuse
normalized placement on the currently loaded image.

Blink, smile and jaw scores deform local image regions. Head roll rotates the
puppet. This is an anchored 2D image rig on a plane, not a 3D head, automatic
landmark extraction from drawings, or general-purpose animation retargeting.
Heavy deformation can stretch nearby artwork. Keep expressions moderate.

## Inspect geometry and tracking

Textured mode maps the source frame onto the face using raw detected landmark
UV coordinates. The surface still follows filtered geometry. This reduces
texture drift from smoothing. Eyes and inner mouth remain holes in the cached
SciPy topology. Rotation exposes the open surface rather than a complete head.

Enable landmark explorer in Inspect. Choose an index from 0 through 467 or
Shift-click near a visible mesh vertex. Region selection highlights landmarks.
Coordinate traces show normalized source x, y and relative z. Region export
includes selected IDs, coordinates and bounded history, not an anatomical map.
Picking is disabled in the lab, emoji and cartoon views.

Tracking health checks cropped-image brightness, sharpness and face framing
twice a second. Hints are heuristics, not MediaPipe confidence values. During
face loss the last view fades for 350 milliseconds, then clears. Filter state
resets on loss and reacquisition starts with current measurements.

## Compare smoothers

Enable Compare raw / Kalman / EMA to replace the main view with three models.
Left mint is raw, center amber is Kalman, right blue is EMA. Change EMA response
to compare stability and lag. EMA adjusts its response for source frame spacing.
The lab uses independent filters and does not alter the main pipeline filter.

Charts track nose coordinates. Roughness is RMS second difference over recent
motion, not a ground-truth jitter measurement. Lag estimates correlate nose
velocity over zero through six source-frame delays. They can be misleading
with periodic or weak motion. The display uses n/a when motion is insufficient.
Cost reports cover each filter calculation, not inference or rendering.

Export lab measurements saves parameters, recent traces and measurements.
The lab needs raw landmarks and adds rendering and analysis work while enabled.
Disable it for the lowest studio cost.

## Play with expressions

Enable expression playground in Play. Match successive smile, jaw, wink and
head-turn targets. Threshold hysteresis and cooldown avoid repeated accidental
hits. Simultaneous blinks do not count as independent winks. Face loss releases
active gestures. Head-turn thresholds use approximate visual yaw, not calibrated
metric pose.

Click the 3D view for keyboard control. S smiles, J opens the jaw, L and R wink,
and A and D turn. Buttons offer a mouse fallback. The game controls only its own
score and target. It does not send OS input or control other applications.

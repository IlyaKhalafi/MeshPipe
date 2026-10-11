# Motion studio

## Adaptive CPU

Enable Adaptive CPU budget in Inspect or start with
`meshpipe run 0 --adaptive-cpu --budget-ms 33`.
The controller averages inference costs and adjusts processing width after
30 samples with hysteresis. Webcam cadence also adapts, dropping stale camera
frames through the existing single-slot capture mailbox. Video timestamps and
playback cadence stay intact. Resolution changes reuse the detector.
This is a processing budget, not a guarantee about rendering or total FPS.

## Landmark-only recording

Enable the privacy checkbox before recording or use `--landmark-only`.
Sessions retain landmarks, raw landmarks, expressions and timestamps but store
no source images. Preview and capture hide the source. Textured and Source Color
views require images and are unavailable for private replay.
Inference still sees source frames in memory. Facial motion remains sensitive.
Legacy recordings continue to load with their images.

## Non-destructive motion editor

Open a performance in Create, then use Motion. Trim start and end use milliseconds
relative to the original recording. Apply edit updates replay without changing
the SQLite file. Undo and redo keep up to 50 edits in memory.
Strength and fades modify expressions. The three curve controls set expression
gain at the recording start, midpoint and end with linear interpolation.
Edit JSON supports up to 256 time and gain points per channel.
Spike removal applies a three-frame median to landmarks and expression scores.
Blend performance synchronizes a second take by normalized duration and combines
landmarks and expressions. Source images always come from the primary take.
Save edit writes a sidecar JSON. Load the same source and blend take before
loading that sidecar. Blend paths are deliberately not embedded or auto-opened.

## Custom GLB avatars

Style includes a bounded glTF 2.0 binary loader for morph-target head meshes.
Try canonical morph head loads a bundled example without downloading an avatar.
Export triangle primitives with named morphs in `mesh.extras.targetNames`.
Channels matching MediaPipe blendshape names work automatically. Map other target
names to channels such as `jawOpen`, `eyeBlinkLeft` and `mouthSmileLeft`, then
save a rig preset. Head rotation follows tracking and the viewport keeps its
orbit and zoom. Rest mesh is centered and scaled for the studio.

Supported data includes multiple meshes, node transforms, triangle indices,
POSITION morphs, opaque base colors and embedded textures with TEXCOORD_0.
Limits include a 32 MiB file, 100000 total vertices, 64 morphs per primitive and
2048-pixel textures. External resources, skeleton skins, sparse accessors,
compression and required extensions fail with an explicit message. Export a
morph-only head rather than a complete skinned humanoid. This is expression
retargeting, not automatic rig discovery or full-body animation.
Bake nonzero default morph weights into the rest mesh before exporting.

## Offline Blender bake

Export Blender motion from Motion after applying edits. Choose export FPS and
maximum linear key reduction error. The JSON contains expressions and unwrapped
head angles, not source images. Build the updated add-on ZIP with
`python tools/build_blender_addon.py` and install it in Blender.

Choose the expression mesh and optional head root. Set Animation JSON to the
exported file. Remap JSON accepts a channel-to-shape-key object, for example
`{"jawOpen": "MouthOpen"}`. Set strength and start frame, stop the live receiver,
then press Bake Saved Performance. Scene FPS conversion is automatic.
New actions preserve existing actions and restore current key values and root
rotation. Reduced keys use linear interpolation. Existing animation can be
restored by selecting its old action. Blender runtime needs a local installation
and a compatible rig. The export protocol is tested without Blender.
Head baking preserves Euler, quaternion and axis-angle rotation modes.

## Expression displacement heatmap

Hold a comfortable neutral face and capture neutral in Style. Heatmap colors
show per-vertex displacement as a fraction of eye distance. Cyan marks little
movement, amber stronger movement and coral values near the color limit.
Stable eye and nose anchors estimate a rigid alignment against the reference,
removing head rotation and translation before comparison. Adjust the scale for
subtle movement. Capture neutral again after camera or identity changes.
These geometry measurements are not emotion recognition or medical analysis.

## Face-attached accessories

Choose Glasses, Crown or Brows in Style, set the attachment landmark, scale and
local x, y and depth offsets, then apply. Glasses default to the upper nose at
168. Crown works well at forehead landmark 10. Save and load anchor presets.
Geometry follows the tracked head basis and attachment point.
Accessories view renders the opaque face and accessory together with normal
depth testing, so temple arms disappear behind the face instead of drawing over
it. This is a 3D AR fitting view, not a source-camera compositing filter or an
invisible camera-image occluder. Accessories are original procedural geometry.

## Browser showcase

Open a recording, optionally apply motion edits, then export browser studio.
Browser and Blender exports run on a background worker with a snapshot of the
current edit and independent read-only connections. Replay stays responsive.
The output folder contains HTML, CSS, JavaScript and quantized facial motion.
No source images, external libraries or network services are required for
playback. Serve the folder with `python -m http.server 8000` and open localhost
in a browser. Upload the static files to a host such as GitHub Pages to share.

Drag or use arrow keys to orbit. Scroll or use plus and minus to zoom.
Switch between a face mesh, golden emoji and paper robot. Raw, Kalman and shared
side-by-side views use one timeline. Motion is downsampled to at most 360 frames
by default, not rerun through MediaPipe in the browser. Source provenance labels
distinguish genuine tracking from synthetic demos. Facial motion remains
sensitive even without images. Only share with recorded people's permission.
Exports label the saved signal Processed, or Edited when an edit is active.
The included demo is labeled Kalman because its generator runs that filter.

## Headless exports

Both exports also work without a display or inference model.

```bash
meshpipe export-web exports/performance.meshpipe --edit edit.json --output exports/web
meshpipe export-animation exports/performance.meshpipe --edit edit.json --fps 30 --output exports/motion.json
```

Use `--blend other.meshpipe` when the edit has a blend weight. Browser export
accepts `--frames` from 2 to 900. Animation export accepts `--tolerance` for
linear key reduction. Omitting an edit exports the unchanged performance.

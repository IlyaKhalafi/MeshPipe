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

Motion includes a bounded glTF 2.0 binary loader for morph-target head meshes.
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

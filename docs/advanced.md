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

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

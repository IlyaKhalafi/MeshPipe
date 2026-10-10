# Third-party assets

## Canonical face

`src/meshpipe/assets/canonical.npy` contains the 468 vertex positions from
Google's MediaPipe canonical face model. Original coordinates remain unchanged.

Source https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj

The MediaPipe Authors. Licensed under Apache License 2.0.
The full license accompanies the asset in `src/meshpipe/assets/LICENSE.apache`.
MeshPipe generates its own triangles with SciPy rather than distributing the
original model's faces.

## Showcase input

Both showcases use actual recorded speech and facial motion from NASA's
"Interview with Antti Pulkkinen" from 2012. NASA created
the footage. Wikimedia Commons lists it as public domain in the United States.

Original source https://svs.gsfc.nasa.gov/10899/

File and license https://commons.wikimedia.org/wiki/File:Interview_with_Antti_Pulkkinen.ogv

The GIFs use a four-second excerpt beginning at two seconds. The face preview
is cropped for readability. MediaPipe processes the decoded video frames, not
an animated photograph or precomputed landmarks. Both the mesh and emoji are
driven by detected landmarks and expression scores. Audio is omitted.

The documentation presentation adds an orbiting viewer camera and labels.
It does not imply endorsement by NASA or the recorded subject.

## Face detector

The face landmarker is downloaded separately from Google's official model
storage. It is not included in this repository. MediaPipe software and the
canonical model are subject to their own license, not MeshPipe's MIT license.

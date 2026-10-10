# Third-party assets

## Canonical face

`src/meshpipe/assets/canonical.npy` contains the 468 vertex positions from
Google's MediaPipe canonical face model. Original coordinates remain unchanged.

Source https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj

Copyright 2020 The MediaPipe Authors. Licensed under Apache License 2.0.
The full license accompanies the asset in `src/meshpipe/assets/LICENSE.apache`.
MeshPipe generates its own triangles with SciPy rather than distributing the
original model's faces.

## Showcase input

The detection showcase uses the public-domain NASA photograph of astronaut
Eileen Collins, distributed by scikit-image as `astronaut.png`. Its source is
https://github.com/scikit-image/scikit-image/blob/v0.19.3/skimage/data/astronaut.png

The input video is generated from this still photograph using affine motion.
It is not a webcam recording or evidence of live expression tracking.
The emoji showcase uses synthetic landmarks and expressions. Neither showcase
measures hardware performance or claims endorsement by NASA.

## Face detector

The face landmarker is downloaded separately from Google's official model
storage. It is not included in this repository. MediaPipe software and the
canonical model are subject to their own license, not MeshPipe's MIT license.

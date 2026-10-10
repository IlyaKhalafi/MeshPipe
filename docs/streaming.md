# Local avatar streaming

MeshPipe sends optional OSC bundles over UDP to `127.0.0.1`. It never sends
images or meshes. Start Local stream in Create or choose an OSC port at startup.
Stop it when no receiver is needed. Default port is 9000.

```bash
meshpipe run 0 --osc-port 9000
python tools/build_blender_addon.py
```

## Blender setup

The build command creates `exports/meshpipe-blender.zip` from this checkout.
It includes the receiver and the same dependency-free OSC decoder used in
MeshPipe. Blender does not need MediaPipe, Open3D or a Python package install.

1. Install the ZIP through Blender's add-on preferences and enable MeshPipe Local Receiver.
2. Open the MeshPipe panel in the 3D View sidebar.
3. Choose a mesh with expression shape keys and optionally a head-root object.
4. Set the same port as MeshPipe and choose Start Local Receiver.
5. Start streaming in MeshPipe. Use Stop Receiver or Escape to release the socket.

Shape-key names must match MediaPipe expression channels exactly. Examples are
`jawOpen`, `eyeBlinkLeft`, `eyeBlinkRight`, `mouthSmileLeft` and `mouthSmileRight`.
The receiver does not create a rig, invent mappings, or insert animation keys.
Use an unanimated test object before applying it to a production scene.

Only received expression keys are driven. Unrelated keys and Basis remain
untouched. A no-face packet or a 500-millisecond timeout sets driven expressions
to zero and returns head rotation to its initial orientation. Stopping restores
original shape-key values and original head rotation mode and values.

The add-on declares Blender 3.6 as its minimum API version. Codec, archive and
shape-key preservation tests run without Blender. Native Blender operation
still needs validation on your Blender version and rig.

## Packet contract

Each packet is an OSC bundle containing float32 messages. Streams are capped
at 30 FPS by default. Packets have an 8192-byte limit. Receivers should discard
malformed, nonfinite, oversized or unsupported data. UDP delivery is best effort.

| Address | Values |
| --- | --- |
| `/meshpipe/frame` | Sequence number and source time in seconds |
| `/meshpipe/tracked` | 1 for detected face or 0 for face loss |
| `/meshpipe/rotation` | Approximate XYZ Euler rotation in radians |
| `/meshpipe/position` | Nose x and y in source-image coordinates and relative z |
| `/meshpipe/blendshape/<name>` | Expression score clamped between 0 and 1 |

Rotation uses the viewer's eye and forehead axes, with image Y inverted and
relative depth reversed. It is not MediaPipe's facial transformation matrix,
world-space pose or a calibrated camera estimate. The Blender receiver composes
it in local XYZ axes with the head root's initial rotation. Different rig axes
may require a parent transform. Position is provided for custom receivers but
the Blender add-on does not translate the head.

Frame sequence numbers use OSC float32 and lose unit precision above 16,777,216.
Restart very long streams rather than depending on sequence accuracy indefinitely.
Replay emits saved expression frames and source timestamps, with the selected
personal calibration applied only when enabled.

The socket binds only to loopback. There is no authentication between local
processes. Do not expose this receiver through port forwarding or treat it as
a secure network service. Credentials, remote services and frame uploads are
not involved.

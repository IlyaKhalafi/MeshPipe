from types import SimpleNamespace

import numpy as np
import pytest

from meshpipe.cartoon import CartoonRig, builtin_template
from meshpipe.demo import demo_frame
from meshpipe.geometry import to_vertices
from meshpipe.pipeline import Frame
from meshpipe.scene import FaceScene


@pytest.mark.parametrize("closed_side", ["left", "right"])
def test_anatomical_eye_position_and_independent_blinks(closed_side):
    image, landmarks, _ = demo_frame(0)
    scores = {f"eyeBlink{closed_side.title()}": 1.0}
    frame = Frame(image, 0, landmarks, to_vertices(landmarks, 4 / 3), None, None, scores, 0)
    transforms = {}
    scene = FaceScene.__new__(FaceScene)
    scene.scene = SimpleNamespace(
        set_geometry_transform=lambda name, matrix: transforms.update({name: matrix.copy()})
    )
    scene._replace = lambda *args: None
    scene._update_avatar(frame)
    horizontal = transforms["head"][:3, 0]
    for side, sign in [("left", 1), ("right", -1)]:
        eye = transforms[f"eye-{side}"]
        assert np.dot(eye[:3, 3], horizontal) == pytest.approx(sign * 0.48)
        expected_height = 0.025 if side == closed_side else 0.23
        assert np.linalg.norm(eye[:3, 1]) == pytest.approx(expected_height)


def test_cartoon_uses_supported_transparency_shader():
    image, landmarks, scores = demo_frame(0)
    frame = Frame(image, 0, landmarks, to_vertices(landmarks, 4 / 3), None, None, scores, 0)
    scene = FaceScene.__new__(FaceScene)
    scene.cartoon = CartoonRig(builtin_template())
    scene.opacity = 1
    captured = []
    scene._replace = lambda *arguments: captured.append(arguments)
    scene._update_cartoon(frame)
    name, mesh, material = captured[0]
    assert name == "face" and len(mesh.vertex_normals) == 4
    assert material.shader == "defaultLitTransparency"
    assert material.has_alpha and material.base_reflectance == 0
    assert np.asarray(material.albedo_img).shape[2] == 4

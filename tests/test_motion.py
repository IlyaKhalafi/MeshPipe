import sqlite3

import numpy as np
import pytest

from meshpipe.config import Config
from meshpipe.demo import demo_frame
from meshpipe.motion import EditedReader, MotionEdit
from meshpipe.pipeline import Processor
from meshpipe.session import SessionReader, SessionRecorder


@pytest.fixture
def performance(tmp_path):
    path = tmp_path / "motion.meshpipe"
    processor = Processor(Config(demo=True))
    recorder = SessionRecorder(path, save_images=False, source_kind="synthetic")
    for index in range(5):
        rgb, points, _ = demo_frame(index / 10)
        frame = processor.process(
            rgb,
            index * 100,
            demo_landmarks=points,
            demo_expressions={"jawOpen": 1 if index == 2 else 0.2},
        )
        recorder.submit(frame)
    recorder.close()
    return path


def test_edits_do_not_mutate_original(performance, tmp_path):
    before = performance.read_bytes()
    reader = EditedReader(SessionReader(performance))
    reader.set_edit(MotionEdit(start_ms=100, end_ms=300, despike=True, strength=2))
    assert reader.duration_ms == 200
    assert reader.frame(1).expressions["jawOpen"] == pytest.approx(0.4)
    assert reader.frame(1).source_available is False
    reader.undo()
    assert reader.duration_ms == 400
    reader.undo(redo=True)
    assert reader.duration_ms == 200
    path = tmp_path / "edit.json"
    reader.edit.save(path, 400)
    assert MotionEdit.load(path, 400) == reader.edit
    reader.close()
    assert before == performance.read_bytes()
    with sqlite3.connect(performance) as connection:
        assert connection.execute("SELECT count(*) FROM frames").fetchone()[0] == 5


def test_fades_curves_blend(performance):
    reader = EditedReader(SessionReader(performance))
    reader.set_blend(performance)
    reader.set_edit(
        MotionEdit(fade_ms=100, blend_weight=0.5, curves={"jawOpen": [[0, 0], [400, 1]]})
    )
    assert reader.frame(0).expressions["jawOpen"] == 0
    assert reader.frame(2).expressions["jawOpen"] == pytest.approx(0.5)
    assert reader.frame(4).expressions["jawOpen"] == 0
    assert np.isfinite(reader.frame(2).vertices).all()
    reader.close()


@pytest.mark.parametrize(
    "edit",
    [
        MotionEdit(start_ms=500),
        MotionEdit(strength=float("nan")),
        MotionEdit(curves={"x": [[0, 1], [0, 2]]}),
    ],
    ids=["outside", "nan", "duplicate-time"],
)
def test_invalid_edits(edit):
    with pytest.raises(ValueError):
        edit.validate(400)


def test_unedited_geometry_reuses_cached_arrays(performance):
    reader = EditedReader(SessionReader(performance))
    original = reader._base_frame(0)
    edited = reader.frame(0)
    assert edited.vertices is original.vertices
    assert edited.normals is original.normals
    reader.close()

import threading

from meshpipe.baking import export_animation
from meshpipe.jobs import ExportJob, MotionSource
from meshpipe.motion import EditedReader, MotionEdit
from meshpipe.session import SessionReader

pytest_plugins = ["test_motion"]


def test_export_worker_owns_reader_and_edit_snapshot(performance, tmp_path):
    reader = EditedReader(SessionReader(performance))
    reader.set_edit(MotionEdit(strength=0.5))
    source = MotionSource.snapshot(reader)
    reader.set_edit(MotionEdit(strength=2))
    reader.close()
    job = ExportJob(source, lambda take: export_animation(take, tmp_path / "motion.json"))
    assert job.done.wait(5)
    assert job.error is None
    assert job.result.is_file()
    assert source.edit.strength == 0.5
    job.close()


def test_export_cancellation_and_failure(performance):
    reader = EditedReader(SessionReader(performance))
    source = MotionSource.snapshot(reader)
    reader.close()
    started, release = threading.Event(), threading.Event()

    def action(take):
        started.set()
        release.wait(2)
        return take.frame(0)

    job = ExportJob(source, action)
    assert started.wait(5)
    job.cancel.set()
    release.set()
    assert job.done.wait(5)
    assert "cancelled" in job.error
    job.close()

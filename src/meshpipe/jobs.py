import copy
import threading
from dataclasses import dataclass
from pathlib import Path

from meshpipe.motion import EditedReader, MotionEdit
from meshpipe.session import SessionReader


@dataclass(frozen=True)
class MotionSource:
    path: Path
    edit: MotionEdit
    blend: Path | None = None

    @classmethod
    def snapshot(cls, reader: EditedReader):
        return cls(
            reader.base.path,
            copy.deepcopy(reader.edit),
            reader.blend.path if reader.blend else None,
        )


class ExportJob:
    """Export an immutable edit snapshot with thread-owned SQLite connections."""

    def __init__(self, source: MotionSource, action):
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.result = None
        self.error = None
        self.thread = threading.Thread(
            target=self._run, args=(source, action), name="meshpipe-export", daemon=True
        )
        self.thread.start()

    def _run(self, source, action):
        reader = None
        try:
            reader = EditedReader(SessionReader(source.path))
            reader.set_edit(source.edit, remember=False)
            if source.blend:
                reader.set_blend(source.blend)
            original_frame = reader.frame

            def frame(index):
                if self.cancel.is_set():
                    raise InterruptedError("Motion export cancelled")
                return original_frame(index)

            reader.frame = frame
            self.result = action(reader)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            try:
                if reader:
                    reader.close()
            finally:
                self.done.set()

    def close(self):
        self.cancel.set()
        self.thread.join(timeout=5)

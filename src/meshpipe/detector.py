import numpy as np
from numpy.typing import NDArray

from meshpipe.config import Config
from meshpipe.model import resolve_model


class FaceDetector:
    def __init__(self, config: Config):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.timestamp = -1
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(resolve_model(config.model)),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=config.detection_confidence,
            min_face_presence_confidence=config.detection_confidence,
            min_tracking_confidence=config.detection_confidence,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb: NDArray, timestamp_ms: int) -> tuple[NDArray | None, dict[str, float]]:
        self.timestamp = max(timestamp_ms, self.timestamp + 1)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.landmarker.detect_for_video(image, self.timestamp)
        if not result.face_landmarks:
            return None, {}
        landmarks = np.array(
            [[point.x, point.y, point.z] for point in result.face_landmarks[0][:468]],
            dtype=np.float64,
        )
        expressions = {
            item.category_name: float(item.score)
            for item in (result.face_blendshapes[0] if result.face_blendshapes else [])
        }
        return landmarks, expressions

    def close(self) -> None:
        self.landmarker.close()

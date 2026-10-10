import numpy as np
from numpy.typing import NDArray


class LandmarkFilter:
    """Independent constant-velocity Kalman filters for every landmark coordinate."""

    def __init__(self, measurement_noise: float = 2e-5, acceleration_noise: float = 0.08):
        if measurement_noise <= 0 or acceleration_noise <= 0:
            raise ValueError("Noise parameters must be positive")
        self.measurement_noise = measurement_noise
        self.acceleration_noise = acceleration_noise
        self.reset()

    def reset(self) -> None:
        self.position: NDArray | None = None
        self.velocity: NDArray | None = None
        self.covariance: NDArray | None = None

    def update(self, measurement: NDArray, dt: float) -> NDArray:
        observed = np.asarray(measurement, dtype=np.float64)
        if observed.ndim != 2 or observed.shape[1] != 3 or not np.isfinite(observed).all():
            raise ValueError("Landmarks must be a finite (N, 3) array")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Time step must be finite and positive")
        if self.position is None or self.position.shape != observed.shape:
            self.position = observed.copy()
            self.velocity = np.zeros_like(observed)
            self.covariance = np.broadcast_to(np.eye(2), (*observed.shape, 2, 2)).copy()
            self.covariance[..., 0, 0] = self.measurement_noise
            return self.position.copy()
        step = min(dt, 0.25)
        transition = np.array([[1.0, step], [0.0, 1.0]])
        process_noise = self.acceleration_noise * np.array(
            [[step**4 / 4, step**3 / 2], [step**3 / 2, step**2]]
        )
        predicted = self.position + step * self.velocity
        covariance = transition @ self.covariance @ transition.T + process_noise
        gain = covariance[..., :, 0] / (covariance[..., 0, 0] + self.measurement_noise)[..., None]
        residual = observed - predicted
        self.position = predicted + gain[..., 0] * residual
        self.velocity += gain[..., 1] * residual
        correction = np.broadcast_to(np.eye(2), covariance.shape).copy()
        correction[..., :, 0] -= gain
        self.covariance = (
            correction @ covariance @ np.swapaxes(correction, -1, -2)
            + self.measurement_noise * gain[..., :, None] * gain[..., None, :]
        )
        return self.position.copy()

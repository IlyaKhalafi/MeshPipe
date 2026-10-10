import numpy as np
import pytest

from meshpipe.filter import LandmarkFilter


def test_initial_observation_and_reset():
    tracker = LandmarkFilter()
    points = np.full((468, 3), 0.5)
    np.testing.assert_array_equal(tracker.update(points, 1 / 30), points)
    tracker.reset()
    np.testing.assert_array_equal(tracker.update(points + 1, 1 / 30), points + 1)


def test_noise_reduction_and_finite_covariance():
    random = np.random.default_rng(42)
    tracker = LandmarkFilter()
    observed = random.normal(0.5, 0.005, (240, 468, 3))
    filtered = np.array([tracker.update(points, 1 / 30) for points in observed])
    assert filtered[30:].var() < observed[30:].var() * 0.65
    assert np.isfinite(tracker.covariance).all()
    assert np.linalg.eigvalsh(tracker.covariance).min() >= -1e-12


def test_constant_velocity_tracks_without_large_lag():
    tracker = LandmarkFilter()
    for index in range(120):
        expected = np.full((8, 3), index / 3000)
        actual = tracker.update(expected, 1 / 30)
    np.testing.assert_allclose(actual, expected, atol=0.0001)


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf])
def test_invalid_timestep(dt):
    with pytest.raises(ValueError):
        LandmarkFilter().update(np.zeros((468, 3)), dt)


@pytest.mark.parametrize("points", [np.zeros((2, 2)), np.full((2, 3), np.nan)])
def test_invalid_landmarks(points):
    with pytest.raises(ValueError):
        LandmarkFilter().update(points, 1 / 30)

"""Unit tests for mppi_nav.utils (no ROS needed)."""
import math

import numpy as np
import pytest

from mppi_nav.utils import (load_path_csv, points_at_spacing, resample_path, resolve_path,
                            save_path_csv, spacing_stats, yaw_from_quaternion)

# Exact contents of smoothing-rpp/waypoints/waypoint.csv (header, a space, no final newline).
REFERENCE_CSV = 'x,y\n0.0,0.0\n2.0,6.0\n4.0,-6.0\n5.0, 4.0'


def simulate_recorder(poses, spacing=0.1):
    """Feed poses to points_at_spacing the same way PathRecorder.odom_callback does."""
    points = [poses[0]]
    for p in poses[1:]:
        points.extend(points_at_spacing(points[-1], p, spacing))
    return np.array(points)


def test_points_at_spacing_straight_line():
    pts = points_at_spacing((0.0, 0.0), (0.35, 0.0), 0.1)
    assert np.allclose(pts, [(0.1, 0.0), (0.2, 0.0), (0.3, 0.0)])


def test_points_at_spacing_too_close_returns_nothing():
    assert points_at_spacing((0.0, 0.0), (0.05, 0.05), 0.1) == []


def test_recorder_spacing_is_exact_on_curved_odom_stream():
    # Odometry at ~45 Hz while driving 0.22 m/s on a 0.5 m radius circle: ~5 mm steps.
    t = np.arange(0.0, 20.0, 1.0 / 45.0)
    r, w = 0.5, 0.22 / 0.5
    poses = list(zip(r * np.sin(w * t), r * (1 - np.cos(w * t))))
    pts = simulate_recorder(poses)
    lo, mean, hi = spacing_stats(pts)
    assert lo == pytest.approx(0.1, abs=1e-9)
    assert hi == pytest.approx(0.1, abs=1e-9)
    # The chord-based points must still lie on the driven circle (error < 1 cm).
    radial_err = np.abs(np.hypot(pts[:, 0], pts[:, 1] - r) - r)
    assert radial_err.max() < 0.01


def test_load_reference_repo_csv(tmp_path):
    f = tmp_path / 'waypoint.csv'
    f.write_text(REFERENCE_CSV)
    pts = load_path_csv(str(f))
    assert pts.shape == (4, 2)
    assert np.allclose(pts[-1], (5.0, 4.0))


def test_load_csv_without_header(tmp_path):
    f = tmp_path / 'p.csv'
    f.write_text('0,0\n0.1,0\n\n0.2,0\n')
    assert load_path_csv(str(f)).shape == (3, 2)


def test_load_csv_bad_line_raises(tmp_path):
    f = tmp_path / 'p.csv'
    f.write_text('x,y\n0,0\nfoo,bar\n')
    with pytest.raises(ValueError, match=':3:'):
        load_path_csv(str(f))


def test_save_load_roundtrip(tmp_path):
    pts = [(0.0, 0.1), (0.0, 0.2), (0.1, 0.2)]  # example from the assignment PDF
    out = save_path_csv(str(tmp_path / 'sub' / 'p.csv'), pts)
    assert open(out).readline().strip() == 'x,y'
    assert np.allclose(load_path_csv(out), pts)


def test_save_never_writes_negative_zero(tmp_path):
    out = save_path_csv(str(tmp_path / 'p.csv'), [(0.0, -0.0), (0.1, -0.00001)])
    assert '-0.0000' not in open(out).read()


def test_resample_sparse_waypoints():
    sparse = np.array([[0.0, 0.0], [2.0, 6.0], [4.0, -6.0], [5.0, 4.0]])
    dense = resample_path(sparse, 0.1)
    lo, mean, hi = spacing_stats(dense)
    assert np.allclose(dense[0], sparse[0]) and np.allclose(dense[-1], sparse[-1])
    assert mean == pytest.approx(0.1, abs=1e-3)
    assert hi <= 0.1 + 1e-3  # corners only make chords shorter, never longer


def test_resample_drops_duplicates():
    dense = resample_path(np.array([[0, 0], [0, 0], [1, 0]], dtype=float), 0.1)
    assert len(dense) == 11


@pytest.mark.parametrize('yaw', [0.0, 0.5, -2.0, math.pi - 1e-6])
def test_yaw_from_quaternion(yaw):
    q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    assert yaw_from_quaternion(*q) == pytest.approx(yaw, abs=1e-9)


def test_resolve_path(tmp_path):
    assert resolve_path('/abs/a.csv', str(tmp_path)) == '/abs/a.csv'
    assert resolve_path('a.csv', str(tmp_path)) == str(tmp_path / 'a.csv')

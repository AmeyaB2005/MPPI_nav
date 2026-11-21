"""Unit tests for mppi_nav.mppi_core (pure NumPy, no ROS needed)."""
import math
import os
import sys

import numpy as np
import pytest

from mppi_nav.mppi_core import (MPPIConfig, MPPIController, compute_weights, rollout,
                                shift_nominal)
from mppi_nav.utils import load_mppi_params, resample_path, scan_to_points

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def straight_path(length=4.0):
    return resample_path(np.array([[0.0, 0.0], [length, 0.0]]), 0.1)


def figure_eight(a=1.2):
    t = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 2000)
    return resample_path(np.column_stack((a * np.sin(t), a * np.sin(t) * np.cos(t))), 0.1)


def make(path=None, **over):
    cfg = MPPIConfig(seed=0, **over)
    return MPPIController(cfg, straight_path() if path is None else path)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_rollout_shape():
    V = np.zeros((7, 34, 2))
    assert rollout(np.zeros(3), V, 0.15).shape == (7, 34, 3)


def test_unicycle_single_step():
    s = rollout(np.array([1.0, 2.0, math.pi / 2]), np.array([[0.2, 0.5]]), 0.1)[0]
    assert s == pytest.approx([1.0, 2.02, math.pi / 2 + 0.05])


def test_rollout_matches_step_by_step_loop():
    rng = np.random.default_rng(1)
    U = rng.normal(size=(20, 2))
    x = np.array([0.3, -0.2, 0.7])
    ref = []
    for v, w in U:                       # the textbook Euler loop
        x = np.array([x[0] + v * math.cos(x[2]) * 0.1, x[1] + v * math.sin(x[2]) * 0.1,
                      x[2] + w * 0.1])
        ref.append(x)
    assert np.allclose(rollout(np.array([0.3, -0.2, 0.7]), U, 0.1), ref)


# --------------------------------------------------------------------------- #
# MPPI pieces
# --------------------------------------------------------------------------- #
def test_weights_sum_to_one_and_favour_low_cost():
    S = np.array([5.0, 1.0, 3.0, 1e6, 1e4])
    w = compute_weights(S, 0.3)
    assert w.sum() == pytest.approx(1.0)
    assert np.argmax(w) == 1
    assert np.all(np.isfinite(w)) and w[3] < 1e-12


def test_noise_is_time_correlated_with_right_sigma():
    c = make(num_samples=4000)
    eps = c.sample_noise(4000, c.cfg.horizon)
    assert eps.shape == (4000, c.cfg.horizon, 2)
    # at the knots the std is sigma; between them interpolation makes it a bit smaller
    assert eps[:, 0].std(axis=0) == pytest.approx([c.cfg.noise_v, c.cfg.noise_w], rel=0.1)
    # neighbouring steps are strongly correlated (white noise would give ~0)
    assert np.corrcoef(eps[:, 3, 1], eps[:, 4, 1])[0, 1] > 0.8


def test_primitives_are_in_every_sample_set():
    c = make()
    V = c.sample_controls()
    P = len(c.primitives)
    assert P == c.cfg.primitive_rates * len(c.cfg.primitive_turn_steps) * 2
    assert np.allclose(V[2:2 + P], c.primitives)
    assert np.allclose(c.primitives[:, :, 0], c.cfg.v_max)       # full speed
    assert c.primitives[:, 0, 1].min() < 0 < c.primitives[:, 0, 1].max()  # both directions


def test_shift_nominal_by_whole_step():
    U = np.arange(10, dtype=float).reshape(5, 2)
    assert np.allclose(shift_nominal(U, 1.0), np.vstack([U[1:], U[-1]]))


# --------------------------------------------------------------------------- #
# Local window / progress (Case 4)
# --------------------------------------------------------------------------- #
def test_progress_never_backward_and_no_jump_at_crossing():
    path = figure_eight()
    c = make(path)
    # indices where the path passes through its centre (the crossing), ~1/4 and ~3/4
    near_centre = np.where(np.linalg.norm(path, axis=1) < 0.06)[0]
    first_pass = near_centre[near_centre < len(path) / 2]
    second_pass = near_centre[near_centre > len(path) / 2]
    assert len(first_pass) and len(second_pass)

    history = []
    for p in path[: first_pass.max() + 3]:      # drive the first part, through the crossing
        history.append(c.update_progress(p))
    assert np.all(np.diff(history) >= 0)
    assert c.progress_idx < second_pass.min() - 10   # matched the 1st pass, not the 2nd

    before = c.progress_idx
    for p in path[before - 15: before][::-1]:  # robot moves backwards
        c.update_progress(p)
    assert c.progress_idx == before


def test_rollout_on_a_later_lap_gets_no_progress():
    # Tiny circle driven twice: one lap = 16 points, shorter than the 20-point
    # cost window, so the window contains the SAME place twice (index i and i+16).
    t = np.linspace(0.0, 4 * np.pi, 2000)
    path = resample_path(0.25 * np.column_stack((np.cos(t), np.sin(t))), 0.1)
    c = make(path)
    start, end = c.window()
    win = path[start:end]
    lap = int(np.argmin(np.linalg.norm(win[8:] - win[0], axis=1))) + 8
    assert np.linalg.norm(win[lap] - win[0]) < 0.06 and lap < len(win)

    # A rollout that just sits at the start spot. Plain "nearest window point" could
    # match it to the later lap (index `lap`) and reward it for a full lap of progress.
    stay = np.repeat(win[:1], 10, axis=0)[None]
    idx, d = c.project_rollouts(stay, win)
    assert idx[0, -1] <= c.cfg.rollout_search                # no progress
    # A rollout that really follows the path advances along it.
    follow = win[:10][None]
    idx_f, _ = c.project_rollouts(follow, win)
    assert idx_f[0, -1] == 9


# --------------------------------------------------------------------------- #
# Obstacles, goal, recovery
# --------------------------------------------------------------------------- #
def test_colliding_sample_gets_high_cost():
    c = make()
    c.set_obstacles(np.column_stack((np.full(21, 1.0), np.linspace(-0.5, 0.5, 21))),
                    robot_xy=np.zeros(2))
    T = c.cfg.horizon
    V = np.stack([np.tile([0.22, 0.0], (T, 1)),            # straight into the wall
                  np.zeros((T, 2))])                         # stop
    S, collided = c.evaluate_costs(rollout(np.zeros(3), V, c.cfg.dt), V)
    assert collided.tolist() == [True, False]
    assert S[0] > S[1] + c.cfg.w_collision / 2


def test_obstacle_points_are_cropped_and_capped():
    c = make(max_obstacle_points=50)
    far = np.array([[10.0, 10.0]])
    near = np.column_stack((np.linspace(0.5, 1.5, 500), np.full(500, 0.3)))
    c.set_obstacles(np.vstack([near, far]), robot_xy=np.zeros(2))
    assert len(c.obstacles) <= 50
    assert np.linalg.norm(c.obstacles, axis=1).max() < c.cfg.obstacle_crop_radius


def test_goal_reached_gives_zero_command():
    path = straight_path(1.0)
    c = make(path)
    for p in path:
        c.update_progress(p)
    r = c.compute(np.array([*path[-1], 0.0]))
    assert r.goal_reached and r.v == 0.0 and r.w == 0.0


def test_closed_loop_tracks_straight_path():
    """Drive 10 s on a free straight path with the controller in the loop."""
    c = make()
    state = np.zeros(3)
    dt = 1.0 / c.cfg.control_rate
    for _ in range(int(10.0 / dt)):
        r = c.compute(state)
        assert 0.0 < r.ess <= c.cfg.num_samples
        state = rollout(state, np.array([[r.v, r.w]]), dt)[0]
        assert abs(state[1]) < 0.05            # cross-track error < 5 cm
        assert abs(state[2]) < 0.35            # heading error < 20 deg
    assert state[0] > 1.5                      # moved forward at a sensible speed


def test_all_collide_triggers_recovery_backwards():
    c = make()
    ring = 0.1 * np.column_stack((np.cos(np.linspace(-1, 1, 30)), np.sin(np.linspace(-1, 1, 30))))
    c.set_obstacles(ring, robot_xy=np.zeros(2))              # obstacle 0.1 m in front
    r = c.compute(np.zeros(3))
    assert r.all_collide and r.v < 0.0


# --------------------------------------------------------------------------- #
# Config + shared helpers
# --------------------------------------------------------------------------- #
def test_yaml_matches_code_defaults():
    params = load_mppi_params(os.path.join(PKG, 'config', 'mppi_params.yaml'))
    cfg = MPPIConfig.from_dict(params)
    default = MPPIConfig()
    for k, v in params.items():
        if hasattr(default, k):
            got, want = getattr(cfg, k), getattr(default, k)
            if isinstance(want, tuple):
                assert tuple(got) == want, k
            else:
                assert float(got) == pytest.approx(float(want)), k


def test_scan_to_points():
    ranges = np.array([1.0, np.inf, np.nan, 0.05, 2.0])
    pts = scan_to_points(ranges, 0.0, math.pi / 2, 0.12, 3.5, (1.0, 1.0, math.pi / 2))
    # beam 0 (straight ahead = +y) and beam 4 (angle 2*pi -> also +y); others dropped
    assert np.allclose(pts, [[1.0, 2.0], [1.0, 3.0]])


def test_sim2d_lidar_sees_wall():
    sys.path.insert(0, PKG)
    from sim2d.world import Box, LidarModel, World
    world = World([Box(cx=1.0, cy=0.0, sx=0.05, sy=2.0)])
    lidar = LidarModel(noise_std=0.0)
    r = world.raycast((0.0, 0.0), 0.0, lidar, np.random.default_rng(0))
    assert r[0] == pytest.approx(0.975)                      # front face of the wall
    assert np.isinf(r[180])                                   # nothing behind

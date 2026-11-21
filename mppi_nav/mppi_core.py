"""MPPI (Model Predictive Path Integral) path tracker. Pure NumPy, no ROS.

The controller is called once per control cycle with the current robot state
(x, y, theta). It returns a velocity command (v, w).

One cycle of MPPI
-----------------
  U   : nominal control sequence, shape (T, 2) = T steps of (v, w).
  1. Sample K noisy copies of U:   V_k = clip(U + eps_k),  eps_k ~ N(0, Sigma)
  2. Simulate each V_k with the unicycle model -> K trajectories of T states.
  3. Give every trajectory a cost S_k (tracking, progress, obstacles, ...).
  4. Weight each sample:            w_k = exp(-(S_k - min S) / lambda) / eta
                                    (eta = sum of the exponentials, so sum w_k = 1)
  5. New nominal sequence:          U = sum_k w_k V_k     (weighted average)
  6. Send U[0] to the robot, shift U by one control period for the next cycle.

Lambda ("temperature") sets how greedy the average is:
  lambda -> 0   : only the single best sample counts (noisy, jumpy)
  lambda -> inf : every sample counts equally (U barely changes)

Path handling
-------------
`progress_idx` is the index of the path point the robot has reached. It only
ever moves forward, and the search for the nearest path point is limited to a
small window around it. This is what stops the robot from "jumping" to a later
part of a path that crosses itself (Case 4).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class MPPIConfig:
    """All MPPI hyperparameters. Defaults match config/mppi_params.yaml."""

    # model / sampling
    dt: float = 0.15
    horizon: int = 34
    num_samples: int = 1000
    temperature: float = 1.0
    noise_v: float = 0.1
    noise_w: float = 0.6
    v_min: float = -0.05
    v_max: float = 0.22
    w_min: float = -2.0
    w_max: float = 2.0
    exploration_ratio: float = 0.1
    noise_knots: int = 6
    primitive_rates: int = 8           # turn rates in the primitive library (0 = off)
    primitive_rate_max: float = 1.5    # [rad/s] largest primitive turn rate
    primitive_turn_steps: tuple = (6, 10)   # how long the primitives turn [model steps]
    use_savgol: bool = True
    savgol_window: int = 9
    savgol_order: int = 3
    control_rate: float = 20.0
    # local path window
    search_back: int = 5
    search_forward: int = 5
    window_length: int = 20
    rollout_search: int = 3
    goal_tolerance: float = 0.1
    # cost weights
    w_track: float = 5.0
    w_progress: float = 15.0
    w_heading: float = 1.0
    w_terminal: float = 1.0
    w_v: float = 0.0
    w_w: float = 0.05
    w_dv: float = 1.0
    w_dw: float = 0.1
    blocked_scale: float = 0.1
    # obstacles
    robot_radius: float = 0.105
    collision_radius: float = 0.18
    inflation_radius: float = 0.5
    obstacle_decay: float = 0.1
    w_obstacle: float = 1.0
    w_collision: float = 1e4
    obstacle_crop_radius: float = 2.0
    obstacle_voxel: float = 0.03
    max_obstacle_points: int = 360
    # recovery / visualisation
    recovery_speed: float = 0.05
    num_vis_samples: int = 20
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'MPPIConfig':
        """Build a config from a dict, ignoring keys that are not MPPI settings
        (e.g. ROS-only parameters such as frame_id)."""
        names = {f.name: f.type for f in fields(cls)}
        known = {k: v for k, v in params.items() if k in names}
        cfg = cls(**known)
        # YAML may give ints where floats are expected (and vice versa): normalise.
        if isinstance(cfg.primitive_turn_steps, (list, tuple)):
            cfg.primitive_turn_steps = tuple(int(n) for n in cfg.primitive_turn_steps)
        for k in ('horizon', 'num_samples', 'noise_knots', 'primitive_rates', 'savgol_window', 'savgol_order', 'search_back',
                  'search_forward', 'window_length', 'rollout_search',
                  'max_obstacle_points', 'num_vis_samples'):
            setattr(cfg, k, int(getattr(cfg, k)))
        return cfg


@dataclass
class MPPIResult:
    """What one control cycle returns."""

    v: float                           # linear velocity command [m/s]
    w: float                           # angular velocity command [rad/s]
    goal_reached: bool
    all_collide: bool                  # True -> recovery command was used
    progress_idx: int
    optimal_traj: np.ndarray           # (T, 3) rollout of the new nominal sequence
    sampled_trajs: np.ndarray          # (n_vis, T, 3) a few sampled rollouts
    min_cost: float = 0.0
    ess: float = 0.0                   # effective sample size = 1 / sum(w_k^2)
    compute_ms: float = 0.0


# --------------------------------------------------------------------------- #
# Building blocks (module-level so they can be unit-tested on their own)
# --------------------------------------------------------------------------- #
def rollout(state: np.ndarray, controls: np.ndarray, dt: float) -> np.ndarray:
    """Simulate the unicycle model for every control sequence at once.

    Model (forward Euler), for t = 0 .. T-1:
        x[t+1]     = x[t] + v[t] * cos(theta[t]) * dt
        y[t+1]     = y[t] + v[t] * sin(theta[t]) * dt
        theta[t+1] = theta[t] + w[t] * dt

    Vectorised over samples and time: theta is a cumulative sum of w*dt, and x, y
    are cumulative sums that use the heading from *before* each step.

    Args:
        state:    (3,) start state (x, y, theta).
        controls: (..., T, 2) control sequences of (v, w).
        dt:       model step [s].
    Returns:
        (..., T, 3) states after each step (the start state is not included).
    """
    v = controls[..., 0]
    w = controls[..., 1]
    theta_after = state[2] + np.cumsum(w * dt, axis=-1)                 # theta[1..T]
    theta_before = np.concatenate(                                       # theta[0..T-1]
        [np.full(theta_after.shape[:-1] + (1,), state[2]), theta_after[..., :-1]], axis=-1)
    x = state[0] + np.cumsum(v * np.cos(theta_before) * dt, axis=-1)
    y = state[1] + np.cumsum(v * np.sin(theta_before) * dt, axis=-1)
    return np.stack([x, y, theta_after], axis=-1)


def compute_weights(costs: np.ndarray, temperature: float) -> np.ndarray:
    """MPPI importance weights w_k = exp(-(S_k - min S) / lambda), normalised to sum to 1.

    Subtracting min S does not change the normalised result (it cancels in the
    division) but keeps exp() from underflowing to 0 for every sample, and means
    huge collision costs simply give weight ~0 instead of causing overflow.
    """
    shifted = costs - costs.min()
    w = np.exp(-shifted / temperature)
    return w / w.sum()


def shift_nominal(U: np.ndarray, steps: float) -> np.ndarray:
    """Warm start: move the nominal sequence `steps` model steps forward in time.

    The control loop (20 Hz = 0.05 s) runs faster than the model step (0.1 s), so
    the shift is usually fractional (0.5 steps). We linearly interpolate U at the
    shifted times; times past the end repeat the last control.
    """
    T = U.shape[0]
    t_new = np.arange(T) + steps
    t_old = np.arange(T)
    return np.column_stack([np.interp(t_new, t_old, U[:, j]) for j in range(U.shape[1])])


def path_headings(path: np.ndarray) -> np.ndarray:
    """Tangent direction of the path at each point (direction to the next point)."""
    d = np.diff(path, axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0])
    return np.append(yaw, yaw[-1]) if len(yaw) else np.zeros(len(path))


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #
class MPPIController:
    """Stateful MPPI path tracker. Call `set_obstacles` when a new scan arrives
    and `compute` once per control cycle."""

    def __init__(self, cfg: MPPIConfig, path: np.ndarray) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.sigma = np.array([cfg.noise_v, cfg.noise_w])
        self.u_lo = np.array([cfg.v_min, cfg.w_min])
        self.u_hi = np.array([cfg.v_max, cfg.w_max])
        self.primitives = self.build_primitives()
        self.obstacles = np.zeros((0, 2))
        self.tree: Optional[cKDTree] = None
        self.set_path(path)

    # ------------------------------------------------------------------ #
    # Inputs
    # ------------------------------------------------------------------ #
    def set_path(self, path: np.ndarray) -> None:
        """Load a new path (N, 2) and reset the controller state."""
        self.path = np.asarray(path, dtype=float)
        if len(self.path) < 2:
            raise ValueError('path needs at least 2 points')
        self.path_yaw = path_headings(self.path)
        # Max path points the robot can pass in one horizon at full speed (progress cap).
        spacing = float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).mean())
        self.max_progress = int(np.ceil(self.cfg.v_max * self.cfg.horizon * self.cfg.dt
                                        / max(spacing, 1e-6)))
        self.progress_idx = 0
        self.goal_reached = False
        self.U = np.zeros((self.cfg.horizon, 2))   # nominal control sequence
        self.last_cmd = np.zeros(2)                # command sent last cycle

    def set_obstacles(self, points: np.ndarray, robot_xy: Optional[np.ndarray] = None) -> None:
        """Store obstacle points (M, 2) in the world frame (e.g. from the lidar).

        To keep the nearest-neighbour queries fast we
          1. drop points farther than `obstacle_crop_radius` from the robot
             (rollouts never reach them),
          2. merge points that fall in the same `obstacle_voxel` grid cell,
          3. cap the count at `max_obstacle_points` (evenly spaced subset).
        A KD-tree is then built once per scan and reused every control cycle.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if robot_xy is not None and len(pts):
            pts = pts[np.linalg.norm(pts - robot_xy, axis=1) <= self.cfg.obstacle_crop_radius]
        if len(pts) and self.cfg.obstacle_voxel > 0:
            cells = np.floor(pts / self.cfg.obstacle_voxel).astype(np.int64)
            _, keep = np.unique(cells, axis=0, return_index=True)
            pts = pts[np.sort(keep)]
        if len(pts) > self.cfg.max_obstacle_points:
            pts = pts[np.linspace(0, len(pts) - 1, self.cfg.max_obstacle_points).astype(int)]
        self.obstacles = pts
        self.tree = cKDTree(pts) if len(pts) else None

    # ------------------------------------------------------------------ #
    # Path progress (local window)
    # ------------------------------------------------------------------ #
    def update_progress(self, xy: np.ndarray) -> int:
        """Advance `progress_idx` to the nearest path point inside the search window.

        Only indices in [progress_idx - search_back, progress_idx + search_forward]
        are considered, and progress_idx never decreases. At a self-crossing the
        other lap's points are far away *in index*, so they are never matched.
        """
        lo = max(0, self.progress_idx - self.cfg.search_back)
        hi = min(len(self.path), self.progress_idx + self.cfg.search_forward + 1)
        nearest = lo + int(np.argmin(np.linalg.norm(self.path[lo:hi] - xy, axis=1)))
        self.progress_idx = max(self.progress_idx, nearest)
        return self.progress_idx

    def path_blocked(self) -> bool:
        """True if any point of the local path window is within collision_radius of
        an obstacle point, i.e. the robot cannot drive along the path itself."""
        if self.tree is None:
            return False
        start, end = self.window()
        d, _ = self.tree.query(self.path[start:end],
                               distance_upper_bound=self.cfg.collision_radius)
        return bool(np.isfinite(d).any())

    def window(self) -> tuple:
        """(start, end) indices of the local path window used by the costs."""
        start = self.progress_idx
        end = min(len(self.path), start + self.cfg.window_length)
        return start, end

    # ------------------------------------------------------------------ #
    # Sampling and cost
    # ------------------------------------------------------------------ #
    def sample_noise(self, K: int, T: int) -> np.ndarray:
        """Time-correlated Gaussian noise eps, shape (K, T, 2).

        Plain white noise (independent at each of the T steps) averages out over
        the horizon: a sample's mean speed has std sigma / sqrt(T), so once the
        robot has slowed down (U ~ 0) almost no sample drives forward far enough to
        find a way around an obstacle, and the robot freezes.
        Instead we draw N(0, sigma) at `noise_knots` evenly spaced knots and linearly
        interpolate between them. Each sample is then a smooth manoeuvre (e.g. "turn
        right for 1 s, then straighten") with the full sigma at every step.
        """
        n = max(2, min(self.cfg.noise_knots, T))
        knots = self.rng.normal(size=(K, n, 2)) * self.sigma
        t_knot = np.linspace(0.0, T - 1, n)
        idx = np.clip(np.searchsorted(t_knot, np.arange(T), side='right') - 1, 0, n - 2)
        frac = ((np.arange(T) - t_knot[idx]) / (t_knot[idx + 1] - t_knot[idx]))[None, :, None]
        return knots[:, idx] * (1.0 - frac) + knots[:, idx + 1] * frac

    def build_primitives(self) -> np.ndarray:
        """Fixed "turn and go" control sequences, shape (P, T, 2).

        Full speed, turning at a constant rate for n steps, then either straight on or
        turning back at half the rate (to come back toward the path), for rates evenly
        spread over [-primitive_rate_max, +primitive_rate_max] and each n in
        primitive_turn_steps.

        Why: to get round a wide obstacle the robot has to commit to one side early.
        Random samples (especially from U ~ 0 when the robot has slowed down) almost
        never contain such a decisive swerve, so the robot can freeze in front of it
        (Case 2: stuck for 68 s). Including these few deterministic options fixes that;
        they are scored like any other sample, so they only win when they are best.
        """
        c = self.cfg
        if c.primitive_rates <= 0:
            return np.zeros((0, c.horizon, 2))
        prims = []
        for rate in np.linspace(-c.primitive_rate_max, c.primitive_rate_max, c.primitive_rates):
            for n in c.primitive_turn_steps:
                for back in (0.0, -0.5 * rate):
                    U = np.empty((c.horizon, 2))
                    U[:, 0] = c.v_max
                    U[:n, 1] = rate
                    U[n:, 1] = back
                    prims.append(U)
        return np.clip(np.array(prims), self.u_lo, self.u_hi)

    def sample_controls(self) -> np.ndarray:
        """K control sequences (K, T, 2) around the nominal sequence U.

        * sample 0 is U itself (no noise), so the current plan is always kept as an option;
        * sample 1 is all zeros (stop / brake);
        * the next P samples are the fixed "turn and go" primitives (build_primitives);
        * the first `exploration_ratio` of the rest are noise around ZERO instead of
          around U. These let the robot discover very different manoeuvres (rotate in
          place, back off) that small noise around U would never reach.
        """
        K, T = self.cfg.num_samples, self.cfg.horizon
        P = len(self.primitives)
        eps = self.sample_noise(K, T)
        base = np.broadcast_to(self.U, (K, T, 2)).copy()
        n_explore = int(self.cfg.exploration_ratio * K)
        base[2 + P:2 + P + n_explore] = 0.0
        V = np.clip(base + eps, self.u_lo, self.u_hi)
        V[0] = np.clip(self.U, self.u_lo, self.u_hi)
        V[1] = 0.0
        V[2:2 + P] = self.primitives
        return V

    def project_rollouts(self, xy: np.ndarray, win: np.ndarray) -> tuple:
        """Match every rollout point to a path point, moving forward along the path only.

        Rollout step t may only match window points idx[t-1] .. idx[t-1] + rollout_search
        (idx[-1] = 0, the robot's own progress point). So a rollout's position along
        the path advances continuously, as it would for a robot really driving the path.
        A rollout that cuts across a self-crossing loop cannot "teleport" to the far
        side of the loop: it stays matched to the start of the loop, gets a large
        tracking error and no progress. This is the per-rollout version of the
        monotonic progress_idx.

        Loops over the T time steps (34), vectorised over the K samples.

        Args:
            xy: (K, T, 2) rollout positions.   win: (W, 2) local path window.
        Returns:
            (idx (K, T) window index per point, dist (K, T) distance to it)
        """
        K, T, _ = xy.shape
        W = len(win)
        steps = np.arange(self.cfg.rollout_search + 1)                     # (m+1,)
        rows = np.arange(K)
        idx = np.zeros(K, dtype=int)
        out_idx = np.empty((K, T), dtype=int)
        out_d = np.empty((K, T))
        for t in range(T):
            cand = np.minimum(idx[:, None] + steps, W - 1)                 # (K, m+1)
            d2 = ((xy[:, t, None, :] - win[cand]) ** 2).sum(-1)            # (K, m+1)
            j = d2.argmin(axis=1)
            idx = cand[rows, j]
            out_idx[:, t] = idx
            out_d[:, t] = np.sqrt(d2[rows, j])
        return out_idx, out_d

    @staticmethod
    def segment_fraction(p: np.ndarray, win: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """How far (0..1) each point p (K, 2) lies along the segment win[idx] -> win[idx+1].

        Projection of p onto the segment: f = (p - a).(b - a) / |b - a|^2, clipped to [0, 1).
        Zero at the last window point (there is no next segment).
        """
        nxt = np.minimum(idx + 1, len(win) - 1)
        a, b = win[idx], win[nxt]
        ab = b - a
        f = ((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-12)
        return np.where(nxt > idx, np.clip(f, 0.0, 0.999), 0.0)

    def evaluate_costs(self, states: np.ndarray, V: np.ndarray) -> tuple:
        """Total cost S_k of every sample, and whether it collides.

        Args:
            states: (K, T, 3) rollouts.  V: (K, T, 2) their controls.
        Returns:
            (costs (K,), collided (K,) bool)
        """
        c = self.cfg
        K, T, _ = states.shape
        xy = states[..., :2]
        start, end = self.window()
        win = self.path[start:end]                                    # (W, 2)

        # --- If the path ahead is blocked, following it exactly is impossible, so the
        #     tracking and heading terms are scaled down. Otherwise moving sideways
        #     (needed to get round a wide obstacle) costs tracking but earns no progress,
        #     and the robot freezes in front of the obstacle (a local minimum).
        #     Nav2's MPPI does the same: its path critics switch off when the path is occupied.
        blocked = self.path_blocked()
        track_scale = c.blocked_scale if blocked else 1.0

        # --- tracking: distance from each rollout point to "its" path point.
        nearest, d_track = self.project_rollouts(xy, win)                  # (K, T) each
        cost = track_scale * c.w_track * d_track.sum(axis=1)

        # --- heading: misalignment with the path tangent at that nearest point
        yaw_err = states[..., 2] - self.path_yaw[start + nearest]
        cost += track_scale * c.w_heading * (1.0 - np.cos(yaw_err)).sum(axis=1)

        # --- progress: reward how far along the window the terminal state got.
        #     Without this, stopping in front of a long obstacle (Case 3) is "cheap".
        #     Capped at the progress of driving the path at full speed, so cutting a
        #     corner (which reaches a later index sooner) earns nothing extra.
        #     Continuous (index + fraction along the next segment): with a whole-number
        #     index, two almost identical samples could differ by a full w_progress,
        #     and the weights would collapse onto a single sample.
        progress = nearest[:, -1] + self.segment_fraction(xy[:, -1], win, nearest[:, -1])
        cost -= c.w_progress * np.minimum(progress, self.max_progress)

        # --- terminal: once the goal is inside the window, pull the final state onto
        #     the last path point so the robot comes to rest there. (Earlier on, a
        #     target point ahead could lie across a small loop and invite a shortcut;
        #     tracking + progress already do the path following.)
        if end == len(self.path):
            cost += c.w_terminal * np.linalg.norm(xy[:, -1] - self.path[-1], axis=1)

        # --- obstacles
        collided = np.zeros(K, dtype=bool)
        if self.tree is not None:
            # distance to the nearest obstacle point; inf if farther than inflation_radius
            d_obs, _ = self.tree.query(xy.reshape(-1, 2),
                                       distance_upper_bound=c.inflation_radius)
            d_obs = d_obs.reshape(K, T)
            hit = d_obs < c.collision_radius                               # (K, T)
            near = np.isfinite(d_obs) & ~hit
            # smooth repulsion: w_obstacle at the collision radius, decaying outwards
            repulse = np.where(
                near, c.w_obstacle * np.exp(-(d_obs - c.collision_radius) / c.obstacle_decay), 0.0)
            cost += repulse.sum(axis=1)
            # collision: a big constant (so any colliding sample gets weight ~0) plus a
            # small part that grows with the number of colliding steps (so that if
            # EVERY sample collides, the ones that collide least still win).
            n_hit = hit.sum(axis=1)
            collided = n_hit > 0
            cost += c.w_collision * (collided + n_hit / T)

        # --- control effort and smoothness (first difference is vs. the last command sent)
        cost += c.w_v * (V[..., 0] ** 2).sum(axis=1) + c.w_w * (V[..., 1] ** 2).sum(axis=1)
        dV = np.diff(np.concatenate([np.broadcast_to(self.last_cmd, (K, 1, 2)), V], axis=1),
                     axis=1)
        cost += c.w_dv * (dV[..., 0] ** 2).sum(axis=1) + c.w_dw * (dV[..., 1] ** 2).sum(axis=1)
        return cost, collided

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def compute(self, state: np.ndarray) -> MPPIResult:
        """Run one MPPI cycle from `state` = (x, y, theta) and return the command."""
        t0 = time.perf_counter()
        c = self.cfg
        state = np.asarray(state, dtype=float)
        self.update_progress(state[:2])

        # Goal: last part of the path reached and close to the final point.
        at_end = self.progress_idx >= len(self.path) - 1 - c.search_back
        if self.goal_reached or (
                at_end and np.linalg.norm(state[:2] - self.path[-1]) < c.goal_tolerance):
            self.goal_reached = True
            self.last_cmd = np.zeros(2)
            return self._result(0.0, 0.0, state, all_collide=False, t0=t0)

        V = self.sample_controls()                         # 1. sample   (K, T, 2)
        states = rollout(state, V, c.dt)                   # 2. simulate (K, T, 3)
        costs, collided = self.evaluate_costs(states, V)   # 3. score    (K,)

        if collided.all():
            # Every option hits something: back away from the nearest obstacle.
            self.U[:] = 0.0
            v, w = self._recovery_command(state)
            self.last_cmd = np.array([v, w])
            return self._result(v, w, state, all_collide=True, t0=t0)

        weights = compute_weights(costs, c.temperature)    # 4. weights  (K,), sum = 1
        U_new = np.einsum('k,ktj->tj', weights, V)         # 5. U = sum_k w_k V_k
        if c.use_savgol and c.horizon >= c.savgol_window:
            U_new = savgol_filter(U_new, c.savgol_window, c.savgol_order, axis=0)
        U_new = np.clip(U_new, self.u_lo, self.u_hi)

        v, w = float(U_new[0, 0]), float(U_new[0, 1])      # 6. apply the first control
        self.last_cmd = U_new[0].copy()
        optimal = rollout(state, U_new, c.dt)
        # warm start for the next cycle: shift by one control period (in model steps)
        self.U = shift_nominal(U_new, (1.0 / c.control_rate) / c.dt)

        vis_idx = self.rng.choice(len(V), size=min(c.num_vis_samples, len(V)), replace=False)
        return MPPIResult(
            v=v, w=w, goal_reached=False, all_collide=False, progress_idx=self.progress_idx,
            optimal_traj=optimal, sampled_trajs=states[vis_idx], min_cost=float(costs.min()),
            ess=float(1.0 / np.sum(weights ** 2)),
            compute_ms=(time.perf_counter() - t0) * 1e3)

    # ------------------------------------------------------------------ #
    def _recovery_command(self, state: np.ndarray) -> tuple:
        """Slow move away from the nearest obstacle point: reverse if it is in front,
        creep forward if it is behind."""
        _, i = self.tree.query(state[:2])
        dx, dy = self.obstacles[i] - state[:2]
        in_front = dx * np.cos(state[2]) + dy * np.sin(state[2]) > 0.0
        v = -self.cfg.recovery_speed if in_front else self.cfg.recovery_speed
        return float(np.clip(v, self.cfg.v_min, self.cfg.v_max)), 0.0

    def _result(self, v: float, w: float, state: np.ndarray, all_collide: bool,
                t0: float) -> MPPIResult:
        """Result for the special cases (goal / recovery): no sampling statistics."""
        traj = rollout(state, np.tile([v, w], (self.cfg.horizon, 1)), self.cfg.dt)
        return MPPIResult(v=v, w=w, goal_reached=self.goal_reached, all_collide=all_collide,
                          progress_idx=self.progress_idx, optimal_traj=traj,
                          sampled_trajs=np.zeros((0, self.cfg.horizon, 3)),
                          compute_ms=(time.perf_counter() - t0) * 1e3)

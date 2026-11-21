"""Minimal 2D world for offline MPPI testing: box obstacles, a ray-cast lidar
and a unicycle robot. No ROS, no physics engine.

The lidar mimics the TurtleBot3 LDS as seen in Gazebo: 360 beams over
[0, 2*pi), range 0.12-3.5 m, ~5 Hz, mounted 0.032 m behind the robot centre.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Box:
    """Rectangle obstacle: centre (cx, cy), size (sx along x, sy along y), rotation yaw."""

    cx: float
    cy: float
    sx: float
    sy: float
    yaw: float = 0.0
    t_added: float = 0.0        # sim time at which it appeared (dynamic obstacles)

    def corners(self) -> np.ndarray:
        """(4, 2) corner points, counter-clockwise."""
        hx, hy = self.sx / 2.0, self.sy / 2.0
        local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return local @ np.array([[c, s], [-s, c]]) + (self.cx, self.cy)

    def segments(self) -> np.ndarray:
        """(4, 4) edges as rows of (x1, y1, x2, y2)."""
        p = self.corners()
        return np.hstack([p, np.roll(p, -1, axis=0)])


@dataclass
class LidarModel:
    """Parameters of the simulated 2D lidar."""

    n_beams: int = 360
    angle_min: float = 0.0
    range_min: float = 0.12
    range_max: float = 3.5
    noise_std: float = 0.01      # [m] Gaussian range noise
    offset_x: float = -0.032     # [m] sensor position in the robot frame (from the TB3 TF)
    rate: float = 5.0            # [Hz]

    @property
    def angle_increment(self) -> float:
        return 2.0 * np.pi / self.n_beams


class World:
    """Holds the obstacles; answers ray-casts and distance queries."""

    def __init__(self, boxes: Optional[List[Box]] = None) -> None:
        self.boxes: List[Box] = list(boxes or [])

    def add_box(self, box: Box) -> None:
        self.boxes.append(box)

    def _segments(self) -> np.ndarray:
        if not self.boxes:
            return np.zeros((0, 4))
        return np.vstack([b.segments() for b in self.boxes])

    def raycast(self, origin: Tuple[float, float], yaw: float, lidar: LidarModel,
                rng: np.random.Generator) -> np.ndarray:
        """Range of every beam (inf = no return), like sensor_msgs/LaserScan.ranges.

        Ray:      o + t * d          (t >= 0)
        Segment:  p + u * (q - p)    (0 <= u <= 1)
        Solving with 2D cross products (a x b = ax*by - ay*bx):
            t = ((p - o) x s) / (d x s),   u = ((p - o) x d) / (d x s),   s = q - p
        All beams x all segments are solved at once with broadcasting.
        """
        angles = yaw + lidar.angle_min + lidar.angle_increment * np.arange(lidar.n_beams)
        segs = self._segments()
        ranges = np.full(lidar.n_beams, np.inf)
        if len(segs):
            d = np.column_stack((np.cos(angles), np.sin(angles)))[:, None, :]   # (B, 1, 2)
            p, s = segs[None, :, :2], (segs[:, 2:] - segs[:, :2])[None]       # (1, S, 2)
            po = p - np.asarray(origin)                                       # (1, S, 2)

            def cross(a, b):
                return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

            denom = cross(d, s)                                               # (B, S)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = cross(po, s) / denom
                u = cross(po, d) / denom
            t = np.where((np.abs(denom) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1), t, np.inf)
            ranges = t.min(axis=1)
        ranges = ranges + rng.normal(0.0, lidar.noise_std, lidar.n_beams)
        ranges[(ranges > lidar.range_max) | (ranges < lidar.range_min)] = np.inf
        return ranges

    def distance(self, xy: np.ndarray) -> float:
        """Exact distance from a point to the nearest obstacle surface (inf if none).
        A point inside a box returns 0."""
        segs = self._segments()
        if not len(segs):
            return np.inf
        p, q = segs[:, :2], segs[:, 2:]
        pq = q - p
        u = np.clip(((xy - p) * pq).sum(1) / (pq ** 2).sum(1), 0.0, 1.0)
        d = np.linalg.norm(p + u[:, None] * pq - xy, axis=1).min()
        for b in self.boxes:                      # inside test (box frame)
            c, s = np.cos(b.yaw), np.sin(b.yaw)
            lx = c * (xy[0] - b.cx) + s * (xy[1] - b.cy)
            ly = -s * (xy[0] - b.cx) + c * (xy[1] - b.cy)
            if abs(lx) <= b.sx / 2 and abs(ly) <= b.sy / 2:
                return 0.0
        return float(d)


def step_unicycle(state: np.ndarray, v: float, w: float, dt: float,
                  v_limit: float = 0.22, w_limit: float = 2.84, substeps: int = 10) -> np.ndarray:
    """Advance the 'real' robot by dt with the commanded (v, w), clipped to the
    burger's limits. Uses small substeps so it is more accurate than the MPPI model."""
    v = float(np.clip(v, -v_limit, v_limit))
    w = float(np.clip(w, -w_limit, w_limit))
    x, y, th = state
    h = dt / substeps
    for _ in range(substeps):
        x += v * np.cos(th) * h
        y += v * np.sin(th) * h
        th += w * h
    return np.array([x, y, np.arctan2(np.sin(th), np.cos(th))])

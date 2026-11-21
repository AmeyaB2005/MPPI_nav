"""Pure-Python helpers: path CSV I/O, path resampling and small geometry functions.

Nothing in this file imports ROS, so the offline simulator (sim2d/) and the
unit tests can use it without a ROS environment. ROS marker helpers live in
markers.py instead.

CSV format (same as the reference smoothing-rpp repo):
    x,y
    0.0,0.0
    0.1,0.0
    ...
"""
from __future__ import annotations

import math
import os
from typing import Iterable, List, Tuple

import numpy as np

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
# CSV I/O
# --------------------------------------------------------------------------- #
def load_path_csv(file_path: str) -> np.ndarray:
    """Load an (N, 2) array of (x, y) points from a CSV file.

    Tolerates what the reference repo's waypoint.csv contains: an optional
    `x,y` header on the first line, spaces around values ("5.0, 4.0"), blank
    lines and a missing trailing newline. Any other unparsable line is an error,
    so a corrupt file fails loudly instead of being silently truncated.
    """
    points: List[Point] = []
    with open(os.path.expanduser(file_path), 'r', encoding='utf-8') as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            fields = [s.strip() for s in line.split(',')]
            try:
                points.append((float(fields[0]), float(fields[1])))
            except (ValueError, IndexError):
                if not points and line_no == 1:
                    continue  # header line, e.g. "x,y"
                raise ValueError(f'{file_path}:{line_no}: cannot parse "{line}" as x,y')
    if not points:
        raise ValueError(f'{file_path}: no points found')
    return np.asarray(points, dtype=float)


def save_path_csv(file_path: str, points: Iterable[Point]) -> str:
    """Write points to CSV with an `x,y` header. Creates parent folders.

    Returns the absolute path written.
    """
    file_path = os.path.abspath(os.path.expanduser(file_path))
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('x,y\n')
        for x, y in points:
            # round(...) + 0.0 turns -0.0 into 0.0 so the file never shows "-0.0000"
            f.write(f'{round(x, 4) + 0.0:.4f},{round(y, 4) + 0.0:.4f}\n')
    return file_path


# --------------------------------------------------------------------------- #
# Path geometry
# --------------------------------------------------------------------------- #
def points_at_spacing(last: Point, current: Point, spacing: float) -> List[Point]:
    """New points placed exactly `spacing` apart, walking from `last` toward `current`.

    Used by the path recorder. Odometry arrives every ~5 mm of travel, so simply
    saving a pose once it is >= 0.1 m away would give gaps of 0.100-0.105 m.
    Instead we step along the straight line last -> current in exact `spacing`
    increments. Each new point becomes the start of the next step, so all
    consecutive saved points are exactly `spacing` apart.

    Returns an empty list if `current` is closer than `spacing` to `last`.
    """
    new_points: List[Point] = []
    lx, ly = last
    while True:
        dx, dy = current[0] - lx, current[1] - ly
        dist = math.hypot(dx, dy)
        if dist < spacing:
            return new_points
        lx += spacing * dx / dist
        ly += spacing * dy / dist
        new_points.append((lx, ly))


def resample_path(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline to (approximately) uniform arc-length `spacing`.

    Used to turn sparse waypoints (e.g. the reference repo's 4-point CSV) into a
    dense path. Keeps the first and last point exactly. Duplicate consecutive
    points are dropped first so the arc-length is strictly increasing.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate(([True], seg > 1e-9))
    pts = pts[keep]
    if len(pts) < 2:
        return pts.copy()
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    n_steps = max(1, int(round(s[-1] / spacing)))
    s_new = np.linspace(0.0, s[-1], n_steps + 1)
    return np.column_stack((np.interp(s_new, s, pts[:, 0]), np.interp(s_new, s, pts[:, 1])))


def spacing_stats(points: np.ndarray) -> Tuple[float, float, float]:
    """(min, mean, max) distance between consecutive points."""
    d = np.linalg.norm(np.diff(np.asarray(points, dtype=float), axis=0), axis=1)
    if d.size == 0:
        return (0.0, 0.0, 0.0)
    return (float(d.min()), float(d.mean()), float(d.max()))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Yaw (rotation about z) of a quaternion, in radians, range (-pi, pi]."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def scan_to_points(ranges: np.ndarray, angle_min: float, angle_increment: float,
                   range_min: float, range_max: float,
                   sensor_pose: Tuple[float, float, float]) -> np.ndarray:
    """Convert a 2D laser scan into (M, 2) obstacle points in the world frame.

    Beams that are inf/nan or outside [range_min, range_max] are dropped.
    `sensor_pose` = (x, y, yaw) of the laser in the world frame.
    Shared by the ROS node and the offline simulator, so both see identical input.
    """
    r = np.asarray(ranges, dtype=float)
    angles = angle_min + angle_increment * np.arange(len(r))
    valid = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    r, angles = r[valid], angles[valid] + sensor_pose[2]
    return np.column_stack((sensor_pose[0] + r * np.cos(angles),
                            sensor_pose[1] + r * np.sin(angles)))


def load_mppi_params(yaml_file: str, node_name: str = 'mppi_tracker') -> dict:
    """Read the `ros__parameters` block of a ROS 2 params YAML as a plain dict."""
    import yaml  # local import: only needed by the offline tools
    with open(os.path.expanduser(yaml_file), 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return dict(data[node_name]['ros__parameters'])


def resolve_path(file_name: str, base_dir: str) -> str:
    """Absolute path for `file_name`: kept if absolute (or ~), else joined to `base_dir`."""
    expanded = os.path.expanduser(file_name)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(os.path.expanduser(base_dir), expanded)

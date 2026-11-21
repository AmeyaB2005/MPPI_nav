"""RViz helpers: build nav_msgs/Path and visualization_msgs/Marker messages.

Colour convention (from the assignment figure):
    recorded path  = green
    traversed path = red
    obstacles      = blue
"""
from __future__ import annotations

from typing import Sequence, Tuple

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

RGBA = Tuple[float, float, float, float]

GREEN: RGBA = (0.0, 0.9, 0.0, 1.0)
RED: RGBA = (1.0, 0.0, 0.0, 1.0)
BLUE: RGBA = (0.1, 0.3, 1.0, 1.0)
YELLOW: RGBA = (1.0, 0.9, 0.0, 1.0)
GREY: RGBA = (0.6, 0.6, 0.6, 0.4)


def make_path_msg(points: Sequence[Tuple[float, float]], frame_id: str, stamp: Time) -> Path:
    """nav_msgs/Path from (x, y) points. Orientation is left as identity."""
    msg = Path()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    for x, y in points:
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
    return msg


def make_marker(points: Sequence[Tuple[float, float]], frame_id: str, stamp: Time,
                ns: str, marker_id: int, marker_type: int, color: RGBA,
                scale: float, z: float = 0.02) -> Marker:
    """Generic Marker (LINE_STRIP, POINTS, SPHERE_LIST, ...) from (x, y) points.

    `scale` is the line width (LINE_STRIP) or the point diameter (POINTS / SPHERE_LIST).
    `z` lifts the marker slightly above the ground so it is not hidden by the grid.
    """
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = marker_type
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = scale
    m.scale.y = scale
    m.scale.z = scale
    m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
    m.points = [Point(x=float(x), y=float(y), z=z) for x, y in points]
    return m

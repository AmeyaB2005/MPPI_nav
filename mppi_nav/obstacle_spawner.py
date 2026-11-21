"""Dynamic obstacle spawner: drops boxes on the path in front of the moving robot.

This is the test harness for the "objects dropped randomly in front of the robot"
part of the assignment. It is not part of the controller: the tracker only ever
sees these boxes through the lidar, exactly like any other obstacle.

Where a box goes
----------------
On the recorded path, `ahead_min`..`ahead_max` metres in front of the robot
(measured ALONG the path, so it really blocks the route on curves too), shifted
sideways by up to `lateral_max`, rotated to the path direction, random size.

When
----
  mode auto   : every `period` seconds, but only while the robot is moving
                (so nothing is dropped during the tracker's start hold)
  /drop_obstacle (std_srvs/Trigger): drop one now, in either mode
At most `max_boxes` dropped boxes exist at once; the oldest is deleted.
A spot is skipped (auto mode tries again 0.5 s later, further along) if it is
  * closer than `min_end_distance` to the end of the path, or
  * closer than `crossing_clearance` to ANOTHER part of the path: near a crossing
    or a tight bend one box would block two passes of the path at once, leaving
    no reasonable way round (seen on a hand-recorded path with sharp turns).

How
---
Gazebo's create/remove services, bridged to ROS by ros_gz_bridge
(ros_gz_interfaces/srv/SpawnEntity and DeleteEntity), called asynchronously.

Frames
------
The robot pose comes from the same odometry topic as the tracker (odom frame,
origin = spawn pose). The path CSV and Gazebo use the world frame, so the pose
is converted with the spawn pose (start_x, start_y, start_yaw):
    p_world = start + R(start_yaw) * p_odom
"""
from __future__ import annotations

import math
import signal
from collections import deque
from typing import Deque, Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Trigger

from mppi_nav.utils import load_path_csv, resample_path, spacing_stats

PATH_SPACING = 0.1          # [m] the path is resampled to this if it is sparser
PROGRESS_BACK = 5           # [points] progress search window, as in mppi_core
PROGRESS_FORWARD = 10


def box_sdf(name: str, size: float, height: float) -> str:
    """SDF of a static blue box (its pose is given separately in the spawn request)."""
    geom = f'<geometry><box><size>{size:.3f} {size:.3f} {height:.3f}</size></box></geometry>'
    return (f'<sdf version="1.8"><model name="{name}"><static>true</static>'
            f'<link name="link"><collision name="collision">{geom}</collision>'
            f'<visual name="visual">{geom}<material><ambient>0.1 0.3 1 1</ambient>'
            f'<diffuse>0.1 0.3 1 1</diffuse></material></visual></link></model></sdf>')


class ObstacleSpawner(Node):
    """Drops boxes on the path ahead of the robot, automatically or on request."""

    def __init__(self) -> None:
        super().__init__('obstacle_spawner')
        any_type = ParameterDescriptor(dynamic_typing=True)
        defaults = {
            'path_file': '', 'start_x': 0.0, 'start_y': 0.0, 'start_yaw': 0.0,
            'odom_topic': '/odometry/filtered',
            'mode': 'auto', 'period': 8.0, 'ahead_min': 0.8, 'ahead_max': 1.5,
            'lateral_max': 0.15, 'size_min': 0.25, 'size_max': 0.4, 'height': 0.4,
            'max_boxes': 3, 'min_end_distance': 1.0, 'crossing_clearance': 0.8,
            'min_speed': 0.05,
            'world': 'default', 'seed': -1,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value, any_type)
        self.p = {name: self.get_parameter(name).value for name in defaults}

        path = load_path_csv(self.p['path_file'])
        if spacing_stats(path)[2] > 1.5 * PATH_SPACING:
            path = resample_path(path, PATH_SPACING)
        self.path = path                                       # world frame
        d = np.diff(path, axis=0)
        self.path_yaw = np.append(np.arctan2(d[:, 1], d[:, 0]), math.atan2(d[-1, 1], d[-1, 0]))
        self.spacing = float(np.linalg.norm(d, axis=1).mean())

        seed = int(self.p['seed'])
        self.rng = np.random.default_rng(None if seed < 0 else seed)
        self.robot_xy: Optional[np.ndarray] = None             # world frame
        self.speed = 0.0
        self.progress = 0
        self.count = 0
        self.boxes: Deque[str] = deque()
        self.last_drop = None

        world = self.p['world']
        self.spawn_cli = self.create_client(SpawnEntity, f'/world/{world}/create')
        self.delete_cli = self.create_client(DeleteEntity, f'/world/{world}/remove')
        self.create_subscription(Odometry, self.p['odom_topic'], self.odom_callback, 1)
        self.create_service(Trigger, '/drop_obstacle', self.drop_service_callback)
        self.create_timer(0.5, self.auto_drop)
        self.get_logger().info(
            f"Obstacle spawner ready (mode={self.p['mode']}, every {self.p['period']} s while "
            f"moving). Drop one now: ros2 service call /drop_obstacle std_srvs/srv/Trigger")

    # ------------------------------------------------------------------ #
    def odom_callback(self, msg: Odometry) -> None:
        """Robot pose -> world frame; advance the (monotonic) progress index on the path."""
        sx, sy, syaw = self.p['start_x'], self.p['start_y'], self.p['start_yaw']
        ox, oy = msg.pose.pose.position.x, msg.pose.pose.position.y
        c, s = math.cos(syaw), math.sin(syaw)
        self.robot_xy = np.array([sx + c * ox - s * oy, sy + s * ox + c * oy])
        self.speed = abs(msg.twist.twist.linear.x)
        lo = max(0, self.progress - PROGRESS_BACK)
        hi = min(len(self.path), self.progress + PROGRESS_FORWARD + 1)
        nearest = lo + int(np.argmin(np.linalg.norm(self.path[lo:hi] - self.robot_xy, axis=1)))
        self.progress = max(self.progress, nearest)

    def auto_drop(self) -> None:
        if self.p['mode'] != 'auto' or self.robot_xy is None or self.speed < self.p['min_speed']:
            return
        now = self.get_clock().now()
        if self.last_drop is None:
            self.last_drop = now          # first drop one period after the robot starts moving
            return
        if (now - self.last_drop).nanoseconds * 1e-9 >= float(self.p['period']):
            self.drop()

    def drop_service_callback(self, _req: Trigger.Request,
                              resp: Trigger.Response) -> Trigger.Response:
        resp.message = self.drop()
        resp.success = resp.message.startswith('Dropped')
        return resp

    def near_other_pass(self, idx: int, xy: np.ndarray) -> bool:
        """True if a part of the path that is far away ALONG the path (not just the
        neighbouring points) comes within crossing_clearance of xy."""
        clearance = float(self.p['crossing_clearance'])
        far = np.abs(np.arange(len(self.path)) - idx) * self.spacing > 2.0 * clearance
        return bool((np.linalg.norm(self.path[far] - xy, axis=1) < clearance).any())

    # ------------------------------------------------------------------ #
    def drop(self) -> str:
        """Spawn one box on the path ahead of the robot. Returns a status message.
        `last_drop` is only updated on success, so auto mode retries a skipped spot."""
        if self.robot_xy is None:
            return self.report('Not dropped: no robot pose yet')
        if not self.spawn_cli.service_is_ready():
            return self.report('Not dropped: Gazebo create service not available')
        ahead = self.rng.uniform(self.p['ahead_min'], self.p['ahead_max'])
        idx = self.progress + int(round(ahead / self.spacing))
        remaining = (len(self.path) - 1 - idx) * self.spacing
        if remaining < self.p['min_end_distance']:
            return self.report('Not dropped: too close to the end of the path', skip=True)

        lateral = self.rng.uniform(-self.p['lateral_max'], self.p['lateral_max'])
        yaw = float(self.path_yaw[idx])
        x = float(self.path[idx, 0] - lateral * math.sin(yaw))
        y = float(self.path[idx, 1] + lateral * math.cos(yaw))
        if self.near_other_pass(idx, np.array([x, y])):
            return self.report('Not dropped: too close to another pass of the path '
                               '(crossing / tight bend)', skip=True)
        size = float(self.rng.uniform(self.p['size_min'], self.p['size_max']))
        self.last_drop = self.get_clock().now()
        self.count += 1
        name = f'dropped_box_{self.count}'

        req = SpawnEntity.Request()
        req.entity_factory.name = name
        req.entity_factory.sdf = box_sdf(name, size, float(self.p['height']))
        req.entity_factory.pose.position.x = x
        req.entity_factory.pose.position.y = y
        req.entity_factory.pose.position.z = float(self.p['height']) / 2.0
        req.entity_factory.pose.orientation.z = math.sin(yaw / 2.0)
        req.entity_factory.pose.orientation.w = math.cos(yaw / 2.0)
        self.spawn_cli.call_async(req)
        self.boxes.append(name)
        while len(self.boxes) > int(self.p['max_boxes']):
            self.remove(self.boxes.popleft())
        return self.report(f'Dropped {name}: {size:.2f} m box, {ahead:.2f} m ahead along the '
                           f'path, {lateral:+.2f} m sideways, at ({x:.2f}, {y:.2f})')

    def remove(self, name: str) -> None:
        if not self.delete_cli.service_is_ready():
            return
        req = DeleteEntity.Request()
        req.entity.name = name
        req.entity.type = Entity.MODEL
        self.delete_cli.call_async(req)
        self.get_logger().info(f'Removed {name} (max_boxes={self.p["max_boxes"]})')

    def report(self, msg: str, skip: bool = False) -> str:
        """Log a status message; skips are rate-limited (auto mode retries every 0.5 s)."""
        if skip:
            self.get_logger().info(msg, throttle_duration_sec=5.0)
        else:
            self.get_logger().info(msg)
        return msg


def main(args: list | None = None) -> None:
    rclpy.init(args=args)
    node = ObstacleSpawner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Ctrl+C reaches a node twice (the terminal signals the whole process group and
        # ros2 launch forwards its own SIGINT). Ignore the second one so it cannot
        # interrupt the clean-up below (e.g. writing the CSV).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

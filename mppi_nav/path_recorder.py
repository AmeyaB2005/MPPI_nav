"""Path recorder node.

Drive the robot with teleop; this node records the robot's (x, y) positions as
points exactly `spacing` metres apart (default 0.1 m) and writes them to CSV.

The pose comes from `odom_topic`: by default the EKF estimate /odometry/filtered
(wheels + IMU gyro), the same source the MPPI tracker uses when replaying, so a
recorded path and its replay are in the same frame. Odometry starts at (0, 0, 0)
at the robot's spawn pose, so points are converted to the world frame with the
spawn pose (start_x, start_y, start_yaw):  p_world = start + R(start_yaw) * p_odom

Saving happens
  * when the node shuts down (Ctrl+C), and
  * whenever the /save_path service (std_srvs/Trigger) is called.

Live visualisation
  * /recorded_path    nav_msgs/Path (transient_local, so late RViz still gets it)
  * /markers/recorded visualization_msgs/Marker, green LINE_STRIP

Parameters
  odom_topic    (str)   Pose source.                          [/odometry/filtered]
  start_x/y/yaw (float) Spawn pose in the world frame.        [0, 0, 0]
  output_file   (str)   CSV file to write. Relative paths are resolved against
                        `output_dir`.                          [recorded.csv]
  output_dir    (str)   Folder for relative output files.     [~/ros2_ws/src/mppi_nav/paths]
  spacing       (float) Distance between saved points [m].    [0.1]
  frame_id      (str)   Frame of the recorded points.         [odom]
  publish_rate  (float) Rate of Path/Marker publishing [Hz].  [2.0]
  marker_width  (float) Line width of the green marker [m].   [0.03]
"""
from __future__ import annotations

import math
import signal
from typing import List

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from nav_msgs.msg import Odometry, Path
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

from mppi_nav.markers import GREEN, make_marker, make_path_msg
from mppi_nav.utils import Point, points_at_spacing, resolve_path, save_path_csv, spacing_stats


class PathRecorder(Node):
    """Records odometry positions at fixed spacing and saves them to CSV."""

    def __init__(self) -> None:
        super().__init__('path_recorder')

        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_yaw', 0.0)
        self.declare_parameter('output_file', 'recorded.csv')
        self.declare_parameter('output_dir', '~/ros2_ws/src/mppi_nav/paths')
        self.declare_parameter('spacing', 0.1)
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('marker_width', 0.03)

        self.output_file: str = resolve_path(
            self.get_parameter('output_file').value, self.get_parameter('output_dir').value)
        self.spacing: float = float(self.get_parameter('spacing').value)
        self.frame_id: str = self.get_parameter('frame_id').value
        self.marker_width: float = float(self.get_parameter('marker_width').value)

        self.points: List[Point] = []
        self.start = tuple(float(self.get_parameter(n).value)
                           for n in ('start_x', 'start_y', 'start_yaw'))
        odom_topic = self.get_parameter('odom_topic').value

        # transient_local = "latched": RViz started later still receives the last message.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(Path, '/recorded_path', latched)
        self.marker_pub = self.create_publisher(Marker, '/markers/recorded', latched)

        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.create_service(Trigger, '/save_path', self.save_service_callback)
        self.create_timer(1.0 / float(self.get_parameter('publish_rate').value), self.publish_viz)

        self.get_logger().info(
            f'Recording {odom_topic} every {self.spacing:.2f} m -> {self.output_file}\n'
            '  Drive with:  ros2 run turtlebot3_teleop teleop_keyboard\n'
            '  Save now:    ros2 service call /save_path std_srvs/srv/Trigger\n'
            '  Ctrl+C here also saves.')

    # ------------------------------------------------------------------ #
    def odom_callback(self, msg: Odometry) -> None:
        """Append new points whenever the robot has moved >= spacing from the last one."""
        ox, oy = msg.pose.pose.position.x, msg.pose.pose.position.y
        sx, sy, syaw = self.start
        c, s = math.cos(syaw), math.sin(syaw)
        current = (sx + c * ox - s * oy, sy + s * ox + c * oy)    # odom -> world
        if not self.points:
            self.points.append(current)  # always keep the very first pose
            self.get_logger().info(f'First point: ({current[0]:.3f}, {current[1]:.3f})')
            return
        self.points.extend(points_at_spacing(self.points[-1], current, self.spacing))

    def publish_viz(self) -> None:
        """Publish the path recorded so far as nav_msgs/Path and a green line marker."""
        if not self.points:
            return
        stamp = self.get_clock().now().to_msg()
        # Points are stored in the world frame; RViz shows the odom frame -> convert back.
        sx, sy, syaw = self.start
        c, s = math.cos(syaw), math.sin(syaw)
        shown = [(c * (x - sx) + s * (y - sy), -s * (x - sx) + c * (y - sy)) for x, y in self.points]
        self.path_pub.publish(make_path_msg(shown, self.frame_id, stamp))
        self.marker_pub.publish(make_marker(
            shown, self.frame_id, stamp, ns='recorded', marker_id=0,
            marker_type=Marker.LINE_STRIP, color=GREEN, scale=self.marker_width))

    # ------------------------------------------------------------------ #
    def save(self) -> str:
        """Write the CSV and return a one-line summary."""
        if len(self.points) < 2:
            return f'Not saved: only {len(self.points)} point(s) recorded. Drive the robot first.'
        path = save_path_csv(self.output_file, self.points)
        lo, mean, hi = spacing_stats(self.points)
        return (f'Saved {len(self.points)} points to {path} '
                f'(spacing min/mean/max = {lo:.3f}/{mean:.3f}/{hi:.3f} m)')

    def save_service_callback(self, _request: Trigger.Request,
                              response: Trigger.Response) -> Trigger.Response:
        """/save_path service: save without stopping the recorder."""
        response.message = self.save()
        response.success = response.message.startswith('Saved')
        self.get_logger().info(response.message)
        return response


def main(args: list | None = None) -> None:
    """Spin the recorder; always save on exit (Ctrl+C or launch shutdown)."""
    rclpy.init(args=args)
    node = PathRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C: fall through to the save below
    finally:
        # Ctrl+C reaches a node twice (the terminal signals the whole process group and
        # ros2 launch forwards its own SIGINT). Ignore the second one so it cannot
        # interrupt the clean-up below (e.g. writing the CSV).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # Logging still works after the context is shut down, so this message is shown.
        node.get_logger().info(node.save())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

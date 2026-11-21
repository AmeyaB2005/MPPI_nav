"""MPPI path tracker node: follows a recorded CSV path and avoids lidar obstacles.

Data flow
---------
  odom_topic (nav_msgs/Odometry) -> robot pose, plus a short history for scan timing.
                                    /odom = raw wheel odometry; /odometry/filtered = EKF
                                    (wheels + IMU gyro, see config/ekf.yaml), the default.
  /scan  (sensor_msgs/LaserScan) -> obstacle points in the odom frame
  timer  (control_rate, 20 Hz)   -> MPPIController.compute() -> /cmd_vel (TwistStamped)

The callbacks only store the latest data. The MPPI controller itself is only
touched from the timer callback, so no locks are needed.

Executor: single-threaded on purpose. With use_sim_time, rclpy handles every
/clock message (~1000 Hz from Gazebo) in Python. With a multi-threaded executor
those wake-ups fought the MPPI step for Python's GIL and one step took ~195 ms
(only ~5 Hz). Single-threaded: ~26 ms per step and a steady 20 Hz. /odom and
/scan messages that arrive during a step are queued and handled right after it;
they carry their own timestamps, so nothing is lost.

Frames
------
Gazebo's diff-drive odometry starts at (0, 0, 0) wherever the robot is spawned.
The path CSV is in the world frame of the recording session. So the path is
moved into the odom frame using the spawn pose (start_x, start_y, start_yaw):
    p_odom = R(-start_yaw) * (p_world - start)
For a path recorded with record.launch.py (robot spawned at the origin) and
replayed from its first point, this is consistent automatically.

Published
---------
  /cmd_vel              geometry_msgs/TwistStamped (or Twist, see cmd_vel_stamped)
  /recorded_path        nav_msgs/Path, latched
  /markers/recorded     green line (latched)
  /markers/traversed    red line, a point every traversed_spacing metres
  /markers/obstacles    blue points (current lidar obstacle points)
  /markers/mppi         MarkerArray: optimal rollout (yellow), sampled rollouts (grey),
                        current progress point on the path (magenta)
"""
from __future__ import annotations

import math
import signal
from collections import deque
from dataclasses import fields
from typing import Deque, Optional, Tuple

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from mppi_nav.markers import BLUE, GREEN, GREY, RED, YELLOW, make_marker, make_path_msg
from mppi_nav.mppi_core import MPPIConfig, MPPIController, MPPIResult
from mppi_nav.utils import (load_path_csv, resample_path, save_path_csv, scan_to_points,
                            spacing_stats, yaw_from_quaternion)

Pose2D = Tuple[float, float, float]
MAGENTA = (1.0, 0.0, 1.0, 1.0)


class MPPITracker(Node):
    """ROS wrapper around mppi_core.MPPIController."""

    def __init__(self) -> None:
        super().__init__('mppi_tracker')
        any_type = ParameterDescriptor(dynamic_typing=True)

        # ---- node parameters ----
        p = {
            'path_file': '',
            'frame_id': 'odom',
            'control_rate': 20.0,
            'sensor_timeout': 0.5,          # [s] stop if /odom or /scan is older than this
            'start_x': 0.0,                 # spawn pose in the path's (world) frame
            'start_y': 0.0,
            'start_yaw': 0.0,
            'path_spacing': 0.1,            # [m] resample the CSV if it is sparser than this
            'cmd_vel_stamped': True,        # Jazzy TB3 bridge expects TwistStamped
            'base_frame': 'base_footprint',
            'laser_frame': 'base_scan',
            'odom_history': 2.0,            # [s] pose history kept to place each scan
            'traversed_spacing': 0.05,      # [m] red marker point spacing
            'traversed_file': '',           # optional CSV of the driven path (written at exit)
            'viz_every': 2,                 # publish rollout markers every N control cycles
            'start_delay': 10.0,            # [s] hold still this long after all sensors are live
            'odom_topic': '/odom',          # robot pose source (/odometry/filtered with the EKF)
            'max_latency': 0.2,             # [s] predict the pose forward at most this far
        }
        for name, default in p.items():
            self.declare_parameter(name, default, any_type)
        # ---- every MPPI parameter, with the defaults from MPPIConfig ----
        for f in fields(MPPIConfig):
            if f.name in p:
                continue
            default = getattr(MPPIConfig, f.name) if f.name != 'seed' else -1
            if isinstance(default, tuple):
                default = list(default)              # ROS parameters take lists, not tuples
            self.declare_parameter(f.name, default, any_type)

        get = lambda n: self.get_parameter(n).value  # noqa: E731
        self.frame_id: str = get('frame_id')
        self.sensor_timeout = Duration(seconds=float(get('sensor_timeout')))
        self.traversed_spacing = float(get('traversed_spacing'))
        self.viz_every = int(get('viz_every'))
        self.cmd_stamped = bool(get('cmd_vel_stamped'))
        self.traversed_file: str = get('traversed_file')

        # ---- path: load, densify if needed, move into the odom frame ----
        path_world = load_path_csv(get('path_file'))
        spacing = float(get('path_spacing'))
        if spacing_stats(path_world)[2] > 1.5 * spacing:
            path_world = resample_path(path_world, spacing)
            self.get_logger().info(f'Path was sparse: resampled to {spacing} m spacing')
        self.start: Pose2D = (float(get('start_x')), float(get('start_y')), float(get('start_yaw')))
        self.path = self.world_to_odom(path_world, self.start)

        # ---- controller ----
        cfg_dict = {f.name: get(f.name) for f in fields(MPPIConfig)
                    if self.has_parameter(f.name)}
        cfg_dict['seed'] = None if int(cfg_dict.get('seed', -1)) < 0 else int(cfg_dict['seed'])
        self.cfg = MPPIConfig.from_dict(cfg_dict)
        self.ctrl = MPPIController(self.cfg, self.path)

        # ---- state written by callbacks, read by the timer ----
        self.pose: Optional[Pose2D] = None
        self.last_odom_time: Optional[Time] = None
        self.last_scan_time: Optional[Time] = None
        self.odom_hist: Deque[Tuple[float, Pose2D]] = deque()
        self.new_scan_points: Optional[np.ndarray] = None
        self.laser_offset: Optional[Pose2D] = None      # laser pose in the base frame
        self.traversed: list = []
        self.goal_logged = False
        self.t_start: Optional[Time] = None           # when tracking actually started
        self.t_ready: Optional[Time] = None           # when all sensors were first live
        self.started = False
        self.start_requested = False
        self.t_created = self.get_clock().now()
        self.start_delay = float(get('start_delay'))    # [s]
        self.cycle = 0
        self.compute_ms: list = []
        self.min_lidar_dist = float('inf')   # closest lidar point to the robot centre [m]
        self.pose_stamp = 0.0                 # [s] timestamp of self.pose
        self.max_pose_age = 0.0               # [s] worst pose age seen (latency monitor)
        self.max_latency = float(get('max_latency'))

        # ---- TF (only for the fixed laser -> base offset) ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.base_frame: str = get('base_frame')
        self.laser_frame: str = get('laser_frame')

        # ---- publishers ----
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.create_publisher(TwistStamped if self.cmd_stamped else Twist,
                                             '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/recorded_path', latched)
        self.recorded_pub = self.create_publisher(Marker, '/markers/recorded', latched)
        self.traversed_pub = self.create_publisher(Marker, '/markers/traversed', 10)
        self.obstacle_pub = self.create_publisher(Marker, '/markers/obstacles', 10)
        self.mppi_pub = self.create_publisher(MarkerArray, '/markers/mppi', 10)

        # ---- subscriptions (one group) and control timer (another group) ----
        sensors = MutuallyExclusiveCallbackGroup()
        self.odom_topic: str = get('odom_topic')
        # Queue depth 1: always process the NEWEST message. With a deeper queue, a slow
        # control step (e.g. Gazebo GUI + RViz rendering on the CPU) lets old messages pile
        # up; the pose then lags by up to depth/rate (20 / 30 Hz = 0.66 s), and while the
        # robot turns every lidar point is rotated by the stale heading.
        newest_only = QoSProfile(depth=1, reliability=qos_profile_sensor_data.reliability)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 1,
                                 callback_group=sensors)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, newest_only,
                                 callback_group=sensors)
        self.create_service(Trigger, '/start_tracking', self.start_service_callback)
        self.create_timer(1.0 / float(get('control_rate')), self.control_step,
                          callback_group=MutuallyExclusiveCallbackGroup())

        self.publish_recorded_path()
        lo, mean, hi = spacing_stats(self.path)
        self.get_logger().info(
            f'Loaded {len(self.path)} path points from {get("path_file")} '
            f'(spacing {mean:.3f} m). Spawn pose {self.start}. '
            f'MPPI: K={self.cfg.num_samples}, T={self.cfg.horizon} x {self.cfg.dt}s, '
            f'lambda={self.cfg.temperature}. Robot holds still until /odom, /scan and TF are '
            f'live, then starts after {self.start_delay:.0f} s '
            '(or call /start_tracking to start now).')

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def world_to_odom(points: np.ndarray, start: Pose2D) -> np.ndarray:
        """Express world-frame points relative to the spawn pose (= odom origin)."""
        c, s = math.cos(start[2]), math.sin(start[2])
        d = points - np.array(start[:2])
        return np.column_stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]))

    def lookup_laser_offset(self) -> None:
        """Laser pose in the base frame, from TF (fixed, so looked up once).
        Falls back to 'laser at the robot centre' with a warning if TF is missing."""
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.laser_frame, Time())
            q = tf.transform.rotation
            self.laser_offset = (tf.transform.translation.x, tf.transform.translation.y,
                                 yaw_from_quaternion(q.x, q.y, q.z, q.w))
            self.get_logger().info(f'Laser offset in {self.base_frame}: {self.laser_offset}')
            # The offset is fixed: stop listening to /tf so its 30 Hz messages do not
            # compete with the control loop for the (single-threaded) executor.
            self.tf_listener.unregister()
        except TransformException as e:
            if self.last_scan_time is not None and \
                    (self.get_clock().now() - self.t_created) > Duration(seconds=3.0):
                self.laser_offset = (0.0, 0.0, 0.0)
                self.get_logger().warn(f'No TF {self.base_frame}->{self.laser_frame} ({e}); '
                                       'assuming the laser is at the robot centre.')

    def predict_pose(self, now: Time) -> np.ndarray:
        """Latency compensation: move the last odometry pose forward from its timestamp
        to `now` with the last command (unicycle model). The MPPI plan then starts from
        where the robot is now, not where it was when the odometry was measured."""
        age = now.nanoseconds * 1e-9 - self.pose_stamp
        self.max_pose_age = max(self.max_pose_age, age)
        if age > self.max_latency:
            self.get_logger().warn(f'Pose is {age * 1e3:.0f} ms old (CPU overloaded?)',
                                   throttle_duration_sec=2.0)
        dt = min(max(age, 0.0), self.max_latency)
        v, w = self.ctrl.last_cmd
        x, y, th = self.pose
        return np.array([x + v * math.cos(th + 0.5 * w * dt) * dt,
                         y + v * math.sin(th + 0.5 * w * dt) * dt,
                         th + w * dt])

    def pose_at(self, stamp_s: float) -> Optional[Pose2D]:
        """Robot pose closest in time to `stamp_s` from the odom history."""
        if not self.odom_hist:
            return None
        return min(self.odom_hist, key=lambda tp: abs(tp[0] - stamp_s))[1]

    # ------------------------------------------------------------------ #
    # Callbacks (store data only)
    # ------------------------------------------------------------------ #
    def odom_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                yaw_from_quaternion(q.x, q.y, q.z, q.w))
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self.odom_hist.append((t, pose))
        while self.odom_hist and t - self.odom_hist[0][0] > \
                float(self.get_parameter('odom_history').value):
            self.odom_hist.popleft()
        self.pose = pose
        self.pose_stamp = t
        self.last_odom_time = self.get_clock().now()

    def scan_callback(self, msg: LaserScan) -> None:
        """Lidar -> obstacle points in the odom frame, using the robot pose at scan time."""
        self.last_scan_time = self.get_clock().now()
        if self.laser_offset is None:
            return
        robot = self.pose_at(Time.from_msg(msg.header.stamp).nanoseconds * 1e-9)
        if robot is None:
            return
        # laser pose in odom = robot pose (+) laser offset
        ox, oy, oyaw = self.laser_offset
        c, s = math.cos(robot[2]), math.sin(robot[2])
        sensor = (robot[0] + c * ox - s * oy, robot[1] + s * ox + c * oy, robot[2] + oyaw)
        self.new_scan_points = scan_to_points(np.asarray(msg.ranges), msg.angle_min,
                                              msg.angle_increment, msg.range_min,
                                              msg.range_max, sensor)

    # ------------------------------------------------------------------ #
    # Control loop
    # ------------------------------------------------------------------ #
    def start_service_callback(self, _req: Trigger.Request,
                               resp: Trigger.Response) -> Trigger.Response:
        """/start_tracking: skip the rest of the start delay."""
        resp.success = not self.started
        resp.message = 'Starting now' if resp.success else 'Already tracking'
        self.start_requested = True  # picked up by the next control step
        return resp

    def ready_to_start(self, now: Time) -> bool:
        """Start gate: hold the robot until every input is live, then `start_delay` more
        seconds, so Gazebo and RViz have finished loading before the robot moves."""
        if self.started:
            return True
        if self.t_ready is None:
            self.t_ready = now
            self.get_logger().info(f'All inputs live. Starting in {self.start_delay:.0f} s '
                                   '(ros2 service call /start_tracking std_srvs/srv/Trigger '
                                   'to start now).')
        waited = (now - self.t_ready).nanoseconds * 1e-9          # [s]
        if self.start_requested or waited >= self.start_delay:
            self.started = True
            self.t_start = now
            self.get_logger().info('>>> Tracking started')
            return True
        self.get_logger().info(f'Starting in {self.start_delay - waited:.0f} s ...',
                               throttle_duration_sec=2.0)
        return False

    def control_step(self) -> None:
        now = self.get_clock().now()
        if self.laser_offset is None:
            self.lookup_laser_offset()

        if self.ctrl.goal_reached:
            self.publish_cmd(0.0, 0.0)
            return

        # Safety: stop if odometry or lidar data is missing or stale.
        stale = [name for name, t in ((self.odom_topic, self.last_odom_time),
                                      ('/scan', self.last_scan_time))
                 if t is None or (now - t) > self.sensor_timeout]
        if stale or self.laser_offset is None:
            self.publish_cmd(0.0, 0.0)
            self.get_logger().warn(f'Waiting for {stale or ["laser TF"]}: robot stopped',
                                   throttle_duration_sec=2.0)
            return
        if not self.ready_to_start(now):
            self.publish_cmd(0.0, 0.0)
            return

        # Hand the newest scan to the controller (only from this thread).
        points, self.new_scan_points = self.new_scan_points, None
        if points is not None:
            self.ctrl.set_obstacles(points, np.array(self.pose[:2]))
            self.publish_obstacles(points)

        state = self.predict_pose(now)
        if self.ctrl.tree is not None:
            self.min_lidar_dist = min(self.min_lidar_dist, float(self.ctrl.tree.query(state[:2])[0]))
        result = self.ctrl.compute(state)
        self.compute_ms.append(result.compute_ms)
        self.get_logger().debug(f'MPPI step {result.compute_ms:.1f} ms, ESS {result.ess:.0f}, '
                                f'progress {result.progress_idx}/{len(self.path) - 1}')

        if result.goal_reached:
            self.publish_cmd(0.0, 0.0)
            self.on_goal(now)
            return
        if result.all_collide:
            self.get_logger().warn('Every MPPI sample collides: recovery '
                                   f'(v={result.v:+.2f} m/s)', throttle_duration_sec=1.0)
        self.publish_cmd(result.v, result.w)

        self.record_traversed(state)
        self.cycle += 1
        if self.cycle % self.viz_every == 0:
            self.publish_mppi(result, now)

    def on_goal(self, now: Time) -> None:
        if self.goal_logged:
            return
        self.goal_logged = True
        self.record_traversed(np.array(self.pose), force=True)
        elapsed = (now - self.t_start).nanoseconds * 1e-9
        ms = np.array(self.compute_ms)
        self.get_logger().info(
            f'Goal reached in {elapsed:.1f} s (sim time). MPPI step: mean {ms.mean():.1f} ms, '
            f'max {ms.max():.1f} ms over {len(ms)} cycles. Closest lidar point to the robot '
            f'centre: {self.min_lidar_dist:.3f} m (collision_radius {self.cfg.collision_radius}). '
            f'Worst pose age: {self.max_pose_age * 1e3:.0f} ms.')
        self.save_traversed()

    def publish_cmd(self, v: float, w: float) -> None:
        if self.cmd_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.base_frame
            twist = msg.twist
        else:
            msg = twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #
    def publish_recorded_path(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(make_path_msg(self.path, self.frame_id, stamp))
        self.recorded_pub.publish(make_marker(self.path, self.frame_id, stamp, 'recorded', 0,
                                              Marker.LINE_STRIP, GREEN, 0.03))

    def record_traversed(self, state: np.ndarray, force: bool = False) -> None:
        """Append the pose every traversed_spacing metres and republish the red line."""
        xy = (float(state[0]), float(state[1]))
        if self.traversed and not force and \
                math.dist(xy, self.traversed[-1]) < self.traversed_spacing:
            return
        self.traversed.append(xy)
        if len(self.traversed) >= 2:
            self.traversed_pub.publish(make_marker(
                self.traversed, self.frame_id, self.get_clock().now().to_msg(), 'traversed', 0,
                Marker.LINE_STRIP, RED, 0.03, z=0.03))

    def publish_obstacles(self, points: np.ndarray) -> None:
        self.obstacle_pub.publish(make_marker(points, self.frame_id,
                                              self.get_clock().now().to_msg(), 'obstacles', 0,
                                              Marker.POINTS, BLUE, 0.05, z=0.05))

    def publish_mppi(self, r: MPPIResult, now: Time) -> None:
        stamp = now.to_msg()
        arr = MarkerArray()
        arr.markers.append(make_marker(r.optimal_traj[:, :2], self.frame_id, stamp, 'optimal', 0,
                                       Marker.LINE_STRIP, YELLOW, 0.025, z=0.04))
        for i, traj in enumerate(r.sampled_trajs):
            arr.markers.append(make_marker(traj[:, :2], self.frame_id, stamp, 'samples', i,
                                           Marker.LINE_STRIP, GREY, 0.005, z=0.035))
        target = self.path[min(r.progress_idx, len(self.path) - 1)]
        arr.markers.append(make_marker([target], self.frame_id, stamp, 'progress', 0,
                                       Marker.SPHERE_LIST, MAGENTA, 0.08, z=0.05))
        self.mppi_pub.publish(arr)

    def save_traversed(self) -> None:
        if self.traversed_file and len(self.traversed) >= 2:
            out = save_path_csv(self.traversed_file, self.traversed)
            self.get_logger().info(f'Traversed path saved to {out}')


def main(args: list | None = None) -> None:
    rclpy.init(args=args)
    node = MPPITracker()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Ctrl+C reaches a node twice (the terminal signals the whole process group and
        # ros2 launch forwards its own SIGINT). Ignore the second one so it cannot
        # interrupt the clean-up below (e.g. writing the CSV).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.save_traversed()
        if rclpy.ok():
            node.publish_cmd(0.0, 0.0)      # leave the robot stopped
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

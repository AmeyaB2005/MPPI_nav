"""Shared simulation setup for record.launch.py and mppi.launch.py.

Recording and replaying a path must use the SAME world, the SAME robot model and
the SAME pose source, otherwise the recorded path and the replay live in slightly
different frames. Both launch files therefore build the simulation here:

  Gazebo server (+ optional GUI) with worlds/<name>.sdf
  TurtleBot3 robot, patched at launch time (see patched_robot_sdf)
  robot_state_publisher, ROS <-> Gazebo bridges, /ground_truth bridge
  robot_localization EKF (wheel odometry + IMU gyro)  -> /odometry/filtered
  RViz with config/mppi.rviz
"""
from __future__ import annotations

import math
import os
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from mppi_nav.utils import load_path_csv

MIN_HEADING_DIST = 0.05  # [m] the spawn yaw points at the first path point at least this far away

CASTER_FRICTION = '<friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction>'
GROUND_TRUTH_PLUGIN = """
    <plugin filename="gz-sim-odometry-publisher-system"
            name="gz::sim::systems::OdometryPublisher">
      <odom_frame>world</odom_frame>
      <robot_base_frame>base_footprint</robot_base_frame>
      <odom_topic>ground_truth</odom_topic>
      <odom_publish_frequency>30</odom_publish_frequency>
      <dimensions>2</dimensions>
    </plugin>
  </model>"""


def resolve_csv(name: str, share: str, paths_dir: str) -> str:
    """Absolute path of a CSV given on the command line (absolute, or a name in paths/)."""
    name = os.path.expanduser(name)
    if os.path.isabs(name):
        return name
    for folder in (os.path.join(share, 'paths'), os.path.expanduser(paths_dir)):
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f'path CSV "{name}" not found in {share}/paths or {paths_dir}')


def spawn_pose_from_path(csv_file: str) -> tuple:
    """(x, y, yaw) of the first path point, facing the first point >= 5 cm away."""
    pts = load_path_csv(csv_file)
    x0, y0 = pts[0]
    for x, y in pts[1:]:
        if math.hypot(x - x0, y - y0) >= MIN_HEADING_DIST:
            return float(x0), float(y0), math.atan2(y - y0, x - x0)
    return float(x0), float(y0), 0.0


def world_file_for(world: str, share: str) -> str:
    """worlds/<world>.sdf of this package, or a path to any .sdf file."""
    world_file = (os.path.expanduser(world) if world.endswith('.sdf')
                  else os.path.join(share, 'worlds', f'{world}.sdf'))
    if not os.path.isfile(world_file):
        raise FileNotFoundError(f'world file not found: {world_file}')
    return world_file


def pose_topic(use_ekf: bool) -> str:
    """Robot pose source used by every node (tracker, recorder, spawner)."""
    return '/odometry/filtered' if use_ekf else '/odom'


def patched_robot_sdf(model_file: str) -> str:
    """TurtleBot3 model SDF with three changes, applied at launch time:

    1. Frictionless back caster. The stock model gives the 5 mm caster ball no
       <friction>, so ODE uses mu = 1: the caster drags sideways whenever the robot
       turns, the wheels slip, and the wheel odometry (/odom) over-counts the turn.
       Measured: up to 1.47 m / 26 deg odom error in one case-1 run. A real ball
       caster rolls freely, so mu = 0 is the realistic value.
    2. The IMU gets <gz_frame_id>imu_link</gz_frame_id> (like the lidar already has),
       so /imu is stamped with a frame that exists in TF (needed by the EKF).
    3. A ground-truth odometry publisher (/ground_truth), used only to measure
       odometry drift and to report honest metrics. The controller never uses it.
    """
    with open(model_file, 'r', encoding='utf-8') as f:
        sdf = f.read()
    caster = sdf.index("<link name='caster_back_link'>")
    surface = sdf.index('<surface>', caster) + len('<surface>')
    sdf = sdf[:surface] + CASTER_FRICTION + sdf[surface:]
    imu_topic = sdf.index('<topic>imu</topic>') + len('<topic>imu</topic>')
    sdf = sdf[:imu_topic] + '<gz_frame_id>imu_link</gz_frame_id>' + sdf[imu_topic:]
    end = sdf.rindex('</model>')
    return sdf[:end] + GROUND_TRUTH_PLUGIN + sdf[end + len('</model>'):]


def sim_actions(world_file: str, x: float, y: float, yaw: float, use_ekf: bool,
                gui: str, use_rviz: str) -> List:
    """Launch actions for the whole simulation (everything except our own nodes)."""
    share = get_package_share_directory('mppi_nav')
    tb3_share = get_package_share_directory('turtlebot3_gazebo')
    gz_share = get_package_share_directory('ros_gz_sim')
    model = os.environ.get('TURTLEBOT3_MODEL', 'burger')

    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r -s -v2 {world_file}',
                          'on_exit_shutdown': 'true'}.items())
    gz_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-g -v2'}.items(),
        condition=IfCondition(gui))
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_share, 'launch', 'robot_state_publisher.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items())
    spawn = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-name', model,
                   '-string', patched_robot_sdf(
                       os.path.join(tb3_share, 'models', f'turtlebot3_{model}', 'model.sdf')),
                   '-x', str(x), '-y', str(y), '-z', '0.01', '-Y', str(yaw)])
    # With the EKF, Gazebo's odom->base_footprint TF must not be bridged (the EKF publishes it).
    bridge_file = (os.path.join(share, 'config', 'bridge_no_tf.yaml') if use_ekf
                   else os.path.join(tb3_share, 'params', f'turtlebot3_{model}_bridge.yaml'))
    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        arguments=['--ros-args', '-p', 'config_file:=' + bridge_file])
    ground_truth_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='ground_truth_bridge',
        output='screen', arguments=['/ground_truth@nav_msgs/msg/Odometry[gz.msgs.Odometry'])
    ekf = Node(
        package='robot_localization', executable='ekf_node', name='ekf_filter_node',
        output='screen', parameters=[os.path.join(share, 'config', 'ekf.yaml')],
        condition=IfCondition(str(use_ekf)))
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='log',
        arguments=['-d', os.path.join(share, 'config', 'mppi.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(use_rviz))
    return [gz_server, gz_client, robot_state_publisher, spawn, bridge,
            ground_truth_bridge, ekf, rviz]

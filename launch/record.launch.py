"""Record a path: Gazebo + TurtleBot3 + RViz + path_recorder.

Usage:
    ros2 launch mppi_nav record.launch.py output:=case4_loop.csv
    # in a second terminal:
    ros2 run turtlebot3_teleop teleop_keyboard

Ctrl+C in the launch terminal saves the CSV. You can also save at any time with
    ros2 service call /save_path std_srvs/srv/Trigger

The simulation is built by mppi_nav.launch_utils.sim_actions, exactly like
mppi.launch.py: same world file, same patched robot (frictionless caster), same
EKF pose (/odometry/filtered). So a path recorded here is replayed in the same frame.

Relative `output` names are written to `output_dir` (default: the package's
source paths/ folder, so recordings end up in the git repo, not in install/).
Replay a recording with:
    ros2 launch mppi_nav mppi.launch.py path:=case4_loop.csv
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mppi_nav.launch_utils import pose_topic, sim_actions, world_file_for


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory('mppi_nav')
    arg = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    x, y, yaw = float(arg('x_pose')), float(arg('y_pose')), float(arg('yaw'))
    use_ekf = arg('use_ekf').lower() == 'true'
    world_file = world_file_for(arg('world'), share)

    recorder = Node(
        package='mppi_nav', executable='path_recorder', name='path_recorder', output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time': True,
            'output_file': arg('output'),
            'output_dir': arg('output_dir'),
            'spacing': float(arg('spacing')),
            'odom_topic': pose_topic(use_ekf),
            'start_x': x, 'start_y': y, 'start_yaw': yaw,
        }])
    hint = LogInfo(msg=f'\n>>> world={world_file}, pose source={pose_topic(use_ekf)}\n'
                       '>>> Drive the robot in another terminal:  '
                       'ros2 run turtlebot3_teleop teleop_keyboard\n'
                       '>>> Ctrl+C here saves the CSV.\n')
    return sim_actions(world_file, x, y, yaw, use_ekf, arg('gui'), arg('use_rviz')) + [recorder, hint]


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument('output', default_value='recorded.csv',
                              description='CSV file name (relative to output_dir) or absolute path'),
        DeclareLaunchArgument('output_dir', default_value='~/ros2_ws/src/mppi_nav/paths',
                              description='Folder for relative output names'),
        DeclareLaunchArgument('spacing', default_value='0.1',
                              description='Distance between recorded points [m]'),
        DeclareLaunchArgument('world', default_value='empty',
                              description='worlds/<name>.sdf to record in (default: empty)'),
        DeclareLaunchArgument('x_pose', default_value='0.0', description='Robot spawn x [m]'),
        DeclareLaunchArgument('y_pose', default_value='0.0', description='Robot spawn y [m]'),
        DeclareLaunchArgument('yaw', default_value='0.0', description='Robot spawn yaw [rad]'),
        DeclareLaunchArgument('use_ekf', default_value='true',
                              description='record the EKF pose (same as the tracker uses)'),
        DeclareLaunchArgument('use_rviz', default_value='true', description='Start RViz'),
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo GUI'),
    ]
    tb3_models = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'models')
    return LaunchDescription(args + [
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', tb3_models),
        OpaqueFunction(function=launch_setup),
    ])

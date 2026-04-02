import os
import launch.logging
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    pkg_semantic_lidar_slam = get_package_share_directory('semantic_lidar_slam')
    config_yaml = os.path.join(pkg_semantic_lidar_slam, 'config', 'livox_360.yaml')
    rviz_config_file = os.path.join(pkg_semantic_lidar_slam, 'rviz', 'lio.rviz')

    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to start RVIZ2'
    )
    rviz_flag = LaunchConfiguration('rviz')

    semantic_lidar_slam_node = Node(
        package='semantic_lidar_slam',
        executable='semantic_lidar_slam_node',
        name='semantic_lidar_slam_node',
        output='screen',
        parameters=[config_yaml],
        arguments=['--ros-args', '--log-level', 'info']
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='super_lio',
        arguments=['-d', rviz_config_file, '--ros-args', '--log-level', 'warn'],
        condition=IfCondition(rviz_flag)
    )

    ld = LaunchDescription()
    ld.add_action(declare_rviz_arg)
    ld.add_action(semantic_lidar_slam_node)
    ld.add_action(rviz2_node)

    return ld
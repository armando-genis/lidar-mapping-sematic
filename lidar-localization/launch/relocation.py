import os
import launch.logging
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    pkg_lio_localization = get_package_share_directory('lio_localization')
    config_yaml = os.path.join(pkg_lio_localization, 'config', 'relocation.yaml')
    rviz_config_file = os.path.join(pkg_lio_localization, 'rviz', 'relocation.rviz')

    declare_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Whether to start RVIZ2'
    )
    rviz_flag = LaunchConfiguration('rviz')

    lio_localization_node = Node(
        package='lio_localization',
        executable='lio_localization_relocation_node',
        name='lio_localization_relocation_node',
        output='screen',
        parameters=[config_yaml],
        arguments=['--ros-args', '--log-level', 'info']
    )

    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='lio_localization_relocation',
        arguments=['-d', rviz_config_file, '--ros-args', '--log-level', 'warn'],
        condition=IfCondition(rviz_flag)
    )

    ld = LaunchDescription()
    ld.add_action(declare_rviz_arg)
    ld.add_action(lio_localization_node)
    ld.add_action(rviz2_node)

    return ld
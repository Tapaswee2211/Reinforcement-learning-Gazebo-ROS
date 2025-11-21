import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'my_bot'
    use_sim_time = LaunchConfiguration('use_sim_time')
    world_file = LaunchConfiguration('world')

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory(package_name), 'launch', 'rsp.launch.py')]
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]
        ),
        launch_arguments={'world': world_file}.items()
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'my_bot'],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        #DeclareLaunchArgument('world', default_value=os.path.join(get_package_share_directory(package_name), 'worlds', 'rect_track.world')),
        DeclareLaunchArgument('world', default_value=os.path.join('/home/tapaswee/my_world1.world')),
        rsp,
        gazebo,
        spawn_entity,
    ])


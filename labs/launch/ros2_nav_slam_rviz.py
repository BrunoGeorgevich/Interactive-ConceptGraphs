from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter
import os


def generate_launch_description():
    return LaunchDescription(
        [
            SetParameter(name="use_sim_time", value=True),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", "rviz/robot_at_virtualhome.rviz"],
                parameters=[{"use_sim_time": True}],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("nav2_bringup"),
                        "launch",
                        "navigation_launch.py",
                    )
                ),
                launch_arguments={"use_sim_time": "True"}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("slam_toolbox"),
                        "launch",
                        "online_async_launch.py",
                    )
                ),
                launch_arguments={
                    "use_sim_time": "True",
                }.items(),
            ),
        ]
    )

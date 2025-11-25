from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch import LaunchDescription
from launch_ros.actions import Node, SetParameter
import os


def generate_launch_description():
    slam_params_file = os.path.join(os.getcwd(), "launch", "slam_params.yaml")
    print("Using SLAM params file: ", slam_params_file)

    return LaunchDescription(
        [
            # Mantemos o sim_time
            SetParameter(name="use_sim_time", value=True),
            DeclareLaunchArgument(
                "slam_params_file",
                default_value=slam_params_file,
                description="Full path to the ROS2 parameters file",
            ),
            # --- REMOVIDO: static_transform_publisher ---
            # A responsabilidade volta para o Python para garantir sincronia temporal.
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
                    "params_file": slam_params_file,
                }.items(),
            ),
        ]
    )

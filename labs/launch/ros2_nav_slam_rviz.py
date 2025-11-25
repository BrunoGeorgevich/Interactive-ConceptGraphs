from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node, SetParameter
from launch import LaunchDescription
import traceback
import os


def generate_launch_description() -> LaunchDescription:
    """
    Generate a ROS2 launch description for navigation, SLAM, and RViz visualization.

    This function creates a launch configuration that includes RViz2 for visualization,
    Nav2 for navigation capabilities, and SLAM Toolbox for simultaneous localization
    and mapping. All components are configured to use simulation time.

    :return: A LaunchDescription object containing all configured nodes and launch files
    :rtype: LaunchDescription
    :raises FileNotFoundError: If the SLAM parameters file cannot be found
    :raises ValueError: If package directories cannot be located
    """
    try:
        slam_params_file = os.path.join(os.getcwd(), "launch", "slam_params.yaml")

        if not os.path.isfile(slam_params_file):
            raise FileNotFoundError(
                f"SLAM parameters file not found: {slam_params_file}"
            )

        print(f"Using SLAM params file: {slam_params_file}")

        nav2_share_dir = get_package_share_directory("nav2_bringup")
        slam_toolbox_share_dir = get_package_share_directory("slam_toolbox")

        return LaunchDescription(
            [
                SetParameter(name="use_sim_time", value=True),
                DeclareLaunchArgument(
                    "slam_params_file",
                    default_value=slam_params_file,
                    description="Full path to the ROS2 parameters file",
                ),
                Node(
                    package="rviz2",
                    executable="rviz2",
                    arguments=["-d", "rviz/robot_at_virtualhome.rviz"],
                    parameters=[{"use_sim_time": True}],
                ),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(
                            nav2_share_dir,
                            "launch",
                            "navigation_launch.py",
                        )
                    ),
                    launch_arguments={"use_sim_time": "True"}.items(),
                ),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(
                            slam_toolbox_share_dir,
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
    except (FileNotFoundError, ValueError, OSError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to generate launch description: {str(e)}")

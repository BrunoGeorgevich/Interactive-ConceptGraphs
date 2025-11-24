import traceback
import subprocess
import numpy as np
import signal
import time
import os

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros
import rclpy

from utils import read_dfs


class ScanPosePublisher(Node):
    """
    ROS2 node for publishing laser scan and odometry data with synchronized transforms.

    This class manages the publication of LaserScan and Odometry messages, as well as the necessary
    static and dynamic transforms for robot localization and mapping in a ROS2 environment.
    """

    def __init__(self, frequency: int) -> None:
        """
        Initialize the ScanPosePublisher node.

        :param frequency: Publishing frequency in Hz
        :type frequency: int
        """
        super().__init__("scan_pose_publisher")
        self.frequency = frequency
        self.scan_publisher = self.create_publisher(LaserScan, "/scan", frequency)
        self.odom_publisher = self.create_publisher(Odometry, "/odom", frequency)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.current_time_ns = self.get_clock().now().nanoseconds
        self.time_increment_ns = int((1.0 / frequency) * 1e9)

        self.attach_base_link_tf()
        self.publish_static_tf()
        self.clock_publisher = self.create_publisher(Clock, "/clock", 10)

    def publish_clock(self, timestamp: Time):
        msg = Clock()
        msg.clock = timestamp.to_msg()
        self.clock_publisher.publish(msg)

    def get_next_timestamp(self) -> Time:
        """
        Generate the next timestamp for message publication.

        :return: Next timestamp
        :rtype: Time
        """
        self.current_time_ns += self.time_increment_ns
        return Time(nanoseconds=self.current_time_ns)

    def attach_base_link_tf(self) -> None:
        """
        Publish static transform between base_link and base_footprint.

        :raises RuntimeError: If transform publication fails
        """
        try:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "base_link"
            t.child_frame_id = "base_footprint"
            self.tf_static_broadcaster.sendTransform(t)
        except (ValueError, AttributeError, RuntimeError) as e:
            traceback.print_exc()
            raise RuntimeError("Failed to attach base_link transform") from e

    def publish_static_tf(self) -> None:
        """
        Publish static transform between base_link and laser_frame.

        :raises RuntimeError: If transform publication fails
        """
        try:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "base_link"
            t.child_frame_id = "laser_frame"
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.046
            t.transform.translation.z = 1.59
            t.transform.rotation.w = 1.0
            self.tf_static_broadcaster.sendTransform(t)
        except (ValueError, AttributeError, RuntimeError) as e:
            traceback.print_exc()
            raise RuntimeError("Failed to publish static transform") from e

    def publish_scan(
        self, ranges: list, timestamp: Time, max_range: float = 15.0
    ) -> None:
        """
        Publish laser scan message.

        :param ranges: List of range measurements
        :type ranges: list
        :param timestamp: Message timestamp
        :type timestamp: Time
        :param max_range: Maximum range value
        :type max_range: float
        :raises RuntimeError: If scan publication fails
        """
        try:
            msg = LaserScan()
            msg.header.stamp = timestamp.to_msg()
            msg.header.frame_id = "laser_frame"
            msg.angle_min = 0.0
            msg.angle_max = 2 * np.pi
            msg.angle_increment = (2 * np.pi) / len(ranges)
            msg.range_min = 0.1
            msg.range_max = max_range
            msg.ranges = ranges
            self.scan_publisher.publish(msg)
        except (ValueError, AttributeError, ZeroDivisionError) as e:
            traceback.print_exc()
            raise RuntimeError("Failed to publish scan") from e

    def publish_tf_and_odom(
        self,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
        timestamp: Time,
    ) -> None:
        """
        Publish transform and odometry messages.

        :param x: X position
        :type x: float
        :param y: Y position
        :type y: float
        :param z: Z position
        :type z: float
        :param roll: Roll angle in radians
        :type roll: float
        :param pitch: Pitch angle in radians
        :type pitch: float
        :param yaw: Yaw angle in radians
        :type yaw: float
        :param timestamp: Message timestamp
        :type timestamp: Time
        :raises RuntimeError: If publication fails
        """
        try:
            q = self.euler_to_quaternion(roll, pitch, yaw)
            t = TransformStamped()
            t.header.stamp = timestamp.to_msg()
            t.header.frame_id = "odom"
            t.child_frame_id = "base_link"
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.tf_broadcaster.sendTransform(t)

            o = Odometry()
            o.header.stamp = timestamp.to_msg()
            o.header.frame_id = "odom"
            o.child_frame_id = "base_link"
            o.pose.pose.position.x = x
            o.pose.pose.position.y = y
            o.pose.pose.position.z = z
            o.pose.pose.orientation = t.transform.rotation
            self.odom_publisher.publish(o)
        except (ValueError, AttributeError, RuntimeError) as e:
            traceback.print_exc()
            raise RuntimeError("Failed to publish transform and odometry") from e

    def euler_to_quaternion(self, roll: float, pitch: float, yaw: float) -> list[float]:
        """
        Convert Euler angles to quaternion.

        :param roll: Roll angle in radians
        :type roll: float
        :param pitch: Pitch angle in radians
        :type pitch: float
        :param yaw: Yaw angle in radians
        :type yaw: float
        :return: Quaternion as [x, y, z, w]
        :rtype: list[float]
        """
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        return [float(qx), float(qy), float(qz), float(qw)]


def map_exists(path: str) -> tuple[bool, str, str]:
    """
    Check if generated map files exist and are valid.

    :param path: Path to check for map files
    :type path: str
    :return: Tuple of (exists, yaml_path, png_path)
    :rtype: tuple[bool, str, str]
    """
    yaml_path = os.path.join(path, "..", "generated_map.yaml")
    png_path = os.path.join(path, "..", "generated_map.png")

    if os.path.exists(yaml_path) and os.path.exists(png_path):
        if os.path.getsize(png_path) > 0:
            return True, yaml_path, png_path
    return False, yaml_path, png_path


def launch_ros_stack(launch_file: str) -> subprocess.Popen:
    """
    Launch ROS2 navigation stack.

    :param launch_file: Path to the launch file
    :type launch_file: str
    :return: Subprocess handle
    :rtype: subprocess.Popen
    :raises OSError: If launch fails
    """
    process = subprocess.Popen(
        ["ros2", "launch", launch_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return process


def kill_ros_stack(process: subprocess.Popen) -> None:
    """
    Terminate ROS2 stack process gracefully.

    :param process: Process to terminate
    :type process: subprocess.Popen
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError, AttributeError):
        traceback.print_exc()


def process_single_home(home_name: str, data_folder: str) -> bool:
    """
    Check if a home needs map generation.

    :param home_name: Name of the home directory
    :type home_name: str
    :param data_folder: Path to the data folder
    :type data_folder: str
    :return: True if processing is needed, False otherwise
    :rtype: bool
    """
    experiment = "Wandering"
    image_data_file = "LogImg.csv"

    path = os.path.join(data_folder, home_name, experiment)
    exists, yaml_full, png_full = map_exists(path)

    if exists:
        print(f"[{home_name}] Map already exists. Skipping.")
        print(f"   -> PNG:  {os.path.abspath(png_full)}")
        return False

    try:
        read_dfs(image_data_file, path)
    except (FileNotFoundError, IOError, ValueError):
        traceback.print_exc()
        return False

    return True


def run_simulation(
    home_name: str,
    log_df,
    sample_freq: int,
    speedup_factor: float,
    data_folder: str,
) -> None:
    """
    Execute map generation simulation for a home.

    :param home_name: Name of the home being processed
    :type home_name: str
    :param log_df: DataFrame containing log data
    :type log_df: Any
    :param sample_freq: Frequency for publishing messages
    :type sample_freq: int
    :param speedup_factor: Factor to speed up simulation
    :type speedup_factor: float
    :param data_folder: Path to the data folder
    :type data_folder: str
    :raises RuntimeError: If simulation fails
    """
    node = ScanPosePublisher(sample_freq)
    real_sleep = (1.0 / sample_freq) / speedup_factor

    print(f"[{home_name}] Generating map ({len(log_df)} rows, {speedup_factor}x)...")

    try:
        print("   -> Warming up SLAM...")
        first_row = log_df.iloc[0]
        for _ in range(15):
            raw_pos = first_row.get("robot_position")
            raw_rot = first_row.get("robot_rotation")
            data = first_row.get("data")

            ros_x, ros_y, ros_z = (
                float(raw_pos[2]),
                float(-raw_pos[0]),
                float(raw_pos[1]),
            )
            ros_yaw = -np.deg2rad(raw_rot[1])
            ros_roll, ros_pitch = np.deg2rad(raw_rot[0]), np.deg2rad(raw_rot[2])
            ranges = [float(x) for x in data]

            sim_time = node.get_next_timestamp()

            node.publish_clock(sim_time)
            node.publish_static_tf()
            node.publish_scan(ranges, sim_time)
            node.publish_tf_and_odom(
                ros_x, ros_y, ros_z, ros_roll, ros_pitch, ros_yaw, sim_time
            )
            rclpy.spin_once(node, timeout_sec=0.001)
            time.sleep(0.1)

        for idx, row in log_df.iterrows():
            if idx % 10 == 0:
                print(f"   Progress: {idx}/{len(log_df)}", end="\r")

            if idx % 20 == 0:
                node.publish_static_tf()

            raw_pos = row.get("robot_position")
            raw_rot = row.get("robot_rotation")
            data = row.get("data")
            if not raw_pos or not raw_rot or not data:
                continue

            ros_x, ros_y, ros_z = (
                float(raw_pos[2]),
                float(-raw_pos[0]),
                float(raw_pos[1]),
            )
            unity_yaw_deg = raw_rot[1]
            ros_yaw = -np.deg2rad(unity_yaw_deg)
            ros_roll, ros_pitch = np.deg2rad(raw_rot[0]), np.deg2rad(raw_rot[2])
            ranges = [float(x) for x in data]

            sim_time = node.get_next_timestamp()
            node.publish_scan(ranges, sim_time)
            node.publish_tf_and_odom(
                ros_x, ros_y, ros_z, ros_roll, ros_pitch, ros_yaw, sim_time
            )

            rclpy.spin_once(node, timeout_sec=0.0001)
            time.sleep(real_sleep)

    except (ValueError, KeyError, AttributeError, RuntimeError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Simulation failed for {home_name}") from e
    finally:
        node.destroy_node()

    path = os.path.join(data_folder, home_name, "Wandering")
    save_path = os.path.join(path, "..")
    os.makedirs(save_path, exist_ok=True)
    map_file = os.path.join(save_path, "generated_map")

    subprocess.run(
        [
            "ros2",
            "run",
            "nav2_map_server",
            "map_saver_cli",
            "-f",
            map_file,
            "--mode",
            "trinary",
            "--fmt",
            "png",
            "--ros-args",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    final_png = map_file + ".png"
    if os.path.exists(final_png) and os.path.getsize(final_png) > 0:
        print(f"[{home_name}] SUCCESS: {final_png}")
    else:
        print(f"[{home_name}] FAILURE: Empty map.")


if __name__ == "__main__":
    SPEEDUP_FACTOR = 5.0
    SAMPLE_FREQ = 10
    LAUNCH_FILE = "launch/ros2_nav_slam_rviz.py"
    DATA_FOLDER = "data"
    START_HOME = 1
    END_HOME = 30
    CLEANUP_TIME = 8

    rclpy.init()

    try:
        for i in range(START_HOME, END_HOME + 1):
            home_name = f"Home{i:02d}"

            experiment = "Wandering"
            path = os.path.join(DATA_FOLDER, home_name, experiment)
            exists, _, _ = map_exists(path)

            if exists:
                print(f"[{home_name}] Already exists. Skipping.")
                continue

            try:
                log_df, _, _ = read_dfs("LogImg.csv", path)
            except (FileNotFoundError, IOError, ValueError):
                traceback.print_exc()
                print(f"[{home_name}] CSV not found.")
                continue

            print(f"\n--- Starting {home_name} ---")
            proc = launch_ros_stack(LAUNCH_FILE)

            time.sleep(6)

            run_simulation(home_name, log_df, SAMPLE_FREQ, SPEEDUP_FACTOR, DATA_FOLDER)

            kill_ros_stack(proc)
            print(f"Waiting {CLEANUP_TIME}s...")
            time.sleep(CLEANUP_TIME)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            if "proc" in locals():
                kill_ros_stack(proc)
        except (NameError, AttributeError):
            traceback.print_exc()
        rclpy.shutdown()

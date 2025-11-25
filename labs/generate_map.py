from typing import Any, TextIO
import numpy as np
import subprocess
import traceback
import signal
import time
import math
import os

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
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
    ROS2 node for publishing laser scan, odometry, and TF transforms with synchronized clock.
    """

    def __init__(self, frequency: int) -> None:
        """
        Initialize the ScanPosePublisher node.

        :param frequency: Publishing frequency in Hz
        :type frequency: int
        :raises ValueError: If frequency is invalid
        """
        super().__init__("scan_pose_publisher")
        self.frequency = frequency

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.clock_publisher = self.create_publisher(Clock, "/clock", 10)
        self.scan_publisher = self.create_publisher(LaserScan, "/scan", qos_profile)
        self.odom_publisher = self.create_publisher(Odometry, "/odom", qos_profile)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.current_time_ns = self.get_clock().now().nanoseconds
        self.time_increment_ns = int((1.0 / frequency) * 1e9)

    def publish_clock(self, timestamp: Time) -> None:
        """
        Publish simulated clock time.

        :param timestamp: ROS time to publish
        :type timestamp: Time
        :raises RuntimeError: If clock publishing fails
        """
        try:
            msg = Clock()
            msg.clock = timestamp.to_msg()
            self.clock_publisher.publish(msg)
        except (RuntimeError, ValueError, AttributeError):
            traceback.print_exc()
            raise RuntimeError("Failed to publish clock message")

    def get_next_timestamp(self) -> Time:
        """
        Get the next timestamp based on configured frequency.

        :return: Next timestamp
        :rtype: Time
        """
        self.current_time_ns += self.time_increment_ns
        return Time(nanoseconds=self.current_time_ns)

    def publish_static_tf(self, timestamp: Time) -> None:
        """
        Publish static transforms between robot frames.

        :param timestamp: Timestamp for the transform
        :type timestamp: Time
        :raises RuntimeError: If transform publishing fails
        """
        try:
            t1 = TransformStamped()
            t1.header.stamp = timestamp.to_msg()
            t1.header.frame_id = "base_link"
            t1.child_frame_id = "laser_frame"
            t1.transform.translation.x = 0.0
            t1.transform.translation.y = 0.046
            t1.transform.translation.z = 1.59
            t1.transform.rotation.w = 1.0

            t2 = TransformStamped()
            t2.header.stamp = timestamp.to_msg()
            t2.header.frame_id = "base_footprint"
            t2.child_frame_id = "base_link"
            t2.transform.rotation.w = 1.0

            self.tf_static_broadcaster.sendTransform([t1, t2])
        except (RuntimeError, ValueError, AttributeError):
            traceback.print_exc()
            raise RuntimeError("Failed to publish static transforms")

    def publish_scan(
        self, ranges: list[float], timestamp: Time, max_range: float = 15.0
    ) -> None:
        """
        Publish laser scan data.

        :param ranges: List of laser range measurements
        :type ranges: list[float]
        :param timestamp: Timestamp for the scan
        :type timestamp: Time
        :param max_range: Maximum range value
        :type max_range: float
        :raises RuntimeError: If scan publishing fails
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
        except (RuntimeError, ValueError, AttributeError, ZeroDivisionError):
            traceback.print_exc()
            raise RuntimeError("Failed to publish laser scan")

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
        Publish transform and odometry data.

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
        :param timestamp: Timestamp for the data
        :type timestamp: Time
        :raises RuntimeError: If publishing fails
        """
        try:
            q = self.euler_to_quaternion(roll, pitch, yaw)

            t = TransformStamped()
            t.header.stamp = timestamp.to_msg()
            t.header.frame_id = "odom"
            t.child_frame_id = "base_footprint"
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
            o.child_frame_id = "base_footprint"
            o.pose.pose.position.x = x
            o.pose.pose.position.y = y
            o.pose.pose.position.z = z
            o.pose.pose.orientation = t.transform.rotation
            self.odom_publisher.publish(o)
        except (RuntimeError, ValueError, AttributeError):
            traceback.print_exc()
            raise RuntimeError("Failed to publish transform and odometry")

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
    Check if generated map files exist.

    :param path: Path to check for map files
    :type path: str
    :return: Tuple of (exists, yaml_path, png_path)
    :rtype: tuple[bool, str, str]
    """
    yaml_path = os.path.join(path, "..", "generated_map.yaml")
    png_path = os.path.join(path, "..", "generated_map.png")
    if (
        os.path.exists(yaml_path)
        and os.path.exists(png_path)
        and os.path.getsize(png_path) > 0
    ):
        return True, yaml_path, png_path
    return False, yaml_path, png_path


def launch_ros_stack(
    launch_file: str, log_file_path: str, save_logs: bool
) -> tuple[subprocess.Popen | None, TextIO | None]:
    """
    Launch ROS2 stack and optionally redirect output to a log file.

    :param launch_file: Path to the ROS2 launch file
    :type launch_file: str
    :param log_file_path: Path to save log output
    :type log_file_path: str
    :param save_logs: Whether to save logs to file
    :type save_logs: bool
    :return: Tuple of (process, log_file)
    :rtype: tuple[subprocess.Popen | None, TextIO | None]
    :raises OSError: If process launch fails
    :raises IOError: If log file cannot be created
    """
    log_file = None
    stdout_target = subprocess.DEVNULL
    stderr_target = subprocess.DEVNULL

    if save_logs:
        try:
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            print(f"   -> Saving ROS logs to: {log_file_path}")
            log_file = open(log_file_path, "w")
            stdout_target = log_file
            stderr_target = subprocess.STDOUT
        except (OSError, IOError):
            traceback.print_exc()
            raise IOError(f"Failed to create log file: {log_file_path}")

    try:
        process = subprocess.Popen(
            ["ros2", "launch", launch_file],
            stdout=stdout_target,
            stderr=stderr_target,
            preexec_fn=os.setsid,
        )
        return process, log_file
    except (OSError, subprocess.SubprocessError):
        if log_file:
            log_file.close()
        traceback.print_exc()
        raise OSError(f"Failed to launch ROS2 stack: {launch_file}")


def kill_ros_stack(process: subprocess.Popen | None) -> None:
    """
    Terminate ROS2 stack process gracefully.

    :param process: Process to terminate
    :type process: subprocess.Popen | None
    """
    if process is None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        traceback.print_exc()


def run_simulation(
    home_name: str,
    log_df: Any,
    sample_freq: int,
    speedup_factor: float,
    data_folder: str,
    jump_threshold: float,
    rotation_threshold: float,
) -> None:
    """
    Run simulation and generate map from logged data.

    :param home_name: Name of the home environment
    :type home_name: str
    :param log_df: DataFrame containing robot log data
    :type log_df: Any
    :param sample_freq: Sampling frequency in Hz
    :type sample_freq: int
    :param speedup_factor: Simulation speedup multiplier
    :type speedup_factor: float
    :param data_folder: Root folder for data storage
    :type data_folder: str
    :param jump_threshold: Position jump detection threshold
    :type jump_threshold: float
    :param rotation_threshold: Rotation change threshold
    :type rotation_threshold: float
    :raises RuntimeError: If simulation fails
    """
    node = ScanPosePublisher(sample_freq)
    real_sleep = (1.0 / sample_freq) / speedup_factor

    last_valid_x = None
    last_valid_y = None
    last_valid_z = None
    prev_yaw = 0.0

    print(f"[{home_name}] Generating map ({len(log_df)} rows, {speedup_factor}x)...")

    try:
        print("   -> Warming up SLAM...")
        first_row = log_df.iloc[0]

        sim_time = node.get_next_timestamp()
        node.publish_clock(sim_time)
        node.publish_static_tf(sim_time)
        rclpy.spin_once(node, timeout_sec=0.01)

        for i in range(15):
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
            node.publish_static_tf(sim_time)
            node.publish_scan(ranges, sim_time)
            node.publish_tf_and_odom(
                ros_x, ros_y, ros_z, ros_roll, ros_pitch, ros_yaw, sim_time
            )
            rclpy.spin_once(node, timeout_sec=0.001)
            time.sleep(0.1)

        last_valid_x, last_valid_y, last_valid_z = ros_x, ros_y, ros_z
        prev_yaw = ros_yaw

        for idx, row in log_df.iterrows():
            if idx % 10 == 0:
                print(f"   Progress: {idx}/{len(log_df)}", end="\r")

            raw_pos = row.get("robot_position")
            raw_rot = row.get("robot_rotation")
            data = row.get("data")
            if not raw_pos or not raw_rot or not data:
                continue

            curr_x, curr_y, curr_z = (
                float(raw_pos[2]),
                float(-raw_pos[0]),
                float(raw_pos[1]),
            )
            ros_yaw = -np.deg2rad(raw_rot[1])
            ros_roll, ros_pitch = np.deg2rad(raw_rot[0]), np.deg2rad(raw_rot[2])
            ranges = [float(x) for x in data]

            sim_time = node.get_next_timestamp()
            node.publish_clock(sim_time)
            node.publish_static_tf(sim_time)

            dist = math.sqrt(
                (curr_x - last_valid_x) ** 2 + (curr_y - last_valid_y) ** 2
            )
            if dist > jump_threshold:
                ros_x, ros_y, ros_z = last_valid_x, last_valid_y, last_valid_z
            else:
                ros_x, ros_y, ros_z = curr_x, curr_y, curr_z
                last_valid_x, last_valid_y, last_valid_z = curr_x, curr_y, curr_z

            delta_yaw = abs(ros_yaw - prev_yaw)
            if delta_yaw > np.pi:
                delta_yaw = 2 * np.pi - delta_yaw

            if delta_yaw > rotation_threshold:
                node.publish_tf_and_odom(
                    ros_x, ros_y, ros_z, ros_roll, ros_pitch, ros_yaw, sim_time
                )
                prev_yaw = ros_yaw
                rclpy.spin_once(node, timeout_sec=0.0001)
                time.sleep(real_sleep)
                continue

            prev_yaw = ros_yaw
            node.publish_scan(ranges, sim_time)
            node.publish_tf_and_odom(
                ros_x, ros_y, ros_z, ros_roll, ros_pitch, ros_yaw, sim_time
            )
            rclpy.spin_once(node, timeout_sec=0.0001)
            time.sleep(real_sleep)

    except (RuntimeError, ValueError, KeyError, IndexError, AttributeError):
        traceback.print_exc()
        raise RuntimeError(f"Simulation failed for {home_name}")
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
    DATA_FOLDER = "/mnt/d/Documentos/Datasets/Robot@VirtualHomeLarge/"
    LAUNCH_FILE = "launch/ros2_nav_slam_rviz.py"

    ROTATION_THRESHOLD = 0.25
    SPEEDUP_FACTOR = 1.0
    JUMP_THRESHOLD = 0.3

    LOG_FOLDER = "logs"
    SAVE_LOGS = False
    SKIP_IF_EXISTS = True

    SAMPLE_FREQ = 10
    CLEANUP_TIME = 2
    WARMUP_TIME = 6
    START_HOME = 1
    END_HOME = 30

    rclpy.init()
    proc = None
    log_file = None

    try:
        for i in range(START_HOME, END_HOME + 1):
            home_name = f"Home{i:02d}"
            path = os.path.join(DATA_FOLDER, home_name, "Wandering")
            if SKIP_IF_EXISTS and map_exists(path)[0]:
                print(f"[{home_name}] Already exists. Skipping.")
                continue

            try:
                log_df, _, _ = read_dfs("LogImg.csv", path)
            except (OSError, IOError, ValueError):
                traceback.print_exc()
                print(f"[{home_name}] CSV error/missing.")
                continue

            print(f"\n--- Starting {home_name} ---")

            log_filename = f"{home_name}_log.txt"
            log_full_path = os.path.join(os.getcwd(), LOG_FOLDER, log_filename)

            proc, log_file = launch_ros_stack(LAUNCH_FILE, log_full_path, SAVE_LOGS)

            time.sleep(WARMUP_TIME)
            try:
                run_simulation(
                    home_name,
                    log_df,
                    SAMPLE_FREQ,
                    SPEEDUP_FACTOR,
                    DATA_FOLDER,
                    JUMP_THRESHOLD,
                    ROTATION_THRESHOLD,
                )
            except (RuntimeError, ValueError):
                traceback.print_exc()

            kill_ros_stack(proc)

            if log_file:
                log_file.close()

            print(f"Waiting {CLEANUP_TIME}s...")
            time.sleep(CLEANUP_TIME)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if proc:
            kill_ros_stack(proc)
        if log_file:
            log_file.close()
        rclpy.shutdown()

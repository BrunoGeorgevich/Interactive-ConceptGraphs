import traceback
import yaml
import cv2
import os

from utils import read_dfs


def world_to_map(
    x: float, y: float, origin: list, resolution: float, image_height: int
) -> tuple:
    """
    Converts world coordinates to map pixel coordinates.

    :param x: World X coordinate (ROS convention)
    :type x: float
    :param y: World Y coordinate (ROS convention)
    :type y: float
    :param origin: Map origin [x, y, theta] from YAML
    :type origin: list
    :param resolution: Map resolution in meters per pixel
    :type resolution: float
    :param image_height: Height of the map image in pixels
    :type image_height: int
    :return: Tuple of pixel coordinates (pixel_x, pixel_y)
    :rtype: tuple
    """
    pixel_x = int((x - origin[0]) / resolution)
    pixel_y = int(image_height - ((y - origin[1]) / resolution))
    return pixel_x, pixel_y


def unity_to_ros_coordinates(
    unity_pos: list, unity_rot: list, experiment: str
) -> tuple:
    """
    Converts Unity coordinate system to ROS coordinate system.

    :param unity_pos: Unity position [x, y, z]
    :type unity_pos: list
    :param unity_rot: Unity rotation [roll, pitch, yaw] in degrees
    :type unity_rot: list
    :param experiment: Experiment type name
    :type experiment: str
    :return: Tuple of (ros_x, ros_y, ros_z, ros_yaw)
    :rtype: tuple
    """
    ros_x = float(unity_pos[2])
    ros_y = float(-unity_pos[0])
    ros_z = float(unity_pos[1])

    unity_yaw_deg = unity_rot[1]
    ros_yaw = -unity_yaw_deg

    if "Custom" in experiment or "Manual" in experiment:
        ros_yaw += 180.0

    return ros_x, ros_y, ros_z, ros_yaw


if __name__ == "__main__":
    DATA_FOLDER: str = "D:\\Documentos\\Datasets\\Robot@VirtualHomeLarge"
    HOME: str = "Home05"
    EXPERIMENT: str = "Wandering"

    image_path: str = os.path.join(DATA_FOLDER, HOME, "generated_map.png")
    data_path: str = os.path.join(DATA_FOLDER, HOME, EXPERIMENT)
    map_path: str = os.path.join(DATA_FOLDER, HOME, "generated_map.yaml")

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")

        log_scan_df, filtered_info_grid_df, virtual_objs_df = read_dfs(
            "LogImg.csv", data_path
        )

        with open(map_path, "r") as f:
            map_data = yaml.safe_load(f)

        origin = map_data["origin"]
        resolution = map_data["resolution"]
        image_height = image.shape[0]

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        for index, row in log_scan_df.iterrows():
            unity_pos = row.get("robot_position")
            unity_rot = row.get("robot_rotation")

            if unity_pos is None or unity_rot is None:
                continue

            ros_x, ros_y, ros_z, ros_yaw = unity_to_ros_coordinates(
                unity_pos, unity_rot, EXPERIMENT
            )

            pixel_x, pixel_y = world_to_map(
                ros_x, ros_y, origin, resolution, image_height
            )

            print(
                f"Unity pos: {unity_pos} -> ROS pos: ({ros_x:.2f}, {ros_y:.2f}) -> "
                f"Image pos: ({pixel_x}, {pixel_y})"
            )

            cv2.circle(image, (pixel_x, pixel_y), 2, (0, 0, 255), -1)

            cv2.imshow("Image", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        traceback.print_exc()
        print(f"Error processing map validation: {str(e)}")

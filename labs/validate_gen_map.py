import traceback
import yaml
import cv2
import os

from utils import read_dfs, unity_to_ros_coordinates, world_to_map


if __name__ == "__main__":
    DATA_FOLDER: str = THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET
    HOME: str = "Home01"
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

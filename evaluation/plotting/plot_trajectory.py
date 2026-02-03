import traceback
import yaml
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from labs.utils import read_dfs, unity_to_ros_coordinates, world_to_map


def generate_trajectory_map(
    data_folder: str,
    home: str,
    experiment: str,
    output_folder: str | None = None,
    stride: int = 20,
) -> None:
    """
    Generate map images with trajectories drawn on top.

    :param data_folder: Root folder path of the dataset
    :type data_folder: str
    :param home: Name of the home environment
    :type home: str
    :param experiment: Name of the experiment
    :type experiment: str
    :param output_folder: Optional custom output folder path for saving images
    :type output_folder: str | None
    :param stride: Step interval for drawing trajectory points
    :type stride: int
    :raises FileNotFoundError: If image or data files cannot be found
    :raises KeyError: If required keys are missing from map data
    :raises yaml.YAMLError: If YAML file cannot be parsed
    :raises ValueError: If stride is less than 1
    :return: None
    :rtype: None
    """
    if stride < 1:
        raise ValueError("Stride must be at least 1")

    image_path: str = os.path.join(data_folder, home, "generated_map.png")
    data_path: str = os.path.join(data_folder, home, experiment)
    map_path: str = os.path.join(data_folder, home, "generated_map.yaml")

    if output_folder is None:
        output_folder = os.path.join("trajectory_maps")

    os.makedirs(output_folder, exist_ok=True)

    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")

        log_scan_df, _, _ = read_dfs("LogImg.csv", data_path)

        with open(map_path, "r") as f:
            map_data = yaml.safe_load(f)

        origin = map_data["origin"]
        resolution = map_data["resolution"]
        image_height = image.shape[0]

        trajectory_image = image.copy()
        points = []

        for step, row in enumerate(log_scan_df.itertuples()):
            if step % stride != 0:
                continue

            unity_pos = getattr(row, "robot_position", None)
            unity_rot = getattr(row, "robot_rotation", None)

            if unity_pos is None or unity_rot is None:
                continue

            ros_x, ros_y, _, _ = unity_to_ros_coordinates(
                unity_pos, unity_rot, experiment
            )
            pixel_x, pixel_y = world_to_map(
                ros_x, ros_y, origin, resolution, image_height
            )

            for i in range(len(points) - 1):
                cv2.line(trajectory_image, points[i], points[i + 1], (255, 0, 0), 1)

            points.append((pixel_x, pixel_y))
            cv2.circle(trajectory_image, (pixel_x, pixel_y), 2, (0, 0, 255), -1)

        output_path: str = os.path.join(
            output_folder, f"{experiment}_{home}_trajectory.png"
        )
        cv2.imwrite(output_path, trajectory_image)
        print(f"Trajectory map saved to: {output_path}")

    except (FileNotFoundError, KeyError, yaml.YAMLError, IOError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Error generating trajectory map: {str(e)}")


def generate_multiple_trajectory_maps(
    data_folder: str,
    home: str,
    experiments: list[str],
    output_folder: str | None = None,
    stride: int = 20,
) -> None:
    """
    Generate trajectory maps for multiple experiments.

    :param data_folder: Root folder path of the dataset
    :type data_folder: str
    :param home: Name of the home environment
    :type home: str
    :param experiments: List of experiment names to process
    :type experiments: list[str]
    :param output_folder: Optional custom output folder path for saving images
    :type output_folder: str | None
    :param stride: Step interval for drawing trajectory points
    :type stride: int
    :raises RuntimeError: If any experiment fails to process
    :return: None
    :rtype: None
    """
    for experiment in experiments:
        try:
            print(f"Processing experiment: {experiment}")
            generate_trajectory_map(
                data_folder, home, experiment, output_folder, stride
            )
        except RuntimeError as e:
            traceback.print_exc()
            print(f"Failed to process experiment {experiment}: {str(e)}")


if __name__ == "__main__":
    DATA_FOLDER: str = "THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET"
    OUTPUT_FOLDER = "trajectory_maps"

    HOMES = [f"Home{i:02d}" for i in range(1, 31)]
    EXPERIMENT: str = "Wandering"

    try:
        for HOME in HOMES:
            generate_trajectory_map(DATA_FOLDER, HOME, EXPERIMENT, OUTPUT_FOLDER)
    except RuntimeError as e:
        traceback.print_exc()
        print(f"Script execution failed: {str(e)}")

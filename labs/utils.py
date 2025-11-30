from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from typing import Any
import pandas as pd
import numpy as np
import traceback
import colorsys
import pickle
import gzip
import csv
import ast
import cv2
import os


def read_log_scan(path: str) -> pd.DataFrame:
    """Reads a log file, extracts scan data, and returns it as a Pandas DataFrame.

    The log file is expected to be a semicolon-separated CSV file. Each row should
    contain a scan ID (including robot position), robot rotation, and other data fields.

    :param path: The path to the log file.
    :type path: str
    :raises FileNotFoundError: If the file specified by `path` does not exist.
    :raises csv.Error: If the file is not a valid CSV file or has an invalid format.
    :raises ValueError: If the data cannot be converted to the expected types (int or float).
    :raises TypeError: If the `path` parameter is not a string.
    :raises IndexError: If there is missing data in rows.
    :return: A Pandas DataFrame containing the extracted scan data.
    :rtype: pandas.DataFrame
    """
    content: list[dict] = []
    if not isinstance(path, str):
        raise TypeError("The path parameter must be a string.")

    try:
        with open(path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=";")
            headers = next(reader)
            headers = [h.strip() for h in headers if h.strip()]

            for row in reader:
                if not row:
                    continue
                try:
                    scan_id_str, robot_rot_str, *data_strs = row
                    scan_id_num = int(scan_id_str.split("(")[0])
                    robot_position = [
                        float(coord)
                        for coord in scan_id_str.split("(")[1]
                        .replace(")", "")
                        .split(", ")
                        if coord
                    ]
                    robot_rotation = [
                        float(angle)
                        for angle in robot_rot_str.replace("(", "")
                        .replace(")", "")
                        .split(", ")
                        if angle
                    ]

                    data = [
                        float(el.replace(",", ".").strip()) for el in data_strs if el
                    ]

                    content.append(
                        {
                            "scan_id": scan_id_num,
                            "robot_position": robot_position,
                            "robot_rotation": robot_rotation,
                            "data": data,
                        }
                    )
                except (ValueError, IndexError) as e:
                    traceback.print_exc()
                    raise ValueError(
                        f"Error processing row: {row}. Invalid format for scan_id, robot position, robot rotation or data. Details: {e}"
                    )

    except FileNotFoundError:
        traceback.print_exc()
        raise FileNotFoundError(f"The file was not found at the specified path: {path}")
    except csv.Error as e:
        traceback.print_exc()
        raise csv.Error(f"Error reading CSV file: {e}")

    return pd.DataFrame(content)


def read_log_scan_v2(path: str) -> pd.DataFrame:
    """Reads a log file, extracts scan data, and returns it as a Pandas DataFrame.

    The log file is expected to be a semicolon-separated CSV file. Each row should
    contain a scan ID (including robot position), robot rotation, and other data fields.

    :param path: The path to the log file.
    :type path: str
    :raises FileNotFoundError: If the file specified by `path` does not exist.
    :raises csv.Error: If the file is not a valid CSV file or has an invalid format.
    :raises ValueError: If the data cannot be converted to the expected types (int or float).
    :raises TypeError: If the `path` parameter is not a string.
    :raises IndexError: If there is missing data in rows.
    :return: A Pandas DataFrame containing the extracted scan data.
    :rtype: pandas.DataFrame
    """
    content: list[dict] = []
    if not isinstance(path, str):
        raise TypeError("The path parameter must be a string.")

    try:
        with open(path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=",")
            headers = next(reader)
            headers = [h.strip() for h in headers if h.strip()]

            for idx, row in enumerate(reader):
                if not row:
                    continue
                try:
                    scan_id_num = idx
                    robot_pos_str = row[0:3]

                    robot_pos_str[0] = robot_pos_str[0][len(str(scan_id_num)) :]

                    robot_rot_str = row[3:6]
                    data_strs = row[13:]

                    robot_position = [
                        float(coord.strip()) for coord in robot_pos_str if coord
                    ]
                    robot_rotation = [
                        float(angle.strip()) for angle in robot_rot_str if angle
                    ]

                    data = [float(el.strip()) for el in data_strs if el]

                    content.append(
                        {
                            "scan_id": scan_id_num,
                            "robot_position": robot_position,
                            "robot_rotation": robot_rotation,
                            "data": data,
                        }
                    )
                except (ValueError, IndexError) as e:
                    traceback.print_exc()
                    raise ValueError(
                        f"Error processing row: {row}. Invalid format for scan_id, robot position, robot rotation or data. Details: {e}"
                    )

    except FileNotFoundError:
        traceback.print_exc()
        raise FileNotFoundError(f"The file was not found at the specified path: {path}")
    except csv.Error as e:
        traceback.print_exc()
        raise csv.Error(f"Error reading CSV file: {e}")

    return pd.DataFrame(content)


def read_log_img_v2(path: str) -> pd.DataFrame:
    """
    Reads a log image file, extracts image and pose data, and returns it as a Pandas DataFrame.

    The log image file is expected to be a comma-separated CSV file. Each row should contain
    a photo ID, robot position, robot rotation, camera position, camera rotation, and room information.

    :param path: The path to the log image file.
    :type path: str
    :raises FileNotFoundError: If the file specified by `path` does not exist.
    :raises csv.Error: If the file is not a valid CSV file or has an invalid format.
    :raises ValueError: If the data cannot be converted to the expected types (float).
    :raises TypeError: If the `path` parameter is not a string.
    :raises IndexError: If there is missing data in rows.
    :return: A Pandas DataFrame containing the extracted image and pose data.
    :rtype: pd.DataFrame
    """
    content: list[dict] = []
    if not isinstance(path, str):
        raise TypeError("The path parameter must be a string.")

    try:
        with open(path, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter=",")
            try:
                headers = next(reader)
            except StopIteration:
                raise ValueError("The CSV file is empty or missing headers.")
            headers = [h.strip() for h in headers if h.strip()]

            for row in reader:
                if not row:
                    continue
                try:
                    if len(row) < 14:
                        raise IndexError("Row does not contain enough columns.")
                    photo_id_str = row[0].replace("-", "").strip()
                    robot_pos_str = row[1:4]
                    robot_rot_str = row[4:7]
                    camera_pos_str = row[7:10]
                    camera_rot_str = row[10:13]
                    room = row[13]
                    robot_position = [
                        float(coord.strip()) for coord in robot_pos_str if coord
                    ]
                    robot_rotation = [
                        float(angle.strip()) for angle in robot_rot_str if angle
                    ]
                    camera_position = [
                        float(coord.strip()) for coord in camera_pos_str if coord
                    ]
                    camera_rotation = [
                        float(angle.strip()) for angle in camera_rot_str if angle
                    ]

                    content.append(
                        {
                            "photoID": photo_id_str,
                            "robotPosition": robot_position,
                            "robotRotation": robot_rotation,
                            "cameraPosition": camera_position,
                            "cameraRotation": camera_rotation,
                            "room": room,
                        }
                    )
                except (ValueError, IndexError) as e:
                    traceback.print_exc()
                    raise ValueError(
                        f"Error processing row: {row}. Invalid format for photoID, robot position, robot rotation, camera position, camera rotation or room. Details: {e}"
                    )

    except FileNotFoundError:
        traceback.print_exc()
        raise FileNotFoundError(f"The file was not found at the specified path: {path}")
    except (csv.Error, OSError) as e:
        traceback.print_exc()
        raise csv.Error(f"Error reading CSV file: {e}")

    return pd.DataFrame(content)


def parse_tuple_or_list(data: str) -> tuple[float, ...] | list[float]:
    """Parses a string into a tuple or list of floats.

    This function attempts to parse a string, removing 'RGBA' if present,
    into a tuple or list of floats.

    :param data: The string to parse.
    :type data: str
    :raises ValueError: If the data cannot be converted to a tuple or list of floats, or if the input is not a string.
    :return: A tuple or list of floats.
    :rtype: tuple[float, ...] | list[float]
    """
    if not isinstance(data, str):
        raise ValueError(f"Expected a string, but received {type(data)} instead.")
    try:
        if "RGBA" in data:
            data = data.replace("RGBA", "")
        return ast.literal_eval(data)
    except (ValueError, SyntaxError) as e:
        raise ValueError(
            f"Could not convert {data} to a tuple or list, it must be a tuple or list. Details: {e}"
        )


def read_virtual_objects(path: str) -> pd.DataFrame:
    """
    Reads virtual object data from a CSV file and parses specific columns.

    This function reads a CSV file, expecting a semicolon (;) as the separator, and parses the 'globalPosition', 'rotation', and 'color' columns.
    It converts the string values in these columns to tuples or lists of floats, parsing 'globalPosition' and 'rotation' as lists of floats, similar to the parsing in read_log_scan.

    :param path: The path to the CSV file.
    :type path: str
    :raises FileNotFoundError: If the file specified by 'path' does not exist.
    :raises pd.errors.EmptyDataError: If the CSV file is empty.
    :raises pd.errors.ParserError: If there's an error while parsing the CSV.
    :raises TypeError: If the path parameter is not a string.
    :raises ValueError: If the data in 'globalPosition' or 'rotation' cannot be converted to the expected types.
    :return: A Pandas DataFrame with parsed 'globalPosition', 'rotation', and 'color' columns.
    :rtype: pd.DataFrame
    """
    if not isinstance(path, str):
        raise TypeError(
            "Expected a string for the path, but received a different type."
        )

    try:
        virtual_objs_df = pd.read_csv(path, sep=";")
    except FileNotFoundError:
        traceback.print_exc()
        raise FileNotFoundError(f"The file was not found in the path: {path}")
    except pd.errors.EmptyDataError:
        traceback.print_exc()
        raise pd.errors.EmptyDataError(f"The file in path: {path} is empty.")
    except pd.errors.ParserError as e:
        traceback.print_exc()
        raise pd.errors.ParserError(
            f"Error parsing the CSV file in path: {path}. Details: {e}"
        )

    try:
        if "globalPosition" in virtual_objs_df.columns:

            def parse_position(val: str) -> list[float]:
                try:
                    val = val.strip()
                    if val.startswith("(") and val.endswith(")"):
                        val = val[1:-1]
                    return [
                        float(coord.strip())
                        for coord in val.split(",")
                        if coord.strip()
                    ]
                except (ValueError, AttributeError) as e:
                    traceback.print_exc()
                    raise ValueError(
                        f"Invalid format for globalPosition: {val}. Details: {e}"
                    )

            virtual_objs_df["globalPosition"] = virtual_objs_df["globalPosition"].apply(
                parse_position
            )

        if "rotation" in virtual_objs_df.columns:

            def parse_rotation(val: str) -> list[float]:
                try:
                    val = val.strip()
                    if val.startswith("(") and val.endswith(")"):
                        val = val[1:-1]
                    return [
                        float(angle.strip())
                        for angle in val.split(",")
                        if angle.strip()
                    ]
                except (ValueError, AttributeError) as e:
                    traceback.print_exc()
                    raise ValueError(
                        f"Invalid format for rotation: {val}. Details: {e}"
                    )

            virtual_objs_df["rotation"] = virtual_objs_df["rotation"].apply(
                parse_rotation
            )

        if "color" in virtual_objs_df.columns:
            virtual_objs_df["color"] = [
                np.uint8(np.array(parse_tuple_or_list(item)[:3]) * 255).tolist()
                for item in virtual_objs_df["color"]
            ]

    except (ValueError, AttributeError) as e:
        traceback.print_exc()
        raise ValueError(
            f"Error processing columns in virtual objects CSV. Details: {e}"
        )

    return virtual_objs_df


def read_dfs(
    image_data_file: str,
    data_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads and processes data from CSV files related to image data.

    This function reads data from specified CSV files, including an info grid file,
    a log scan file, and a virtual objects file. It filters the info grid data
    to include only rows where the 'photoID' column contains 'rgb' (case-insensitive).

    :param image_data_file: Name of the image data file (CSV).
    :type image_data_file: str
    :param data_path: Path to the directory containing the data files.
    :type data_path: str
    :raises FileNotFoundError: If any of the specified files do not exist.
    :raises KeyError: If the 'photoID' column is missing in the info grid file.
    :raises TypeError: If `read_log_scan` or `read_virtual_objects` do not return a pandas dataframe
    :return: A tuple containing the log scan dataframe, the filtered info grid dataframe, and the virtual objects dataframe.
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    try:
        info_grid_path = os.path.join(data_path, image_data_file)
        log_scan_path = os.path.join(data_path, "LogScan.csv")
        virtual_objs_path = os.path.join(data_path, "VirtualObjects.csv")

        info_grid_data = ""
        if os.path.exists(info_grid_path):
            with open(info_grid_path, "r", encoding="utf-8") as file:
                info_grid_data = file.read()
        else:
            raise FileNotFoundError(f"LogImg.csv not found in path: {info_grid_path}")

        if ";" in info_grid_data:
            info_grid_df = pd.read_csv(info_grid_path, sep=";")
        else:
            info_grid_df = read_log_img_v2(info_grid_path)

        log_scan_data = ""

        if os.path.exists(log_scan_path):
            with open(log_scan_path, "r", encoding="utf-8") as file:
                log_scan_data = file.read()
        else:
            raise FileNotFoundError(f"LogScan.csv not found in path: {log_scan_path}")

        if ";" in log_scan_data:
            log_scan_df = read_log_scan(log_scan_path)
        else:
            log_scan_df = read_log_scan_v2(log_scan_path)

        virtual_objs_df = None
        if os.path.exists(virtual_objs_path):
            virtual_objs_df = read_virtual_objects(virtual_objs_path)

        filtered_info_grid_df = info_grid_df[
            info_grid_df["photoID"].str.lower().str.contains("rgb")
        ]

        return log_scan_df, filtered_info_grid_df, virtual_objs_df

    except FileNotFoundError as e:
        traceback.print_exc()
        raise FileNotFoundError(
            f"One or more files not found in path: {data_path} - {e}"
        )
    except KeyError:
        traceback.print_exc()
        raise KeyError("The DataFrame must contain a 'photoID' column.")
    except TypeError:
        traceback.print_exc()
        raise TypeError(
            "The functions `read_log_scan` and `read_virtual_objects` must return a pandas DataFrame"
        )
    except Exception as e:
        traceback.print_exc()
        raise Exception(
            f"An unexpected error occurred while reading the CSV files: {e}"
        )


def get_row_data(
    remove_underline: bool, data_path: str, log_scan_df: pd.DataFrame, row: dict
) -> tuple[np.ndarray, Any, Any, Any, Any, Any, Any]:
    """Extracts image data and scan information from a given row.

    This function retrieves image paths, reads image data (RGB, mask, depth), extracts scan data,
    and returns them along with position, rotation, camera position, camera rotation, and room information.

    :param remove_underline: If True, removes an underscore from the rgb path before creating the depth path
    :type remove_underline: bool
    :param data_path: The base path where image files are located.
    :type data_path: str
    :param log_scan_df: DataFrame containing scan data, with 'scan_id' and 'data' columns.
    :type log_scan_df: pd.DataFrame
    :param row: A dictionary containing information about the current image: 'robotPosition',
                'robotRotation', 'cameraPosition', 'cameraRotation', 'room', and 'photoID'.
    :type row: dict
    :raises KeyError: If the row dictionary is missing required keys.
    :raises FileNotFoundError: If any of the image files (RGB, mask, depth) are not found.
    :raises ValueError: If there's an error converting scan_id to an integer or if the photoID has a wrong format.
    :raises IndexError: If the log_scan_df does not contain the required scan_id or the data cannot be accessed.
    :raises cv2.error: If cv2.imread cannot read the image file or cv2.split or cv2.merge fails.
    :return: A tuple containing the combined image (RGB, mask, depth), scan data,
             robot position, robot rotation, camera position, camera rotation, and room.
    :rtype: tuple[np.ndarray, Any, Any, Any, Any, Any, Any]
    """
    try:

        robot_position = row["robotPosition"]
        robot_rotation = row["robotRotation"]
        camera_position = row["cameraPosition"]
        camera_rotation = row["cameraRotation"]
        room = row["room"]
        photo_id = row["photoID"]

        rgb_path = os.path.join(data_path, photo_id)
        mask_path = rgb_path.replace("rgb", "mask")
        depth_path = rgb_path.replace("rgb", "depth")

        if remove_underline:
            depth_path = rgb_path.replace("_rgb", "depth")

        try:
            scan_id = int(photo_id.split("_")[0])
        except (ValueError, IndexError) as e:
            print(f"Error processing photoID: {photo_id}, {e}")
            traceback.print_exc()
            raise ValueError(f"Invalid photoID format: {photo_id}") from e

        try:
            scan = log_scan_df[log_scan_df["scan_id"] == scan_id]["data"].values[0]
        except IndexError as e:
            print(f"Scan ID {scan_id} not found in log_scan_df")
            traceback.print_exc()
            raise IndexError(f"Scan ID {scan_id} not found in log_scan_df") from e

        try:
            rgb_image = cv2.imread(rgb_path)
            mask_image = cv2.imread(mask_path)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        except cv2.error as e:
            print(f"Error reading image: {e}")
            traceback.print_exc()
            raise FileNotFoundError(
                f"Could not read image file at: {rgb_path}, {mask_path} or {depth_path}"
            ) from e

        if rgb_image is None or mask_image is None or depth_image is None:
            raise FileNotFoundError(
                f"Could not read image file at: {rgb_path}, {mask_path} or {depth_path}"
            )

        depth_image = cv2.split(depth_image)[-1]
        depth_image = cv2.merge([depth_image, depth_image, depth_image])

        return (
            (rgb_image, mask_image, depth_image),
            scan,
            robot_position,
            robot_rotation,
            camera_position,
            camera_rotation,
            room,
        )

    except (KeyError, FileNotFoundError, ValueError, IndexError, cv2.error) as e:
        print(f"An error occurred while processing the data: {e}")
        traceback.print_exc()
        raise


def generate_distinct_color(existing_colors) -> tuple[int, int, int]:
    """
    Generate a distinct color that is not too similar to existing colors.

    :param existing_colors: List of existing colors to avoid similarity with.
    :type existing_colors: list[tuple[int, int, int]]
    :return: A tuple representing the RGB color.
    :rtype: tuple[int, int, int]
    """

    for _ in range(1000):
        h = np.random.randint(0, 360) / 360.0
        s = np.random.uniform(0.6, 1.0)
        v = np.random.uniform(0.7, 0.9)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        rgb = (int(r * 255), int(g * 255), int(b * 255))
        max_diff = max(abs(r - g), abs(r - b), abs(g - b))
        if max_diff < 0.2:
            continue
        distinct = True
        for existing in existing_colors:
            distance = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, existing)) ** 0.5
            if distance < 120:
                distinct = False
                break
        white_distance = sum((c - 255) ** 2 for c in rgb) ** 0.5
        black_distance = sum(c**2 for c in rgb) ** 0.5
        gray_similarity = 1 - (max(rgb) - min(rgb)) / 255
        if white_distance < 150 or black_distance < 150 or gray_similarity > 0.7:
            distinct = False
        if distinct:
            return rgb
    import random

    return (
        random.randint(50, 180),
        random.randint(50, 180),
        random.randint(50, 180),
    )


def draw_right_angle_arrow(
    ax,
    pos,
    edge,
    v_spacing,
    color="black",
    width=1,
    alpha=1.0,
    arrowstyle="->",
    linestyle="-",
    edge_type="",
):
    """
    Draw a right-angle arrow with custom routing based on edge type.

    :param ax: Matplotlib axis.
    :param pos: Dictionary of node positions.
    :param edge: Tuple (source, target).
    :param color: Arrow color.
    :param width: Line width.
    :param alpha: Transparency.
    :param arrowstyle: Style of the arrow.
    :param linestyle: Line style.
    :param edge_type: Type of edge routing ('env_cluster', 'cluster_child', or default).
    :param v_spacing: Vertical spacing parameter.
    """
    node1, node2 = edge
    if edge_type == "env_cluster":
        p0 = pos[node1]
        p1 = (p0[0], p0[1] - v_spacing / 2)
        p2 = (pos[node2][0], p1[1])
        p3 = pos[node2]
        path = [p0, p1, p2, p3]
    elif edge_type == "cluster_child":
        p0 = pos[node1]
        p1 = (p0[0], p0[1] - v_spacing / 2)
        p2 = (pos[node2][0], p1[1])
        # The vertical segment ends above the group node
        p3 = (pos[node2][0], pos[node2][1] + v_spacing / 4)
        path = [p0, p1, p2, p3]
    else:
        p0 = pos[node1]
        p1 = (p0[0], (p0[1] + pos[node2][1]) / 2)
        p2 = (pos[node2][0], p1[1])
        p3 = pos[node2]
        path = [p0, p1, p2, p3]

    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
        if i == len(path) - 2:
            arrow = FancyArrowPatch(
                start,
                (end[0], end[1]),
                arrowstyle=arrowstyle,
                color=color,
                linewidth=width,
                alpha=alpha,
                linestyle=linestyle,
                connectionstyle="arc3,rad=0",
            )
            ax.add_patch(arrow)
        else:
            plt.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=color,
                linewidth=width,
                alpha=alpha,
                linestyle=linestyle,
            )


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


def load_pkl_gz_result(result_path: str) -> dict:
    """
    Loads the result file from a compressed pickle.

    :param result_path: Path to the compressed pickle result file
    :type result_path: str
    :raises RuntimeError: If loading the result file fails
    :return: Dictionary containing the loaded results
    :rtype: dict
    """
    try:
        potential_path = os.path.realpath(result_path)
        if potential_path != result_path:
            result_path = potential_path

        with gzip.open(result_path, "rb") as f:
            results = pickle.load(f)

        if not isinstance(results, dict):
            raise ValueError(
                "Results should be a dictionary! other types are not supported!"
            )

        return results
    except (OSError, IOError, ValueError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load result file: {e}") from e

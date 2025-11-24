from agno.models.openrouter import OpenRouter
from collections import Counter
from dotenv import load_dotenv
from joblib import Parallel, delayed
from agno.agent import Agent
from textwrap import dedent
from typing import Tuple
import numpy as np
import traceback
import random
import pickle
import gzip
import yaml
import ast
import cv2
import os


def save_watershed_map(
    filepath: str,
    merged_regions: dict,
    region_mask: np.ndarray,
    merged_colors: dict,
    width: int,
    height: int,
) -> bool:
    """
    Saves the Watershed map data to a compressed pickle file.

    :param filepath: Path to save the file.
    :type filepath: str
    :param merged_regions: Dictionary of merged regions.
    :type merged_regions: dict
    :param region_mask: Numpy array mask of regions.
    :type region_mask: np.ndarray
    :param merged_colors: Dictionary of region colors.
    :type merged_colors: dict
    :param width: Width of the map.
    :type width: int
    :param height: Height of the map.
    :type height: int
    :raises ValueError: If dimensions or region mask are invalid.
    :raises OSError: If file cannot be saved.
    :raises IOError: If file cannot be saved.
    :raises TypeError: If data types are invalid.
    :return: True if saved successfully.
    :rtype: bool
    """
    try:
        if not filepath.endswith(".pkl.gz"):
            if filepath.endswith(".pkl"):
                filepath += ".gz"
            else:
                filepath += ".pkl.gz"

        if region_mask.size == 0 or width <= 0 or height <= 0:
            raise ValueError("Invalid dimensions or empty region mask")

        watershed_data = {
            "width": width,
            "height": height,
            "region_mask": region_mask,
            "merged_regions": merged_regions,
            "merged_colors": merged_colors,
        }

        with gzip.open(filepath, "wb") as f:
            pickle.dump(watershed_data, f)

        return True

    except (OSError, IOError, ValueError, TypeError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to save Watershed map: {e}") from e


def load_watershed_map(filepath: str) -> tuple:
    """
    Loads Watershed map data from a compressed pickle file.

    :param filepath: Path to the file.
    :type filepath: str
    :raises FileNotFoundError: If file does not exist.
    :raises ValueError: If file structure is invalid.
    :raises KeyError: If required keys are missing.
    :raises OSError: If file cannot be read.
    :raises IOError: If file cannot be read.
    :return: Tuple of merged_regions, region_mask, merged_colors, width, height.
    :rtype: tuple
    """
    try:
        if not filepath.endswith(".pkl.gz"):
            if filepath.endswith(".pkl"):
                filepath += ".gz"
            else:
                filepath += ".pkl.gz"

        with gzip.open(filepath, "rb") as f:
            watershed_data = pickle.load(f)

        required_keys = [
            "width",
            "height",
            "region_mask",
            "merged_regions",
            "merged_colors",
        ]
        if not all(key in watershed_data for key in required_keys):
            raise ValueError("Invalid pickle structure: missing required keys")

        width = watershed_data["width"]
        height = watershed_data["height"]
        region_mask = watershed_data["region_mask"]
        merged_regions = watershed_data["merged_regions"]
        merged_colors = watershed_data["merged_colors"]

        return merged_regions, region_mask, merged_colors, width, height

    except (FileNotFoundError, ValueError, KeyError, OSError, IOError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load Watershed map: {e}") from e


def reconstruct_watershed_image(
    merged_regions: dict,
    region_mask: np.ndarray,
    merged_colors: dict,
    width: int,
    height: int,
    doors_mask: np.ndarray,
    map_image: np.ndarray,
) -> np.ndarray:
    """
    Reconstructs the Watershed image from loaded data.

    :param merged_regions: Dictionary of merged regions.
    :type merged_regions: dict
    :param region_mask: Numpy array mask of regions.
    :type region_mask: np.ndarray
    :param merged_colors: Dictionary of region colors.
    :type merged_colors: dict
    :param width: Width of the image.
    :type width: int
    :param height: Height of the image.
    :type height: int
    :param doors_mask: Numpy array mask of doors.
    :type doors_mask: np.ndarray
    :param map_image: Original map image.
    :type map_image: np.ndarray
    :raises ValueError: If dimensions are invalid.
    :return: Reconstructed image as a numpy array.
    :rtype: np.ndarray
    """
    try:
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions")

        if region_mask.shape != (height, width):
            raise ValueError("Region mask dimensions do not match image dimensions")

        reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                merged_region_id = region_mask[y, x]
                if merged_region_id in merged_colors:
                    reconstructed_image[y, x] = merged_colors[merged_region_id]

        # reconstructed_image[doors_mask > 0] = (0, 0, 255)
        reconstructed_image[map_image == 0] = (0, 0, 0)

        try:
            if reconstructed_image.shape[:2] != map_image.shape:
                raise ValueError(
                    "Shape mismatch between reconstructed_image and map_image."
                )

            mask_floor = map_image > 0
            mask_black = np.all(reconstructed_image == [0, 0, 0], axis=-1)
            mask = mask_floor & mask_black
            reconstructed_image[mask] = (200, 200, 200)
        except (ValueError, IndexError) as e:
            traceback.print_exc()
            raise RuntimeError(
                f"Failed to update floor pixels in reconstructed image: {e}"
            ) from e

        for merged_id, region_data in merged_regions.items():
            region_pixels = np.where(region_mask == int(merged_id))

            if len(region_pixels[0]) > 0:
                center_y = int(np.mean(region_pixels[0]))
                center_x = int(np.mean(region_pixels[1]))

                center_y = max(0, min(height - 1, center_y))
                center_x = max(0, min(width - 1, center_x))

                dominant_class = region_data.get("dominant_class", "unknown")

                text_size = 0.3
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    dominant_class, cv2.FONT_HERSHEY_SIMPLEX, text_size, thickness
                )

                text_x = center_x - text_width // 2
                text_y = center_y + text_height // 2

                cv2.putText(
                    reconstructed_image,
                    dominant_class,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_size,
                    (0, 0, 0),
                    thickness + 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    reconstructed_image,
                    dominant_class,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_size,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        return reconstructed_image

    except ValueError as e:
        traceback.print_exc()
        raise RuntimeError(f"Error reconstructing Watershed image: {e}") from e


def segment_map_watershed(
    image: np.ndarray,
    door_size_px: int = 17,
    min_wall_thickness: int = 5,
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Segments the map using Watershed algorithm ensuring doors only exist between walls with minimum thickness.

    :param image: Input grayscale image.
    :type image: np.ndarray
    :param door_size_px: Door kernel size in pixels.
    :type door_size_px: int
    :param min_wall_thickness: Minimum wall thickness for door detection.
    :type min_wall_thickness: int
    :raises ValueError: If image dimensions are invalid.
    :return: Tuple of colored_map, final_mask, regions_dict, doors_mask.
    :rtype: tuple[np.ndarray, np.ndarray, dict, np.ndarray]
    """
    height, width = image.shape[:2]

    bin_img = np.zeros_like(image)
    bin_img[image > 0] = 255

    walls = cv2.bitwise_not(bin_img)

    wall_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (min_wall_thickness, min_wall_thickness)
    )
    thick_walls = cv2.erode(walls, wall_kernel, iterations=1)
    thick_walls = cv2.dilate(thick_walls, wall_kernel, iterations=1)

    robust_floor_map = cv2.bitwise_not(thick_walls)

    door_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (door_size_px, door_size_px)
    )

    sure_rooms = cv2.morphologyEx(robust_floor_map, cv2.MORPH_OPEN, door_kernel)

    doors_mask_raw = cv2.subtract(robust_floor_map, sure_rooms)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    doors_mask = cv2.morphologyEx(doors_mask_raw, cv2.MORPH_OPEN, clean_kernel)

    doors_mask = cv2.bitwise_and(doors_mask, bin_img)

    ret, markers = cv2.connectedComponents(sure_rooms)
    markers = markers + 1

    sure_bg = np.zeros_like(image)
    sure_bg[image == 0] = 255

    unknown = cv2.subtract(bin_img, sure_rooms)
    markers[unknown == 255] = 0
    markers[image == 0] = -1

    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    region_mask = markers.copy()
    region_mask[region_mask <= 1] = 0
    region_mask[image == 0] = 0

    floor_pixels = image > 0
    unlabeled_floor = (region_mask == 0) & floor_pixels

    if np.any(unlabeled_floor):
        labeled_y, labeled_x = np.where(region_mask > 0)
        if len(labeled_y) > 0:
            labeled_coords = np.column_stack((labeled_y, labeled_x))
            labeled_values = region_mask[labeled_y, labeled_x]
            empty_y, empty_x = np.where(unlabeled_floor)
            for i in range(len(empty_y)):
                uy, ux = empty_y[i], empty_x[i]
                dists = (labeled_coords[:, 0] - uy) ** 2 + (
                    labeled_coords[:, 1] - ux
                ) ** 2
                nearest_idx = np.argmin(dists)
                region_mask[uy, ux] = labeled_values[nearest_idx]

    final_mask = np.full_like(region_mask, -1)
    valid_regions = region_mask > 1
    final_mask[valid_regions] = region_mask[valid_regions] - 2

    unique_ids = np.unique(final_mask)
    unique_ids = unique_ids[unique_ids >= 0]

    colored_map = np.zeros((height, width, 3), dtype=np.uint8)
    regions_dict = {}

    for uid in unique_ids:
        color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
        mask_this_region = final_mask == uid
        colored_map[mask_this_region] = color

        ys, xs = np.where(mask_this_region)
        cX, cY = 0, 0
        polygon = []
        if len(xs) > 0:
            cX, cY = int(np.mean(xs)), int(np.mean(ys))
            region_u8 = np.zeros((height, width), dtype=np.uint8)
            region_u8[mask_this_region] = 255
            contours, _ = cv2.findContours(
                region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                polygon = largest.reshape(-1, 2).tolist()

        regions_dict[uid] = {
            "polygon": polygon,
            "center": (cX, cY),
            "color": color,
            "pixels": list(zip(ys, xs)),
        }

    return colored_map, final_mask, regions_dict, doors_mask


def load_pkl_gz_result(result_path: str) -> dict:
    """
    Loads the result file from a compressed pickle.

    :param result_path: Path to the result file.
    :type result_path: str
    :raises ValueError: If loaded data is not a dictionary.
    :raises OSError: If file cannot be read.
    :raises IOError: If file cannot be read.
    :return: Loaded results dictionary.
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


class RoomData:
    """
    Represents room data and provides coordinate conversion utilities.
    """

    def __init__(self, room_data: dict):
        """
        Initializes RoomData with room information.

        :param room_data: Dictionary containing room information.
        :type room_data: dict
        :raises KeyError: If required keys are missing.
        :raises ValueError: If pose cannot be parsed.
        """
        try:
            self.class_name = room_data["room_class"]
            self.description = room_data["room_description"]
            raw_pose = tuple(ast.literal_eval(room_data["pose"]))
            u_x = raw_pose[0]
            u_y = raw_pose[1]
            u_z = raw_pose[2]
            self.ros_x = float(u_z)
            self.ros_y = float(-u_x)
            if len(raw_pose) >= 5:
                self.ros_yaw = -raw_pose[4]
            else:
                self.ros_yaw = 0
        except (KeyError, ValueError, SyntaxError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize RoomData: {e}") from e

    def get_map_coordinates(
        self, origin: Tuple[float, float], resolution: float, map_image: np.ndarray
    ) -> Tuple[int, int]:
        """
        Converts ROS world coordinates to map pixel coordinates.

        :param origin: Origin of the map.
        :type origin: tuple[float, float]
        :param resolution: Map resolution.
        :type resolution: float
        :param map_image: Map image array.
        :type map_image: np.ndarray
        :return: Pixel coordinates (x, y).
        :rtype: tuple[int, int]
        """
        pixel_x = int((self.ros_x - origin[0]) / resolution)
        pixel_y = int(map_image.shape[0] - ((self.ros_y - origin[1]) / resolution))
        return pixel_x, pixel_y


def resolve_dominant_class_with_bias(
    class_counts: dict, bias_threshold: float = 1.10
) -> str:
    """
    Returns the dominant class applying a penalty to transitioning class.

    :param class_counts: Dictionary of class counts.
    :type class_counts: dict
    :param bias_threshold: Threshold multiplier for transitioning bias.
    :type bias_threshold: float
    :return: Dominant class name.
    :rtype: str
    """
    if not class_counts:
        return "unknown"

    sorted_classes = sorted(
        class_counts.items(), key=lambda item: item[1], reverse=True
    )

    top_class, top_count = sorted_classes[0]

    if top_class != "transitioning":
        return top_class

    if len(sorted_classes) > 1:
        runner_up_class, runner_up_count = sorted_classes[1]

        if top_count <= (runner_up_count * bias_threshold):
            return runner_up_class

    return "transitioning"


def analyze_boundary_permeability(
    region_id_a: int,
    region_id_b: int,
    region_mask: np.ndarray,
    doors_mask: np.ndarray,
    door_ratio_threshold: float = 0.30,
) -> bool:
    """
    Decides if two regions should be merged by analyzing boundary proportion.

    :param region_id_a: First region ID.
    :type region_id_a: int
    :param region_id_b: Second region ID.
    :type region_id_b: int
    :param region_mask: Region mask array.
    :type region_mask: np.ndarray
    :param doors_mask: Doors mask array.
    :type doors_mask: np.ndarray
    :param door_ratio_threshold: Threshold for door ratio.
    :type door_ratio_threshold: float
    :return: True if merge is allowed, False if blocked.
    :rtype: bool
    """
    mask_a = (region_mask == region_id_a).astype(np.uint8)
    mask_b = (region_mask == region_id_b).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    dilated_a = cv2.dilate(mask_a, kernel, iterations=1)

    common_boundary = (dilated_a == 1) & (mask_b == 1)
    total_boundary_pixels = np.count_nonzero(common_boundary)

    if total_boundary_pixels == 0:
        return True

    door_overlap = common_boundary & (doors_mask > 0)
    door_pixels = np.count_nonzero(door_overlap)

    if door_pixels == 0:
        return True

    door_ratio = door_pixels / total_boundary_pixels

    if door_ratio > door_ratio_threshold:
        return False

    return True


def absorb_unvisited_regions(
    region_mask: np.ndarray, regions_dict: dict, region_class_counts: dict
) -> Tuple[np.ndarray, dict]:
    """
    Absorbs unvisited regions into adjacent visited neighbors.

    :param region_mask: Region mask array.
    :type region_mask: np.ndarray
    :param regions_dict: Dictionary of regions data.
    :type regions_dict: dict
    :param region_class_counts: Dictionary of region class counts.
    :type region_class_counts: dict
    :return: Updated region_mask and regions_dict.
    :rtype: tuple[np.ndarray, dict]
    """
    height, width = region_mask.shape

    visited_ids = set()
    for rid, counts in region_class_counts.items():
        if sum(counts.values()) > 0:
            visited_ids.add(rid)

    print(
        f"Initial State: {len(regions_dict)} regions, {len(visited_ids)} are visited."
    )

    max_iterations = 20

    for i in range(max_iterations):
        changes = 0

        current_ids = np.unique(region_mask)
        current_ids = current_ids[current_ids >= 0]

        unvisited_present = [uid for uid in current_ids if uid not in visited_ids]
        if not unvisited_present:
            break

        for unvisited_id in unvisited_present:

            mask_u = (region_mask == unvisited_id).astype(np.uint8)
            dilated = cv2.dilate(mask_u, np.ones((3, 3), np.uint8))

            neighbors = np.unique(region_mask[dilated == 1])
            neighbors = neighbors[neighbors != unvisited_id]
            neighbors = neighbors[neighbors >= 0]

            best_neighbor = -1

            valid_neighbors = [n for n in neighbors if n in visited_ids]

            if valid_neighbors:
                best_neighbor = valid_neighbors[0]

            if best_neighbor != -1:
                region_mask[region_mask == unvisited_id] = best_neighbor
                changes += 1

        print(f"Absorption iteration {i+1}: {changes} regions absorbed.")
        if changes == 0:
            break

    new_regions_dict = {}
    final_ids = np.unique(region_mask)
    final_ids = final_ids[final_ids >= 0]

    for uid in final_ids:
        mask_this = region_mask == uid
        ys, xs = np.where(mask_this)
        if len(xs) > 0:
            cX, cY = int(np.mean(xs)), int(np.mean(ys))

            new_regions_dict[uid] = {
                "center": (cX, cY),
                "pixels": list(zip(ys, xs)),
                "color": regions_dict.get(uid, {}).get("color", (128, 128, 128)),
            }

    return region_mask, new_regions_dict


def promote_connecting_doors(
    region_mask: np.ndarray,
    doors_mask: np.ndarray,
    regions_dict: dict,
    min_area: int = 30,
) -> Tuple[np.ndarray, dict, set]:
    """
    Promotes door areas to transitioning regions if they connect distinct fragments.

    :param region_mask: Region mask array.
    :type region_mask: np.ndarray
    :param doors_mask: Doors mask array.
    :type doors_mask: np.ndarray
    :param regions_dict: Dictionary of regions data.
    :type regions_dict: dict
    :param min_area: Minimum area threshold.
    :type min_area: int
    :return: Updated region_mask, regions_dict, and set of new transitioning IDs.
    :rtype: tuple[np.ndarray, dict, set]
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        doors_mask, connectivity=8
    )

    current_max_id = np.max(region_mask)
    next_id = current_max_id + 1
    new_transitioning_ids = set()

    kernel = np.ones((3, 3), np.uint8)

    print(f"Analyzing {num_labels - 1} door components for connectivity...")

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            doors_mask[labels == i] = 0
            continue

        this_door_mask = (labels == i).astype(np.uint8)

        dilated_door = cv2.dilate(this_door_mask, kernel, iterations=1)

        neighbor_ids = np.unique(region_mask[dilated_door == 1])

        valid_neighbors = [nid for nid in neighbor_ids if nid >= 0]

        if len(valid_neighbors) >= 2:
            region_mask[this_door_mask == 1] = next_id

            ys, xs = np.where(this_door_mask == 1)
            cX, cY = int(np.mean(xs)), int(np.mean(ys))

            regions_dict[next_id] = {
                "dominant_class": "transitioning",
                "center": (cX, cY),
                "pixels": list(zip(ys, xs)),
                "color": (128, 128, 128),
                "room_descriptions": [
                    "Structural passageway connecting distinct regions."
                ],
            }

            new_transitioning_ids.add(next_id)
            next_id += 1

    print(
        f"Promoted {len(new_transitioning_ids)} door areas to 'transitioning' (Distinct connections confirmed)."
    )
    return region_mask, regions_dict, new_transitioning_ids


def split_regions_by_doors(
    region_mask: np.ndarray,
    doors_mask: np.ndarray,
    regions_dict: dict,
    min_fragment_size: int = 50,
) -> Tuple[np.ndarray, dict]:
    """
    Splits regions if internal doors cause topological disconnection.

    :param region_mask: Region mask array.
    :type region_mask: np.ndarray
    :param doors_mask: Doors mask array.
    :type doors_mask: np.ndarray
    :param regions_dict: Dictionary of regions data.
    :type regions_dict: dict
    :param min_fragment_size: Minimum fragment size threshold.
    :type min_fragment_size: int
    :return: Updated region_mask and regions_dict.
    :rtype: tuple[np.ndarray, dict]
    """
    height, width = region_mask.shape
    current_max_id = np.max(region_mask)
    next_id = current_max_id + 1

    original_ids = list(regions_dict.keys())
    split_count = 0

    print("Checking for regions that need splitting by internal doors...")

    for region_id in original_ids:
        mask_region = (region_mask == region_id).astype(np.uint8)

        internal_door_pixels = (mask_region == 1) & (doors_mask > 0)

        if not np.any(internal_door_pixels):
            continue

        cut_mask = cv2.subtract(mask_region, internal_door_pixels.astype(np.uint8))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cut_mask, connectivity=4
        )

        if num_labels > 2:
            valid_parts = []
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_fragment_size:
                    valid_parts.append(i)

            if len(valid_parts) >= 2:
                print(
                    f"  -> Splitting Region {region_id} into {len(valid_parts)} parts due to internal door."
                )

                part1_label = valid_parts[0]
                mask_p1 = labels == part1_label

                region_mask[region_mask == region_id] = -1
                region_mask[mask_p1] = region_id

                ys, xs = np.where(mask_p1)
                cX, cY = int(np.mean(xs)), int(np.mean(ys))
                regions_dict[region_id]["pixels"] = list(zip(ys, xs))
                regions_dict[region_id]["center"] = (cX, cY)

                for k in range(1, len(valid_parts)):
                    part_label = valid_parts[k]
                    mask_pk = labels == part_label

                    region_mask[mask_pk] = next_id

                    ys, xs = np.where(mask_pk)
                    cX, cY = int(np.mean(xs)), int(np.mean(ys))

                    base_color = regions_dict[region_id].get("color", (128, 128, 128))
                    regions_dict[next_id] = {
                        "center": (cX, cY),
                        "pixels": list(zip(ys, xs)),
                        "color": base_color,
                    }

                    next_id += 1
                split_count += 1

    print(f"Total regions split: {split_count}")
    return region_mask, regions_dict


def ensure_trajectory_connectivity(
    region_mask: np.ndarray,
    room_data_list: list,
    origin: Tuple[float, float],
    resolution: float,
    map_image: np.ndarray,
) -> np.ndarray:
    """
    Ensures trajectory points fall within valid regions by filling gaps.

    :param region_mask: Region mask array.
    :type region_mask: np.ndarray
    :param room_data_list: List of RoomData objects.
    :type room_data_list: list
    :param origin: Map origin coordinates.
    :type origin: tuple[float, float]
    :param resolution: Map resolution.
    :type resolution: float
    :param map_image: Map image array.
    :type map_image: np.ndarray
    :return: Updated region_mask.
    :rtype: np.ndarray
    """
    height, width = region_mask.shape
    changes = 0

    traj_pixels = []
    for room_data in room_data_list:
        x, y = room_data.get_map_coordinates(origin, resolution, map_image)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        traj_pixels.append((y, x))

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    traj_pixels.append((ny, nx))

    search_radius = 5

    for y, x in traj_pixels:
        current_val = region_mask[y, x]

        if current_val < 0:
            found_neighbor = -1
            min_dist = 999

            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        n_val = region_mask[ny, nx]
                        if n_val >= 0:
                            dist = dy**2 + dx**2
                            if dist < min_dist:
                                min_dist = dist
                                found_neighbor = n_val

            if found_neighbor != -1:
                region_mask[y, x] = found_neighbor
                changes += 1

    print(f"Trajectory Safety: Filled {changes} gaps along the robot path.")
    return region_mask


def fill_remaining_gaps(region_mask: np.ndarray) -> np.ndarray:
    """
    Removes boundary lines between colored regions by assigning them to the majority neighbor.

    :param region_mask: Region mask array.
    :type region_mask: np.ndarray
    :return: Updated region_mask.
    :rtype: np.ndarray
    """
    height, width = region_mask.shape

    valid_mask = (region_mask >= 0).astype(np.uint8)

    if np.all(valid_mask):
        return region_mask

    kernel = np.ones((3, 3), np.uint8)
    has_neighbor = cv2.dilate(valid_mask, kernel) & (~valid_mask)

    gaps_y, gaps_x = np.where(has_neighbor)

    if len(gaps_y) == 0:
        return region_mask

    for i in range(len(gaps_y)):
        y, x = gaps_y[i], gaps_x[i]
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    val = region_mask[ny, nx]
                    if val >= 0:
                        neighbors.append(val)

        if neighbors:
            most_common = Counter(neighbors).most_common(1)[0][0]
            region_mask[y, x] = most_common

    return region_mask


if __name__ == "__main__":
    load_dotenv()

    SELECTED_HOUSE = 2
    PREFFIX = "offline"
    MAP_PATH = (
        f"D:\\Documentos\\Datasets\\Robot@VirtualHomeLarge\\Home{SELECTED_HOUSE:02d}"
    )
    OBJECTS_PATH = f"data\\{PREFFIX}_house_{SELECTED_HOUSE}_map\\pcd_{PREFFIX}_house_{SELECTED_HOUSE}_map.pkl.gz"

    DOOR_SIZE_PX = 18
    MIN_WALL_THICKNESS = 3
    DOOR_RATIO_THRESH = 0.30
    BIAS_THRESH = 1.10

    print("Loading objects and LLM agent...")
    llm_agent = Agent(
        model=OpenRouter(
            id="google/gemini-2.5-flash-lite",
            api_key=os.environ["OPENROUTER_API_KEY"],
        ),
        system_message=dedent(
            """
            PERSONA:
            A natural language processing expert, focused on following strict instructions.
            TASK:
            Generate a **single paragraph description** of a specific environment.
            MANDATORY REQUIREMENTS:
            - The output **MUST be a single paragraph only**.
            - The output **MUST begin with the exact phrase "It's a [environment class]..."**.
            - Simple, clear sentences are **REQUIRED**.
            """
        ),
    )

    print("Loading map data and settings...")
    prompt_mask = "The environment is a {} and these are its descriptions: {}"
    results = load_pkl_gz_result(OBJECTS_PATH)
    room_data_list = [RoomData(room_data) for room_data in results["room_data_list"]]

    map_image_path = os.path.join(MAP_PATH, "generated_map.png")
    map_settings_path = os.path.join(MAP_PATH, "generated_map.yaml")

    map_image = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
    with open(map_settings_path, "r") as f:
        map_settings = yaml.safe_load(f)

    print("Pre-processing map image...")
    map_image[map_image < 250] = 0
    contours, _ = cv2.findContours(
        map_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            cv2.drawContours(map_image, [contour], -1, (0, 0, 0), -1)

    try:
        white_pixels = np.where(map_image > 0)
        if len(white_pixels[0]) > 0:
            min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
            min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
            padding = 5
            min_y = max(0, min_y - padding)
            min_x = max(0, min_x - padding)
            max_y = min(map_image.shape[0] - 1, max_y + padding)
            max_x = min(map_image.shape[1] - 1, max_x + padding)
            map_image = map_image[min_y : max_y + 1, min_x : max_x + 1]
    except (IndexError, ValueError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Error cropping image: {str(e)}") from e

    print("Generating Topological Segmentation (Morphological Watershed)...")
    watershed_image, region_mask, regions_dict, doors_mask = segment_map_watershed(
        map_image, door_size_px=DOOR_SIZE_PX, min_wall_thickness=MIN_WALL_THICKNESS
    )
    origin = map_settings["origin"]
    resolution = map_settings["resolution"]
    height, width = region_mask.shape

    print("Counting trajectory points per region (Ignoring points on doors)...")
    region_class_counts = {}
    for room_data in room_data_list:
        x, y = room_data.get_map_coordinates(origin, resolution, watershed_image)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        if doors_mask[y, x] > 0:
            continue
        region_id = region_mask[y, x]
        if region_id == -1:
            continue
        if region_id not in region_class_counts:
            region_class_counts[region_id] = {}
        room_class = room_data.class_name
        if room_class not in region_class_counts[region_id]:
            region_class_counts[region_id][room_class] = 0
        region_class_counts[region_id][room_class] += 1

    print("Absorbing unvisited regions into visited neighbors...")
    region_mask, regions_dict = absorb_unvisited_regions(
        region_mask, regions_dict, region_class_counts
    )

    print("Splitting large regions intersected by doors...")
    region_mask, regions_dict = split_regions_by_doors(
        region_mask, doors_mask, regions_dict, min_fragment_size=50
    )

    print("Recalculating points for new fragments...")
    region_class_counts = {}
    for room_data in room_data_list:
        x, y = room_data.get_map_coordinates(origin, resolution, watershed_image)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        if doors_mask[y, x] > 0:
            continue
        region_id = region_mask[y, x]
        if region_id == -1:
            continue
        if region_id not in region_class_counts:
            region_class_counts[region_id] = {}
        room_class = room_data.class_name
        if room_class not in region_class_counts[region_id]:
            region_class_counts[region_id][room_class] = 0
        region_class_counts[region_id][room_class] += 1

    print("Promoting connecting doors to 'transitioning' fragments...")
    region_mask, regions_dict, promoted_ids = promote_connecting_doors(
        region_mask, doors_mask, regions_dict, min_area=600
    )
    structural_door_regions = set(promoted_ids)

    print("Determining dominant classes...")
    class_colors = {
        "kitchen": (255, 99, 71),
        "bathroom": (135, 206, 250),
        "bedroom": (186, 85, 211),
        "living room": (60, 179, 113),
        "office": (255, 215, 0),
        "hallway": (255, 140, 0),
        "laundry room": (70, 130, 180),
        "transitioning": (128, 128, 128),
    }
    remaining_ids = list(regions_dict.keys())
    for region_id in remaining_ids:
        if region_id in structural_door_regions:
            continue
        counts = region_class_counts.get(region_id, {})
        if not counts:
            regions_dict[region_id]["dominant_class"] = "unknown"
            continue
        dominant_class = resolve_dominant_class_with_bias(
            counts, bias_threshold=BIAS_THRESH
        )
        regions_dict[region_id]["dominant_class"] = dominant_class

    print("Rebuilding adjacency graph...")
    adjacency_list = {}
    height, width = region_mask.shape
    for y in range(height):
        for x in range(width):
            current_region = region_mask[y, x]
            if current_region == -1:
                continue
            if current_region not in adjacency_list:
                adjacency_list[current_region] = set()
            for dy, dx in [(0, 1), (1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_region = region_mask[ny, nx]
                    if neighbor_region != current_region and neighbor_region != -1:
                        adjacency_list[current_region].add(neighbor_region)
                        if neighbor_region not in adjacency_list:
                            adjacency_list[neighbor_region] = set()
                        adjacency_list[neighbor_region].add(current_region)

    print("Resolving 'unknown' regions by adopting neighbor classes...")
    unknown_resolved_count = 0
    max_unknown_iters = 5

    for i in range(max_unknown_iters):
        changes = 0
        for region_id in regions_dict:
            current_class = regions_dict[region_id].get("dominant_class", "unknown")

            if current_class == "unknown":
                neighbors = adjacency_list.get(region_id, set())
                valid_neighbors = []

                for neighbor in neighbors:
                    if neighbor in regions_dict:
                        n_class = regions_dict[neighbor].get(
                            "dominant_class", "unknown"
                        )
                        if n_class != "unknown":
                            valid_neighbors.append(n_class)

                if valid_neighbors:
                    most_common = Counter(valid_neighbors).most_common(1)[0][0]
                    regions_dict[region_id]["dominant_class"] = most_common
                    changes += 1
                    unknown_resolved_count += 1

        if changes == 0:
            break

    print(f"Resolved {unknown_resolved_count} unknown regions.")

    print("Merging connected regions...")
    merged_regions = {}
    region_mapping = {}
    next_merged_id = 0
    visited = set()
    sorted_ids = sorted(list(regions_dict.keys()))

    for region_id in sorted_ids:
        if region_id in visited:
            continue
        if "dominant_class" not in regions_dict[region_id]:
            continue

        dominant_class = regions_dict[region_id]["dominant_class"]
        queue = [region_id]
        component = []
        visited.add(region_id)

        while queue:
            current = queue.pop(0)
            component.append(current)
            if current in adjacency_list:
                for neighbor in adjacency_list[current]:
                    if neighbor not in visited:
                        if (
                            neighbor in regions_dict
                            and "dominant_class" in regions_dict[neighbor]
                        ):
                            neighbor_class = regions_dict[neighbor]["dominant_class"]
                            should_merge = False

                            if (
                                dominant_class == "transitioning"
                                and neighbor_class == "transitioning"
                            ):
                                should_merge = True

                            elif (
                                dominant_class == "kitchen"
                                and neighbor_class == "kitchen"
                            ):
                                should_merge = True

                            elif neighbor_class == dominant_class:
                                should_merge = analyze_boundary_permeability(
                                    current,
                                    neighbor,
                                    region_mask,
                                    doors_mask,
                                    door_ratio_threshold=DOOR_RATIO_THRESH,
                                )

                            if should_merge:
                                visited.add(neighbor)
                                queue.append(neighbor)

        room_descriptions = []
        combined_pixels = []
        for orig_region in component:
            region_mapping[orig_region] = next_merged_id
            if "pixels" in regions_dict[orig_region]:
                combined_pixels.extend(regions_dict[orig_region]["pixels"])
            for room_data in room_data_list:
                x, y = room_data.get_map_coordinates(
                    origin, resolution, watershed_image
                )
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                if region_mask[y, x] == orig_region:
                    if hasattr(room_data, "description") and room_data.description:
                        room_descriptions.append(room_data.description)

        merged_regions[next_merged_id] = {
            "original_regions": component,
            "dominant_class": dominant_class,
            "pixels": combined_pixels,
            "room_descriptions": room_descriptions,
        }
        next_merged_id += 1

    print("Generating room descriptions using LLM...")

    def summarize_region_descriptions(
        rid: int, rdata: dict, pmask: str, agent: Agent
    ) -> tuple:
        """
        Summarizes region descriptions using LLM agent.

        :param rid: Region ID.
        :type rid: int
        :param rdata: Region data dictionary.
        :type rdata: dict
        :param pmask: Prompt mask string.
        :type pmask: str
        :param agent: LLM Agent instance.
        :type agent: Agent
        :return: Tuple of region ID and description.
        :rtype: tuple
        """
        d_class = rdata["dominant_class"]
        combined_desc = "\n".join(
            [d for d in rdata["room_descriptions"] if isinstance(d, str)]
        )
        if not combined_desc.strip():
            return (rid, f"It's a {d_class} with no specific details observed.")
        try:
            return (rid, agent.run(pmask.format(d_class, combined_desc)).content)
        except (RuntimeError, ValueError, KeyError):
            traceback.print_exc()
            return (rid, f"It's a {d_class}.")

    llm_results = Parallel(n_jobs=-1)(
        delayed(summarize_region_descriptions)(
            rid, merged_regions[rid], prompt_mask, llm_agent
        )
        for rid in merged_regions
    )
    for rid, desc in llm_results:
        merged_regions[rid]["summarized_description"] = desc

    print("Updating final region mappings...")
    for y in range(height):
        for x in range(width):
            original_region = region_mask[y, x]
            if original_region in region_mapping:
                region_mask[y, x] = region_mapping[original_region]
            else:
                if original_region != -1:
                    region_mask[y, x] = -1

    print("Generating colors...")
    merged_colors = {}
    for merged_id in merged_regions:
        merged_colors[merged_id] = (
            np.random.randint(50, 256),
            np.random.randint(50, 256),
            np.random.randint(50, 256),
        )

    print("Applying final connectivity fixes...")

    region_mask = fill_remaining_gaps(region_mask)

    region_mask = ensure_trajectory_connectivity(
        region_mask,
        room_data_list,
        origin,
        resolution,
        watershed_image,
    )

    for uid in regions_dict:
        ys, xs = np.where(region_mask == uid)
        if len(xs) > 0:
            regions_dict[uid]["pixels"] = list(zip(ys, xs))

    print("Saving Watershed map...")
    save_watershed_map(
        "watershed_map.pkl", merged_regions, region_mask, merged_colors, width, height
    )
    print("Reconstructing final image...")
    reconstructed_image = reconstruct_watershed_image(
        merged_regions, region_mask, merged_colors, width, height, doors_mask, map_image
    )

    cv2.namedWindow("Reconstructed Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Reconstructed Image", reconstructed_image)
    cv2.namedWindow("Watershed Raw", cv2.WINDOW_NORMAL)
    cv2.imshow("Watershed Raw", watershed_image)

    print("Generating trajectory image...")
    trajectory_image = cv2.cvtColor(map_image.copy(), cv2.COLOR_GRAY2BGR)
    trajectory_image = (trajectory_image * 0.6).astype(np.uint8)
    for room_data in room_data_list:
        x, y = room_data.get_map_coordinates(origin, resolution, trajectory_image)
        x, y = max(0, min(x, width - 1)), max(0, min(y, height - 1))
        cls_name = room_data.class_name
        rgb_color = class_colors.get(cls_name, (128, 128, 128))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        cv2.circle(trajectory_image, (x, y), 4, bgr_color, -1)
        cv2.circle(trajectory_image, (x, y), 4, (0, 0, 0), 1)

    cv2.namedWindow("Robot Trajectory", cv2.WINDOW_NORMAL)
    cv2.imshow("Robot Trajectory", trajectory_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# TODO: Resolver o problema de merge das regiões unknown
# TODO: Simplificar a geração do mapa
# TODO: Utilizar os objetos para fazer um veredito da classe do cômodo
# TODO: Corrigir alguns problemas de geração do mapa topológico para as casas: 5, 14, 18, 27

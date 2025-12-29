from typing import Tuple, Dict, Any, List, Optional
from agno.models.openrouter import OpenRouter
from joblib import Parallel, delayed
from collections import Counter
from dotenv import load_dotenv
from agno.agent import Agent
from textwrap import dedent
import numpy as np
import traceback
import random
import pickle
import gzip
import yaml
import json
import ast
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from labs.utils import load_pkl_gz_result


MAP_BINARY_THRESHOLD = 250
MIN_CONTOUR_AREA = 100
CROP_PADDING = 5
TRAJECTORY_IMAGE_DIMMING = 0.6
DEFAULT_PARAMS = {
    "door_size_px": 18,
    "door_ratio_thresh": 0.30,
    "bias_thresh": 2.0,
    "min_fragment_size": 50,
    "min_door_area": 60,
    "max_unknown_iters": 5,
    "min_noise_area": 10,
    "door_dist_factor": 1.1,
    "door_mask_scale": 0.1,
}
CLASS_COLORS = {
    "kitchen": (255, 99, 71),
    "bathroom": (135, 206, 250),
    "bedroom": (186, 85, 211),
    "living room": (60, 179, 113),
    "office": (255, 215, 0),
    "hallway": (255, 140, 0),
    "laundry room": (70, 130, 180),
    "transitioning": (128, 128, 128),
}


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

    :param filepath: Path where the compressed pickle file will be saved
    :type filepath: str
    :param merged_regions: Dictionary containing merged region information
    :type merged_regions: dict
    :param region_mask: NumPy array representing the region mask
    :type region_mask: np.ndarray
    :param merged_colors: Dictionary mapping region IDs to their colors
    :type merged_colors: dict
    :param width: Width of the map in pixels
    :type width: int
    :param height: Height of the map in pixels
    :type height: int
    :raises RuntimeError: If saving the Watershed map fails due to file I/O or data errors
    :return: True if the save operation was successful
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

    :param filepath: Path to the compressed pickle file
    :type filepath: str
    :raises RuntimeError: If loading the Watershed map fails due to file not found or invalid structure
    :return: Tuple containing merged regions, region mask, merged colors, width, and height
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
    Reconstructs the Watershed image from loaded data with region labels.

    :param merged_regions: Dictionary containing merged region information
    :type merged_regions: dict
    :param region_mask: NumPy array representing the region mask
    :type region_mask: np.ndarray
    :param merged_colors: Dictionary mapping region IDs to their colors
    :type merged_colors: dict
    :param width: Width of the map in pixels
    :type width: int
    :param height: Height of the map in pixels
    :type height: int
    :param doors_mask: Binary mask indicating door locations
    :type doors_mask: np.ndarray
    :param map_image: Original grayscale map image
    :type map_image: np.ndarray
    :raises RuntimeError: If reconstruction fails due to dimension mismatch or data errors
    :return: Reconstructed color image with region labels
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


def filter_noise_contours(mask: np.ndarray, min_area: int) -> np.ndarray:
    """
    Removes small noise blobs from a binary mask using contour filtering.

    :param mask: Binary mask to filter
    :type mask: np.ndarray
    :param min_area: Minimum contour area to keep
    :type min_area: int
    :return: Cleaned binary mask
    :rtype: np.ndarray
    """
    if min_area <= 0:
        return mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    return clean_mask


def segment_map_watershed(
    image: np.ndarray,
    door_size_px: int = 17,
    min_noise_area: int = 10,
    door_dist_factor: float = 1.1,
    door_mask_scale: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Segments the map using Watershed algorithm with distance transform and contour filtering.

    :param image: Grayscale input map image
    :type image: np.ndarray
    :param door_size_px: Expected door size in pixels
    :type door_size_px: int
    :param min_noise_area: Minimum area for noise filtering
    :type min_noise_area: int
    :param door_dist_factor: Distance threshold factor for room identification
    :type door_dist_factor: float
    :param door_mask_scale: Scale factor for door mask thickness
    :type door_mask_scale: float
    :return: Tuple containing colored map, region mask, regions dictionary, and doors mask
    :rtype: Tuple[np.ndarray, np.ndarray, dict, np.ndarray]
    """
    height, width = image.shape[:2]

    bin_img = np.zeros_like(image)
    bin_img[image > 0] = 255

    bin_img = filter_noise_contours(bin_img, min_noise_area)

    dist_transform = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)

    dist_thresh = (door_size_px / 2.0) * door_dist_factor

    _, sure_rooms_float = cv2.threshold(dist_transform, dist_thresh, 255, 0)
    sure_rooms = sure_rooms_float.astype(np.uint8)

    sure_rooms = filter_noise_contours(sure_rooms, min_noise_area)

    unknown = cv2.subtract(bin_img, sure_rooms)

    ret, markers = cv2.connectedComponents(sure_rooms)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers[image == 0] = -1

    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    region_mask = markers.copy()
    region_mask[region_mask <= 1] = 0
    region_mask[image == 0] = 0

    ws_boundaries = (markers == -1).astype(np.uint8)
    floor_boundaries = cv2.bitwise_and(ws_boundaries, bin_img)

    if np.count_nonzero(floor_boundaries) > 0:
        dist_from_boundary = cv2.distanceTransform(1 - floor_boundaries, cv2.DIST_L2, 3)
        mask_thresh = door_size_px * door_mask_scale
        doors_mask_raw = (dist_from_boundary <= mask_thresh).astype(np.uint8) * 255
        doors_mask_raw = cv2.bitwise_and(doors_mask_raw, bin_img)
    else:
        doors_mask_raw = np.zeros_like(bin_img)

    doors_mask = filter_noise_contours(doors_mask_raw, min_noise_area)

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


class RoomData:
    """
    Represents room data and provides coordinate conversion utilities.
    """

    def __init__(self, room_data: dict):
        """
        Initializes RoomData with room information.

        :param room_data: Dictionary containing room data including class, description, and pose
        :type room_data: dict
        :raises RuntimeError: If initialization fails due to missing or invalid data
        """
        try:
            self.class_name = room_data["room_class"]
            self.description = room_data["room_description"]
            raw_pose = tuple(ast.literal_eval(room_data["pose"]))
            u_x = raw_pose[0]
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

        :param origin: Origin coordinates of the map in world frame
        :type origin: Tuple[float, float]
        :param resolution: Resolution of the map in meters per pixel
        :type resolution: float
        :param map_image: Map image for dimension reference
        :type map_image: np.ndarray
        :return: Tuple containing pixel x and y coordinates
        :rtype: Tuple[int, int]
        """
        pixel_x = int((self.ros_x - origin[0]) / resolution)
        pixel_y = int(map_image.shape[0] - ((self.ros_y - origin[1]) / resolution))
        return pixel_x, pixel_y


def resolve_dominant_class_with_bias(
    class_counts: dict, bias_threshold: float = 1.10
) -> str:
    """
    Returns the dominant class applying a penalty to transitioning class.

    :param class_counts: Dictionary mapping class names to their counts
    :type class_counts: dict
    :param bias_threshold: Threshold factor for penalizing transitioning class
    :type bias_threshold: float
    :return: Name of the dominant class
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

    :param region_id_a: ID of the first region
    :type region_id_a: int
    :param region_id_b: ID of the second region
    :type region_id_b: int
    :param region_mask: NumPy array representing the region mask
    :type region_mask: np.ndarray
    :param doors_mask: Binary mask indicating door locations
    :type doors_mask: np.ndarray
    :param door_ratio_threshold: Threshold for door pixel ratio to determine merging
    :type door_ratio_threshold: float
    :return: True if regions should be merged, False otherwise
    :rtype: bool
    """
    mask_a = (region_mask == region_id_a).astype(np.uint8)
    mask_b = (region_mask == region_id_b).astype(np.uint8)

    overlaps = 0
    overlaps += np.count_nonzero((mask_a[:-1, :] == 1) & (mask_b[1:, :] == 1))
    overlaps += np.count_nonzero((mask_a[1:, :] == 1) & (mask_b[:-1, :] == 1))
    overlaps += np.count_nonzero((mask_a[:, :-1] == 1) & (mask_b[:, 1:] == 1))
    overlaps += np.count_nonzero((mask_a[:, 1:] == 1) & (mask_b[:, :-1] == 1))

    total_boundary_pixels = overlaps

    if total_boundary_pixels == 0:
        return True

    kernel = np.ones((3, 3), np.uint8)
    dilated_a = cv2.dilate(mask_a, kernel, iterations=1)

    common_boundary = (dilated_a == 1) & (mask_b == 1)

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

    :param region_mask: NumPy array representing the region mask
    :type region_mask: np.ndarray
    :param regions_dict: Dictionary containing region information
    :type regions_dict: dict
    :param region_class_counts: Dictionary mapping region IDs to class counts
    :type region_class_counts: dict
    :return: Tuple containing updated region mask and regions dictionary
    :rtype: Tuple[np.ndarray, dict]
    """
    visited_ids = set()
    for rid, counts in region_class_counts.items():
        if sum(counts.values()) > 0:
            visited_ids.add(rid)

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

    :param region_mask: NumPy array representing the region mask
    :type region_mask: np.ndarray
    :param doors_mask: Binary mask indicating door locations
    :type doors_mask: np.ndarray
    :param regions_dict: Dictionary containing region information
    :type regions_dict: dict
    :param min_area: Minimum area for door regions to be promoted
    :type min_area: int
    :return: Tuple containing updated region mask, regions dictionary, and set of new transitioning IDs
    :rtype: Tuple[np.ndarray, dict, set]
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        doors_mask, connectivity=8
    )

    current_max_id = np.max(region_mask)
    next_id = current_max_id + 1
    new_transitioning_ids = set()

    kernel = np.ones((3, 3), np.uint8)

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

    return region_mask, regions_dict, new_transitioning_ids


def split_regions_by_doors(
    region_mask: np.ndarray,
    doors_mask: np.ndarray,
    regions_dict: dict,
    min_fragment_size: int = 50,
) -> Tuple[np.ndarray, dict]:
    """
    Splits regions if internal doors cause topological disconnection.

    :param region_mask: NumPy array representing the region mask
    :type region_mask: np.ndarray
    :param doors_mask: Binary mask indicating door locations
    :type doors_mask: np.ndarray
    :param regions_dict: Dictionary containing region information
    :type regions_dict: dict
    :param min_fragment_size: Minimum size for split fragments to be considered valid
    :type min_fragment_size: int
    :return: Tuple containing updated region mask and regions dictionary
    :rtype: Tuple[np.ndarray, dict]
    """
    current_max_id = np.max(region_mask)
    next_id = current_max_id + 1

    original_ids = list(regions_dict.keys())
    split_count = 0

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

    :param region_mask: NumPy array representing the region mask
    :type region_mask: np.ndarray
    :param room_data_list: List of RoomData objects containing trajectory information
    :type room_data_list: list
    :param origin: Origin coordinates of the map in world frame
    :type origin: Tuple[float, float]
    :param resolution: Resolution of the map in meters per pixel
    :type resolution: float
    :param map_image: Original map image for coordinate conversion
    :type map_image: np.ndarray
    :return: Updated region mask with filled trajectory gaps
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

    return region_mask


def fill_remaining_gaps(region_mask: np.ndarray) -> np.ndarray:
    """
    Removes boundary lines between colored regions by assigning them to the majority neighbor.

    :param region_mask: NumPy array representing the region mask
    :type region_mask: np.ndarray
    :return: Region mask with filled gaps
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


def run_segmentation_pipeline(
    params: Dict[str, Any],
    map_image: np.ndarray,
    room_data_list: list,
    map_origin: Tuple[float, float],
    map_resolution: float,
    llm_agent: Agent | None = None,
    generate_descriptions: bool = False,
    prompt_mask: str = "",
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray, dict, np.ndarray]:
    """
    Encapsulates the entire segmentation pipeline from watershed to region merging.

    :param params: Dictionary containing segmentation parameters
    :type params: Dict[str, Any]
    :param map_image: Grayscale input map image
    :type map_image: np.ndarray
    :param room_data_list: List of RoomData objects containing trajectory information
    :type room_data_list: list
    :param map_origin: Origin coordinates of the map in world frame
    :type map_origin: Tuple[float, float]
    :param map_resolution: Resolution of the map in meters per pixel
    :type map_resolution: float
    :param llm_agent: Agent for generating room descriptions
    :type llm_agent: Agent | None
    :param generate_descriptions: Flag to enable description generation
    :type generate_descriptions: bool
    :param prompt_mask: Template string for LLM prompts
    :type prompt_mask: str
    :return: Tuple containing watershed image, reconstructed image, merged regions, region mask, merged colors, and doors overlay
    :rtype: Tuple[np.ndarray, np.ndarray, dict, np.ndarray, dict, np.ndarray]
    """
    watershed_image, region_mask, regions_dict, doors_mask = segment_map_watershed(
        map_image,
        door_size_px=params["door_size_px"],
        min_noise_area=params["min_noise_area"],
        door_dist_factor=params.get("door_dist_factor", 1.1),
        door_mask_scale=params.get("door_mask_scale", 0.4),
    )
    height, width = region_mask.shape

    region_class_counts = {}
    for room_data in room_data_list:
        x, y = room_data.get_map_coordinates(
            map_origin, map_resolution, watershed_image
        )
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

    region_mask, regions_dict = absorb_unvisited_regions(
        region_mask, regions_dict, region_class_counts
    )

    region_mask, regions_dict = split_regions_by_doors(
        region_mask,
        doors_mask,
        regions_dict,
        min_fragment_size=params["min_fragment_size"],
    )

    region_class_counts = {}
    for room_data in room_data_list:
        x, y = room_data.get_map_coordinates(
            map_origin, map_resolution, watershed_image
        )
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

    region_mask, regions_dict, promoted_ids = promote_connecting_doors(
        region_mask, doors_mask, regions_dict, min_area=params["min_door_area"]
    )
    structural_door_regions = set(promoted_ids)

    remaining_ids = list(regions_dict.keys())
    for region_id in remaining_ids:
        if region_id in structural_door_regions:
            continue
        counts = region_class_counts.get(region_id, {})
        if not counts:
            regions_dict[region_id]["dominant_class"] = "unknown"
            continue
        dominant_class = resolve_dominant_class_with_bias(
            counts, bias_threshold=params["bias_thresh"]
        )

        if dominant_class == "hallway":
            dominant_class = "transitioning"

        if dominant_class == "laundry room":
            dominant_class = "kitchen"

        regions_dict[region_id]["dominant_class"] = dominant_class

    adjacency_list = {}
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

    for _ in range(params["max_unknown_iters"]):
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
        if changes == 0:
            break

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
                                    door_ratio_threshold=params["door_ratio_thresh"],
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
                    map_origin, map_resolution, watershed_image
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

    if generate_descriptions and llm_agent:

        def summarize_desc(rid: int, rdata: dict, pmask: str, agt: Agent) -> tuple:
            """
            Summarizes room descriptions using the LLM agent.

            :param rid: Region ID
            :type rid: int
            :param rdata: Region data dictionary
            :type rdata: dict
            :param pmask: Prompt template string
            :type pmask: str
            :param agt: LLM agent for generation
            :type agt: Agent
            :return: Tuple containing region ID and summarized description
            :rtype: tuple
            """
            d_cls = rdata["dominant_class"]
            c_desc = "\n".join(
                [d for d in rdata["room_descriptions"] if isinstance(d, str)]
            )
            if not c_desc.strip():
                return (rid, f"It's a {d_cls} with no specific details observed.")
            try:
                return (rid, agt.run(pmask.format(d_cls, c_desc)).content)
            except (RuntimeError, ValueError, KeyError):
                traceback.print_exc()
                return (rid, f"It's a {d_cls}.")

        llm_results = Parallel(n_jobs=-1)(
            delayed(summarize_desc)(rid, merged_regions[rid], prompt_mask, llm_agent)
            for rid in merged_regions
        )
        for rid, desc in llm_results:
            merged_regions[rid]["summarized_description"] = desc

    for y in range(height):
        for x in range(width):
            original_region = region_mask[y, x]
            if original_region in region_mapping:
                region_mask[y, x] = region_mapping[original_region]
            else:
                if original_region != -1:
                    region_mask[y, x] = -1

    merged_colors = {}
    for merged_id in merged_regions:
        merged_colors[merged_id] = (
            np.random.randint(50, 256),
            np.random.randint(50, 256),
            np.random.randint(50, 256),
        )

    region_mask = fill_remaining_gaps(region_mask)
    region_mask = ensure_trajectory_connectivity(
        region_mask, room_data_list, map_origin, map_resolution, watershed_image
    )

    reconstructed_image = reconstruct_watershed_image(
        merged_regions, region_mask, merged_colors, width, height, doors_mask, map_image
    )

    doors_overlay = reconstructed_image.copy()

    red_doors = np.zeros_like(reconstructed_image)
    red_doors[doors_mask > 0] = (0, 0, 255)

    door_indices = doors_mask > 0
    try:
        if np.any(door_indices):
            doors_overlay[door_indices] = cv2.addWeighted(
                reconstructed_image[door_indices], 0.7, red_doors[door_indices], 0.3, 0
            )
    except (ValueError, cv2.error):
        traceback.print_exc()

    return (
        watershed_image,
        reconstructed_image,
        merged_regions,
        region_mask,
        merged_colors,
        doors_overlay,
    )


def get_profile_path(house_id: int, base_path: str = "") -> str:
    """
    Returns the profile path for a given house.

    :param house_id: ID of the house
    :type house_id: int
    :param base_path: Base directory path for profiles
    :type base_path: str
    :return: File path to the house profile JSON
    :rtype: str
    """
    return os.path.join(base_path, "profiles", f"Home{house_id:02d}.json")


def load_profile(house_id: int, default_params: dict, base_path: str = "") -> dict:
    """
    Loads profile parameters from disk or returns default.

    :param house_id: ID of the house
    :type house_id: int
    :param default_params: Default parameters to use if profile not found
    :type default_params: dict
    :param base_path: Base directory path for profiles
    :type base_path: str
    :return: Dictionary containing profile parameters
    :rtype: dict
    """
    path = get_profile_path(house_id, base_path)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
                params = default_params.copy()
                params.update(loaded)
                print(f"Loaded profile from {path}")
                return params
        except (OSError, IOError, json.JSONDecodeError):
            traceback.print_exc()
            print(f"Error loading profile from {path}")
    else:
        print(f"No profile found at {path}, using defaults.")
    return default_params


def save_profile(house_id: int, params: dict) -> None:
    """
    Saves parameters to disk.

    :param house_id: ID of the house
    :type house_id: int
    :param params: Parameters dictionary to save
    :type params: dict
    """
    os.makedirs("profiles", exist_ok=True)
    path = get_profile_path(house_id)
    try:
        with open(path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Profile saved to {path}")
    except (OSError, IOError):
        traceback.print_exc()
        print(f"Error saving profile to {path}")


def load_house_context(
    house_id: int,
    base_path: str,
    prefixes: List[str],
    map_binary_threshold: int,
    min_contour_area: int,
    crop_padding: int,
) -> Dict[str, Any]:
    """
    Loads all necessary data for a specific house ID and multiple prefixes.

    :param house_id: ID of the house
    :type house_id: int
    :param base_path: Base directory path containing house data
    :type base_path: str
    :param prefixes: List of prefix strings for different map views
    :type prefixes: List[str]
    :param map_binary_threshold: Threshold for binary map conversion
    :type map_binary_threshold: int
    :param min_contour_area: Minimum contour area for noise filtering
    :type min_contour_area: int
    :param crop_padding: Padding to apply when cropping the map
    :type crop_padding: int
    :return: Dictionary mapping prefixes to their loaded context data
    :rtype: Dict[str, Any]
    """
    context = {}
    print(f"\n--- Loading Context for House {house_id} ---")

    for prefix in prefixes:
        objects_path = f"{base_path}\\outputs\\Home{house_id:02d}\\Wandering\\exps\\{prefix}_house_{house_id}_map\\pcd_{prefix}_house_{house_id}_map.pkl.gz"
        map_image_filename = "generated_map.png"
        map_config_filename = "generated_map.yaml"

        print(f"[{prefix}] Loading objects from: {objects_path}")
        try:
            results = load_pkl_gz_result(objects_path)
            room_data_list = [
                RoomData(room_data) for room_data in results["room_data_list"]
            ]

            exp_map_path = f"{base_path}\\Home{house_id:02d}\\{map_image_filename}"
            exp_config_path = f"{base_path}\\Home{house_id:02d}\\{map_config_filename}"

            if not os.path.exists(exp_map_path):
                print(f"[{prefix}] Map not found at {exp_map_path}, skipping.")
                continue

            map_image = cv2.imread(exp_map_path, cv2.IMREAD_GRAYSCALE)
            with open(exp_config_path, "r") as f:
                map_settings = yaml.safe_load(f)

            map_image[map_image < map_binary_threshold] = 0
            contours, _ = cv2.findContours(
                map_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if cv2.contourArea(contour) < min_contour_area:
                    cv2.drawContours(map_image, [contour], -1, (0, 0, 0), -1)

            white_pixels = np.where(map_image > 0)
            if len(white_pixels[0]) > 0:
                min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
                min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
                min_y = max(0, min_y - crop_padding)
                min_x = max(0, min_x - crop_padding)
                max_y = min(map_image.shape[0] - 1, max_y + crop_padding)
                max_x = min(map_image.shape[1] - 1, max_x + crop_padding)
                map_image = map_image[min_y : max_y + 1, min_x : max_x + 1]

            origin = map_settings["origin"]
            resolution = map_settings["resolution"]

            context[prefix] = {
                "map_image": map_image,
                "room_data_list": room_data_list,
                "origin": origin,
                "resolution": resolution,
            }

        except (RuntimeError, ValueError, KeyError, OSError, IOError):
            print(f"[{prefix}] Error loading data")
            traceback.print_exc()

    return context


def stack_images(images: List[np.ndarray]) -> np.ndarray:
    """
    Horizontally stacks images, resizing them to the maximum height found.

    :param images: List of images to stack
    :type images: List[np.ndarray]
    :return: Horizontally concatenated image
    :rtype: np.ndarray
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    processed_images = []
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        processed_images.append(img)

    max_h = max(img.shape[0] for img in processed_images)
    resized_images = []

    for img in processed_images:
        h, w = img.shape[:2]
        if h != max_h:
            scale = max_h / h
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_NEAREST)
        resized_images.append(img)

    return cv2.hconcat(resized_images)


def process_house_segmentation(
    house_id: int,
    house_context: Dict[str, Any],
    params: Dict[str, Any],
    prefixes: List[str],
    llm_agent: Optional[Agent],
    generate_descriptions: bool,
    prompt_mask: str,
    class_colors: Dict[str, Tuple[int, int, int]],
    trajectory_dimming: float,
) -> Dict[str, Any]:
    """
    Executes the segmentation pipeline for a specific house across all provided prefixes.
    This function serves as the shared core logic for both experimental and standard modes.

    :param house_id: Identifier of the house being processed
    :type house_id: int
    :param house_context: Dictionary containing loaded map and room data for the house
    :type house_context: Dict[str, Any]
    :param params: Dictionary of segmentation parameters
    :type params: Dict[str, Any]
    :param prefixes: List of map prefixes to process (e.g., 'improved', 'online')
    :type prefixes: List[str]
    :param llm_agent: The LLM agent instance for description generation
    :type llm_agent: Optional[Agent]
    :param generate_descriptions: Whether to generate room descriptions using LLM
    :type generate_descriptions: bool
    :param prompt_mask: Template string for the LLM prompt
    :type prompt_mask: str
    :param class_colors: Dictionary mapping room classes to RGB colors for visualization
    :type class_colors: Dict[str, Tuple[int, int, int]]
    :param trajectory_dimming: Factor to dim the map background for trajectory visualization
    :type trajectory_dimming: float
    :return: A dictionary mapping prefixes to their processing results (images and data)
    :rtype: Dict[str, Any]
    """
    results = {}

    print(f"Processing House {house_id}...")

    for prefix in prefixes:
        if prefix not in house_context:
            continue

        ctx = house_context[prefix]

        (
            watershed_img,
            reconstructed_img,
            merged_regions,
            region_mask,
            merged_colors,
            doors_overlay_img,
        ) = run_segmentation_pipeline(
            params,
            ctx["map_image"],
            ctx["room_data_list"],
            ctx["origin"],
            ctx["resolution"],
            llm_agent=llm_agent,
            generate_descriptions=generate_descriptions,
            prompt_mask=prompt_mask,
        )

        height, width = region_mask.shape
        trajectory_image = cv2.cvtColor(ctx["map_image"].copy(), cv2.COLOR_GRAY2BGR)
        trajectory_image = (trajectory_image * trajectory_dimming).astype(np.uint8)

        for room_data in ctx["room_data_list"]:
            x, y = room_data.get_map_coordinates(
                ctx["origin"], ctx["resolution"], trajectory_image
            )
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            cls_name = room_data.class_name
            rgb_color = class_colors.get(cls_name, (128, 128, 128))
            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

            cv2.circle(trajectory_image, (x, y), 4, bgr_color, -1)
            cv2.circle(trajectory_image, (x, y), 4, (0, 0, 0), 1)

        # cv2.putText(
        #     reconstructed_img,
        #     prefix,
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 0, 255),
        #     2,
        # )

        results[prefix] = {
            "watershed_img": watershed_img,
            "reconstructed_img": reconstructed_img,
            "merged_regions": merged_regions,
            "region_mask": region_mask,
            "merged_colors": merged_colors,
            "doors_overlay_img": doors_overlay_img,
            "trajectory_img": trajectory_image,
            "width": width,
            "height": height,
        }

    return results


def run_experiment_mode(
    initial_house_id: int,
    base_path: str,
    prefixes: List[str],
    default_params: Dict[str, Any],
    trackbar_config: List[Dict[str, Any]],
    llm_agent: Agent,
    gen_desc: bool,
    prompt_mask: str,
    class_colors: Dict[str, Tuple[int, int, int]],
    traj_dimming: float,
    map_bin_thresh: int,
    min_cnt_area: int,
    crop_pad: int,
    profile_path: str = "",
):
    """
    Runs the interactive experiment mode with OpenCV GUI.

    :param initial_house_id: Starting house ID
    :type initial_house_id: int
    :param base_path: Base directory for dataset
    :type base_path: str
    :param prefixes: List of map prefixes
    :type prefixes: List[str]
    :param default_params: Default segmentation parameters
    :type default_params: Dict[str, Any]
    :param trackbar_config: Configuration for UI trackbars
    :type trackbar_config: List[Dict[str, Any]]
    :param llm_agent: LLM Agent instance
    :type llm_agent: Agent
    :param gen_desc: Whether to generate descriptions
    :type gen_desc: bool
    :param prompt_mask: Prompt mask string
    :type prompt_mask: str
    :param class_colors: Color mapping for classes
    :type class_colors: Dict[str, Tuple[int, int, int]]
    :param traj_dimming: Dimming factor for trajectory
    :type traj_dimming: float
    :param map_bin_thresh: Map binary threshold
    :type map_bin_thresh: int
    :param min_cnt_area: Minimum contour area
    :type min_cnt_area: int
    :param crop_pad: Crop padding
    :type crop_pad: int
    :param profile_path: Path to save/load profiles
    :type profile_path: str
    """
    print("Starting EXPERIMENT_MODE. Open the 'Controls' window to adjust parameters.")
    print(
        "Controls:\n 'S' - Save Profile\n 'A' - Prev House\n 'D' - Next House\n 'Q'/'ESC' - Quit"
    )

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 450)

    selected_house = initial_house_id
    initial_params = load_profile(selected_house, default_params, profile_path)

    house_context = load_house_context(
        selected_house, base_path, prefixes, map_bin_thresh, min_cnt_area, crop_pad
    )

    def nothing(x: int) -> None:
        """
        Dummy callback for trackbar.
        """
        pass

    for cfg in trackbar_config:
        initial_val = initial_params.get(cfg["param"], default_params[cfg["param"]])
        trackbar_pos = int(initial_val * cfg["scale"])
        cv2.createTrackbar(cfg["label"], "Controls", trackbar_pos, cfg["max"], nothing)

    last_params = {}
    force_update = True

    while True:
        current_params = {}
        for cfg in trackbar_config:
            val = cv2.getTrackbarPos(cfg["label"], "Controls")
            val = max(cfg["min"], val)
            real_val = val / cfg["scale"]
            if cfg["type"] is int:
                real_val = int(real_val)
            current_params[cfg["param"]] = real_val

        if current_params != last_params or force_update:

            results = process_house_segmentation(
                house_id=selected_house,
                house_context=house_context,
                params=current_params,
                prefixes=prefixes,
                llm_agent=llm_agent,
                generate_descriptions=gen_desc,
                prompt_mask=prompt_mask,
                class_colors=class_colors,
                trajectory_dimming=traj_dimming,
            )

            stack_reconstructed = []
            stack_watershed = []
            stack_doors = []
            stack_trajectory = []

            for prefix in prefixes:
                if prefix in results:
                    data = results[prefix]
                    stack_reconstructed.append(data["reconstructed_img"])
                    stack_watershed.append(data["watershed_img"])
                    stack_doors.append(data["doors_overlay_img"])
                    stack_trajectory.append(data["trajectory_img"])

            if stack_reconstructed:
                cv2.imshow("Reconstructed Image", stack_images(stack_reconstructed))
                cv2.imshow("Watershed Raw", stack_images(stack_watershed))
                cv2.imshow("Doors Overlay", stack_images(stack_doors))
                cv2.imshow("Robot Trajectory", stack_images(stack_trajectory))

            last_params = current_params.copy()
            force_update = False

        key = cv2.waitKey(100)

        if key == 27 or key == ord("q"):
            break

        elif key == ord("s"):
            save_profile(selected_house, current_params)

        elif key == ord("a") or key == ord("d"):
            new_house = selected_house
            if key == ord("a"):
                new_house = max(1, selected_house - 1)
            else:
                new_house = min(30, selected_house + 1)

            if new_house != selected_house:
                selected_house = new_house
                new_profile = load_profile(selected_house, default_params)
                for cfg in trackbar_config:
                    val = new_profile.get(cfg["param"], default_params[cfg["param"]])
                    pos = int(val * cfg["scale"])
                    cv2.setTrackbarPos(cfg["label"], "Controls", pos)

                house_context = load_house_context(
                    selected_house,
                    base_path,
                    prefixes,
                    map_bin_thresh,
                    min_cnt_area,
                    crop_pad,
                )
                force_update = True

    cv2.destroyAllWindows()
    print("Experiment Mode ended.")


def run_standard_mode(
    house_id: int,
    base_path: str,
    prefixes: List[str],
    default_params: Dict[str, Any],
    llm_agent: Agent,
    gen_desc: bool,
    prompt_mask: str,
    class_colors: Dict[str, Tuple[int, int, int]],
    traj_dimming: float,
    map_bin_thresh: int,
    min_cnt_area: int,
    crop_pad: int,
    filename_template: str,
    save_results: bool = True,
    profile_path: str = "",
) -> Dict[str, Any]:
    """
    Runs the standard non-interactive mode. Processes the selected house and optionally saves results.

    :param house_id: ID of the house to process
    :type house_id: int
    :param base_path: Base directory path containing house data
    :type base_path: str
    :param prefixes: List of prefix strings for different map views
    :type prefixes: List[str]
    :param default_params: Default parameters for segmentation
    :type default_params: Dict[str, Any]
    :param llm_agent: Agent for generating room descriptions
    :type llm_agent: Agent
    :param gen_desc: Flag to enable description generation
    :type gen_desc: bool
    :param prompt_mask: Template string for LLM prompts
    :type prompt_mask: str
    :param class_colors: Dictionary mapping room classes to RGB colors
    :type class_colors: Dict[str, Tuple[int, int, int]]
    :param traj_dimming: Dimming factor for trajectory visualization
    :type traj_dimming: float
    :param map_bin_thresh: Threshold for binary map conversion
    :type map_bin_thresh: int
    :param min_cnt_area: Minimum contour area for noise filtering
    :type min_cnt_area: int
    :param crop_pad: Padding to apply when cropping the map
    :type crop_pad: int
    :param filename_template: Template for output filenames
    :type filename_template: str
    :param save_results: Whether to save the results to disk (pickle and images), defaults to True
    :type save_results: bool
    :param profile_path: Path to save/load profiles
    :type profile_path: str
    :return: Dictionary containing the processing results for all prefixes
    :rtype: Dict[str, Any]
    """
    print(f"Running STANDARD mode for House {house_id}...")

    params = load_profile(house_id, default_params, profile_path)

    house_context = load_house_context(
        house_id, base_path, prefixes, map_bin_thresh, min_cnt_area, crop_pad
    )

    if not house_context:
        print(f"Failed to load context for House {house_id}. Exiting.")
        return {}

    results = process_house_segmentation(
        house_id=house_id,
        house_context=house_context,
        params=params,
        prefixes=prefixes,
        llm_agent=llm_agent,
        generate_descriptions=gen_desc,
        prompt_mask=prompt_mask,
        class_colors=class_colors,
        trajectory_dimming=traj_dimming,
    )

    if save_results:
        output_dir = os.path.join("segmentation_results", f"Home{house_id:02d}")
        os.makedirs(output_dir, exist_ok=True)

        for prefix, data in results.items():
            base_filename = filename_template.format(prefix=prefix)
            pkl_path = os.path.join(output_dir, base_filename)
            img_path = pkl_path.replace(".pkl", ".png").replace(".gz", "")

            print(f"[{prefix}] Saving results to {pkl_path}...")

            save_success = save_watershed_map(
                filepath=pkl_path,
                merged_regions=data["merged_regions"],
                region_mask=data["region_mask"],
                merged_colors=data["merged_colors"],
                width=data["width"],
                height=data["height"],
            )

            if save_success:
                cv2.imwrite(img_path, data["reconstructed_img"])
                print(f"[{prefix}] Saved image to {img_path}")
            else:
                print(f"[{prefix}] Failed to save pickle data.")
    else:
        print("Skipping save to disk as requested.")

    print("Standard mode execution finished.")
    return results


if __name__ == "__main__":
    load_dotenv()

    EXPERIMENT_MODE = False
    SAVE_OUTPUTS_TO_DISK = True

    GENERATE_DESCRIPTIONS = True
    SELECTED_HOUSE = 1
    PREFIXES = ["improved", "online", "offline"]
    DATASET_BASE_PATH = "THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET"
    OUTPUT_FILENAME_TEMPLATE = "watershed_map_{prefix}.pkl"

    TRACKBAR_CONFIG = [
        {
            "label": "Door Size",
            "param": "door_size_px",
            "min": 1000,
            "max": 3000,
            "scale": 100,
            "type": int,
        },
        {
            "label": "Door Ratio %",
            "param": "door_ratio_thresh",
            "min": 0,
            "max": 300,
            "scale": 100,
            "type": float,
        },
        {
            "label": "Bias %",
            "param": "bias_thresh",
            "min": 0,
            "max": 1000,
            "scale": 100,
            "type": float,
        },
        {
            "label": "Frag Size",
            "param": "min_fragment_size",
            "min": 1,
            "max": 1000,
            "scale": 1,
            "type": int,
        },
        {
            "label": "Min Door Area",
            "param": "min_door_area",
            "min": 1,
            "max": 100,
            "scale": 1,
            "type": int,
        },
        {
            "label": "Unk Iters",
            "param": "max_unknown_iters",
            "min": 0,
            "max": 100,
            "scale": 1,
            "type": int,
        },
        {
            "label": "Min Noise Area",
            "param": "min_noise_area",
            "min": 0,
            "max": 1500,
            "scale": 100,
            "type": int,
        },
        {
            "label": "Door Dist %",
            "param": "door_dist_factor",
            "min": 0,
            "max": 200,
            "scale": 100,
            "type": float,
        },
        {
            "label": "Door Mask %",
            "param": "door_mask_scale",
            "min": 0,
            "max": 200,
            "scale": 100,
            "type": float,
        },
    ]

    LLM_MODEL_ID = "openai/gpt-oss-120b:nitro"
    SYSTEM_MESSAGE_TEXT = dedent(
        """
        PERSONA:
        A natural language processing expert.
        """
    )
    PROMPT_MASK = "The environment is a {} and these are its descriptions: {}"

    print("Loading LLM agent...")
    llm_agent = Agent(
        model=OpenRouter(
            id=LLM_MODEL_ID,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        ),
        system_message=SYSTEM_MESSAGE_TEXT,
    )

    if EXPERIMENT_MODE:
        run_experiment_mode(
            initial_house_id=SELECTED_HOUSE,
            base_path=DATASET_BASE_PATH,
            prefixes=PREFIXES,
            default_params=DEFAULT_PARAMS,
            trackbar_config=TRACKBAR_CONFIG,
            llm_agent=llm_agent,
            gen_desc=GENERATE_DESCRIPTIONS,
            prompt_mask=PROMPT_MASK,
            class_colors=CLASS_COLORS,
            traj_dimming=TRAJECTORY_IMAGE_DIMMING,
            map_bin_thresh=MAP_BINARY_THRESHOLD,
            min_cnt_area=MIN_CONTOUR_AREA,
            crop_pad=CROP_PADDING,
            profile_path=DATASET_BASE_PATH,
        )
    else:
        final_results = run_standard_mode(
            house_id=SELECTED_HOUSE,
            base_path=DATASET_BASE_PATH,
            prefixes=PREFIXES,
            default_params=DEFAULT_PARAMS,
            llm_agent=llm_agent,
            gen_desc=GENERATE_DESCRIPTIONS,
            prompt_mask=PROMPT_MASK,
            class_colors=CLASS_COLORS,
            traj_dimming=TRAJECTORY_IMAGE_DIMMING,
            map_bin_thresh=MAP_BINARY_THRESHOLD,
            min_cnt_area=MIN_CONTOUR_AREA,
            crop_pad=CROP_PADDING,
            filename_template=OUTPUT_FILENAME_TEMPLATE,
            save_results=SAVE_OUTPUTS_TO_DISK,
            profile_path=DATASET_BASE_PATH,
        )

        if final_results:
            print(f"Successfully processed {len(final_results)} map versions.")

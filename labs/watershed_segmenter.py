from xml import dom
from agno.models.openrouter import OpenRouter
from collections import Counter
from dotenv import load_dotenv
from joblib import Parallel, delayed
from agno.agent import Agent
from textwrap import dedent
from typing import Tuple, Dict, Any
import numpy as np
import traceback
import random
import pickle
import gzip
import yaml
import json
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
    Removes small noise blobs from a binary mask using findContours instead of morphology.
    """
    if min_area <= 0:
        return mask

    # Find contours on the binary mask
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
    door_mask_scale: float = 0.4,  # New parameter for mask size
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Segments the map using Watershed algorithm without morphology.
    Uses Distance Transform and Contour filtering.
    """
    height, width = image.shape[:2]

    # Binary image (Floor = 255, Walls = 0)
    bin_img = np.zeros_like(image)
    bin_img[image > 0] = 255

    # 1. Clean input noise from the floor map first
    bin_img = filter_noise_contours(bin_img, min_noise_area)

    # 2. Identify 'Sure Rooms' (Foreground) using Distance Transform
    # Euclidean distance to the nearest zero pixel (wall)
    dist_transform = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)

    # CRITICAL FIX: The threshold must be strictly LARGER than the door radius.
    dist_thresh = (door_size_px / 2.0) * door_dist_factor

    _, sure_rooms_float = cv2.threshold(dist_transform, dist_thresh, 255, 0)
    sure_rooms = sure_rooms_float.astype(np.uint8)

    # Clean sure_rooms noise
    sure_rooms = filter_noise_contours(sure_rooms, min_noise_area)

    # 3. Identify Unknown Region (Doors + Wall borders)
    # The area that is floor (bin_img) but NOT sure_rooms is the uncertain area
    unknown = cv2.subtract(bin_img, sure_rooms)

    # 4. Markers for Watershed
    ret, markers = cv2.connectedComponents(sure_rooms)
    markers = markers + 1  # Background is 1
    markers[unknown == 255] = 0  # Unknown is 0
    markers[image == 0] = -1  # Walls/Void

    # 5. Run Watershed
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    # 6. Extract Result
    region_mask = markers.copy()
    region_mask[region_mask <= 1] = 0  # Remove background and boundaries
    region_mask[image == 0] = 0  # Ensure walls are 0

    # 7. Extract Doors Mask (Using Watershed Boundaries)
    ws_boundaries = (markers == -1).astype(np.uint8)
    floor_boundaries = cv2.bitwise_and(ws_boundaries, bin_img)

    # Expand boundaries to create the door mask using Distance Transform
    # door_mask_scale allows controlling the thickness of the door region
    if np.count_nonzero(floor_boundaries) > 0:
        dist_from_boundary = cv2.distanceTransform(1 - floor_boundaries, cv2.DIST_L2, 3)
        # Logic: door_size_px * scale. Default 0.4 approx matches previous hardcoded / 2.5
        mask_thresh = door_size_px * door_mask_scale
        doors_mask_raw = (dist_from_boundary <= mask_thresh).astype(np.uint8) * 255
        doors_mask_raw = cv2.bitwise_and(
            doors_mask_raw, bin_img
        )  # Ensure it's on floor
    else:
        doors_mask_raw = np.zeros_like(bin_img)

    # Filter noise from doors mask
    doors_mask = filter_noise_contours(doors_mask_raw, min_noise_area)

    # Fill unlabeled floor gaps
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

    # Re-normalize IDs
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
            # Contour for polygon extraction
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
        """
        pixel_x = int((self.ros_x - origin[0]) / resolution)
        pixel_y = int(map_image.shape[0] - ((self.ros_y - origin[1]) / resolution))
        return pixel_x, pixel_y


def resolve_dominant_class_with_bias(
    class_counts: dict, bias_threshold: float = 1.10
) -> str:
    """
    Returns the dominant class applying a penalty to transitioning class.
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
    """
    mask_a = (region_mask == region_id_a).astype(np.uint8)
    mask_b = (region_mask == region_id_b).astype(np.uint8)

    # Shift-based neighbor check (No Morphology)
    overlaps = 0
    # Shift Up
    overlaps += np.count_nonzero((mask_a[:-1, :] == 1) & (mask_b[1:, :] == 1))
    # Shift Down
    overlaps += np.count_nonzero((mask_a[1:, :] == 1) & (mask_b[:-1, :] == 1))
    # Shift Left
    overlaps += np.count_nonzero((mask_a[:, :-1] == 1) & (mask_b[:, 1:] == 1))
    # Shift Right
    overlaps += np.count_nonzero((mask_a[:, 1:] == 1) & (mask_b[:, :-1] == 1))

    total_boundary_pixels = overlaps

    if total_boundary_pixels == 0:
        return True

    # Check door overlap using dilation (as a geometric expansion tool, not filtering)
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
    """
    height, width = region_mask.shape

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
            # Dilation used for neighbor finding
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

        # Dilation for connectivity check
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
    """
    height, width = region_mask.shape
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
    """
    height, width = region_mask.shape

    valid_mask = (region_mask >= 0).astype(np.uint8)

    if np.all(valid_mask):
        return region_mask

    kernel = np.ones((3, 3), np.uint8)
    # Dilation for gap check
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
    llm_agent: Agent = None,
    generate_descriptions: bool = False,
    prompt_mask: str = "",
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray, dict, np.ndarray]:
    """
    Encapsulates the entire segmentation pipeline.
    """

    # 1. Watershed Segmentation
    watershed_image, region_mask, regions_dict, doors_mask = segment_map_watershed(
        map_image,
        door_size_px=params["door_size_px"],
        min_noise_area=params["min_noise_area"],
        door_dist_factor=params.get("door_dist_factor", 1.1),
        door_mask_scale=params.get("door_mask_scale", 0.4),  # Pass new parameter
    )
    height, width = region_mask.shape

    # 2. Count Trajectory Points
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

    # 3. Absorb Unvisited
    region_mask, regions_dict = absorb_unvisited_regions(
        region_mask, regions_dict, region_class_counts
    )

    # 4. Split by Doors
    region_mask, regions_dict = split_regions_by_doors(
        region_mask,
        doors_mask,
        regions_dict,
        min_fragment_size=params["min_fragment_size"],
    )

    # 5. Recalculate Points (after split)
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

    # 6. Promote Doors
    region_mask, regions_dict, promoted_ids = promote_connecting_doors(
        region_mask, doors_mask, regions_dict, min_area=params["min_door_area"]
    )
    structural_door_regions = set(promoted_ids)

    # 7. Determine Dominant Classes
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

        # Force "hallway" to be "transitioning" to trigger merge
        if dominant_class == "hallway":
            dominant_class = "transitioning"

        if dominant_class == "laundry room":
            dominant_class = "kitchen"

        regions_dict[region_id]["dominant_class"] = dominant_class

    # 8. Rebuild Adjacency
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

    # 9. Resolve Unknowns
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

    # 10. Merge Connected
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

    # 11. LLM Generation
    if generate_descriptions and llm_agent:

        def summarize_desc(rid, rdata, pmask, agt):
            d_cls = rdata["dominant_class"]
            c_desc = "\n".join(
                [d for d in rdata["room_descriptions"] if isinstance(d, str)]
            )
            if not c_desc.strip():
                return (rid, f"It's a {d_cls} with no specific details observed.")
            try:
                return (rid, agt.run(pmask.format(d_cls, c_desc)).content)
            except Exception:
                return (rid, f"It's a {d_cls}.")

        llm_results = Parallel(n_jobs=-1)(
            delayed(summarize_desc)(rid, merged_regions[rid], prompt_mask, llm_agent)
            for rid in merged_regions
        )
        for rid, desc in llm_results:
            merged_regions[rid]["summarized_description"] = desc

    # 12. Update Mask Mappings
    for y in range(height):
        for x in range(width):
            original_region = region_mask[y, x]
            if original_region in region_mapping:
                region_mask[y, x] = region_mapping[original_region]
            else:
                if original_region != -1:
                    region_mask[y, x] = -1

    # 13. Generate Colors
    merged_colors = {}
    for merged_id in merged_regions:
        merged_colors[merged_id] = (
            np.random.randint(50, 256),
            np.random.randint(50, 256),
            np.random.randint(50, 256),
        )

    # 14. Fix Connectivity
    region_mask = fill_remaining_gaps(region_mask)
    region_mask = ensure_trajectory_connectivity(
        region_mask, room_data_list, map_origin, map_resolution, watershed_image
    )

    # 15. Reconstruct Image
    reconstructed_image = reconstruct_watershed_image(
        merged_regions, region_mask, merged_colors, width, height, doors_mask, map_image
    )

    # 16. Create Doors Overlay Image (Fusion)
    # Background: Reconstructed image
    # Overlay: Red doors
    doors_overlay = reconstructed_image.copy()

    # Create red mask for doors
    red_doors = np.zeros_like(reconstructed_image)
    red_doors[doors_mask > 0] = (0, 0, 255)  # Red BGR

    # Blend: alpha=0.7 base, beta=0.3 overlay
    # Only apply where doors exist to keep other colors crisp
    # Fix for potential crashes during blending
    door_indices = doors_mask > 0
    try:
        if np.any(door_indices):
            doors_overlay[door_indices] = cv2.addWeighted(
                reconstructed_image[door_indices], 0.7, red_doors[door_indices], 0.3, 0
            )
    except Exception:
        # Safe fallback if blending fails (e.g., shape mismatch, empty mask)
        pass

    return (
        watershed_image,
        reconstructed_image,
        merged_regions,
        region_mask,
        merged_colors,
        doors_overlay,
    )


def get_profile_path(house_id: int) -> str:
    """Returns the profile path for a given house."""
    return os.path.join("profiles", f"Home{house_id:02d}.json")


def load_profile(house_id: int, default_params: dict) -> dict:
    """Loads profile params from disk or returns default."""
    path = get_profile_path(house_id)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
                # Update defaults with loaded to ensure all keys exist
                params = default_params.copy()
                params.update(loaded)
                print(f"Loaded profile from {path}")
                return params
        except Exception as e:
            print(f"Error loading profile: {e}")
    else:
        print(f"No profile found at {path}, using defaults.")
    return default_params


def save_profile(house_id: int, params: dict):
    """Saves params to disk."""
    os.makedirs("profiles", exist_ok=True)
    path = get_profile_path(house_id)
    try:
        with open(path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Profile saved to {path}")
    except Exception as e:
        print(f"Error saving profile: {e}")


if __name__ == "__main__":
    load_dotenv()

    # ==========================================
    #             GLOBAL CONFIGURATION
    # ==========================================

    EXPERIMENT_MODE = True  # <--- SET TO TRUE FOR INTERACTIVE MODE

    # --- Paths and Dataset Selection ---
    SELECTED_HOUSE = 3
    PREFFIX = "offline"
    DATASET_BASE_PATH = "D:\\Documentos\\Datasets\\Robot@VirtualHomeLarge"
    MAP_PATH = f"{DATASET_BASE_PATH}\\Home{SELECTED_HOUSE:02d}"
    OBJECTS_PATH = f"D:\\Documentos\\Datasets\\Robot@VirtualHomeLarge\\outputs\\Home{SELECTED_HOUSE:02d}\\Wandering\\exps\\{PREFFIX}_house_{SELECTED_HOUSE}_map\\pcd_{PREFFIX}_house_{SELECTED_HOUSE}_map.pkl.gz"
    MAP_IMAGE_FILENAME = "generated_map.png"
    MAP_CONFIG_FILENAME = "generated_map.yaml"
    OUTPUT_FILENAME = "watershed_map.pkl"

    # --- Map Pre-processing ---
    MAP_BINARY_THRESHOLD = 250
    MIN_CONTOUR_AREA = 100
    CROP_PADDING = 5

    # --- New Constants ---
    MIN_NOISE_AREA = 10
    DOOR_DIST_FACTOR = 1.1
    DOOR_MASK_SCALE = (
        0.4  # NEW: Controls the thickness/radius of door overlay (approx 1/2.5)
    )

    # --- Initial Parameters ---
    DEFAULT_PARAMS = {
        "door_size_px": 18,
        "door_ratio_thresh": 0.30,
        "bias_thresh": 2.0,
        "min_fragment_size": 50,
        "min_door_area": 600,
        "max_unknown_iters": 5,
        "min_noise_area": MIN_NOISE_AREA,
        "door_dist_factor": DOOR_DIST_FACTOR,
        "door_mask_scale": DOOR_MASK_SCALE,  # Included in defaults
    }

    # --- Trackbar Configuration (Ranges & Types) ---
    # scale: Multiplier for float values on the UI (Trackbars are int only)
    # type: 'int' or 'float' for the resulting parameter value
    TRACKBAR_CONFIG = [
        {
            "label": "Door Size",
            "param": "door_size_px",
            "min": 1,
            "max": 4000,
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
            "max": 10000,
            "scale": 1,
            "type": int,
        },
        {
            "label": "Min Door Area",
            "param": "min_door_area",
            "min": 1,
            "max": 10000,
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

    # Load parameters from profile if exists
    INITIAL_PARAMS = load_profile(SELECTED_HOUSE, DEFAULT_PARAMS)

    # --- LLM Agent Settings ---
    GENEREATE_DESCRIPTIONS = False
    LLM_MODEL_ID = "google/gemini-2.5-flash-lite"
    SYSTEM_MESSAGE_TEXT = dedent(
        """
        PERSONA:
        A natural language processing expert.
        """
    )
    PROMPT_MASK = "The environment is a {} and these are its descriptions: {}"

    # --- Visualization Settings ---
    TRAJECTORY_IMAGE_DIMMING = 0.6
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

    # ==========================================
    #                 EXECUTION
    # ==========================================

    print("Loading objects and LLM agent...")
    llm_agent = Agent(
        model=OpenRouter(
            id=LLM_MODEL_ID,
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        ),
        system_message=SYSTEM_MESSAGE_TEXT,
    )

    print("Loading map data and settings...")
    results = load_pkl_gz_result(OBJECTS_PATH)
    room_data_list = [RoomData(room_data) for room_data in results["room_data_list"]]

    map_image_path = os.path.join(MAP_PATH, MAP_IMAGE_FILENAME)
    map_settings_path = os.path.join(MAP_PATH, MAP_CONFIG_FILENAME)

    map_image = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
    with open(map_settings_path, "r") as f:
        map_settings = yaml.safe_load(f)

    # --- Pre-processing Image ---
    print("Pre-processing map image...")
    map_image[map_image < MAP_BINARY_THRESHOLD] = 0
    contours, _ = cv2.findContours(
        map_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            cv2.drawContours(map_image, [contour], -1, (0, 0, 0), -1)

    try:
        white_pixels = np.where(map_image > 0)
        if len(white_pixels[0]) > 0:
            min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
            min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
            min_y = max(0, min_y - CROP_PADDING)
            min_x = max(0, min_x - CROP_PADDING)
            max_y = min(map_image.shape[0] - 1, max_y + CROP_PADDING)
            max_x = min(map_image.shape[1] - 1, max_x + CROP_PADDING)
            map_image = map_image[min_y : max_y + 1, min_x : max_x + 1]
    except (IndexError, ValueError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Error cropping image: {str(e)}") from e

    origin = map_settings["origin"]
    resolution = map_settings["resolution"]

    # ==========================================
    #          INTERACTIVE LOOP MODE
    # ==========================================
    if EXPERIMENT_MODE:
        print(
            "Starting EXPERIMENT_MODE. Open the 'Controls' window to adjust parameters."
        )
        print("Controls: Press 's' to Save Profile, 'q' or 'ESC' to Quit.")

        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Controls", 400, 450)

        def nothing(x):
            pass

        # Create Trackbars dynamically from config
        for cfg in TRACKBAR_CONFIG:
            # Calculate initial position based on loaded params
            initial_val = INITIAL_PARAMS.get(cfg["param"], DEFAULT_PARAMS[cfg["param"]])
            trackbar_pos = int(initial_val * cfg["scale"])
            cv2.createTrackbar(
                cfg["label"], "Controls", trackbar_pos, cfg["max"], nothing
            )

        last_params = {}

        while True:
            # 1. Read current trackbar values dynamically
            current_params = {}
            for cfg in TRACKBAR_CONFIG:
                val = cv2.getTrackbarPos(cfg["label"], "Controls")
                # Enforce minimum
                val = max(cfg["min"], val)

                # Convert back to actual parameter value
                real_val = val / cfg["scale"]
                if cfg["type"] is int:
                    real_val = int(real_val)

                current_params[cfg["param"]] = real_val

            # 2. Check if params changed
            if current_params != last_params:
                print(f"Parameters changed. Running pipeline...")

                (
                    watershed_img,
                    reconstructed_img,
                    merged_regions,
                    region_mask,
                    merged_colors,
                    doors_overlay_img,
                ) = run_segmentation_pipeline(
                    current_params,
                    map_image,
                    room_data_list,
                    origin,
                    resolution,
                    llm_agent=llm_agent,
                    generate_descriptions=False,
                    prompt_mask=PROMPT_MASK,
                )

                # Generate Trajectory Image
                height, width = region_mask.shape
                trajectory_image = cv2.cvtColor(map_image.copy(), cv2.COLOR_GRAY2BGR)
                trajectory_image = (trajectory_image * TRAJECTORY_IMAGE_DIMMING).astype(
                    np.uint8
                )
                for room_data in room_data_list:
                    x, y = room_data.get_map_coordinates(
                        origin, resolution, trajectory_image
                    )
                    x, y = max(0, min(x, width - 1)), max(0, min(y, height - 1))
                    cls_name = room_data.class_name
                    rgb_color = CLASS_COLORS.get(cls_name, (128, 128, 128))
                    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
                    cv2.circle(trajectory_image, (x, y), 4, bgr_color, -1)
                    cv2.circle(trajectory_image, (x, y), 4, (0, 0, 0), 1)

                # Show Results
                cv2.namedWindow("Reconstructed Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Reconstructed Image", reconstructed_img)
                cv2.namedWindow("Watershed Raw", cv2.WINDOW_NORMAL)
                cv2.imshow("Watershed Raw", watershed_img)
                cv2.namedWindow("Doors Overlay", cv2.WINDOW_NORMAL)
                cv2.imshow("Doors Overlay", doors_overlay_img)
                cv2.namedWindow("Robot Trajectory", cv2.WINDOW_NORMAL)
                cv2.imshow("Robot Trajectory", trajectory_image)

                last_params = current_params.copy()

            # 3. Handle Key Exits
            key = cv2.waitKey(100)  # Wait 100ms

            if key == 27 or key == ord("q"):  # ESC or q
                print("Exiting without saving (unless 's' was pressed previously)...")
                break

            if key == ord("s"):
                print("Saving profile...")
                save_profile(SELECTED_HOUSE, current_params)

        cv2.destroyAllWindows()
        print("Experiment Mode ended.")

    # ==========================================
    #             NORMAL RUN MODE
    # ==========================================
    else:
        print("Running in STANDARD mode (Single Pass)...")
        (
            watershed_img,
            reconstructed_img,
            merged_regions,
            region_mask,
            merged_colors,
            doors_overlay_img,
        ) = run_segmentation_pipeline(
            INITIAL_PARAMS,
            map_image,
            room_data_list,
            origin,
            resolution,
            llm_agent=llm_agent,
            generate_descriptions=GENEREATE_DESCRIPTIONS,
            prompt_mask=PROMPT_MASK,
        )

        print("Saving Watershed map...")
        height, width = region_mask.shape
        save_watershed_map(
            OUTPUT_FILENAME,
            merged_regions,
            region_mask,
            merged_colors,
            width,
            height,
        )

        # Trajectory Image (Visual only)
        print("Generating trajectory image...")
        trajectory_image = cv2.cvtColor(map_image.copy(), cv2.COLOR_GRAY2BGR)
        trajectory_image = (trajectory_image * TRAJECTORY_IMAGE_DIMMING).astype(
            np.uint8
        )
        for room_data in room_data_list:
            x, y = room_data.get_map_coordinates(origin, resolution, trajectory_image)
            x, y = max(0, min(x, width - 1)), max(0, min(y, height - 1))
            cls_name = room_data.class_name
            rgb_color = CLASS_COLORS.get(cls_name, (128, 128, 128))
            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
            cv2.circle(trajectory_image, (x, y), 4, bgr_color, -1)
            cv2.circle(trajectory_image, (x, y), 4, (0, 0, 0), 1)

        cv2.namedWindow("Reconstructed Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Reconstructed Image", reconstructed_img)
        cv2.namedWindow("Watershed Raw", cv2.WINDOW_NORMAL)
        cv2.imshow("Watershed Raw", watershed_img)
        cv2.namedWindow("Doors Overlay", cv2.WINDOW_NORMAL)
        cv2.imshow("Doors Overlay", doors_overlay_img)
        cv2.namedWindow("Robot Trajectory", cv2.WINDOW_NORMAL)
        cv2.imshow("Robot Trajectory", trajectory_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

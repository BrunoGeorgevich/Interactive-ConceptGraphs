from agno.models.openrouter import OpenRouter
from agno.agent import Agent
from typing import List, Tuple
from dotenv import load_dotenv
from textwrap import dedent
import numpy as np
import traceback
import random
import pickle
import gzip
import yaml
import ast
import cv2
import os
from joblib import Parallel, delayed


def save_voronoi_map(
    filepath: str,
    merged_regions: dict,
    region_mask: np.ndarray,
    merged_colors: dict,
    width: int,
    height: int,
) -> bool:
    """
    Saves the Voronoi map data to a compressed pickle file for later reconstruction.

    :param filepath: Path where the compressed pickle file will be saved.
    :type filepath: str
    :param merged_regions: Dictionary containing merged region data.
    :type merged_regions: dict
    :param region_mask: Array containing region IDs for each pixel.
    :type region_mask: np.ndarray
    :param merged_colors: Dictionary mapping region IDs to RGB colors.
    :type merged_colors: dict
    :param width: Width of the image.
    :type width: int
    :param height: Height of the image.
    :type height: int
    :raises OSError: If the file cannot be written.
    :raises ValueError: If the input data is invalid.
    :raises TypeError: If the data contains non-serializable types.
    :return: True if the map was saved successfully.
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

        voronoi_data = {
            "width": width,
            "height": height,
            "region_mask": region_mask,
            "merged_regions": merged_regions,
            "merged_colors": merged_colors,
        }

        with gzip.open(filepath, "wb") as f:
            pickle.dump(voronoi_data, f)

        return True

    except (OSError, IOError) as e:
        traceback.print_exc()
        print(f"Error saving Voronoi map: {e}")
        return False
    except (ValueError, TypeError) as e:
        traceback.print_exc()
        print(f"Invalid data for saving: {e}")
        return False


def load_voronoi_map(filepath: str) -> tuple[dict, np.ndarray, dict, int, int] | None:
    """
    Loads Voronoi map data from a compressed pickle file.

    :param filepath: Path to the compressed pickle file to load.
    :type filepath: str
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the pickle data is invalid or corrupted.
    :raises OSError: If the file cannot be read.
    :return: Tuple containing (merged_regions, region_mask, merged_colors, width, height) or None if failed.
    :rtype: tuple[dict, np.ndarray, dict, int, int] | None
    """
    try:
        if not filepath.endswith(".pkl.gz"):
            if filepath.endswith(".pkl"):
                filepath += ".gz"
            else:
                filepath += ".pkl.gz"

        with gzip.open(filepath, "rb") as f:
            voronoi_data = pickle.load(f)

        required_keys = [
            "width",
            "height",
            "region_mask",
            "merged_regions",
            "merged_colors",
        ]
        if not all(key in voronoi_data for key in required_keys):
            raise ValueError("Invalid pickle structure: missing required keys")

        width = voronoi_data["width"]
        height = voronoi_data["height"]

        if width <= 0 or height <= 0:
            raise ValueError("Invalid dimensions in saved data")

        region_mask = voronoi_data["region_mask"]

        if region_mask.shape != (height, width):
            raise ValueError("Region mask dimensions do not match saved dimensions")

        merged_regions = voronoi_data["merged_regions"]
        merged_colors = voronoi_data["merged_colors"]

        return merged_regions, region_mask, merged_colors, width, height

    except FileNotFoundError as e:
        traceback.print_exc()
        print(f"Voronoi map file not found: {e}")
        return None
    except (ValueError, KeyError) as e:
        traceback.print_exc()
        print(f"Invalid or corrupted Voronoi map data: {e}")
        return None
    except (OSError, IOError) as e:
        traceback.print_exc()
        print(f"Error reading Voronoi map file: {e}")
        return None


def reconstruct_voronoi_image(
    merged_regions: dict,
    region_mask: np.ndarray,
    merged_colors: dict,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Reconstructs the Voronoi image from loaded data.

    :param merged_regions: Dictionary containing merged region data.
    :type merged_regions: dict
    :param region_mask: Array containing region IDs for each pixel.
    :type region_mask: np.ndarray
    :param merged_colors: Dictionary mapping region IDs to RGB colors.
    :type merged_colors: dict
    :param width: Width of the image.
    :type width: int
    :param height: Height of the image.
    :type height: int
    :raises ValueError: If the input data is invalid.
    :return: Reconstructed Voronoi image as numpy array.
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
        print(f"Error reconstructing Voronoi image: {e}")
        return np.zeros((height, width, 3), dtype=np.uint8)


def segment_map_with_voronoi(
    image: np.ndarray, points: List[Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Segments the map using Voronoi regions and assigns colors to each region.
    Uses flood fill to color each region, limited to the white areas of the map.
    Only considers the largest contour for each Voronoi region. Smaller contours
    are assigned to other regions if geometrically connected, or colored white if isolated.

    :param image: The input binary image where white pixels (>0) represent the map.
    :type image: np.ndarray
    :param points: List of points (x, y) to use as Voronoi cell centers.
    :type points: List[Tuple[int, int]]
    :raises ValueError: If no valid points are provided.
    :return: A tuple containing the colored image, region mask, and region dictionary.
    :rtype: Tuple[np.ndarray, np.ndarray, dict]
    """
    if not points:
        raise ValueError("No points provided for Voronoi segmentation")

    height, width = image.shape[:2]
    colored_map = np.zeros((height, width, 3), dtype=np.uint8)

    num_regions = len(points)
    region_colors = [
        (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        for _ in range(num_regions)
    ]

    region_map = np.full((height, width), -1, dtype=np.int32)

    y_coords, x_coords = np.where(image > 0)

    batch_size = 1000
    for i in range(0, len(y_coords), batch_size):
        batch_points = np.array(
            [
                (x_coords[j], y_coords[j])
                for j in range(i, min(i + batch_size, len(y_coords)))
            ]
        )

        distances = np.zeros((len(batch_points), len(points)))
        for j, point in enumerate(points):
            distances[:, j] = np.sqrt(np.sum((batch_points - point) ** 2, axis=1))

        closest_indices = np.argmin(distances, axis=1)

        for j in range(len(batch_points)):
            x, y = batch_points[j]
            region_map[y, x] = closest_indices[j]

    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    region_masks = []
    for i in range(num_regions):
        region_mask = np.zeros((height, width), dtype=np.uint8)
        region_mask[region_map == i] = 255
        region_masks.append(region_mask)

    valid_area_mask = np.zeros((height, width), dtype=np.uint8)
    valid_area_mask[image > 0] = 255

    processed_region_map = np.full((height, width), -1, dtype=np.int32)
    region_polygons = {}

    for i in range(num_regions):
        try:
            contours, _ = cv2.findContours(
                region_masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

                region_polygons[i] = largest_contour.reshape(-1, 2).tolist()

                largest_contour_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(largest_contour_mask, [largest_contour], 0, 255, -1)

                largest_contour_mask = cv2.bitwise_and(
                    largest_contour_mask, valid_area_mask
                )

                processed_region_map[largest_contour_mask > 0] = i

                for contour in contours:
                    if np.array_equal(contour, largest_contour):
                        continue

                    small_contour_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.drawContours(small_contour_mask, [contour], 0, 255, -1)
                    small_contour_mask = cv2.bitwise_and(
                        small_contour_mask, valid_area_mask
                    )

                    is_connected = False
                    for j in range(num_regions):
                        if j == i:
                            continue

                        dilated_mask = cv2.dilate(
                            small_contour_mask, np.ones((3, 3), np.uint8), iterations=1
                        )
                        intersection = cv2.bitwise_and(dilated_mask, region_masks[j])

                        if np.any(intersection > 0):
                            processed_region_map[small_contour_mask > 0] = j
                            is_connected = True
                            break

                    if not is_connected:
                        processed_region_map[small_contour_mask > 0] = -2
            else:
                region_polygons[i] = []
        except (cv2.error, ValueError):
            traceback.print_exc()
            processed_region_map[region_map == i] = i
            region_polygons[i] = []

    for i in range(num_regions):
        try:
            region_mask = np.zeros((height, width), dtype=np.uint8)
            region_mask[processed_region_map == i] = 255

            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                for contour in contours:
                    temp_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.drawContours(temp_mask, [contour], 0, 255, -1)

                    temp_mask = cv2.bitwise_and(temp_mask, valid_area_mask)
                    temp_mask = cv2.bitwise_and(temp_mask, region_mask)

                    seed_points = np.where(temp_mask > 0)
                    if len(seed_points[0]) > 0:
                        seed_y, seed_x = seed_points[0][0], seed_points[1][0]

                        mask[:] = 0

                        temp_colored_map = colored_map.copy()

                        cv2.floodFill(
                            temp_colored_map,
                            mask,
                            (seed_x, seed_y),
                            region_colors[i],
                            loDiff=(0, 0, 0),
                            upDiff=(0, 0, 0),
                            flags=8 | (255 << 8),
                        )

                        colored_diff = cv2.subtract(temp_colored_map, colored_map)
                        colored_diff_mask = (
                            np.any(colored_diff > 0, axis=2).astype(np.uint8) * 255
                        )
                        colored_diff_mask = cv2.bitwise_and(
                            colored_diff_mask, valid_area_mask
                        )
                        colored_diff_mask = cv2.bitwise_and(
                            colored_diff_mask, region_mask
                        )

                        for c in range(3):
                            colored_map[:, :, c] = np.where(
                                colored_diff_mask > 0,
                                temp_colored_map[:, :, c],
                                colored_map[:, :, c],
                            )
        except (cv2.error, ValueError):
            traceback.print_exc()
            seed_points = np.where((processed_region_map == i) & (image > 0))
            if len(seed_points[0]) > 0:
                for idx in range(min(10, len(seed_points[0]))):
                    seed_y, seed_x = seed_points[0][idx], seed_points[1][idx]
                    if (
                        image[seed_y, seed_x] > 0
                        and processed_region_map[seed_y, seed_x] == i
                    ):
                        mask[:] = 0

                        temp_colored_map = colored_map.copy()

                        cv2.floodFill(
                            temp_colored_map,
                            mask,
                            (seed_x, seed_y),
                            region_colors[i],
                            loDiff=(0, 0, 0),
                            upDiff=(0, 0, 0),
                            flags=8 | (255 << 8),
                        )

                        colored_diff = cv2.subtract(temp_colored_map, colored_map)
                        colored_diff_mask = (
                            np.any(colored_diff > 0, axis=2).astype(np.uint8) * 255
                        )
                        colored_diff_mask = cv2.bitwise_and(
                            colored_diff_mask, valid_area_mask
                        )
                        region_mask = np.zeros((height, width), dtype=np.uint8)
                        region_mask[processed_region_map == i] = 255
                        colored_diff_mask = cv2.bitwise_and(
                            colored_diff_mask, region_mask
                        )

                        for c in range(3):
                            colored_map[:, :, c] = np.where(
                                colored_diff_mask > 0,
                                temp_colored_map[:, :, c],
                                colored_map[:, :, c],
                            )

                        break

    white_regions = np.zeros((height, width), dtype=np.uint8)
    white_regions[processed_region_map == -2] = 255
    white_regions = cv2.bitwise_and(white_regions, valid_area_mask)

    colored_map[white_regions > 0] = [255, 255, 255]

    regions_dict = {}
    for i in range(num_regions):
        regions_dict[i] = {
            "polygon": region_polygons.get(i, []),
            "center": points[i],
            "color": region_colors[i],
        }

    return colored_map, processed_region_map, regions_dict


def load_pkl_gz_result(result_path: str) -> dict:
    """
    Loads the result file and returns objects, background objects, and class colors.

    :param result_path: Path to the gzipped pickle result file.
    :type result_path: str
    :raises ValueError: If the loaded results are not a dictionary.
    :return: A dictionary containing the results.
    :rtype: dict
    """
    potential_path = os.path.realpath(result_path)
    if potential_path != result_path:
        print(f"Resolved symlink for result_path: {result_path} -> \n{potential_path}")
        result_path = potential_path

    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if not isinstance(results, dict):
        raise ValueError(
            "Results should be a dictionary! other types are not supported!"
        )

    return results


class RoomData:
    def __init__(self, room_data: dict):
        self.class_name = room_data["room_class"]
        self.description = room_data["room_description"]
        pose = tuple(ast.literal_eval(room_data["pose"]))
        self.y, self.z, self.x, self.pitch, self.yaw, self.roll = pose
        self.yaw = -np.radians(self.yaw) + np.pi
        self.pitch = np.radians(self.pitch)
        self.roll = np.radians(self.roll)

    def get_map_coordinates(
        self, origin: Tuple[float, float], resolution: float, map_image: np.ndarray
    ) -> Tuple[int, int]:
        return (
            int(-(self.x + origin[0]) / resolution),
            int(map_image.shape[0] - ((self.y - origin[1]) / resolution)),
        )


if __name__ == "__main__":
    load_dotenv()
    MAP_PATH = "D:\\Documentos\\Datasets\\Robot@VirtualHomeLarge\\Home01"
    OBJECTS_PATH = "r_mapping_6_stride15\\pcd_r_mapping_6_stride15.pkl.gz"

    print("Loading objects and LLM agent...")
    llm_agent = Agent(
        model=OpenRouter(
            id="google/gemini-2.5-flash-lite",
            api_key=os.environ["OPENROUTER_API_KEY"],
        ),
        system_message=dedent(
            """
PERSONA:
A natural language processing expert, focused on following strict instructions and generating vivid and cohesive textual descriptions.

TASK:
Generate a **single paragraph description** of a specific environment, starting with "It's a [environment class]...", incorporating physical details, objects, spatial arrangement, visual aspects, and atmosphere, all from multiple descriptions provided and **strictly adhering to all format and content restrictions**. The output **MUST exclusively describe the provided `[environment class]`**, disregarding any information pertaining to different environment classes.

MANDATORY REQUIREMENTS:

- The output **MUST be a single paragraph only** – absolutely no exceptions, bullet points, or multiple paragraphs are permitted.
- The output **MUST begin with the exact phrase "It's a [environment class]..."** – this precise opening format is non-negotiable.
- The provided `[environment class]` **MUST be preserved and explicitly mentioned** within the opening phrase and throughout the description.
- All analysis and description **MUST exclusively reflect the specified `[environment class]`**, ignoring any data that describes other types of environments.
- Simple, clear sentences are **REQUIRED** – avoid complex grammatical structures or jargon.
- A conversational and engaging tone is **MANDATORY**, as if directly describing the environment to someone present.

STRICT GUIDELINES:

- Carefully observe elements that are repeatedly mentioned across different descriptions – these elements take **absolute priority**.
- Consider unique details that enrich understanding of the `[environment class]`, but only if they directly align with and enhance the description of that specific environment type.
- Critically evaluate all information; reject any elements that appear in isolation or contradict the nature of the specified `[environment class]`.
- Develop text that flows naturally and coherently, creating a vivid mental image.
- Maintain a balance between descriptive detail and conciseness – the output should be neither excessively verbose nor overly sparse.
- The input will contain several descriptions of the **same environment and its class**; utilize **ALL relevant provided information** pertaining to the specified `[environment class]`.

CONTENT REQUIREMENTS:

- Naturally incorporate physical characteristics of the `[environment class]` – these are essential for a complete picture.
- Describe objects present within the `[environment class]` and their precise spatial arrangement – positioning and layout are crucial.
- Mention relevant visual aspects: colors, materials, specific lighting conditions, and overall aesthetic – be specific and sensory.
- Convey the atmosphere and primary functionality of the `[environment class]` – capture its essence and purpose.
- Avoid repetitive language and sentence structures; vary phrasing within the simplicity constraints to maintain engagement.

FORBIDDEN ACTIONS:

- **Do NOT deviate from the single paragraph format under any circumstances.**
- **Do NOT ignore or alter the specified `[environment class]` in the output.**
- **Do NOT create multiple paragraphs, bullet points, or any form of list.**
- **Do NOT use overly complex, academic, or obscure language.**
- **Do NOT include contradictory information about the `[environment class]` or information pertaining to *other* environment classes.**
- **Do NOT omit or alter the mandatory opening phrase structure.**

QUALITY CONTROL:

- Prioritize consistently mentioned elements across all relevant descriptions.
- Include unique details only if they genuinely enhance understanding and fit the specified `[environment class]`.
- Ensure every sentence contributes meaningfully to the overall environmental picture of the `[environment class]`.
- Verify that the final output meticulously respects **all mandatory formatting and content requirements**.
- Confirm that the `[environment class]` is properly integrated and consistently described throughout the entire single paragraph.
"""
        ),
    )

    print("Loading map data and settings...")
    prompt_mask = "The environment is a {} and these are its descriptions: {}"
    results = load_pkl_gz_result(OBJECTS_PATH)

    room_data_list = [RoomData(room_data) for room_data in results["room_data_list"]]

    map_image_path = os.path.join(MAP_PATH, "processed_map.png")
    map_settings_path = os.path.join(MAP_PATH, "processed_map.yaml")
    map_image = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
    map_settings = yaml.safe_load(open(map_settings_path, "r"))

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
        print(f"Error cropping image: {str(e)}")

    print("Applying morphological operations...")
    processed_image = cv2.morphologyEx(
        map_image,
        cv2.MORPH_ERODE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=10,
    )

    print("Finding contours and calculating centers...")
    contours, _ = cv2.findContours(
        processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    centers = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        try:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if map_image[cY, cX] > 0:
                    centers.append((cX, cY))
                else:
                    y_coords, x_coords = np.where(map_image > 0)
                    if len(y_coords) > 0:
                        distances = np.sqrt((x_coords - cX) ** 2 + (y_coords - cY) ** 2)
                        nearest_idx = np.argmin(distances)
                        centers.append((x_coords[nearest_idx], y_coords[nearest_idx]))
        except (ZeroDivisionError, IndexError):
            traceback.print_exc()
            continue

    print("Generating Voronoi segmentation...")
    voronoi_image, region_mask, regions_dict = segment_map_with_voronoi(
        map_image, centers
    )

    origin = map_settings["origin"]
    resolution = map_settings["resolution"]

    print("Analyzing region classes...")
    region_class_counts = {}

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

    for room_data in room_data_list:
        x, y = room_data.get_map_coordinates(origin, resolution, voronoi_image)
        room_class = room_data.class_name

        if room_class not in class_colors:
            room_class = "transitioning"

        region_id = region_mask[y, x]
        if region_id not in region_class_counts:
            region_class_counts[region_id] = {}

        if room_class not in region_class_counts[region_id]:
            region_class_counts[region_id][room_class] = 0

        region_class_counts[region_id][room_class] += 1

    print("Determining dominant classes for regions...")
    for region_id, class_counts in region_class_counts.items():
        if class_counts:
            dominant_class = max(class_counts, key=class_counts.get)
            if region_id in regions_dict:
                regions_dict[region_id]["dominant_class"] = dominant_class

    print("Building region adjacency graph...")
    merged_regions = {}
    region_mapping = {}
    next_merged_id = 0

    adjacency_list = {}
    height, width = region_mask.shape

    for y in range(height):
        for x in range(width):
            current_region = region_mask[y, x]
            if current_region not in adjacency_list:
                adjacency_list[current_region] = set()

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_region = region_mask[ny, nx]
                    if neighbor_region != current_region:
                        adjacency_list[current_region].add(neighbor_region)

    print("Merging connected regions with same class...")
    visited = set()

    for region_id in regions_dict:
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
                    if (
                        neighbor not in visited
                        and neighbor in regions_dict
                        and "dominant_class" in regions_dict[neighbor]
                        and regions_dict[neighbor]["dominant_class"] == dominant_class
                    ):
                        visited.add(neighbor)
                        queue.append(neighbor)

        room_descriptions = []
        for orig_region in component:
            for room_data in room_data_list:
                x, y = room_data.get_map_coordinates(origin, resolution, voronoi_image)
                if region_mask[y, x] == orig_region:
                    if hasattr(room_data, "description") and room_data.description:
                        room_descriptions.append(room_data.description)

        merged_regions[next_merged_id] = {
            "original_regions": component,
            "dominant_class": dominant_class,
            "pixels": [],
            "room_descriptions": room_descriptions,
        }

        for orig_region in component:
            region_mapping[orig_region] = next_merged_id
            if "pixels" in regions_dict[orig_region]:
                merged_regions[next_merged_id]["pixels"].extend(
                    regions_dict[orig_region]["pixels"]
                )

        next_merged_id += 1

    print("Generating room descriptions using LLM...")

    def summarize_region_descriptions(region_id, region_data, prompt_mask, llm_agent):
        dominant_class = region_data["dominant_class"]
        combined_description = "\n".join(
            [d for d in region_data["room_descriptions"] if isinstance(d, str)]
        )
        return (
            region_id,
            llm_agent.run(
                prompt_mask.format(dominant_class, combined_description)
            ).content,
        )

    results = Parallel(n_jobs=-1)(
        delayed(summarize_region_descriptions)(
            region_id, merged_regions[region_id], prompt_mask, llm_agent
        )
        for region_id in merged_regions
    )

    for region_id, description in results:
        merged_regions[region_id]["summarized_description"] = description

    print("Updating region mappings...")
    for y in range(height):
        for x in range(width):
            original_region = region_mask[y, x]
            if original_region in region_mapping:
                region_mask[y, x] = region_mapping[original_region]

    regions_dict = merged_regions

    new_voronoi_image = np.zeros_like(voronoi_image)

    print("Generating random colors for merged regions...")
    merged_colors = {}
    for merged_id in merged_regions:
        merged_colors[merged_id] = (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256),
        )

    print("Saving Voronoi map...")
    save_success = save_voronoi_map(
        "voronoi_map.pkl", merged_regions, region_mask, merged_colors, width, height
    )

    print("Loading and reconstructing Voronoi map...")
    voronoi_map_data = load_voronoi_map("voronoi_map.pkl")

    if voronoi_map_data is None:
        print("Failed to load Voronoi map")
        exit()

    merged_regions, region_mask, merged_colors, width, height = voronoi_map_data

    print("Reconstructing final image...")
    reconstructed_image = reconstruct_voronoi_image(
        merged_regions, region_mask, merged_colors, width, height
    )

    if save_success:
        print("Voronoi map saved successfully to voronoi_map.pkl.gz")
    else:
        print("Failed to save Voronoi map")

    print("Processing complete!")

    cv2.namedWindow("reconstructed_image", cv2.WINDOW_NORMAL)
    cv2.imshow("reconstructed_image", reconstructed_image)
    cv2.namedWindow("processed_image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("voronoi_image", cv2.WINDOW_NORMAL)
    cv2.imshow("processed_image", processed_image)
    cv2.imshow("voronoi_image", voronoi_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

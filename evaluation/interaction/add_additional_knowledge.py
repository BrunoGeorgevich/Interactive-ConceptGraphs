from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import traceback
import threading
import pickle
import gzip
import json
import csv
import sys
import os

import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from conceptgraph.interaction.schemas import SystemConfig
from conceptgraph.interaction.system import InteractionSystem
from labs.utils import unity_to_ros_coordinates

try:
    from conceptgraph.interaction.watershed_segmenter import (
        load_house_context,
        process_house_segmentation,
        load_profile,
        DEFAULT_PARAMS as WS_DEFAULT_PARAMS,
        CLASS_COLORS as WS_CLASS_COLORS,
        TRAJECTORY_IMAGE_DIMMING,
        MAP_BINARY_THRESHOLD,
        MIN_CONTOUR_AREA,
        CROP_PADDING,
    )
except ImportError:
    raise ImportError(
        "Watershed segmenter module not found. Ensure that the ConceptGraphs package is correctly installed."
    )

"""
Script for analyzing false negative results from the online processing evaluation
and generating queries to add missing objects to the knowledge base.

This module processes the evaluation results for all 30 homes using the 'online'
processing type, identifies objects that were not correlated (false negatives),
and generates knowledge base insertion queries following the INTENTION_INTERPRETATION_PROMPT format.
"""


def get_false_negatives_csv_path(output_dir: str, home_id: int) -> str:
    """
    Constructs the path to the objects CSV file containing false negative data for a given home.

    :param output_dir: Directory containing output files.
    :type output_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :raises ValueError: If output_dir is empty or home_id is negative.
    :return: Path to the objects CSV file.
    :rtype: str
    """
    if not output_dir or home_id < 0:
        raise ValueError("Invalid output_dir or home_id for CSV path construction.")
    return os.path.join(output_dir, f"Home{home_id:02d}_online_objects.csv")


def get_virtual_objects_csv_path(database_path: str, home_id: int) -> str:
    """
    Constructs the path to the virtual objects CSV file for a given home.

    :param database_path: Base path to the database.
    :type database_path: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :raises ValueError: If database_path is empty or home_id is negative.
    :return: Path to the virtual objects CSV file.
    :rtype: str
    """
    if not database_path or home_id < 0:
        raise ValueError("Invalid database_path or home_id for CSV path construction.")
    return os.path.join(database_path, f"Home{home_id:02d}", "VirtualObjects.csv")


def read_false_negatives_from_csv(csv_path: str) -> list[dict]:
    """
    Reads the objects CSV file and extracts false negative entries.

    A false negative is identified when 'matched' column is 'False' and
    'processed_class' is 'None'.

    :param csv_path: Path to the objects CSV file.
    :type csv_path: str
    :raises FileNotFoundError: If the CSV file does not exist.
    :raises RuntimeError: If there is an error reading or parsing the CSV file.
    :return: List of dictionaries containing false negative object data.
    :rtype: list[dict]
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    false_negatives = []
    try:
        with open(csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=";")
            for row in reader:
                if str(row.get("matched", "True").lower()) == "false":
                    candidate = {
                        "virtual_idx": int(row.get("virtual_idx", -1)),
                        "virtual_type": row.get("virtual_type", "Unknown"),
                        "true_votes": int(row.get("true_votes", 0)),
                        "false_votes": int(row.get("false_votes", 0)),
                        "match": row.get("match", "false"),
                        "coordinates": [
                            float(row.get("x_coordinate", 0.0)),
                            float(row.get("y_coordinate", 0.0)),
                            float(row.get("z_coordinate", 0.0)),
                        ],
                    }
                    false_negatives.append(candidate)

    except (OSError, csv.Error, KeyError, ValueError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Error reading false negatives from CSV: {e}")

    return false_negatives


def get_virtual_object_coordinates(
    virtual_objects_df: object, virtual_type: str
) -> tuple[float, float, float] | None:
    """
    Retrieves ROS coordinates for a virtual object type from the virtual objects dataframe.

    :param virtual_objects_df: Pandas DataFrame containing virtual objects data.
    :type virtual_objects_df: object
    :param virtual_type: The type/class of the virtual object to search for.
    :type virtual_type: str
    :return: Tuple of (x, y, z) ROS coordinates or None if not found.
    :rtype: tuple[float, float, float] | None
    """
    try:
        matching_rows = virtual_objects_df[
            virtual_objects_df["type"].str.lower() == virtual_type.lower()
        ]

        if matching_rows.empty:
            return None

        row = matching_rows.iloc[0]
        rx, ry, rz, _ = unity_to_ros_coordinates(
            row["globalPosition"], row["rotation"], ""
        )
        return (rx, ry, rz)

    except (KeyError, AttributeError, TypeError, ValueError):
        traceback.print_exc()
        return None


def get_all_virtual_object_coordinates(
    virtual_objects_df: object, virtual_type: str
) -> list[tuple[float, float, float]]:
    """
    Retrieves all ROS coordinates for a virtual object type from the virtual objects dataframe.

    :param virtual_objects_df: Pandas DataFrame containing virtual objects data.
    :type virtual_objects_df: object
    :param virtual_type: The type/class of the virtual object to search for.
    :type virtual_type: str
    :return: List of tuples containing (x, y, z) ROS coordinates for all matching objects.
    :rtype: list[tuple[float, float, float]]
    """
    coordinates_list = []
    try:
        matching_rows = virtual_objects_df[
            virtual_objects_df["type"].str.lower() == virtual_type.lower()
        ]

        if matching_rows.empty:
            return coordinates_list

        for _, row in matching_rows.iterrows():
            rx, ry, rz, _ = unity_to_ros_coordinates(
                row["globalPosition"], row["rotation"], ""
            )
            coordinates_list.append((rx, ry, rz))

    except (KeyError, AttributeError, TypeError, ValueError):
        traceback.print_exc()

    return coordinates_list


def world_to_map_coordinates(
    world_coords: tuple[float, float, float],
    origin: tuple[float, float],
    resolution: float,
    height_img: int,
) -> tuple[int, int]:
    """
    Converts world coordinates to map pixel coordinates.

    :param world_coords: World coordinates (x, y, z).
    :type world_coords: tuple[float, float, float]
    :param origin: Map origin coordinates (x, y).
    :type origin: tuple[float, float]
    :param resolution: Map resolution in meters per pixel.
    :type resolution: float
    :param height_img: Height of the map image.
    :type height_img: int
    :return: Pixel coordinates (x, y).
    :rtype: tuple[int, int]
    """
    world_x_map = world_coords[2]
    world_y_map = -world_coords[0]
    pixel_x = int((world_x_map - origin[0]) / resolution)
    pixel_y = int(height_img - ((world_y_map - origin[1]) / resolution))
    return pixel_x, pixel_y


def generate_unique_room_names(merged_regions: dict) -> dict:
    """
    Generates unique room names based on class types and occurrence count.

    :param merged_regions: Dictionary of merged region data.
    :type merged_regions: dict
    :return: Mapping of region IDs to unique room names.
    :rtype: dict
    """
    id_to_name = {}
    class_counters = {}
    sorted_ids = sorted(list(merged_regions.keys()))

    for rid in sorted_ids:
        data = merged_regions[rid]
        base_class = data.get("dominant_class", "unknown")

        if base_class not in class_counters:
            class_counters[base_class] = 1
        else:
            class_counters[base_class] += 1

        unique_name = f"{base_class} {class_counters[base_class]}"
        id_to_name[rid] = unique_name

    return id_to_name


def get_or_compute_watershed_data(
    house_id: int, prefix: str, base_path: str, local_dir: str
) -> tuple[dict, np.ndarray, dict, int, int, tuple[float, float], float]:
    """
    Retrieves cached watershed segmentation data for a given house and prefix, or computes it if unavailable.

    :param house_id: Identifier for the house.
    :type house_id: int
    :param prefix: Data prefix for context and segmentation.
    :type prefix: str
    :param base_path: Base path to the dataset.
    :type base_path: str
    :param local_dir: Local directory for caching segmentation data.
    :type local_dir: str
    :return: Tuple containing merged regions, region mask, merged colors, width, height, origin, and resolution.
    :rtype: tuple[dict, np.ndarray, dict, int, int, tuple[float, float], float]
    :raises ValueError: If context loading fails or the prefix is not found in the context.
    """
    os.makedirs(local_dir, exist_ok=True)
    filename = f"Home{house_id:02d}_{prefix}_watershed.pkl.gz"
    local_path = os.path.join(local_dir, filename)

    context = None
    try:
        context = load_house_context(
            house_id=house_id,
            base_path=base_path,
            prefixes=[prefix],
            map_binary_threshold=MAP_BINARY_THRESHOLD,
            min_contour_area=MIN_CONTOUR_AREA,
            crop_padding=CROP_PADDING,
        )
    except (KeyError, OSError, TypeError) as e:
        traceback.print_exc()
        raise ValueError(f"Could not load context for prefix '{prefix}': {e}") from e

    if prefix not in context:
        raise ValueError(f"Could not load context for prefix '{prefix}'.")

    map_origin = context[prefix]["origin"]
    map_resolution = context[prefix]["resolution"]

    if os.path.exists(local_path):
        try:
            with gzip.open(local_path, "rb") as f:
                cached_data = pickle.load(f)
            return (
                cached_data["merged_regions"],
                cached_data["region_mask"],
                cached_data["merged_colors"],
                cached_data["width"],
                cached_data["height"],
                map_origin,
                map_resolution,
            )
        except (OSError, pickle.UnpicklingError) as e:
            traceback.print_exc()
            print(f"Failed to load cache, recomputing: {e}")

    print(f"Cache missing. Computing Watershed data for House {house_id} ({prefix})...")

    params = load_profile(house_id, WS_DEFAULT_PARAMS, base_path=base_path)
    results = process_house_segmentation(
        house_id=house_id,
        house_context=context,
        params=params,
        prefixes=[prefix],
        llm_agent=None,
        generate_descriptions=False,
        prompt_mask="",
        class_colors=WS_CLASS_COLORS,
        trajectory_dimming=TRAJECTORY_IMAGE_DIMMING,
    )

    res_data = results[prefix]
    data_to_save = {
        "merged_regions": res_data["merged_regions"],
        "region_mask": res_data["region_mask"],
        "merged_colors": res_data["merged_colors"],
        "width": res_data["width"],
        "height": res_data["height"],
    }

    try:
        with gzip.open(local_path, "wb") as f:
            pickle.dump(data_to_save, f)
    except (OSError, pickle.PicklingError) as e:
        traceback.print_exc()
        print(f"Failed to save cache: {e}")

    return (
        res_data["merged_regions"],
        res_data["region_mask"],
        res_data["merged_colors"],
        res_data["width"],
        res_data["height"],
        map_origin,
        map_resolution,
    )


def get_room_name_from_coordinates(
    world_coords: tuple[float, float, float],
    origin: tuple[float, float],
    resolution: float,
    region_mask: np.ndarray,
    unique_names: dict[int, str],
) -> str:
    """
    Determines the room name based on world coordinates and segmentation mask.

    If the coordinates fall outside valid regions or in an unknown area,
    finds the nearest known room by expanding search radius.

    :param world_coords: World coordinates (x, y, z).
    :type world_coords: tuple[float, float, float]
    :param origin: Map origin coordinates (x, y).
    :type origin: tuple[float, float]
    :param resolution: Map resolution in meters per pixel.
    :type resolution: float
    :param region_mask: Region mask array containing region IDs (H, W).
    :type region_mask: np.ndarray
    :param unique_names: Dictionary mapping region IDs to readable names.
    :type unique_names: dict[int, str]
    :return: Room name from the nearest known room.
    :rtype: str
    """
    h, w = region_mask.shape[:2]
    px, py = world_to_map_coordinates(world_coords, origin, resolution, h)

    if 0 <= px < w and 0 <= py < h:
        region_id = region_mask[py, px]
        if region_id >= 0:
            room_name = unique_names.get(int(region_id), None)
            if room_name is not None:
                return room_name

    return _find_nearest_known_room(px, py, w, h, region_mask, unique_names)


def _find_nearest_known_room(
    px: int,
    py: int,
    width: int,
    height: int,
    region_mask: np.ndarray,
    unique_names: dict[int, str],
    max_search_radius: int = 500,
) -> str:
    """
    Finds the nearest known room by expanding search radius from given pixel coordinates.

    :param px: Pixel x coordinate.
    :type px: int
    :param py: Pixel y coordinate.
    :type py: int
    :param width: Width of the region mask.
    :type width: int
    :param height: Height of the region mask.
    :type height: int
    :param region_mask: Region mask array containing region IDs (H, W).
    :type region_mask: np.ndarray
    :param unique_names: Dictionary mapping region IDs to readable names.
    :type unique_names: dict[int, str]
    :param max_search_radius: Maximum radius to search for a known room.
    :type max_search_radius: int
    :return: Room name from the nearest known room or default if none found.
    :rtype: str
    """
    default_room = "Unknown Area"

    if not unique_names:
        return default_room

    for radius in range(1, max_search_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue

                nx, ny = px + dx, py + dy

                if 0 <= nx < width and 0 <= ny < height:
                    region_id = region_mask[ny, nx]
                    if region_id >= 0:
                        room_name = unique_names.get(int(region_id), None)
                        if room_name is not None:
                            return room_name

    if unique_names:
        return next(iter(unique_names.values()))

    return default_room


def enrich_false_negatives(
    false_negatives: list[dict],
    virtual_objects_df: object,
) -> list[dict]:
    """
    Removes duplicate false negative entries based on virtual_type and coordinates.

    Duplicates are identified when both the virtual_type (case-insensitive) and
    coordinates (within tolerance) match.

    :param false_negatives: List of false negative dictionaries.
    :type false_negatives: list[dict]
    :param virtual_objects_df: Pandas DataFrame containing virtual objects data.
    :type virtual_objects_df: object
    :return: List of unique false negative dictionaries with coordinates.
    :rtype: list[dict]
    """
    unique_false_negatives = []

    for fn_obj in false_negatives:
        virtual_idx = fn_obj["virtual_idx"]
        virtual_obj = virtual_objects_df.iloc[virtual_idx]

        fn_obj_with_coords = fn_obj.copy()
        fn_obj_with_coords["coordinates"] = virtual_obj["globalPosition"]
        unique_false_negatives.append(fn_obj_with_coords)

    return unique_false_negatives


def get_knowledge_log_path(
    results_output_dir: str, home_id: int, entry_index: int
) -> str:
    """
    Constructs the path to the knowledge entry log file.

    :param results_output_dir: Base directory for results output.
    :type results_output_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param entry_index: Index of the entry.
    :type entry_index: int
    :return: Path to the knowledge entry log JSON file.
    :rtype: str
    """
    home_folder = os.path.join(results_output_dir, f"Home{home_id:02d}")
    return os.path.join(home_folder, f"entry_{entry_index:04d}.json")


def check_knowledge_entry_exists(
    results_output_dir: str, home_id: int, entry_index: int
) -> bool:
    """
    Checks if a knowledge entry log file already exists.

    :param results_output_dir: Base directory for results output.
    :type results_output_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param entry_index: Index of the entry.
    :type entry_index: int
    :return: True if the log file exists, False otherwise.
    :rtype: bool
    """
    log_path = get_knowledge_log_path(results_output_dir, home_id, entry_index)
    return os.path.exists(log_path)


def save_knowledge_entry_log(
    results_output_dir: str,
    home_id: int,
    entry_index: int,
    entry: dict,
    response: str | None,
    error: str | None,
) -> None:
    """
    Saves a knowledge entry log to a JSON file in the home subfolder.

    :param results_output_dir: Base directory for results output.
    :type results_output_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param entry_index: Index of the entry.
    :type entry_index: int
    :param entry: Knowledge entry dictionary containing the query.
    :type entry: dict
    :param response: Model response text or None if failed.
    :type response: str | None
    :param error: Error message or None if successful.
    :type error: str | None
    """
    home_folder = os.path.join(results_output_dir, f"Home{home_id:02d}")
    os.makedirs(home_folder, exist_ok=True)

    log_path = get_knowledge_log_path(results_output_dir, home_id, entry_index)

    log_data = {
        "home_id": home_id,
        "entry_index": entry_index,
        "input_query": entry.get("query", ""),
        "entry_data": entry,
        "output_response": response,
        "error": error,
        "status": "success" if error is None else "failed",
    }

    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
    except (OSError, TypeError, ValueError) as e:
        traceback.print_exc()
        print(f"Error saving log for Home {home_id:02d} entry {entry_index}: {e}")


def generate_knowledge_entry(
    virtual_type: str,
    home_id: int,
    entry_index: int,
    coordinates: tuple[float, float, float],
    room_name: str,
) -> dict:
    """
    Generates a knowledge entry dictionary for a false negative object.

    This entry follows the ADDITIONAL_INFORMATION format from INTENTION_INTERPRETATION_PROMPT.

    :param virtual_type: The type/class of the virtual object.
    :type virtual_type: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param entry_index: Index of the entry for unique identification.
    :type entry_index: int
    :param coordinates: Tuple of (x, y, z) ROS coordinates.
    :type coordinates: tuple[float, float, float]
    :param room_name: Name of the room where the object is located.
    :type room_name: str
    :return: Dictionary containing the knowledge entry.
    :rtype: dict
    """
    return {
        "type": "object",
        "class_name": virtual_type.lower().replace(
            "kitchenfurniture", "kitchen furniture"
        ),
        "description": f"Unmapped {virtual_type} object from virtual environment evaluation",
        "room_name": room_name,
        "coordinates": list(coordinates),
        "source": "evaluation_false_negative",
        "home_id": home_id,
        "entry_index": entry_index,
        "query": f"Add the object {virtual_type} in the coordinates {coordinates[0]:.2f}, {coordinates[1]:.2f}, {coordinates[2]:.2f}",
    }


def insert_knowledge_entry_to_system(
    system: InteractionSystem,
    entry: dict,
    lock: threading.Lock,
    success_list: list,
    error_list: list,
    results_output_dir: str,
    skip_existing: bool,
) -> None:
    """
    Inserts a single knowledge entry into the system using the generated query.

    :param system: Instance of InteractionSystem for processing queries.
    :type system: InteractionSystem
    :param entry: Knowledge entry dictionary containing the query.
    :type entry: dict
    :param lock: Threading lock for synchronization.
    :type lock: threading.Lock
    :param success_list: Shared list to store successful insertions.
    :type success_list: list
    :param error_list: Shared list to store failed insertions.
    :type error_list: list
    :param results_output_dir: Base directory for results output.
    :type results_output_dir: str
    :param skip_existing: Whether to skip entries with existing log files.
    :type skip_existing: bool
    """
    home_id = entry.get("home_id", 0)
    entry_index = entry.get("entry_index", 0)
    room_name = entry.get("room_name", "Unknown Area")

    if skip_existing and check_knowledge_entry_exists(
        results_output_dir, home_id, entry_index
    ):
        with lock:
            success_list.append(
                {
                    "entry": entry,
                    "response": "SKIPPED - Log already exists",
                    "skipped": True,
                }
            )
        return

    try:
        query = entry.get("query", "")
        coordinates = entry.get("coordinates", [0.0, 0.0, 0.0])
        query_with_room = f"{query} Located at {room_name}"

        response = system.process_query(
            query=query_with_room,
            user_pose=(coordinates[0], coordinates[1], coordinates[2]),
        )

        response_text = response.text_response

        save_knowledge_entry_log(
            results_output_dir=results_output_dir,
            home_id=home_id,
            entry_index=entry_index,
            entry=entry,
            response=response_text,
            error=None,
        )

        with lock:
            success_list.append(
                {
                    "entry": entry,
                    "response": response_text,
                    "skipped": False,
                }
            )

    except (RuntimeError, KeyError, ValueError, AttributeError) as e:
        traceback.print_exc()
        error_msg = str(e)

        save_knowledge_entry_log(
            results_output_dir=results_output_dir,
            home_id=home_id,
            entry_index=entry_index,
            entry=entry,
            response=None,
            error=error_msg,
        )

        with lock:
            error_list.append(
                {
                    "entry": entry,
                    "error": error_msg,
                }
            )


def remove_unused_object_virtual_df(
    virtual_objects_df: pd.DataFrame, unused_objects_classes: list[str]
) -> pd.DataFrame:
    """
    Removes rows from the virtual objects DataFrame that match the unused object classes.

    :param virtual_objects_df: Pandas DataFrame containing virtual objects data.
    :type virtual_objects_df: pd.DataFrame
    :param unused_objects_classes: List of object classes to be removed.
    :type unused_objects_classes: list[str]
    :return: Filtered Pandas DataFrame with unused object classes removed.
    :rtype: pd.DataFrame
    """
    if virtual_objects_df.empty or not unused_objects_classes:
        return virtual_objects_df

    filtered_df = virtual_objects_df[
        ~virtual_objects_df["type"]
        .str.lower()
        .isin([cls.lower() for cls in unused_objects_classes])
    ].reset_index(drop=True)

    return filtered_df


def process_home_false_negatives(
    home_id: int,
    output_dir: str,
    database_path: str,
    results_list: list,
    lock: threading.Lock,
    progress_counter: list[int],
    total_homes: int,
    local_data_dir: str,
    coordinate_tolerance: float = 0.1,
    objects_to_remove: list[str] = [],
) -> None:
    """
    Processes false negatives for a single home and generates knowledge entries.

    :param home_id: Identifier for the home.
    :type home_id: int
    :param output_dir: Directory containing evaluation output files.
    :type output_dir: str
    :param database_path: Base path to the database containing virtual objects.
    :type database_path: str
    :param results_list: Shared list to store results from all homes.
    :type results_list: list
    :param lock: Threading lock for synchronization.
    :type lock: threading.Lock
    :param progress_counter: List containing the current progress count.
    :type progress_counter: list[int]
    :param total_homes: Total number of homes to process.
    :type total_homes: int
    :param local_data_dir: Local directory for caching watershed data.
    :type local_data_dir: str
    :param coordinate_tolerance: Tolerance for coordinate comparison when removing duplicates.
    :type coordinate_tolerance: float
    :param objects_to_remove: List of object classes to be excluded from processing.
    :type objects_to_remove: list[str]
    """
    home_results = {
        "home_id": home_id,
        "false_negatives_count": 0,
        # "enriched_false_negatives_count": 0,
        "knowledge_entries": [],
        "error": None,
    }

    try:
        csv_path = get_false_negatives_csv_path(output_dir, home_id)

        if not os.path.exists(csv_path):
            home_results["error"] = f"CSV file not found: {csv_path}"
            with lock:
                results_list.append(home_results)
                progress_counter[0] += 1
                print_progress(progress_counter[0], total_homes, home_id, "SKIPPED")
            return

        false_negatives = read_false_negatives_from_csv(csv_path)
        home_results["false_negatives_count"] = len(false_negatives)

        # virtual_objects_path = get_virtual_objects_csv_path(database_path, home_id)
        # virtual_objects_df = read_virtual_objects(virtual_objects_path)
        # virtual_objects_df = remove_unused_object_virtual_df(
        #     virtual_objects_df, objects_to_remove
        # )

        # enriched_false_negatives = enrich_false_negatives(
        #     false_negatives, virtual_objects_df
        # )
        # home_results["enriched_false_negatives_count"] = len(enriched_false_negatives)

        merged_regions = None
        region_mask = None
        unique_names = None
        map_origin = None
        map_resolution = None

        if false_negatives:
            try:
                (
                    merged_regions,
                    region_mask,
                    _,
                    _,
                    _,
                    map_origin,
                    map_resolution,
                ) = get_or_compute_watershed_data(
                    house_id=home_id,
                    prefix="online",
                    base_path=database_path,
                    local_dir=local_data_dir,
                )
                unique_names = generate_unique_room_names(merged_regions)
            except (ValueError, OSError, pickle.UnpicklingError) as e:
                traceback.print_exc()
                print(f"Warning: Could not load watershed data for Home {home_id}: {e}")

        knowledge_entries = []

        # unique_false_negatives = []
        # unique_virtual_idxs = {}
        # for fn_obj in false_negatives:
        #     if fn_obj["virtual_idx"] not in unique_virtual_idxs:
        #         unique_false_negatives.append(fn_obj)
        #         unique_virtual_idxs[fn_obj["virtual_idx"]] = True

        for idx, fn_obj in enumerate(false_negatives):
            coordinates = fn_obj.get("coordinates", (-1.0, -1.0, -1.0))

            room_name = "Unknown Area"
            if (
                region_mask is not None
                and unique_names is not None
                and map_origin is not None
                and map_resolution is not None
                and coordinates != (-1.0, -1.0, -1.0)
            ):
                room_name = get_room_name_from_coordinates(
                    world_coords=coordinates,
                    origin=map_origin,
                    resolution=map_resolution,
                    region_mask=region_mask,
                    unique_names=unique_names,
                )

            entry = generate_knowledge_entry(
                virtual_type=fn_obj["virtual_type"],
                home_id=home_id,
                entry_index=idx,
                coordinates=coordinates,
                room_name=room_name,
            )
            knowledge_entries.append(entry)

        home_results["knowledge_entries"] = knowledge_entries

        with lock:
            results_list.append(home_results)
            progress_counter[0] += 1
            print_progress(
                progress_counter[0],
                total_homes,
                home_id,
                f"FN: {len(false_negatives)} | Unique: {len(false_negatives)}",
            )

    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as e:
        traceback.print_exc()
        home_results["error"] = str(e)
        with lock:
            results_list.append(home_results)
            progress_counter[0] += 1
            print_progress(progress_counter[0], total_homes, home_id, "ERROR")


def print_progress(current: int, total: int, home_id: int, status: str) -> None:
    """
    Prints progress information for the current home processing.

    :param current: Current number of processed homes.
    :type current: int
    :param total: Total number of homes to process.
    :type total: int
    :param home_id: Identifier of the currently processed home.
    :type home_id: int
    :param status: Status message for the current home.
    :type status: str
    """
    bar_length = 40
    if total == 0:
        return
    filled_length = int(bar_length * current // total)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)
    percent = (current / total) * 100
    print(
        f"\rProgress: |{bar}| {percent:.1f}% ({current}/{total}) "
        f"- Home {home_id:02d}: {status}",
        end="",
        flush=True,
    )
    if current == total:
        print()


def save_knowledge_entries_to_json(results: list[dict], output_path: str) -> None:
    """
    Saves all generated knowledge entries to a JSON file.

    :param results: List of results from all homes containing knowledge entries.
    :type results: list[dict]
    :param output_path: Path to the output JSON file.
    :type output_path: str
    :raises RuntimeError: If there is an error writing the JSON file.
    """
    try:
        all_entries = []
        summary = {
            "total_homes_processed": len(results),
            "total_false_negatives": 0,
            "total_unique_false_negatives": 0,
            "homes_with_errors": 0,
            "per_home_summary": [],
        }

        for result in results:
            home_summary = {
                "home_id": result["home_id"],
                "false_negatives_count": result["false_negatives_count"],
                # "enriched_false_negatives_count": result.get(
                #     "enriched_false_negatives_count", 0
                # ),
                "error": result["error"],
            }
            summary["per_home_summary"].append(home_summary)
            summary["total_false_negatives"] += result["false_negatives_count"]
            # summary["total_unique_false_negatives"] += result.get(
            #     "enriched_false_negatives_count", 0
            # )

            if result["error"]:
                summary["homes_with_errors"] += 1

            all_entries.extend(result.get("knowledge_entries", []))

        output_data = {
            "summary": summary,
            "knowledge_entries": all_entries,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)

        print(f"\nKnowledge entries saved to: {output_path}")
        print(
            f"Total false negatives across all homes: {summary['total_false_negatives']}"
        )
        print(
            f"Total unique false negatives: {summary['total_unique_false_negatives']}"
        )
        print(f"Homes with errors: {summary['homes_with_errors']}")

    except (OSError, TypeError, ValueError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Error saving knowledge entries to JSON: {e}")


def save_knowledge_entries_to_csv(results: list[dict], output_path: str) -> None:
    """
    Saves all generated knowledge entries to a CSV file.

    :param results: List of results from all homes containing knowledge entries.
    :type results: list[dict]
    :param output_path: Path to the output CSV file.
    :type output_path: str
    :raises RuntimeError: If there is an error writing the CSV file.
    """
    try:
        all_entries = []
        for result in results:
            for entry in result.get("knowledge_entries", []):
                entry_flat = {
                    "home_id": entry.get("home_id", ""),
                    "entry_index": entry.get("entry_index", ""),
                    "type": entry.get("type", ""),
                    "class_name": entry.get("class_name", ""),
                    "description": entry.get("description", ""),
                    "room_name": entry.get("room_name", ""),
                    "coordinates_x": entry.get("coordinates", [-1, -1, -1])[0],
                    "coordinates_y": entry.get("coordinates", [-1, -1, -1])[1],
                    "coordinates_z": entry.get("coordinates", [-1, -1, -1])[2],
                    "source": entry.get("source", ""),
                    "query": entry.get("query", ""),
                }
                all_entries.append(entry_flat)

        unique_all_entries = []

        for entry in all_entries:
            if entry not in unique_all_entries:
                unique_all_entries.append(entry)

        if unique_all_entries:
            fieldnames = [
                "home_id",
                "entry_index",
                "type",
                "class_name",
                "description",
                "room_name",
                "coordinates_x",
                "coordinates_y",
                "coordinates_z",
                "source",
                "query",
            ]
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL
                )
                writer.writeheader()
                writer.writerows(unique_all_entries)

            print(f"Knowledge entries CSV saved to: {output_path}")

    except (OSError, csv.Error, TypeError, ValueError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Error saving knowledge entries to CSV: {e}")


def insert_knowledge_entries_to_database(
    results: list[dict],
    database_path: str,
    max_workers: int,
    results_output_dir: str,
    skip_existing: bool,
) -> dict:
    """
    Inserts all knowledge entries into the database using InteractionSystem in threads.

    :param results: List of results from all homes containing knowledge entries.
    :type results: list[dict]
    :param database_path: Base path to the database.
    :type database_path: str
    :param max_workers: Maximum number of concurrent threads for insertion.
    :type max_workers: int
    :param results_output_dir: Base directory for results output and logs.
    :type results_output_dir: str
    :param skip_existing: Whether to skip entries with existing log files.
    :type skip_existing: bool
    :return: Dictionary containing insertion statistics.
    :rtype: dict
    """
    insertion_stats = {
        "total_entries": 0,
        "successful_insertions": 0,
        "failed_insertions": 0,
        "skipped_insertions": 0,
        "per_home_stats": [],
    }

    for result in results:
        home_id = result["home_id"]
        knowledge_entries = result.get("knowledge_entries", [])

        if not knowledge_entries or result.get("error"):
            continue

        home_stats = {
            "home_id": home_id,
            "total_entries": len(knowledge_entries),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }

        try:
            config = SystemConfig(
                house_id=home_id,
                dataset_base_path=database_path,
                prefix="online",
                qdrant_url="http://localhost:6333",
                force_recreate_table=False,
                local_data_dir="data",
                debug_input_path=os.path.join("data", "input_debug.txt"),
                debug_output_path=os.path.join("data", "output_debug.txt"),
            )
            system = InteractionSystem(config)

            success_list = []
            error_list = []
            lock = threading.Lock()

            print(
                f"\nInserting {len(knowledge_entries)} entries for Home {home_id:02d}..."
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for entry in knowledge_entries:
                    futures.append(
                        executor.submit(
                            insert_knowledge_entry_to_system,
                            system=system,
                            entry=entry,
                            lock=lock,
                            success_list=success_list,
                            error_list=error_list,
                            results_output_dir=results_output_dir,
                            skip_existing=skip_existing,
                        )
                    )

                for future in as_completed(futures):
                    try:
                        future.result()
                    except (RuntimeError, KeyError, ValueError) as e:
                        traceback.print_exc()
                        print(f"\nInsertion thread failed: {e}")

            skipped_count = sum(1 for s in success_list if s.get("skipped", False))
            actual_success = len(success_list) - skipped_count

            home_stats["successful"] = actual_success
            home_stats["failed"] = len(error_list)
            home_stats["skipped"] = skipped_count

            insertion_stats["total_entries"] += len(knowledge_entries)
            insertion_stats["successful_insertions"] += actual_success
            insertion_stats["failed_insertions"] += len(error_list)
            insertion_stats["skipped_insertions"] += skipped_count

            print(
                f"Home {home_id:02d}: {actual_success} successful, "
                f"{len(error_list)} failed, {skipped_count} skipped"
            )

        except (RuntimeError, KeyError, ValueError, AttributeError) as e:
            traceback.print_exc()
            home_stats["failed"] = len(knowledge_entries)
            insertion_stats["failed_insertions"] += len(knowledge_entries)
            print(f"Error initializing system for Home {home_id:02d}: {e}")

        insertion_stats["per_home_stats"].append(home_stats)

    return insertion_stats


def run_false_negative_analysis(
    output_dir: str,
    home_ids: list[int],
    max_workers: int,
    results_output_dir: str,
    database_path: str,
    insert_to_database: bool,
    skip_existing_knowledge: bool,
    local_data_dir: str,
    coordinate_tolerance: float = 0.1,
    prepare_additional_knowledge: bool = False,
    objects_to_remove: list[str] = None,
) -> None:
    """
    Runs the false negative analysis for all specified homes using thread pool.

    :param output_dir: Directory containing evaluation output files.
    :type output_dir: str
    :param home_ids: List of home identifiers to process.
    :type home_ids: list[int]
    :param max_workers: Maximum number of concurrent threads.
    :type max_workers: int
    :param results_output_dir: Directory to save the analysis results.
    :type results_output_dir: str
    :param database_path: Base path to the database containing virtual objects.
    :type database_path: str
    :param insert_to_database: Whether to insert entries into the database.
    :type insert_to_database: bool
    :param skip_existing_knowledge: Whether to skip entries with existing log files.
    :type skip_existing_knowledge: bool
    :param local_data_dir: Local directory for caching watershed data.
    :type local_data_dir: str
    :param coordinate_tolerance: Tolerance for coordinate comparison when removing duplicates.
    :type coordinate_tolerance: float
    :param prepare_additional_knowledge: Whether to generate additional knowledge entries or read from JSON.
    :type prepare_additional_knowledge: bool
    :param objects_to_remove: List of object classes to remove from the virtual objects DataFrame.
    :type objects_to_remove: list[str]
    """
    os.makedirs(results_output_dir, exist_ok=True)

    results_list = []
    lock = threading.Lock()
    progress_counter = [0]
    total_homes = len(home_ids)

    print(
        f"Starting false negative analysis for {total_homes} homes (online prefix only)"
    )
    print(f"Output directory: {output_dir}")
    print(f"Results will be saved to: {results_output_dir}")
    print(f"Coordinate tolerance for deduplication: {coordinate_tolerance}")
    print(f"Skip existing knowledge entries: {skip_existing_knowledge}")
    print("-" * 60)

    print_progress(0, total_homes, 0, "Starting...")

    json_output_path = os.path.join(
        results_output_dir, "additional_knowledge_entries.json"
    )
    csv_output_path = os.path.join(
        results_output_dir, "additional_knowledge_entries.csv"
    )

    if prepare_additional_knowledge:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for home_id in home_ids:
                futures.append(
                    executor.submit(
                        process_home_false_negatives,
                        home_id=home_id,
                        output_dir=output_dir,
                        database_path=database_path,
                        results_list=results_list,
                        lock=lock,
                        progress_counter=progress_counter,
                        total_homes=total_homes,
                        local_data_dir=local_data_dir,
                        coordinate_tolerance=coordinate_tolerance,
                        objects_to_remove=objects_to_remove,
                    )
                )

            for future in as_completed(futures):
                try:
                    future.result()
                except (RuntimeError, KeyError, ValueError) as e:
                    traceback.print_exc()
                    print(f"\nThread failed with exception: {e}")

        results_list_sorted = sorted(results_list, key=lambda x: x["home_id"])

        try:
            save_knowledge_entries_to_json(results_list_sorted, json_output_path)
            save_knowledge_entries_to_csv(results_list_sorted, csv_output_path)
        except RuntimeError as e:
            traceback.print_exc()
            print(f"Error saving results: {e}")

        knowledge_entries = []
        for result in results_list_sorted:
            knowledge_entries.extend(result.get("knowledge_entries", []))
    else:
        try:
            with open(json_output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            knowledge_entries = data.get("knowledge_entries", [])
        except (OSError, ValueError, KeyError) as e:
            traceback.print_exc()
            print(f"Error loading knowledge entries from JSON: {e}")
            knowledge_entries = []

    if insert_to_database:
        print("\n" + "=" * 60)
        print("INSERTING KNOWLEDGE ENTRIES INTO DATABASE")
        print("=" * 60)

        # Group entries by home_id for insertion statistics
        entries_by_home = {}
        for entry in knowledge_entries:
            home_id = entry.get("home_id", 0)
            entries_by_home.setdefault(home_id, []).append(entry)

        insertion_stats = {
            "total_entries": len(knowledge_entries),
            "successful_insertions": 0,
            "failed_insertions": 0,
            "skipped_insertions": 0,
            "per_home_stats": [],
        }

        for home_id, entries in entries_by_home.items():
            home_stats = {
                "home_id": home_id,
                "total_entries": len(entries),
                "successful": 0,
                "failed": 0,
                "skipped": 0,
            }
            try:
                config = SystemConfig(
                    house_id=home_id,
                    dataset_base_path=database_path,
                    prefix="online",
                    qdrant_url="http://localhost:6333",
                    force_recreate_table=False,
                    local_data_dir="data",
                    debug_input_path=os.path.join("data", "input_debug.txt"),
                    debug_output_path=os.path.join("data", "output_debug.txt"),
                )
                system = InteractionSystem(config)

                success_list = []
                error_list = []
                lock = threading.Lock()

                print(f"\nInserting {len(entries)} entries for Home {home_id:02d}...")

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for entry in entries:
                        futures.append(
                            executor.submit(
                                insert_knowledge_entry_to_system,
                                system=system,
                                entry=entry,
                                lock=lock,
                                success_list=success_list,
                                error_list=error_list,
                                results_output_dir=results_output_dir,
                                skip_existing=skip_existing_knowledge,
                            )
                        )

                    for future in as_completed(futures):
                        try:
                            future.result()
                        except (RuntimeError, KeyError, ValueError) as e:
                            traceback.print_exc()
                            print(f"\nInsertion thread failed: {e}")

                skipped_count = sum(1 for s in success_list if s.get("skipped", False))
                actual_success = len(success_list) - skipped_count

                home_stats["successful"] = actual_success
                home_stats["failed"] = len(error_list)
                home_stats["skipped"] = skipped_count

                insertion_stats["successful_insertions"] += actual_success
                insertion_stats["failed_insertions"] += len(error_list)
                insertion_stats["skipped_insertions"] += skipped_count

                print(
                    f"Home {home_id:02d}: {actual_success} successful, "
                    f"{len(error_list)} failed, {skipped_count} skipped"
                )

            except (RuntimeError, KeyError, ValueError, AttributeError) as e:
                traceback.print_exc()
                home_stats["failed"] = len(entries)
                insertion_stats["failed_insertions"] += len(entries)
                print(f"Error initializing system for Home {home_id:02d}: {e}")

            insertion_stats["per_home_stats"].append(home_stats)

        insertion_stats_path = os.path.join(
            results_output_dir, "insertion_statistics.json"
        )
        try:
            with open(insertion_stats_path, "w", encoding="utf-8") as f:
                json.dump(insertion_stats, f, indent=4, ensure_ascii=False)
            print(f"\nInsertion statistics saved to: {insertion_stats_path}")
        except (OSError, TypeError, ValueError) as e:
            traceback.print_exc()
            print(f"Error saving insertion statistics: {e}")

    print("\n" + "=" * 60)
    print("FALSE NEGATIVE ANALYSIS COMPLETE")
    print("=" * 60)

    if prepare_additional_knowledge:
        total_fn = 0
        # total_unique_fn = 0
        homes_with_fn = 0
        homes_with_errors = 0
        for result in results_list:
            total_fn += result.get("false_negatives_count", 0)
            # total_unique_fn += result.get("enriched_false_negatives_count", 0)
            if result.get("false_negatives_count", 0) > 0:
                homes_with_fn += 1
            if result.get("error"):
                homes_with_errors += 1

        print(f"Total homes processed: {len(results_list)}")
        print(f"Homes with false negatives: {homes_with_fn}")
        print(f"Total false negatives: {total_fn}")
        # print(f"Total unique false negatives: {total_unique_fn}")
        print(f"Homes with errors: {homes_with_errors}")


if __name__ == "__main__":
    load_dotenv()
    DATABASE_PATH: str = THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET
    EVALUATION_OUTPUT_DIR: str = os.path.join(DATABASE_PATH, "evaluation_results")
    RESULTS_OUTPUT_DIR: str = os.path.join(
        DATABASE_PATH, "additional_knowledge_results"
    )
    LOCAL_DATA_DIR: str = "data"
    HOME_IDS: list[int] = list(range(1, 31))
    MAX_WORKERS: int = 100
    PREPARE_ADDITIONAL_KNOWLEDGE: bool = True
    INSERT_TO_DATABASE: bool = True
    SKIP_EXISTING_KNOWLEDGE: bool = True
    COORDINATE_TOLERANCE: float = 0.1
    OBJECTS_TO_REMOVE: list[str] = ["wall", "floor", "ceiling", "door", "decoration"]

    try:
        run_false_negative_analysis(
            output_dir=EVALUATION_OUTPUT_DIR,
            home_ids=HOME_IDS,
            max_workers=MAX_WORKERS,
            results_output_dir=RESULTS_OUTPUT_DIR,
            database_path=DATABASE_PATH,
            insert_to_database=INSERT_TO_DATABASE,
            skip_existing_knowledge=SKIP_EXISTING_KNOWLEDGE,
            local_data_dir=LOCAL_DATA_DIR,
            coordinate_tolerance=COORDINATE_TOLERANCE,
            prepare_additional_knowledge=PREPARE_ADDITIONAL_KNOWLEDGE,
            objects_to_remove=OBJECTS_TO_REMOVE,
        )
    except (RuntimeError, KeyboardInterrupt) as e:
        traceback.print_exc()
        if isinstance(e, KeyboardInterrupt):
            print("\nProgram interrupted by user. Exiting gracefully.")

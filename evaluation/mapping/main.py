from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from agno.models.openrouter import OpenRouter
from scipy.spatial import cKDTree
from dotenv import load_dotenv
from agno.agent import Agent
from openai import OpenAI
from typing import Union
import multiprocessing
import numpy as np
import traceback
import threading
import yaml
import csv
import cv2
import os

from labs.utils import (
    read_virtual_objects,
    unity_to_ros_coordinates,
    world_to_map,
    load_pkl_gz_result,
)

from evaluation.prompts import OBJECT_COMPARISON_PROMPT


def virtual_objects_csv_path(database_path: str, home_id: int) -> str:
    """
    Constructs the file path for the VirtualObjects.csv file.

    :param database_path: The base path to the database.
    :type database_path: str
    :param home_id: The identifier of the home.
    :type home_id: int
    :raises ValueError: If database_path is empty or home_id is negative.
    :return: The absolute path to the CSV file.
    :rtype: str
    """
    if not database_path or home_id < 0:
        raise ValueError("Invalid database_path or home_id for CSV path construction.")
    return os.path.join(database_path, f"Home{home_id:02d}", "VirtualObjects.csv")


def map_image_path(database_path: str, home_id: int) -> str:
    """
    Constructs the file path for the generated map image.

    :param database_path: The base path to the database.
    :type database_path: str
    :param home_id: The identifier of the home.
    :type home_id: int
    :raises ValueError: If database_path is empty or home_id is negative.
    :return: The absolute path to the map image.
    :rtype: str
    """
    if not database_path or home_id < 0:
        raise ValueError(
            "Invalid database_path or home_id for map image path construction."
        )
    return os.path.join(database_path, f"Home{home_id:02d}", "generated_map.png")


def read_map_properties_yaml(map_path: str) -> dict:
    """
    Reads the map properties from a YAML file associated with the map image.

    :param map_path: The path to the map image file.
    :type map_path: str
    :raises FileNotFoundError: If the YAML file does not exist.
    :raises yaml.YAMLError: If the YAML file cannot be parsed.
    :return: A dictionary containing map properties.
    :rtype: dict
    """
    map_properties_path = map_path.replace(".png", ".yaml")
    try:
        with open(map_properties_path, "r") as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Error reading map properties YAML: {e}")


def match_virtual_to_processed(
    virt_idx: int,
    processed_indices: list[int],
    virtual_obj: dict,
    parsed_processed_objects_props: list[dict],
    llm_agent: Union[Agent, callable],
    tp: list,
    fn: list,
    matched_processed_indices: set[int],
    progress_counter: list[int],
    total: int,
    lock: threading.Lock,
    min_votes: int,
    object_votes: list,
    to_replace_object_classes: dict[str, str],
) -> None:
    """
    Matches a virtual object to processed objects using the LLM agent.

    :param virt_idx: Index of the virtual object.
    :type virt_idx: int
    :param processed_indices: List of indices of processed objects within distance.
    :type processed_indices: list[int]
    :param virtual_obj: Properties of the virtual object.
    :type virtual_obj: dict
    :param parsed_processed_objects_props: List of processed objects' properties.
    :type parsed_processed_objects_props: list[dict]
    :param llm_agent: The language model agent for object comparison.
    :type llm_agent: Union[Agent, callable]
    :param tp: List to store true positive matches.
    :type tp: list
    :param fn: List to store false negatives.
    :type fn: list
    :param matched_processed_indices: Set of processed indices already claimed.
    :type matched_processed_indices: set[int]
    :param progress_counter: List containing the current progress count.
    :type progress_counter: list[int]
    :param total: Total number of virtual objects.
    :type total: int
    :param lock: Threading lock for synchronization.
    :type lock: threading.Lock
    :param min_votes: Minimum number of votes required for decision.
    :type min_votes: int
    :param object_votes: List to store voting results for each object.
    :type object_votes: list
    :param to_replace_object_classes: Mapping of object classes to replace.
    :type to_replace_object_classes: dict[str, str]
    :raises RuntimeError: If the LLM agent fails to run.
    :return: None
    :rtype: None
    """
    found_match: bool = False

    if processed_indices:
        for proc_idx in processed_indices:
            with lock:
                if proc_idx in matched_processed_indices:
                    continue
            processed_obj = parsed_processed_objects_props[proc_idx]
            caption = (
                processed_obj.get("consolidated_caption", '{"Unknown}')
                .split("}")[-2]
                .split('"')[-2]
            )
            prompt_message: str = (
                f"\n<PROCESSED_OBJECT> Class: {processed_obj.get('class_name', 'Unknown')}, "
                f"Caption: {caption} </PROCESSED_OBJECT>\n"
                f"<VIRTUAL_OBJECT> Type: {to_replace_object_classes.get(virtual_obj.get('type').lower(), virtual_obj.get('type'))} </VIRTUAL_OBJECT>"
            )
            results: list[str] = []
            try:
                with ThreadPoolExecutor(max_workers=min_votes) as local_executor:
                    futures = [
                        local_executor.submit(
                            lambda: (
                                llm_agent.run(message=prompt_message)
                                if isinstance(llm_agent, Agent)
                                else llm_agent(prompt_message)
                            )
                        )
                        for _ in range(min_votes)
                    ]
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except (AttributeError, KeyError) as e:
                            traceback.print_exc()
                            raise RuntimeError(f"Failed to run LLM agent: {e}")
            except (RuntimeError, OSError) as e:
                traceback.print_exc()
                raise RuntimeError(f"Parallel LLM execution failed: {e}")

            true_count: int = sum(
                1
                for r in results
                if isinstance(r.content, str) and r.content.strip().lower() == "true"
            )
            false_count: int = min_votes - true_count

            if true_count >= (min_votes // 2 + 1):
                with lock:
                    if proc_idx not in matched_processed_indices:
                        matched_processed_indices.add(proc_idx)
                        tp.append((virtual_obj, processed_obj))
                        object_votes.append(
                            {
                                "virtual_type": virtual_obj.get("type", "Unknown"),
                                "processed_class": processed_obj.get(
                                    "class_name", "Unknown"
                                ),
                                "true_votes": true_count,
                                "false_votes": false_count,
                                "matched": True,
                            }
                        )
                        found_match = True
                if found_match:
                    break

    if not found_match:
        with lock:
            fn.append(virtual_obj)
            object_votes.append(
                {
                    "virtual_type": virtual_obj.get("type", "Unknown"),
                    "processed_class": "None",
                    "true_votes": 0,
                    "false_votes": 0,
                    "matched": False,
                }
            )
    with lock:
        progress_counter[0] += 1
        print_progress_bar(progress_counter[0], total)


def print_progress_bar(current: int, total: int) -> None:
    """
    Prints a progress bar to the console.

    :param current: Current progress count.
    :type current: int
    :param total: Total count for completion.
    :type total: int
    :return: None
    :rtype: None
    """
    bar_length: int = 40
    if total == 0:
        return
    filled_length: int = int(bar_length * current // total)
    bar: str = "=" * filled_length + "-" * (bar_length - filled_length)
    percent: float = (current / total) * 100
    print(f"\rProgress: |{bar}| {percent:.1f}% ({current}/{total})", end="", flush=True)
    if current == total:
        print()


def is_already_processed(output_dir: str, home_id: int, processing_type: str) -> bool:
    """
    Checks if a home and processing type combination has already been processed.

    :param output_dir: Directory where output CSV files are saved.
    :type output_dir: str
    :param home_id: The identifier of the home.
    :type home_id: int
    :param processing_type: The type of processing.
    :type processing_type: str
    :return: True if already processed, False otherwise.
    :rtype: bool
    """
    csv_path: str = os.path.join(
        output_dir, f"Home{home_id:02d}_{processing_type}_objects.csv"
    )
    return os.path.exists(csv_path)


def evaluate_home(
    home_id: int,
    database_path: str,
    processing_type: str,
    to_remove_object_classes: list[str],
    to_replace_object_classes: dict[str, str],
    llm_agent: Union[Agent, callable],
    distance_threshold_meters: float,
    min_votes: int,
    output_dir: str,
) -> dict:
    """
    Evaluates mapping for a single home and processing type.

    :param home_id: The identifier of the home.
    :type home_id: int
    :param database_path: The base path to the database.
    :type database_path: str
    :param processing_type: The type of processing.
    :type processing_type: str
    :param to_remove_object_classes: Classes of objects to remove.
    :type to_remove_object_classes: list[str]
    :param to_replace_object_classes: Mapping of object classes to replace.
    :type to_replace_object_classes: dict[str, str]
    :param llm_agent: The language model agent for object comparison.
    :type llm_agent: Union[Agent, callable]
    :param distance_threshold_meters: Distance threshold in meters.
    :type distance_threshold_meters: float
    :param min_votes: Minimum number of votes required for decision.
    :type min_votes: int
    :param output_dir: Directory to save output CSV files.
    :type output_dir: str
    :raises RuntimeError: If data loading or processing fails.
    :return: Dictionary containing evaluation metrics.
    :rtype: dict
    """
    try:
        virtual_objects = read_virtual_objects(
            virtual_objects_csv_path(database_path, home_id)
        )
        map_path: str = map_image_path(database_path, home_id)
        map_image = cv2.imread(map_path)
        map_properties: dict = read_map_properties_yaml(map_path)
        processed_data = load_pkl_gz_result(
            os.path.join(
                database_path,
                "outputs",
                f"Home{home_id:02d}",
                "Wandering",
                "exps",
                f"{processing_type}_house_{home_id}_map",
                f"pcd_{processing_type}_house_{home_id}_map.pkl.gz",
            )
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        traceback.print_exc()
        raise RuntimeError(
            f"Error loading data for Home {home_id}, Type {processing_type}: {e}"
        )

    processed_objects: list = processed_data.get("objects", [])

    try:
        filtered_processed_objects: list = []
        for obj in processed_objects:
            obj_dict: dict = (
                obj if isinstance(obj, dict) else getattr(obj, "__dict__", {})
            )
            class_name: str | None = obj_dict.get("class_name")
            if not class_name:
                continue
            if any(
                to_remove_class in class_name.lower()
                for to_remove_class in to_remove_object_classes
            ):
                continue
            filtered_processed_objects.append(obj)
        processed_objects = filtered_processed_objects
    except (KeyError, AttributeError, TypeError) as e:
        traceback.print_exc()
        raise RuntimeError(
            f"Filtering error for processed objects in Home {home_id}: {e}"
        )

    try:
        mask = (
            ~virtual_objects["type"]
            .astype(str)
            .apply(
                lambda t: any(
                    to_remove_class in t.lower()
                    for to_remove_class in to_remove_object_classes
                )
            )
        )
        virtual_objects = virtual_objects.loc[mask].reset_index(drop=True)
        # virtual_objects["type"] = (
        #     virtual_objects["type"]
        #     .astype(str)
        #     .apply(
        #         lambda t: (
        #             to_replace_object_classes[t.lower()]
        #             if t.lower() in to_replace_object_classes
        #             else t
        #         )
        #     )
        # )
    except (KeyError, AttributeError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Filtering error for Home {home_id}: {e}")

    print(
        f"[Home {home_id:02d} - {processing_type}] PROCESSED: {len(processed_objects)} | VIRTUAL: {len(virtual_objects)}"
    )

    origin: list | tuple = map_properties["origin"]
    resolution: float = map_properties["resolution"]
    image_height: int = map_image.shape[0]

    parsed_virtual_objects_coords: list = []
    parsed_virtual_objects_props: list = []

    for _, row in virtual_objects.iterrows():
        try:
            ros_x, ros_y, _, _ = unity_to_ros_coordinates(
                row["globalPosition"], row["rotation"], ""
            )
            pixel_x, pixel_y = world_to_map(
                ros_x, ros_y, origin, resolution, image_height
            )
            parsed_virtual_objects_coords.append((pixel_x, pixel_y))
            parsed_virtual_objects_props.append(row.to_dict())
        except (KeyError, TypeError, ValueError):
            traceback.print_exc()
            continue

    parsed_processed_objects_coords: list = []
    parsed_processed_objects_props: list = []

    for obj in processed_objects:
        try:
            obj_dict: dict = (
                obj if isinstance(obj, dict) else getattr(obj, "__dict__", {})
            )
            class_name: str | None = obj_dict.get("class_name")
            if not class_name:
                continue
            centroid = None
            if (
                "pcd_np" in obj_dict
                and isinstance(obj_dict["pcd_np"], np.ndarray)
                and obj_dict["pcd_np"].size > 0
            ):
                centroid = np.mean(obj_dict["pcd_np"], axis=0)
            elif "bbox_np" in obj_dict and isinstance(obj_dict["bbox_np"], np.ndarray):
                centroid = np.mean(obj_dict["bbox_np"], axis=0)
            if centroid is not None and len(centroid) >= 2:
                ros_x, ros_y = float(centroid[2]), -float(centroid[0])
                pixel_x, pixel_y = world_to_map(
                    ros_x, ros_y, origin, resolution, image_height
                )
                parsed_processed_objects_coords.append((pixel_x, pixel_y))
                parsed_processed_objects_props.append(obj_dict)
        except (KeyError, TypeError, ValueError):
            traceback.print_exc()
            continue

    distance_threshold: int = int(distance_threshold_meters / resolution)
    parsed_processed_objects_coords_np = np.array(parsed_processed_objects_coords)
    parsed_virtual_objects_coords_np = np.array(parsed_virtual_objects_coords)

    if len(parsed_processed_objects_coords_np) == 0:
        print(f"[Home {home_id:02d} - {processing_type}] No processed objects found.")
        return {
            "home_id": home_id,
            "processing_type": processing_type,
            "tp": 0,
            "fn": len(parsed_virtual_objects_props),
            "fp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    tree = cKDTree(parsed_processed_objects_coords_np)
    neighbors = tree.query_ball_point(
        parsed_virtual_objects_coords_np, r=distance_threshold
    )

    tp: list = []
    fn: list = []
    matched_processed_indices: set[int] = set()
    progress_counter: list[int] = [0]
    total_virtual_objects: int = len(parsed_virtual_objects_props)
    lock = threading.Lock()
    object_votes: list = []

    print_progress_bar(0, total_virtual_objects)

    cpu_count: int = multiprocessing.cpu_count()
    max_workers: int = max(1, (cpu_count - 1) // min_votes)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for virt_idx, processed_indices in enumerate(neighbors):
            virtual_obj = parsed_virtual_objects_props[virt_idx]
            futures.append(
                executor.submit(
                    match_virtual_to_processed,
                    virt_idx=virt_idx,
                    processed_indices=processed_indices,
                    virtual_obj=virtual_obj,
                    parsed_processed_objects_props=parsed_processed_objects_props,
                    llm_agent=llm_agent,
                    tp=tp,
                    fn=fn,
                    matched_processed_indices=matched_processed_indices,
                    progress_counter=progress_counter,
                    total=total_virtual_objects,
                    lock=lock,
                    min_votes=min_votes,
                    object_votes=object_votes,
                    to_replace_object_classes=to_replace_object_classes,
                )
            )
        wait(futures)

    total_processed_count: int = len(parsed_processed_objects_props)
    matched_count: int = len(matched_processed_indices)
    fp_count: int = total_processed_count - matched_count

    print(
        f"[Home {home_id:02d} - {processing_type}] TP: {len(tp)} | FN: {len(fn)} | FP: {fp_count}"
    )

    precision: float = (
        len(tp) / (len(tp) + fp_count) if (len(tp) + fp_count) > 0 else 0.0
    )
    recall: float = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
    f1_score: float = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_path: str = os.path.join(
        output_dir, f"Home{home_id:02d}_{processing_type}_objects.csv"
    )
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "virtual_type",
                "processed_class",
                "true_votes",
                "false_votes",
                "matched",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(object_votes)
    except (OSError, IOError) as e:
        traceback.print_exc()
        raise RuntimeError(
            f"Error writing CSV for Home {home_id}, Type {processing_type}: {e}"
        )

    return {
        "home_id": home_id,
        "processing_type": processing_type,
        "tp": len(tp),
        "fn": len(fn),
        "fp": fp_count,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def run_full_evaluation(
    database_path: str,
    processing_types: list[str],
    home_ids: list[int],
    to_remove_object_classes: list[str],
    to_replace_object_classes: dict[str, str],
    llm_agent: Union[Agent, callable],
    distance_threshold_meters: float,
    min_votes: int,
    output_dir: str,
) -> None:
    """
    Runs evaluation for all homes and processing types, then saves summary CSV.

    :param database_path: The base path to the database.
    :type database_path: str
    :param processing_types: List of processing types to evaluate.
    :type processing_types: list[str]
    :param home_ids: List of home identifiers to evaluate.
    :type home_ids: list[int]
    :param to_remove_object_classes: Classes of objects to remove.
    :type to_remove_object_classes: list[str]
    :param to_replace_object_classes: Mapping of object classes to replace.
    :type to_replace_object_classes: dict[str, str]
    :param llm_agent: The language model agent for object comparison.
    :type llm_agent: Union[Agent, callable]
    :param distance_threshold_meters: Distance threshold in meters.
    :type distance_threshold_meters: float
    :param min_votes: Minimum number of votes required for decision.
    :type min_votes: int
    :param output_dir: Directory to save output CSV files.
    :type output_dir: str
    :raises RuntimeError: If evaluation or CSV writing fails.
    :return: None
    :rtype: None
    """
    all_results: list[dict] = []

    for home_id in home_ids:
        for processing_type in processing_types:
            if is_already_processed(output_dir, home_id, processing_type):
                print(
                    f"[Home {home_id:02d} - {processing_type}] Already processed, skipping..."
                )
                continue
            try:
                result = evaluate_home(
                    home_id=home_id,
                    database_path=database_path,
                    processing_type=processing_type,
                    to_remove_object_classes=to_remove_object_classes,
                    to_replace_object_classes=to_replace_object_classes,
                    llm_agent=llm_agent,
                    distance_threshold_meters=distance_threshold_meters,
                    min_votes=min_votes,
                    output_dir=output_dir,
                )
                all_results.append(result)
            except RuntimeError:
                traceback.print_exc()
                continue

    summary_csv_path: str = os.path.join(output_dir, "summary_all_homes.csv")
    try:
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "home_id",
                "processing_type",
                "tp",
                "fn",
                "fp",
                "precision",
                "recall",
                "f1_score",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSummary saved to: {summary_csv_path}")
    except (OSError, IOError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Error writing summary CSV: {e}")


if __name__ == "__main__":
    DATABASE_PATH: str = os.path.join(
        "D:", "Documentos", "Datasets", "Robot@VirtualHomeLarge"
    )
    PROCESSING_TYPES: list[str] = ["original", "improved", "offline", "online"]
    HOME_IDS: list[int] = list(range(1, 31))
    TO_REMOVE_OBJECT_CLASSES: list[str] = [
        "wall",
        "floor",
        "ceiling",
        "door",
        "decoration",
    ]
    TO_REPLACE_OBJECT_CLASSES: dict[str, str] = {
        "standard": "shelf or cabinet or standard",
        "kitchenfurniture": "shelf or cabinet or furniture",
        "drying": "towel drying rack",
        "burner": "stove or cooktop or burner",
    }
    DISTANCE_THRESHOLD_METERS: float = 1
    MIN_VOTES: int = 1
    OUTPUT_DIR: str = os.path.join(DATABASE_PATH, "evaluation_results")
    OPENROUTER_MODEL_ID: str = "openai/gpt-oss-120b"
    OPENROUTER_TEMPERATURE: float = 0.0
    OPENROUTER_TOP_P: float = 0.1
    OPENROUTER_MAX_TOKENS: int = 16000

    load_dotenv()

    # llm_agent = Agent(
    #     model=OpenRouter(
    #         id=OPENROUTER_MODEL_ID,
    #         api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    #         temperature=OPENROUTER_TEMPERATURE,
    #         top_p=OPENROUTER_TOP_P,
    #         max_tokens=OPENROUTER_MAX_TOKENS,
    #     ),
    #     system_message=OBJECT_COMPARISON_PROMPT,
    # )

    openai_client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
    )

    def llm_agent(message: str) -> str:
        return (
            openai_client.chat.completions.create(
                model=OPENROUTER_MODEL_ID,
                messages=[
                    {"role": "system", "content": OBJECT_COMPARISON_PROMPT},
                    {"role": "user", "content": message},
                ],
                temperature=OPENROUTER_TEMPERATURE,
                top_p=OPENROUTER_TOP_P,
                max_tokens=OPENROUTER_MAX_TOKENS,
            )
            .choices[0]
            .message
        )

    try:
        run_full_evaluation(
            database_path=DATABASE_PATH,
            processing_types=PROCESSING_TYPES,
            home_ids=HOME_IDS,
            to_remove_object_classes=TO_REMOVE_OBJECT_CLASSES,
            to_replace_object_classes=TO_REPLACE_OBJECT_CLASSES,
            llm_agent=llm_agent,
            distance_threshold_meters=DISTANCE_THRESHOLD_METERS,
            min_votes=MIN_VOTES,
            output_dir=OUTPUT_DIR,
        )
    except (RuntimeError, KeyboardInterrupt) as e:
        traceback.print_exc()
        if isinstance(e, KeyboardInterrupt):
            print("\nProgram interrupted by user. Exiting gracefully.")

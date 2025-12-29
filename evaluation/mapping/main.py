from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import cKDTree
from dotenv import load_dotenv
from agno.agent import Agent
from openai import OpenAI
from typing import Union
import numpy as np
import traceback
import threading
import yaml
import json
import csv
import time
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


def map_image_path(database_path: str, home_id: int) -> str:
    """
    Constructs the path to the generated map image for a given home.

    :param database_path: Base path to the database.
    :type database_path: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :raises ValueError: If database_path is empty or home_id is negative.
    :return: Path to the generated map image.
    :rtype: str
    """
    if not database_path or home_id < 0:
        raise ValueError(
            "Invalid database_path or home_id for map image path construction."
        )
    return os.path.join(database_path, f"Home{home_id:02d}", "generated_map.png")


def read_map_properties_yaml(map_path: str) -> dict:
    """
    Reads the YAML file containing map properties.

    :param map_path: Path to the map image file.
    :type map_path: str
    :raises RuntimeError: If the YAML file cannot be read or parsed.
    :return: Dictionary with map properties.
    :rtype: dict
    """
    map_properties_path = map_path.replace(".png", ".yaml")
    try:
        with open(map_properties_path, "r") as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Error reading map properties YAML: {e}")


def get_paths(output_dir: str, home_id: int, processing_type: str) -> dict[str, str]:
    """
    Generates output file paths for a given home and processing type.

    :param output_dir: Directory for output files.
    :type output_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param processing_type: Type of processing.
    :type processing_type: str
    :return: Dictionary with paths for objects CSV, detailed CSV, and metrics JSON.
    :rtype: dict[str, str]
    """
    base_name = f"Home{home_id:02d}_{processing_type}"
    return {
        "objects_csv": os.path.join(output_dir, f"{base_name}_objects.csv"),
        "detailed_csv": os.path.join(
            output_dir, f"{base_name}_detailed_comparisons.csv"
        ),
        "metrics_json": os.path.join(output_dir, f"{base_name}_metrics.json"),
    }


def is_already_processed(output_dir: str, home_id: int, processing_type: str) -> bool:
    """
    Checks if the metrics JSON file already exists for a given home and processing type.

    :param output_dir: Directory for output files.
    :type output_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param processing_type: Type of processing.
    :type processing_type: str
    :return: True if metrics JSON exists, False otherwise.
    :rtype: bool
    """
    paths = get_paths(output_dir, home_id, processing_type)
    return os.path.exists(paths["metrics_json"])


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
    all_comparisons_log: list,
    to_replace_object_classes: dict[str, str],
) -> None:
    """
    Performs sequential voting comparison with early stopping between a virtual object and processed objects.

    :param virt_idx: Index of the virtual object.
    :type virt_idx: int
    :param processed_indices: Indices of candidate processed objects.
    :type processed_indices: list[int]
    :param virtual_obj: Virtual object properties.
    :type virtual_obj: dict
    :param parsed_processed_objects_props: List of processed object properties.
    :type parsed_processed_objects_props: list[dict]
    :param llm_agent: LLM agent or callable for comparison.
    :type llm_agent: Union[Agent, callable]
    :param tp: List to store true positives.
    :type tp: list
    :param fn: List to store false negatives.
    :type fn: list
    :param matched_processed_indices: Set of already matched processed indices.
    :type matched_processed_indices: set[int]
    :param progress_counter: List containing progress counter.
    :type progress_counter: list[int]
    :param total: Total number of virtual objects.
    :type total: int
    :param lock: Threading lock for synchronization.
    :type lock: threading.Lock
    :param min_votes: Minimum number of votes for majority.
    :type min_votes: int
    :param object_votes: List to store voting results.
    :type object_votes: list
    :param all_comparisons_log: List to store detailed comparison logs.
    :type all_comparisons_log: list
    :param to_replace_object_classes: Mapping for object class replacements.
    :type to_replace_object_classes: dict[str, str]
    :return: None
    :rtype: None
    """
    try:
        found_match: bool = False
        majority_threshold: int = (min_votes // 2) + 1

        virtual_type_processed = to_replace_object_classes.get(
            virtual_obj.get("type").lower(), virtual_obj.get("type")
        )

        virtual_obj_coordinates = virtual_obj.get("globalPosition", [0, 0, 0])

        if processed_indices:
            for proc_idx in processed_indices:
                with lock:
                    if proc_idx in matched_processed_indices:
                        continue

                processed_obj = parsed_processed_objects_props[proc_idx]
                raw_caption = processed_obj.get("consolidated_caption", '{"Unknown}')
                try:
                    caption = raw_caption.split("}")[-2].split('"')[-2]
                except IndexError:
                    caption = raw_caption

                prompt_message: str = (
                    f"\n<PROCESSED_OBJECT> Class: {processed_obj.get('class_name', 'Unknown')}, "
                    f"Caption: {caption} </PROCESSED_OBJECT>\n"
                    f"<VIRTUAL_OBJECT> Type: {virtual_type_processed} </VIRTUAL_OBJECT>"
                )

                current_true_votes: int = 0
                current_false_votes: int = 0
                collected_reasonings: list[str] = []

                for _ in range(min_votes):
                    try:
                        response_content = None
                        reasoning_content = "N/A"

                        if isinstance(llm_agent, Agent):
                            resp = llm_agent.run(message=prompt_message)
                            response_content = resp.content
                            reasoning_content = getattr(resp, "reasoning", "N/A")
                        else:
                            resp = llm_agent(prompt_message)
                            response_content = resp.content
                            reasoning_content = (
                                getattr(resp, "reasoning", None)
                                or getattr(resp, "reasoning_content", None)
                                or "N/A"
                            )

                        collected_reasonings.append(
                            str(reasoning_content) if reasoning_content else "N/A"
                        )

                        if (
                            isinstance(response_content, str)
                            and response_content.strip().lower() == "true"
                        ):
                            current_true_votes += 1
                        else:
                            current_false_votes += 1

                        if current_true_votes >= majority_threshold:
                            break
                        if current_false_votes >= majority_threshold:
                            break

                    except (AttributeError, TypeError, ValueError) as e:
                        traceback.print_exc()
                        collected_reasonings.append(f"ERROR: {str(e)}")
                        current_false_votes += 1
                        if current_false_votes >= majority_threshold:
                            break

                is_match: bool = current_true_votes >= majority_threshold
                reasoning_trace_str = " ||| ".join(collected_reasonings)

                with lock:
                    all_comparisons_log.append(
                        {
                            "virtual_idx": virt_idx,
                            "virtual_type": virtual_obj.get("type", "Unknown"),
                            "processed_idx": proc_idx,
                            "processed_class": processed_obj.get(
                                "class_name", "Unknown"
                            ),
                            "processed_caption": caption,
                            "prompt_message": prompt_message.replace("\n", " "),
                            "reasoning_trace": reasoning_trace_str.replace("\n", " "),
                            "true_votes": current_true_votes,
                            "false_votes": current_false_votes,
                            "is_match_attempt": is_match,
                        }
                    )

                if is_match:
                    with lock:
                        if proc_idx not in matched_processed_indices:
                            matched_processed_indices.add(proc_idx)
                            tp.append((virtual_obj, processed_obj))
                            object_votes.append(
                                {
                                    "virtual_idx": virt_idx,
                                    "virtual_type": virtual_obj.get("type", "Unknown"),
                                    "processed_class": processed_obj.get(
                                        "class_name", "Unknown"
                                    ),
                                    "true_votes": current_true_votes,
                                    "false_votes": current_false_votes,
                                    "matched": True,
                                    "x_coordinate": virtual_obj_coordinates[0],
                                    "y_coordinate": virtual_obj_coordinates[1],
                                    "z_coordinate": virtual_obj_coordinates[2],
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
                        "virtual_idx": virt_idx,
                        "virtual_type": virtual_obj.get("type", "Unknown"),
                        "processed_class": "None",
                        "true_votes": 0,
                        "false_votes": 0,
                        "matched": False,
                        "x_coordinate": virtual_obj_coordinates[0],
                        "y_coordinate": virtual_obj_coordinates[1],
                        "z_coordinate": virtual_obj_coordinates[2],
                    }
                )

    except (KeyError, AttributeError, TypeError, ValueError) as e:
        traceback.print_exc()
        with lock:
            fn.append(virtual_obj)
        raise RuntimeError(f"CRITICAL ERROR processing Virtual Object {virt_idx}: {e}")

    finally:
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
    Evaluates a single home by matching virtual and processed objects, computing metrics, and saving results.

    :param home_id: Identifier for the home.
    :type home_id: int
    :param database_path: Base path to the database.
    :type database_path: str
    :param processing_type: Type of processing.
    :type processing_type: str
    :param to_remove_object_classes: List of object classes to ignore.
    :type to_remove_object_classes: list[str]
    :param to_replace_object_classes: Mapping for object class replacements.
    :type to_replace_object_classes: dict[str, str]
    :param llm_agent: LLM agent or callable for comparison.
    :type llm_agent: Union[Agent, callable]
    :param distance_threshold_meters: Distance threshold in meters for matching.
    :type distance_threshold_meters: float
    :param min_votes: Minimum number of votes for majority.
    :type min_votes: int
    :param output_dir: Directory for output files.
    :type output_dir: str
    :raises RuntimeError: If data loading, filtering, or output writing fails.
    :return: Dictionary with evaluation metrics.
    :rtype: dict
    """
    try:
        virtual_objects = read_virtual_objects(
            virtual_objects_csv_path(database_path, home_id)
        )
        map_path = map_image_path(database_path, home_id)
        map_image = cv2.imread(map_path)
        map_properties = read_map_properties_yaml(map_path)
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
        raise RuntimeError(f"Error loading data for Home {home_id}: {e}")

    processed_objects = processed_data.get("objects", [])

    try:
        filtered = []
        for obj in processed_objects:
            d = obj if isinstance(obj, dict) else getattr(obj, "__dict__", {})
            cls = d.get("class_name")
            if cls and not any(r in cls.lower() for r in to_remove_object_classes):
                filtered.append(obj)
        processed_objects = filtered
    except (KeyError, AttributeError, TypeError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Filtering error: {e}")

    try:
        mask = (
            ~virtual_objects["type"]
            .astype(str)
            .apply(lambda t: any(r in t.lower() for r in to_remove_object_classes))
        )
        virtual_objects = virtual_objects.loc[mask].reset_index(drop=True)
    except (KeyError, AttributeError, TypeError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Virtual filtering error: {e}")

    print(
        f"[Home {home_id:02d} - {processing_type}] PROCESSED: {len(processed_objects)} | VIRTUAL: {len(virtual_objects)}"
    )

    origin = map_properties["origin"]
    resolution = map_properties["resolution"]
    image_height = map_image.shape[0]

    parsed_virtual_objects_coords = []
    parsed_virtual_objects_props = []
    for _, row in virtual_objects.iterrows():
        try:
            rx, ry, _, _ = unity_to_ros_coordinates(
                row["globalPosition"], row["rotation"], ""
            )
            px, py = world_to_map(rx, ry, origin, resolution, image_height)
            parsed_virtual_objects_coords.append((px, py))
            parsed_virtual_objects_props.append(row.to_dict())
        except (KeyError, AttributeError, TypeError, ValueError):
            traceback.print_exc()
            continue

    parsed_processed_objects_coords = []
    parsed_processed_objects_props = []
    for obj in processed_objects:
        try:
            d = obj if isinstance(obj, dict) else getattr(obj, "__dict__", {})
            if not d.get("class_name"):
                continue
            cent = None
            if (
                "pcd_np" in d
                and isinstance(d["pcd_np"], np.ndarray)
                and d["pcd_np"].size > 0
            ):
                cent = np.mean(d["pcd_np"], axis=0)
            elif "bbox_np" in d and isinstance(d["bbox_np"], np.ndarray):
                cent = np.mean(d["bbox_np"], axis=0)

            if cent is not None and len(cent) >= 2:
                rx, ry = float(cent[2]), -float(cent[0])
                px, py = world_to_map(rx, ry, origin, resolution, image_height)
                parsed_processed_objects_coords.append((px, py))
                parsed_processed_objects_props.append(d)
        except (KeyError, AttributeError, TypeError, ValueError):
            traceback.print_exc()
            continue

    distance_threshold = int(distance_threshold_meters / resolution)
    parsed_processed_objects_coords_np = np.array(parsed_processed_objects_coords)
    parsed_virtual_objects_coords_np = np.array(parsed_virtual_objects_coords)

    paths = get_paths(output_dir, home_id, processing_type)
    os.makedirs(output_dir, exist_ok=True)

    if len(parsed_processed_objects_coords_np) == 0:
        print(f"[Home {home_id:02d} - {processing_type}] No processed objects found.")
        metrics = {
            "home_id": home_id,
            "processing_type": processing_type,
            "tp": 0,
            "fn": len(parsed_virtual_objects_props),
            "fp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }
        with open(paths["metrics_json"], "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        return metrics

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
    all_comparisons_log: list = []

    print_progress_bar(0, total_virtual_objects)

    max_workers = 100

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
                    all_comparisons_log=all_comparisons_log,
                    to_replace_object_classes=to_replace_object_classes,
                )
            )
        for future in as_completed(futures):
            try:
                future.result()
            except (RuntimeError, KeyError, AttributeError, TypeError, ValueError) as e:
                traceback.print_exc()
                print(f"Thread failed with exception: {e}")

    total_processed_count = len(parsed_processed_objects_props)
    matched_count = len(matched_processed_indices)
    fp_count = total_processed_count - matched_count

    print(
        f"\n[Home {home_id:02d} - {processing_type}] TP: {len(tp)} | FN: {len(fn)} | FP: {fp_count}"
    )

    precision = len(tp) / (len(tp) + fp_count) if (len(tp) + fp_count) > 0 else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics = {
        "home_id": home_id,
        "processing_type": processing_type,
        "tp": len(tp),
        "fn": len(fn),
        "fp": fp_count,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

    try:
        with open(paths["objects_csv"], "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "virtual_idx",
                "virtual_type",
                "processed_class",
                "true_votes",
                "false_votes",
                "matched",
                "x_coordinate",
                "y_coordinate",
                "z_coordinate",
            ]
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL
            )
            writer.writeheader()
            writer.writerows(object_votes)

        with open(paths["detailed_csv"], "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "virtual_idx",
                "virtual_type",
                "processed_idx",
                "processed_class",
                "processed_caption",
                "prompt_message",
                "reasoning_trace",
                "true_votes",
                "false_votes",
                "is_match_attempt",
            ]
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL
            )
            writer.writeheader()
            writer.writerows(all_comparisons_log)

        with open(paths["metrics_json"], "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

    except (OSError, KeyError, AttributeError, TypeError, ValueError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Error writing output files: {e}")

    return metrics


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
    Runs the full evaluation for all specified homes and processing types.

    :param database_path: Base path to the database.
    :type database_path: str
    :param processing_types: List of processing types.
    :type processing_types: list[str]
    :param home_ids: List of home identifiers.
    :type home_ids: list[int]
    :param to_remove_object_classes: List of object classes to ignore.
    :type to_remove_object_classes: list[str]
    :param to_replace_object_classes: Mapping for object class replacements.
    :type to_replace_object_classes: dict[str, str]
    :param llm_agent: LLM agent or callable for comparison.
    :type llm_agent: Union[Agent, callable]
    :param distance_threshold_meters: Distance threshold in meters for matching.
    :type distance_threshold_meters: float
    :param min_votes: Minimum number of votes for majority.
    :type min_votes: int
    :param output_dir: Directory for output files.
    :type output_dir: str
    :return: None
    :rtype: None
    """
    all_results = []

    for home_id in home_ids:
        for processing_type in processing_types:
            paths = get_paths(output_dir, home_id, processing_type)

            if is_already_processed(output_dir, home_id, processing_type):
                print(
                    f"[Home {home_id:02d} - {processing_type}] Already processed. Loading metrics..."
                )
                try:
                    with open(paths["metrics_json"], "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                        all_results.append(metrics)
                    continue
                except (
                    OSError,
                    json.JSONDecodeError,
                    KeyError,
                    AttributeError,
                    TypeError,
                    ValueError,
                ) as e:
                    traceback.print_exc()
                    print(f"Corrupted metrics file found. Reprocessing. Error: {e}")

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
            except (RuntimeError, KeyError, AttributeError, TypeError, ValueError) as e:
                traceback.print_exc()
                print(f"Failed to process Home {home_id} - {processing_type}: {e}")
                continue

    summary_csv_path = os.path.join(output_dir, "summary_all_homes.csv")
    try:
        if all_results:
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
                writer = csv.DictWriter(
                    csvfile, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL
                )
                writer.writeheader()
                writer.writerows(all_results)
            print(f"\nSummary saved to: {summary_csv_path}")
    except (OSError, KeyError, AttributeError, TypeError, ValueError) as e:
        traceback.print_exc()
        print(f"Error writing summary: {e}")


if __name__ == "__main__":
    """
    Main entry point for running the evaluation script.
    """
    DATABASE_PATH: str = "THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET"
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
    MIN_VOTES: int = 3
    OUTPUT_DIR: str = os.path.join(DATABASE_PATH, "evaluation_results")
    OPENROUTER_MODEL_ID: str = "openai/gpt-oss-120b"
    OPENROUTER_TEMPERATURE: float = 0.0
    OPENROUTER_TOP_P: float = 0.1
    OPENROUTER_MAX_TOKENS: int = 16000

    load_dotenv()

    openai_client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,
    )

    def llm_agent_func(message: str) -> any:
        """
        Calls the OpenAI client with retries for LLM-based comparison.

        :param message: Message to send to the LLM.
        :type message: str
        :raises Exception: If all retries fail.
        :return: LLM response message.
        :rtype: any
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
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
            except (OSError, AttributeError, TypeError, ValueError) as e:
                traceback.print_exc()
                if attempt == max_retries - 1:
                    raise RuntimeError(f"LLM agent call failed after retries: {e}")
                time.sleep(2)

    try:
        run_full_evaluation(
            database_path=DATABASE_PATH,
            processing_types=PROCESSING_TYPES,
            home_ids=HOME_IDS,
            to_remove_object_classes=TO_REMOVE_OBJECT_CLASSES,
            to_replace_object_classes=TO_REPLACE_OBJECT_CLASSES,
            llm_agent=llm_agent_func,
            distance_threshold_meters=DISTANCE_THRESHOLD_METERS,
            min_votes=MIN_VOTES,
            output_dir=OUTPUT_DIR,
        )
    except (RuntimeError, KeyboardInterrupt) as e:
        traceback.print_exc()
        if isinstance(e, KeyboardInterrupt):
            print("\nProgram interrupted by user. Exiting gracefully.")

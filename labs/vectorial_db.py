import trace
from qdrant_client import QdrantClient
from flashrank import Ranker, RerankRequest
from joblib import Parallel, delayed
from functools import lru_cache
from dotenv import load_dotenv
from typing import Any, Tuple, Dict, List
import traceback
import threading
import numpy as np
import pickle
import gzip
import json
import time
import math
import cv2
import os

from agno.document import Document as AgnoDocument
from agno.embedder.openai import OpenAIEmbedder
from agno.models.openrouter import OpenRouter
from agno.models.lmstudio import LMStudio
from agno.vectordb.search import SearchType
from agno.vectordb.qdrant import Qdrant
from agno.agent import Agent

from conceptgraph.slam.slam_classes import MapObjectList
from prompts import AGENT_PROMPT_V3, INTENTION_INTERPRETATION_PROMPT

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    from watershed_segmenter import (
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
    from labs.watershed_segmenter import (
        load_house_context,
        process_house_segmentation,
        load_profile,
        DEFAULT_PARAMS as WS_DEFAULT_PARAMS,
        WS_CLASS_COLORS,
        TRAJECTORY_IMAGE_DIMMING,
        MAP_BINARY_THRESHOLD,
        MIN_CONTOUR_AREA,
        CROP_PADDING,
    )

console = Console()

PREFFIX = "online"
SELECTED_HOUSE = 1
FORCE_RECREATE_TABLE = False
QDRANT_URL = "http://localhost:6333"

REMOTE_MODEL_ID = "openai/gpt-oss-120b:nitro"
LOCAL_MODEL_ID = "openai/gpt-oss-20b"

DATASET_BASE_PATH = r"D:\Documentos\Datasets\Robot@VirtualHomeLarge"
LOCAL_DATA_DIR = "data"
DEBUG_INPUT_FILE_PATH = os.path.join(LOCAL_DATA_DIR, "input_debug.txt")
DEBUG_OUTPUT_FILE_PATH = os.path.join(LOCAL_DATA_DIR, "output_debug.txt")

DEFAULT_USER_POSE = (0.0, 0.0, 0.0)


def log_debug_interaction(
    file_path: str,
    stage: str,
    system_prompt: str = "",
    user_input: str = "",
    content: str = "",
    mode: str = "a",
) -> None:
    """
    Logs the exact content sent to the model in a text file for debugging purposes.

    :param file_path: Path to the debug log file.
    :type file_path: str
    :param stage: Stage identifier for the log entry.
    :type stage: str
    :param system_prompt: System instructions or prompt.
    :type system_prompt: str
    :param user_input: User input or dynamic content.
    :type user_input: str
    :param content: Model response or output.
    :type content: str
    :param mode: File opening mode.
    :type mode: str
    :return: None
    :rtype: None
    """
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        separator = "=" * 60

        with open(file_path, mode, encoding="utf-8") as f:
            f.write(f"\n{separator}\n")
            f.write(f"[{timestamp}] STAGE: {stage}\n")
            f.write(f"{separator}\n")
            if system_prompt:
                f.write("--- SYSTEM INSTRUCTIONS / PROMPT ---\n")
                f.write(f"{system_prompt}\n")
            if user_input:
                f.write("\n--- USER INPUT / DYNAMIC CONTENT ---\n")
                f.write(f"{user_input}\n")
            if content:
                f.write("\n--- MODEL RESPONSE / OUTPUT ---\n")
                f.write(f"{content}\n")
            f.write(f"{separator}\n")
    except (IOError, OSError) as e:
        traceback.print_exc()
        console.print(f"[bold red][DEBUG LOG ERROR][/bold red]: {e}")


@lru_cache(maxsize=1000)
def get_embedder(preffix: str) -> OpenAIEmbedder:
    """
    Returns an embedder instance based on the provided prefix.

    :param preffix: Prefix identifier for embedder configuration.
    :type preffix: str
    :return: Configured embedder instance.
    :rtype: OpenAIEmbedder
    :raises ValueError: If the prefix is unknown.
    """
    if preffix == "offline":
        return OpenAIEmbedder(
            id="text-embedding-qwen3-embedding-0.6b",
            api_key="",
            base_url="http://localhost:1234/v1",
            dimensions=1024,
        )
    elif preffix == "online" or preffix == "improved":
        return OpenAIEmbedder(
            id="qwen/qwen3-embedding-8b",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            dimensions=4096,
        )
    else:
        raise ValueError(f"Unknown preffix for embedder: {preffix}")


def create_qdrant_vector_db(collection: str, url: str = QDRANT_URL) -> Qdrant:
    """
    Creates and returns a Qdrant vector database instance.

    :param collection: Collection name.
    :type collection: str
    :param url: Qdrant server URL.
    :type url: str
    :return: Configured Qdrant vector database.
    :rtype: Qdrant
    :raises ValueError: If database creation fails.
    """
    try:
        vector_db = Qdrant(
            collection=collection,
            url=url,
            embedder=get_embedder(PREFFIX),
            distance="cosine",
            search_type=SearchType.hybrid,
        )
        return vector_db
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Failed to create Qdrant VectorDb: {e}")


def check_collection_exists_and_not_empty(
    collection_name: str, url: str, vector_size: int
) -> bool:
    """
    Checks if a collection exists and contains items with the correct vector size.

    :param collection_name: Name of the collection.
    :type collection_name: str
    :param url: Qdrant server URL.
    :type url: str
    :param vector_size: Expected vector dimensionality.
    :type vector_size: int
    :return: True if collection exists and is valid, False otherwise.
    :rtype: bool
    """
    try:
        client = QdrantClient(url=url)
        if client.collection_exists(collection_name=collection_name):
            count_res = client.count(collection_name=collection_name, exact=True)
            vector_size_check = (
                client.get_collection(collection_name)
                .config.params.vectors["dense"]
                .size
            )

            if vector_size_check != vector_size:
                console.print(
                    f"[yellow]Collection '{collection_name}' has vector size "
                    f"{vector_size_check}, expected {vector_size}.[/yellow]"
                )
                return False

            if count_res.count > 0:
                console.print(
                    f"[green]Collection '{collection_name}' exists with "
                    f"{count_res.count} items.[/green]"
                )
                return True
            else:
                console.print(
                    f"[yellow]Collection '{collection_name}' exists but is empty.[/yellow]"
                )
                return False
        else:
            console.print(
                f"[yellow]Collection '{collection_name}' does not exist.[/yellow]"
            )
            return False
    except Exception as e:
        traceback.print_exc()
        console.print(f"[bold red]Error checking collection status:[/bold red] {e}")
        return False


def delete_collection_if_exists(collection_name: str, url: str) -> None:
    """
    Deletes a collection if it exists in Qdrant.

    :param collection_name: Name of the collection to delete.
    :type collection_name: str
    :param url: Qdrant server URL.
    :type url: str
    :return: None
    :rtype: None
    """
    try:
        client = QdrantClient(url=url)
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)
            console.print(f"[green]Collection '{collection_name}' deleted.[/green]")
    except Exception as e:
        traceback.print_exc()
        console.print(f"[bold red]Error deleting collection:[/bold red] {e}")


def rerank_with_flashrank(
    query: str, documents: list[AgnoDocument], top_k: int = 10
) -> list[AgnoDocument]:
    """
    Reranks documents using FlashRank based on a query.

    :param query: Query string for reranking.
    :type query: str
    :param documents: List of documents to rerank.
    :type documents: list[AgnoDocument]
    :param top_k: Number of top documents to return.
    :type top_k: int
    :return: Reranked list of documents.
    :rtype: list[AgnoDocument]
    :raises ValueError: If reranking fails.
    """
    if not documents:
        return []
    try:
        ranker = Ranker()
        docs_to_rerank = [
            {"id": idx, "text": doc.content} for idx, doc in enumerate(documents)
        ]
        rerank_request = RerankRequest(query=query, passages=docs_to_rerank)
        response = ranker.rerank(rerank_request)

        for idx, doc in enumerate(documents):
            doc.meta_data["previous_score"] = doc.reranking_score
            doc.meta_data["rerank_score"] = float(response[idx]["score"])
            doc.reranking_score = float(response[idx]["score"])

        return sorted(documents, key=lambda d: d.reranking_score, reverse=True)[:top_k]
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Error during FlashRank reranking: {e}") from e


def sanitize_metadata(data: dict) -> dict:
    """
    Sanitizes metadata by removing callable objects and filtering valid types.

    :param data: Raw metadata dictionary.
    :type data: dict
    :return: Sanitized metadata dictionary.
    :rtype: dict
    """
    sanitized = {}
    for k, v in data.items():
        if callable(v):
            continue
        elif isinstance(v, dict):
            sanitized[k] = sanitize_metadata(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            sanitized[k] = v
        elif isinstance(v, (list, tuple)):
            sanitized[k] = [
                item
                for item in v
                if isinstance(item, (str, int, float, bool)) or item is None
            ]
    return sanitized


def process_object_to_doc(obj_key: str, obj: dict) -> AgnoDocument | None:
    """
    Converts an object dictionary into an AgnoDocument for vectorization.

    :param obj_key: Unique key for the object.
    :type obj_key: str
    :param obj: Object data dictionary.
    :type obj: dict
    :return: AgnoDocument instance or None if invalid.
    :rtype: AgnoDocument | None
    :raises ValueError: If class_id determination fails.
    """
    object_tag = obj.get("object_tag", "") or obj.get("class_name", "")

    object_caption = obj.get("object_caption", "") or obj.get(
        "consolidated_caption", ""
    )
    object_caption = (
        object_caption.replace("'", '"')
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )
    try:
        loaded_json = json.loads(object_caption)
        if isinstance(loaded_json, dict):
            object_caption = loaded_json.get("consolidated_caption", object_caption)
    except json.decoder.JSONDecodeError:
        pass

    if not object_tag.strip() or not object_caption.strip():
        return None

    room_context = f" Located in {obj.get('room_name', 'an unknown area')}."
    text_to_embed = f"{object_tag}: {object_caption}{room_context}"

    metadata = {
        k: v
        for k, v in obj.items()
        if k
        not in [
            "embedding",
            "bbox",
            "pcd_np",
            "bbox_np",
            "pcd_color_np",
            "mask",
            "image_idx",
            "mask_idx",
            "color_path",
            "class_id",
            "captions",
            "num_detections",
            "xyxy",
            "conf",
            "n_points",
            "contain_number",
            "is_background",
            "num_obj_in_class",
            "curr_obj_num",
            "new_counter",
            "is_dirty",
        ]
    }

    positions = obj.get("pcd_np", None)

    if positions is None:
        positions = obj.get("bbox_np", None)

    if positions is None:
        return None

    metadata["centroid"] = np.mean(positions, axis=0).tolist()
    metadata = sanitize_metadata(metadata)
    try:
        class_id_values = obj.get("class_id", [])
        if isinstance(class_id_values, (list, tuple)) and class_id_values:
            counts = {}
            for val in class_id_values:
                if isinstance(val, int):
                    counts[val] = counts.get(val, 0) + 1
            if counts:
                most_common_class_id = max(counts, key=counts.get)
                metadata["class_id"] = most_common_class_id
            else:
                metadata["class_id"] = None
        elif isinstance(class_id_values, int):
            metadata["class_id"] = class_id_values
        else:
            metadata["class_id"] = None
    except (TypeError, ValueError) as e:
        traceback.print_exc()
        raise ValueError("Failed to determine the most common class_id value.") from e

    metadata["id"] = str(obj.get("id", obj_key))

    return AgnoDocument(
        content=text_to_embed, meta_data=metadata, id=str(obj.get("id", obj_key))
    )


def retrieve(vector_db: Qdrant, query: str, limit: int = 100) -> list[Any]:
    """
    Retrieves documents from the vector database based on search type.

    :param vector_db: Qdrant vector database instance.
    :type vector_db: Qdrant
    :param query: Query string for retrieval.
    :type query: str
    :param limit: Maximum number of results to return.
    :type limit: int
    :return: List of retrieved documents.
    :rtype: list[Any]
    :raises ValueError: If search type is unsupported.
    """
    filters = None
    if vector_db.search_type == SearchType.vector:
        results = vector_db._run_vector_search_sync(query, limit, filters)
    elif vector_db.search_type == SearchType.keyword:
        results = vector_db._run_keyword_search_sync(query, limit, filters)
    elif vector_db.search_type == SearchType.hybrid:
        results = vector_db._run_hybrid_search_sync(query, limit, filters)
    else:
        raise ValueError(f"Unsupported search type: {vector_db.search_type}")

    search_results = vector_db._build_search_results(results, query)
    for sr, r in zip(search_results, results):
        sr.reranking_score = r.score
    return search_results


def populate_qdrant_from_objects(
    objects: MapObjectList | dict | List[dict], collection_name: str
) -> int:
    """
    Populates Qdrant collection with documents created from objects.

    :param objects: Map object list, dictionary, or list of dictionaries.
    :type objects: MapObjectList | dict | List[dict]
    :param collection_name: Target collection name.
    :type collection_name: str
    :return: Number of documents inserted.
    :rtype: int
    :raises ValueError: If no valid documents are created.
    """
    vector_db = create_qdrant_vector_db(collection=collection_name)
    vector_db.create()

    iterable = objects
    if isinstance(objects, dict):
        iterable = objects.values()

    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(process_object_to_doc)(str(idx), obj)
        for idx, obj in enumerate(iterable)
    )
    docs_to_insert = [d for d in results if d is not None]

    if not docs_to_insert:
        raise ValueError("No documents created from input objects.")

    console.print(
        f"[cyan]Inserting {len(docs_to_insert)} documents into Qdrant collection "
        f"'{collection_name}'...[/cyan]"
    )

    def insert_single_doc(doc: AgnoDocument) -> None:
        try:
            vector_db.insert([doc])
        except Exception as e:
            traceback.print_exc()
            console.print(
                f"[bold red]Failed to insert doc {getattr(doc, 'id', '?')}:[/bold red] {e}"
            )

    Parallel(n_jobs=-1, backend="threading")(
        delayed(insert_single_doc)(doc) for doc in docs_to_insert
    )

    return len(docs_to_insert)


def query_relevant_chunks(
    collection_name: str,
    queries: str | list[str],
    rerank_query: str | None = None,
    top_k: int = 10,
    confidence_threshold: float = 0.5,
    rerank: bool = True,
    rerank_top_k: int = 10,
) -> list[tuple[str, dict, float]]:
    """
    Queries relevant chunks from the vector database with optional reranking.

    :param collection_name: Collection to query.
    :type collection_name: str
    :param queries: Single query string or list of queries.
    :type queries: str | list[str]
    :param rerank_query: Query string for reranking.
    :type rerank_query: str | None
    :param top_k: Number of top results per query.
    :type top_k: int
    :param confidence_threshold: Minimum score threshold.
    :type confidence_threshold: float
    :param rerank: Whether to apply reranking.
    :type rerank: bool
    :param rerank_top_k: Number of documents to keep after reranking.
    :type rerank_top_k: int
    :return: List of tuples containing content, metadata, and score.
    :rtype: list[tuple[str, dict, float]]
    """
    vector_db = create_qdrant_vector_db(collection=collection_name)
    if isinstance(queries, str):
        queries = [queries]

    results_batches = Parallel(n_jobs=-1, backend="threading")(
        delayed(retrieve)(vector_db, q, limit=top_k) for q in queries
    )
    results = [doc for batch in results_batches for doc in batch]

    unique_docs = {}
    for doc in results:
        metadata = getattr(doc, "meta_data", {})
        doc_id = metadata.get("id", None)
        score = getattr(doc, "reranking_score", None)

        if doc_id is None or score is None:
            console.print(
                f"[yellow]Warning: Document missing 'id' or 'reranking_score'; skipping: {doc}[/yellow]"
            )
            continue

        if doc_id and (
            doc_id not in unique_docs or score > unique_docs[doc_id].reranking_score
        ):
            unique_docs[doc_id] = doc

    results = list(unique_docs.values())

    if not results:
        return []

    if rerank:
        results = rerank_with_flashrank(rerank_query, results, top_k=rerank_top_k)

    final_chunks = []
    for doc in results:
        score = doc.meta_data.get("previous_score", 0.0)
        doc.meta_data["score"] = score
        if score >= 0.75:
            final_chunks.append((doc.content, doc.meta_data, score))

        score = doc.reranking_score
        doc.meta_data["score"] = score

        if score >= confidence_threshold:
            final_chunks.append((doc.content, doc.meta_data, score))

    return final_chunks


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


def world_to_map_coordinates(
    world_coords: Tuple[float, float, float],
    origin: Tuple[float, float],
    resolution: float,
    height_img: int,
) -> Tuple[int, int]:
    """
    Converts world coordinates to map pixel coordinates.

    :param world_coords: World coordinates (x, y, z).
    :type world_coords: Tuple[float, float, float]
    :param origin: Map origin coordinates (x, y).
    :type origin: Tuple[float, float]
    :param resolution: Map resolution in meters per pixel.
    :type resolution: float
    :param height_img: Height of the map image.
    :type height_img: int
    :return: Pixel coordinates (x, y).
    :rtype: Tuple[int, int]
    """
    world_x_map = world_coords[2]
    world_y_map = -world_coords[0]
    pixel_x = int((world_x_map - origin[0]) / resolution)
    pixel_y = int(height_img - ((world_y_map - origin[1]) / resolution))
    return pixel_x, pixel_y


def map_to_world_coordinates(
    pixel_coords: Tuple[int, int],
    origin: Tuple[float, float],
    resolution: float,
    height_img: int,
) -> Tuple[float, float, float]:
    """
    Converts map pixel coordinates to world coordinates.

    :param pixel_coords: Pixel coordinates (x, y).
    :type pixel_coords: Tuple[int, int]
    :param origin: Map origin coordinates (x, y).
    :type origin: Tuple[float, float]
    :param resolution: Map resolution in meters per pixel.
    :type resolution: float
    :param height_img: Height of the map image.
    :type height_img: int
    :return: World coordinates (x, y, z).
    :rtype: Tuple[float, float, float]
    """
    px, py = pixel_coords
    world_y_map = ((height_img - py) * resolution) + origin[1]
    world_x_map = (px * resolution) + origin[0]

    raw_x = -world_y_map
    raw_y = 0.0
    raw_z = world_x_map

    return (raw_x, raw_y, raw_z)


def draw_user_on_map(
    image: np.ndarray,
    user_pos: Tuple[float, float, float],
    origin: Tuple[float, float],
    resolution: float,
) -> np.ndarray:
    """
    Draws user position on the map image.

    :param image: Base map image.
    :type image: np.ndarray
    :param user_pos: User world coordinates (x, y, z).
    :type user_pos: Tuple[float, float, float]
    :param origin: Map origin coordinates (x, y).
    :type origin: Tuple[float, float]
    :param resolution: Map resolution in meters per pixel.
    :type resolution: float
    :return: Map image with user marker.
    :rtype: np.ndarray
    """
    h, w = image.shape[:2]
    img_copy = image.copy()
    px, py = world_to_map_coordinates(user_pos, origin, resolution, h)

    if 0 <= px < w and 0 <= py < h:
        cv2.circle(img_copy, (px, py), 6, (255, 0, 0), -1)
        cv2.circle(img_copy, (px, py), 4, (0, 0, 255), -1)
        cv2.putText(
            img_copy, "USER", (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3
        )
        cv2.putText(
            img_copy,
            "USER",
            (px + 8, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
        )

    return img_copy


def reconstruct_map_debug_image(
    merged_regions: dict,
    region_mask: np.ndarray,
    merged_colors: dict,
    width: int,
    height: int,
    unique_names: dict,
) -> np.ndarray:
    """
    Reconstructs a debug map image with colored regions and labels.

    :param merged_regions: Dictionary of merged region data.
    :type merged_regions: dict
    :param region_mask: Region mask array.
    :type region_mask: np.ndarray
    :param merged_colors: Mapping of region IDs to colors.
    :type merged_colors: dict
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :param unique_names: Mapping of region IDs to room names.
    :type unique_names: dict
    :return: Reconstructed debug map image.
    :rtype: np.ndarray
    """
    try:
        reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                mid = region_mask[y, x]
                if mid in merged_colors:
                    reconstructed_image[y, x] = merged_colors[mid]

        for mid, data in merged_regions.items():
            pts = np.where(region_mask == int(mid))
            if len(pts[0]) > 0:
                cy, cx = int(np.mean(pts[0])), int(np.mean(pts[1]))
                cls_name = unique_names.get(mid, data.get("dominant_class", "unknown"))
                cv2.putText(
                    reconstructed_image,
                    cls_name,
                    (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.2,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    reconstructed_image,
                    cls_name,
                    (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return reconstructed_image
    except Exception:
        traceback.print_exc()
        return np.zeros((height, width, 3), dtype=np.uint8)


def get_object_map_coordinates(
    obj: dict, origin: Tuple[float, float], resolution: float, height_img: int
) -> Tuple[int, int]:
    """
    Gets map pixel coordinates for an object.

    :param obj: Object data dictionary.
    :type obj: dict
    :param origin: Map origin coordinates (x, y).
    :type origin: Tuple[float, float]
    :param resolution: Map resolution in meters per pixel.
    :type resolution: float
    :param height_img: Height of the map image.
    :type height_img: int
    :return: Pixel coordinates (x, y) or (-1, -1) if invalid.
    :rtype: Tuple[int, int]
    """
    if "pcd_np" in obj and len(obj["pcd_np"]) > 0:
        centroid = np.mean(obj["pcd_np"], axis=0)
    elif "bbox_np" in obj:
        centroid = np.mean(obj["bbox_np"], axis=0)
    else:
        return -1, -1
    return world_to_map_coordinates(tuple(centroid), origin, resolution, height_img)


def inject_objects_into_map(
    objects: MapObjectList | dict | List[dict],
    reconstructed_image: np.ndarray,
    region_mask: np.ndarray,
    merged_regions: dict,
    unique_names: dict,
    origin: Tuple[float, float],
    resolution: float,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Injects objects into the map image and enriches them with room information.

    :param objects: Map object list, dictionary, or list of dictionaries.
    :type objects: MapObjectList | dict | List[dict]
    :param reconstructed_image: Base reconstructed map image.
    :type reconstructed_image: np.ndarray
    :param region_mask: Region mask array.
    :type region_mask: np.ndarray
    :param merged_regions: Dictionary of merged region data.
    :type merged_regions: dict
    :param unique_names: Mapping of region IDs to room names.
    :type unique_names: dict
    :param origin: Map origin coordinates (x, y).
    :type origin: Tuple[float, float]
    :param resolution: Map resolution in meters per pixel.
    :type resolution: float
    :return: Tuple of visualization image and enriched objects list.
    :rtype: Tuple[np.ndarray, List[dict]]
    """
    img_viz = reconstructed_image.copy()
    h, w = img_viz.shape[:2]

    iterable_objects = []
    if isinstance(objects, MapObjectList):
        iterable_objects = list(objects)
    elif isinstance(objects, dict):
        iterable_objects = list(objects.values())
    elif isinstance(objects, list):
        iterable_objects = objects

    console.print(
        f"\n[cyan]--- Mapping {len(iterable_objects)} Objects to Regions ---[/cyan]"
    )

    region_contours = {}
    region_centers = {}
    for rid in merged_regions.keys():
        mask_rid = (region_mask == int(rid)).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_rid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            region_contours[rid] = cnts
            pts = np.where(region_mask == int(rid))
            if len(pts[0]) > 0:
                cy = int(np.mean(pts[0]))
                cx = int(np.mean(pts[1]))
                region_centers[rid] = (cx, cy)

    enriched_objects = []

    for obj in iterable_objects:
        obj_dict = obj if isinstance(obj, dict) else obj.__dict__
        class_name = obj_dict.get("class_name")
        if not class_name:
            enriched_objects.append(obj_dict)
            continue

        px, py = get_object_map_coordinates(obj_dict, origin, resolution, h)

        final_rid = -1
        if 0 <= px < w and 0 <= py < h:
            rid_at_pixel = region_mask[py, px]
            if rid_at_pixel in merged_regions:
                final_rid = rid_at_pixel
        if final_rid == -1:
            min_dist = float("inf")
            point_float = (float(px), float(py))
            for rid, cnts in region_contours.items():
                for cnt in cnts:
                    dist = abs(cv2.pointPolygonTest(cnt, point_float, True))
                    if dist < min_dist:
                        min_dist = dist
                        final_rid = rid

        room_name = unique_names.get(final_rid, "Unmapped Space")
        obj_dict["room_name"] = room_name
        obj_dict["region_center"] = region_centers.get(final_rid, (0, 0))
        obj_dict["region_id"] = int(final_rid)

        if 0 <= px < w and 0 <= py < h:
            cv2.circle(img_viz, (px, py), 3, (0, 0, 0), -1)
            cv2.circle(img_viz, (px, py), 2, (0, 255, 255), -1)
            text_pos = (px + 5, py - 5)
            cv2.putText(
                img_viz,
                class_name,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img_viz,
                class_name,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        enriched_objects.append(obj_dict)

    return img_viz, enriched_objects


class MapNavigator(threading.Thread):
    """
    Interactive map navigator thread for visualizing and controlling user position.
    """

    def __init__(
        self,
        window_name: str,
        base_image: np.ndarray,
        origin: Tuple[float, float],
        resolution: float,
    ) -> None:
        """
        Initializes the map navigator.

        :param window_name: Name of the display window.
        :type window_name: str
        :param base_image: Base map image.
        :type base_image: np.ndarray
        :param origin: Map origin coordinates (x, y).
        :type origin: Tuple[float, float]
        :param resolution: Map resolution in meters per pixel.
        :type resolution: float
        """
        super().__init__(daemon=True)
        self.window_name = window_name
        self.base_image = base_image
        self.origin = origin
        self.resolution = resolution
        self.height, self.width = base_image.shape[:2]

        self._user_pos = DEFAULT_USER_POSE
        self._lock = threading.Lock()
        self._running = True
        self._should_exit = False

    def run(self) -> None:
        """
        Main thread loop for rendering the map and handling input.

        :return: None
        :rtype: None
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while self._running:
            with self._lock:
                current_pos = self._user_pos

            img_to_show = draw_user_on_map(
                self.base_image, current_pos, self.origin, self.resolution
            )
            cv2.imshow(self.window_name, img_to_show)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                self._should_exit = True
                self._running = False

        cv2.destroyWindow(self.window_name)

    def mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: Any
    ) -> None:
        """
        Handles mouse click events to move the user position.

        :param event: OpenCV mouse event type.
        :type event: int
        :param x: Mouse x coordinate.
        :type x: int
        :param y: Mouse y coordinate.
        :type y: int
        :param flags: Additional mouse event flags.
        :type flags: int
        :param param: Additional callback parameters.
        :type param: Any
        :return: None
        :rtype: None
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= y < self.height and 0 <= x < self.width:
                pixel_color = self.base_image[y, x]
                if np.all(pixel_color == 0):
                    console.print(
                        "[yellow][UI Warning] Cannot move to black (invalid) area.[/yellow]"
                    )
                    return
            new_pos = map_to_world_coordinates(
                (x, y), self.origin, self.resolution, self.height
            )
            with self._lock:
                self._user_pos = new_pos
            console.print(
                f"[green][UI] User moved to: {new_pos[0]:.2f}, "
                f"{new_pos[1]:.2f}, {new_pos[2]:.2f}[/green]"
            )

    def move_to_coordinate(self, world_coords: Tuple[float, float, float]) -> None:
        """
        Moves user to specified world coordinates with validation.

        :param world_coords: Target world coordinates (x, y, z).
        :type world_coords: Tuple[float, float, float]
        :return: None
        :rtype: None
        """
        px, py = world_to_map_coordinates(
            world_coords, self.origin, self.resolution, self.height
        )
        if not (0 <= px < self.width and 0 <= py < self.height):
            console.print(
                "[yellow][UI Warning] Target coordinates out of map bounds.[/yellow]"
            )
            return

        target_color = self.base_image[py, px]

        if not np.all(target_color == 0):
            with self._lock:
                self._user_pos = world_coords
            return

        console.print(
            "[yellow][UI] Target is in invalid area. Searching for nearest valid point...[/yellow]"
        )

        max_radius = 50
        found_valid = False
        best_px, best_py = px, py

        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue

                    nx, ny = px + dx, py + dy

                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if not np.all(self.base_image[ny, nx] == 0):
                            best_px, best_py = nx, ny
                            found_valid = True
                            break
                if found_valid:
                    break
            if found_valid:
                break
        if found_valid:
            corrected_world_pos = map_to_world_coordinates(
                (best_px, best_py), self.origin, self.resolution, self.height
            )
            with self._lock:
                self._user_pos = corrected_world_pos
            console.print(
                "[green][UI] Adjusted position to nearest valid area.[/green]"
            )
        else:
            console.print(
                "[bold red][UI Error] Could not find valid area near target.[/bold red]"
            )

    @property
    def user_pos(self) -> Tuple[float, float, float]:
        """
        Gets current user position.

        :return: User world coordinates (x, y, z).
        :rtype: Tuple[float, float, float]
        """
        with self._lock:
            return self._user_pos

    @user_pos.setter
    def user_pos(self, val: Tuple[float, float, float]) -> None:
        """
        Sets user position.

        :param val: New user world coordinates (x, y, z).
        :type val: Tuple[float, float, float]
        :return: None
        :rtype: None
        """
        with self._lock:
            self._user_pos = val

    def should_exit(self) -> bool:
        """
        Checks if the navigator should exit.

        :return: True if exit requested, False otherwise.
        :rtype: bool
        """
        return self._should_exit

    def stop(self) -> None:
        """
        Stops the navigator thread.

        :return: None
        :rtype: None
        """
        self._running = False


class SceneGraphNode:
    """
    Represents a node in the scene graph corresponding to an object.
    """

    def __init__(self, obj_dict: dict) -> None:
        """
        Initializes a scene graph node from an object dictionary.

        :param obj_dict: Object data dictionary.
        :type obj_dict: dict
        """
        self.name = obj_dict.get("class_name", "unknown_object")
        self.room = obj_dict.get("room_name", "Unknown Area")

        raw_caption = obj_dict.get("consolidated_caption", "") or obj_dict.get(
            "object_caption", ""
        )
        caption_clean = (
            raw_caption.replace("'", '"')
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        try:
            loaded = json.loads(caption_clean)
            if isinstance(loaded, dict):
                caption_clean = loaded.get("consolidated_caption", caption_clean)
        except Exception:
            pass
        self.caption = caption_clean

        self.centroid = None
        if "pcd_np" in obj_dict and len(obj_dict["pcd_np"]) > 0:
            self.centroid = np.mean(obj_dict["pcd_np"], axis=0)
        elif "bbox_np" in obj_dict:
            self.centroid = np.mean(obj_dict["bbox_np"], axis=0)


class SceneGraphManager:
    """
    Manages the scene graph structure for all objects in the environment.
    """

    def __init__(self, enriched_objects: List[dict]) -> None:
        """
        Initializes the scene graph manager.

        :param enriched_objects: List of enriched object dictionaries.
        :type enriched_objects: List[dict]
        """
        self.nodes: List[SceneGraphNode] = [
            SceneGraphNode(obj) for obj in enriched_objects
        ]
        self.nodes_by_room: Dict[str, List[SceneGraphNode]] = {}
        for node in self.nodes:
            if node.room not in self.nodes_by_room:
                self.nodes_by_room[node.room] = []
            self.nodes_by_room[node.room].append(node)

    @lru_cache(maxsize=128)
    def get_room_centers(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Computes and returns the centroids of each room.

        :return: Mapping of room names to their centroid coordinates.
        :rtype: Dict[str, Tuple[float, float, float]]
        """
        room_centers = {}
        for room, nodes in self.nodes_by_room.items():
            centroids = [node.centroid for node in nodes if node.centroid is not None]
            if centroids:
                mean_centroid = np.mean(centroids, axis=0)
                room_centers[room] = (
                    mean_centroid[0],
                    mean_centroid[1],
                    mean_centroid[2],
                )
            else:
                room_centers[room] = (0.0, 0.0, 0.0)
        return room_centers

    def get_text_representation(self, user_pos: Tuple[float, float, float]) -> str:
        """
        Generates a text representation of the scene graph sorted by distance.

        :param user_pos: User world coordinates (x, y, z).
        :type user_pos: Tuple[float, float, float]
        :return: XML-formatted scene graph string.
        :rtype: str
        """
        output = "<SCENE_GRAPH>\n"
        user_arr = np.array(user_pos)

        room_centers = self.get_room_centers()

        for room in sorted(self.nodes_by_room.keys()):
            room_items = []
            for node in self.nodes_by_room[room]:
                dist_val = float("inf")
                dist_str = "N/A"
                if node.centroid is not None:
                    dist_val = np.linalg.norm(node.centroid - user_arr)
                    dist_str = f"{dist_val:.2f}m"
                room_items.append((dist_val, node, dist_str))

            room_items.sort(key=lambda x: x[0])

            room_center = room_centers.get(room, (0.0, 0.0))
            output += f'<ROOM id="{room}" center="{room_center}"> '
            for _, node, dist_str in room_items:
                output += "<OBJECT> "
                output += f"Class: {node.name} | "
                output += f"Description: {node.caption} | "
                output += f"Distance: {dist_str} | "
                output += "</OBJECT> "
            output += "</ROOM>\n"

        output += "</SCENE_GRAPH>"
        return output


def get_or_compute_watershed_data(
    house_id: int, prefix: str, base_path: str, local_dir: str
) -> Tuple[Dict, np.ndarray, Dict, int, int]:
    """
    Retrieves cached watershed data or computes it if unavailable.

    :param house_id: House identifier.
    :type house_id: int
    :param prefix: Data prefix.
    :type prefix: str
    :param base_path: Base dataset path.
    :type base_path: str
    :param local_dir: Local cache directory.
    :type local_dir: str
    :return: Tuple of merged regions, region mask, merged colors, width, height.
    :rtype: Tuple[Dict, np.ndarray, Dict, int, int]
    :raises ValueError: If context loading fails.
    """
    os.makedirs(local_dir, exist_ok=True)
    filename = f"Home{house_id:02d}_{prefix}_watershed.pkl.gz"
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        console.print(
            f"[cyan]Loading cached Watershed data from {local_path}...[/cyan]"
        )
        try:
            with gzip.open(local_path, "rb") as f:
                data = pickle.load(f)
            return (
                data["merged_regions"],
                data["region_mask"],
                data["merged_colors"],
                data["width"],
                data["height"],
            )
        except Exception as e:
            traceback.print_exc()
            console.print(f"[yellow]Failed to load cache: {e}. Recomputing...[/yellow]")

    console.print(
        f"[cyan]Cache missing. Computing Watershed data for House {house_id} "
        f"({prefix})...[/cyan]"
    )
    context = load_house_context(
        house_id=house_id,
        base_path=base_path,
        prefixes=[prefix],
        map_binary_threshold=MAP_BINARY_THRESHOLD,
        min_contour_area=MIN_CONTOUR_AREA,
        crop_padding=CROP_PADDING,
    )

    if prefix not in context:
        raise ValueError(f"Could not load context for prefix '{prefix}'.")

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

    with gzip.open(local_path, "wb") as f:
        pickle.dump(data_to_save, f)

    return (
        res_data["merged_regions"],
        res_data["region_mask"],
        res_data["merged_colors"],
        res_data["width"],
        res_data["height"],
    )


def get_user_room_name(
    user_pos: Tuple[float, float, float],
    origin: Tuple[float, float],
    resolution: float,
    region_mask: np.ndarray,
    unique_names: Dict[int, str],
) -> str:
    """
    Determines the current room name based on user position and segmentation mask.

    :param user_pos: User world coordinates (x, y, z).
    :type user_pos: Tuple[float, float, float]
    :param origin: Map origin coordinates (x, y).
    :type origin: Tuple[float, float]
    :param resolution: Map resolution in meters per pixel.
    :type resolution: float
    :param region_mask: Region mask array containing region IDs (H, W).
    :type region_mask: np.ndarray
    :param unique_names: Dictionary mapping region IDs to readable names.
    :type unique_names: Dict[int, str]
    :return: Room name or 'Unknown Area'.
    :rtype: str
    """
    h, w = region_mask.shape[:2]
    px, py = world_to_map_coordinates(user_pos, origin, resolution, h)

    if 0 <= px < w and 0 <= py < h:
        region_id = region_mask[py, px]
        if region_id < 0:
            return "Hallway/Unknown Area"
        return unique_names.get(int(region_id), "Unknown Room")

    return "Outside Map"


if __name__ == "__main__":
    load_dotenv()

    COLLECTION_NAME = f"House_{SELECTED_HOUSE:02d}"
    console.print(
        Panel(
            f"[bold cyan]Session for {COLLECTION_NAME} (Prefix: {PREFFIX})[/bold cyan]",
            title="[bold green]Initialization[/bold green]",
            expand=False,
        )
    )

    try:
        with open(DEBUG_INPUT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write("=== New Session Started ===\n")
    except Exception:
        pass
    try:
        with open(DEBUG_OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
            f.write("=== New Session Started ===\n")
    except Exception:
        pass

    console.print("[cyan]Initializing Interpreter Agent...[/cyan]")
    if PREFFIX == "offline":
        interpreter_model = LMStudio(
            id=LOCAL_MODEL_ID,
            temperature=0.0,
            top_p=0.1,
            reasoning_effort="high",
            max_tokens=16000,
        )
    else:
        interpreter_model = OpenRouter(
            id=REMOTE_MODEL_ID,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.0,
            top_p=0.1,
            reasoning_effort="high",
            max_tokens=16000,
        )

    interpreter_agent = Agent(
        model=interpreter_model,
        instructions=INTENTION_INTERPRETATION_PROMPT,
        markdown=True,
        description="Intent Interpreter",
    )

    console.print("[cyan]Initializing Bot Agent...[/cyan]")
    if PREFFIX == "offline":
        bot_model = LMStudio(id=LOCAL_MODEL_ID)
    else:
        bot_model = OpenRouter(
            id=REMOTE_MODEL_ID, api_key=os.getenv("OPENROUTER_API_KEY")
        )

    bot_agent = Agent(
        model=bot_model,
        markdown=True,
        description="Smart Wheelchair Navigator",
        instructions=AGENT_PROMPT_V3,
    )

    try:
        (merged_regions, region_mask, merged_colors, width, height) = (
            get_or_compute_watershed_data(
                house_id=SELECTED_HOUSE,
                prefix=PREFFIX,
                base_path=DATASET_BASE_PATH,
                local_dir=LOCAL_DATA_DIR,
            )
        )

        unique_names = generate_unique_room_names(merged_regions)

        base_map_img = reconstruct_map_debug_image(
            merged_regions, region_mask, merged_colors, width, height, unique_names
        )

        console.print("[cyan]Fetching map metadata for visualization...[/cyan]")
        temp_context = load_house_context(
            house_id=SELECTED_HOUSE,
            base_path=DATASET_BASE_PATH,
            prefixes=[PREFFIX],
            map_binary_threshold=MAP_BINARY_THRESHOLD,
            min_contour_area=MIN_CONTOUR_AREA,
            crop_padding=CROP_PADDING,
        )
        map_origin = temp_context[PREFFIX]["origin"]
        map_resolution = temp_context[PREFFIX]["resolution"]

    except Exception as e:
        console.print(f"[bold red]Critical Error loading map data:[/bold red] {e}")
        traceback.print_exc()
        exit(1)

    exists_and_valid = check_collection_exists_and_not_empty(
        COLLECTION_NAME, QDRANT_URL, get_embedder(PREFFIX).dimensions
    )

    if not exists_and_valid:
        client = QdrantClient(url=QDRANT_URL)
        try:
            console.print(
                f"[yellow]Deleting the existing collection {COLLECTION_NAME} if it exists...[/yellow]"
            )
            client.delete_collection(collection_name=COLLECTION_NAME)
        except Exception:
            traceback.print_exc()
            raise

    enriched_objects = []

    if FORCE_RECREATE_TABLE:
        console.print(
            f"[yellow]FORCE_RECREATE_TABLE is True. Cleaning up '{COLLECTION_NAME}'...[/yellow]"
        )
        delete_collection_if_exists(COLLECTION_NAME, QDRANT_URL)

    OBJECTS_PATH = os.path.join(
        DATASET_BASE_PATH,
        "outputs",
        f"Home{SELECTED_HOUSE:02d}",
        "Wandering",
        "exps",
        f"{PREFFIX}_house_{SELECTED_HOUSE}_map",
        f"pcd_{PREFFIX}_house_{SELECTED_HOUSE}_map.pkl.gz",
    )

    console.print(f"[cyan]Loading objects from: {OBJECTS_PATH}[/cyan]")
    if not os.path.exists(OBJECTS_PATH):
        console.print(
            f"[bold red]Error: Objects file not found at {OBJECTS_PATH}[/bold red]"
        )
    else:
        with gzip.open(OBJECTS_PATH, "rb") as f:
            raw_results = pickle.load(f)

        raw_objects = raw_results["objects"]

        console.print("[cyan]Mapping objects to segmented map...[/cyan]")

        debug_map_with_objects, enriched_objects = inject_objects_into_map(
            raw_objects,
            reconstructed_image=base_map_img,
            region_mask=region_mask,
            merged_regions=merged_regions,
            unique_names=unique_names,
            origin=map_origin,
            resolution=map_resolution,
        )

        obj_debug_path = os.path.join(
            LOCAL_DATA_DIR, f"{COLLECTION_NAME}_{PREFFIX}_objects_debug.png"
        )
        img_to_save = draw_user_on_map(
            debug_map_with_objects, DEFAULT_USER_POSE, map_origin, map_resolution
        )
        cv2.imwrite(obj_debug_path, img_to_save)
        console.print(f"[green]Map with objects saved to {obj_debug_path}[/green]")

        if FORCE_RECREATE_TABLE or not exists_and_valid:
            console.print("[cyan]Populating Qdrant with enriched objects...[/cyan]")
            populate_qdrant_from_objects(enriched_objects, COLLECTION_NAME)
            console.print("[green]Vector database populated successfully.[/green]")

    console.print("[cyan]Initializing Scene Graph Manager...[/cyan]")
    scene_manager = SceneGraphManager(enriched_objects)

    console.print("[cyan]Initializing Interactive Map Navigator...[/cyan]")
    navigator = MapNavigator(
        "Interactive Map (Press 'q' to quit)", base_map_img, map_origin, map_resolution
    )
    navigator.start()

    chat_history = []
    active_rag_context = []
    last_bot_message = ""

    console.print(
        Panel(
            "[bold green]Starting Chat Session[/bold green]\n"
            "Click on the map window to move the user. Type your query below.",
            title="[bold cyan]Ready[/bold cyan]",
            expand=False,
        )
    )

    while True:
        try:
            if navigator.should_exit():
                console.print("[yellow]\nExiting program via map window.[/yellow]")
                break

            try:
                current_prompt_pos = navigator.user_pos
                current_room_name = get_user_room_name(
                    user_pos=current_prompt_pos,
                    origin=map_origin,
                    resolution=map_resolution,
                    region_mask=region_mask,
                    unique_names=unique_names,
                )
                query = Prompt.ask(
                    f"\n[cyan][Pos: {current_prompt_pos[0]:.2f}, "
                    f"{current_prompt_pos[1]:.2f}, {current_prompt_pos[2]:.2f}] "
                    f"Room: {current_room_name}[/cyan] Query (or 'q')"
                )
            except EOFError:
                break

            if query.lower() in ["q", "quit", "exit"]:
                navigator.stop()
                break

            if navigator.should_exit():
                break

            if query.lower().startswith("move"):
                try:
                    coords_str = query.lower().replace("move", "").strip()
                    parts = [float(x.strip()) for x in coords_str.split(",")]
                    if len(parts) == 3:
                        navigator.user_pos = tuple(parts)
                        time.sleep(0.1)
                        console.print(
                            f"[green]User moved to: {navigator.user_pos}[/green]"
                        )
                        continue
                    else:
                        console.print("[yellow]Invalid format.[/yellow]")
                        continue
                except ValueError:
                    console.print("[yellow]Invalid coordinates.[/yellow]")
                    continue

            current_user_pos = navigator.user_pos
            current_room_name = get_user_room_name(
                user_pos=current_user_pos,
                origin=map_origin,
                resolution=map_resolution,
                region_mask=region_mask,
                unique_names=unique_names,
            )
            scene_tree = scene_manager.get_text_representation(current_user_pos)

            intention_input = (
                f"<CURRENT_ROOM>{current_room_name}</CURRENT_ROOM>\n"
                f"<SCENE_GRAPH_SUMMARY>\n{scene_tree}\n</SCENE_GRAPH_SUMMARY>\n\n"
                f"<LAST_BOT_MESSAGE>{last_bot_message}</LAST_BOT_MESSAGE>\n\n"
                f"<USER_QUERY>{query}</USER_QUERY>"
            )
            log_debug_interaction(
                DEBUG_INPUT_FILE_PATH,
                "INTERPRETER",
                system_prompt=INTENTION_INTERPRETATION_PROMPT,
                user_input=intention_input,
                mode="w",
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Interpreter processing...", total=None)
                rag_decision_response = interpreter_agent.run(intention_input).content
                progress.remove_task(task)

            console.print(
                f"[magenta][INTERPRETER] Response:[/magenta] {rag_decision_response}"
            )
            log_debug_interaction(
                DEBUG_OUTPUT_FILE_PATH,
                "INTERPRETER",
                content=str(rag_decision_response),
                mode="w",
            )

            try:
                rag_decision_json = (
                    rag_decision_response.replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                result = json.loads(rag_decision_json)
            except (json.JSONDecodeError, Exception) as e:
                console.print(f"[bold red]Interpreter Error:[/bold red] {e}")
                result = {
                    "state": "UNCLEAR",
                    "direct_response": "I couldn't understand that. Could you repeat?",
                }

            state = result.get("state", "NEW_REQUEST")
            direct_response = result.get("direct_response")

            console.print(
                f"\n[bold magenta][INTERPRETER][/bold magenta] State: [cyan]{state}[/cyan]"
            )
            if result.get("intent_explanation"):
                console.print(
                    f"[bold magenta][INTERPRETER][/bold magenta] Intent: [yellow]{result.get('intent_explanation')}[/yellow]"
                )
            if result.get("rerank_query"):
                console.print(
                    f"[bold magenta][INTERPRETER][/bold magenta] Rerank Query: [yellow]{result.get('rerank_query')}[/yellow]"
                )

            if state in ["END_CONVERSATION", "UNCLEAR", "SCENE_GRAPH_QUERY"]:
                console.print(
                    f"\n[bold green]Assistant:[/bold green] {direct_response}\n{'-'*30}"
                )
                last_bot_message = direct_response or ""
                chat_history.append({"user": query, "bot": direct_response})
                if state == "END_CONVERSATION":
                    navigator.stop()
                    break
                continue
            elif state == "TAKE_ME_TO_ROOM":
                if "selected_room" in rag_decision_response:
                    try:
                        selected_room = result.get("selected_room", None)
                        if selected_room is None:
                            console.print(
                                "[bold red]Error: Selected room missing in response.[/bold red]"
                            )
                            rag_docs = []
                            active_rag_context = []
                            continue

                        center_coordinates = selected_room.get("center_coordinates", None)
                        if center_coordinates is None:
                            console.print(
                                "[bold red]Error: Center coordinates missing in response.[/bold red]"
                            )
                            rag_docs = []
                            active_rag_context = []
                            continue

                        console.print(
                            f"\n[bold green][NAVIGATION][/bold green] Room selected: [yellow]{selected_room}[/yellow]"
                        )
                        console.print(
                            f"[bold green][NAVIGATION][/bold green] Center Coordinates: [cyan]{center_coordinates}[/cyan]"
                        )

                        navigator.move_to_coordinate(tuple(center_coordinates))
                        time.sleep(0.1)
                        rag_docs = []
                        active_rag_context = []
                        continue
                    except (KeyError, TypeError, ValueError) as e:
                        traceback.print_exc()
                        console.print(
                            f"[bold red]Error parsing selected room:[/bold red] {e}"
                        )

            if state == "CONTINUE_REQUEST":
                console.print(
                    "[bold blue][RAG][/bold blue] Using ACTIVE MEMORY (Skipping Search)"
                )
                rag_docs = active_rag_context
            else:
                queries = result.get("rag_queries", [])
                rerank_query = result.get("rerank_query", "")

                if queries:
                    console.print(
                        f"[bold blue][RAG][/bold blue] Executing queries: [cyan]{queries}[/cyan]"
                    )
                    relevant_docs = query_relevant_chunks(
                        collection_name=COLLECTION_NAME,
                        queries=queries,
                        rerank_query=rerank_query,
                        top_k=10,
                        confidence_threshold=0.5,
                        rerank=False,
                        rerank_top_k=10,
                    )

                    docs_to_remove = []
                    for idx, doc in enumerate(relevant_docs):
                        centroid = doc[1].get("centroid", None)
                        distance = (
                            math.dist(centroid, current_user_pos) if centroid else None
                        )
                        if distance is None:
                            docs_to_remove.append(idx)

                        doc[1]["distance_to_user"] = distance

                    for idx in reversed(docs_to_remove):
                        relevant_docs.pop(idx)

                    rag_docs = (
                        [chunk[1] for chunk in relevant_docs] if relevant_docs else []
                    )
                    active_rag_context = rag_docs
                else:
                    console.print(
                        "[bold blue][RAG][/bold blue] No queries generated (Localization/Graph lookup)."
                    )
                    rag_docs = []
                    active_rag_context = []

            rag_context_str = json.dumps(rag_docs, indent=2, ensure_ascii=False)
            history_str = "\n".join(
                [f"User: {h['user']}\nBot: {h['bot']}" for h in chat_history]
            )
            intent_explanation = result.get("intent_explanation", "")

            bot_input_prompt = f"""
<INPUT_DATA>
    <USER_QUERY>{query}</USER_QUERY>
    <CURRENT_ROOM>{current_room_name}</CURRENT_ROOM>
    <CURRENT_STATE>{state}</CURRENT_STATE>
    <CHAT_HISTORY>
    {history_str}
    </CHAT_HISTORY>
    <INTERPRETED_INTENT>
    {intent_explanation}
    </INTERPRETED_INTENT>
    <RETRIEVED_CONTEXT_RAG>
    {rag_context_str}
    </RETRIEVED_CONTEXT_RAG>
</INPUT_DATA>
"""
            log_debug_interaction(
                DEBUG_INPUT_FILE_PATH,
                "BOT",
                system_prompt=AGENT_PROMPT_V3,
                user_input=bot_input_prompt,
                mode="a",
            )

            console.print("[cyan]Generating response...[/cyan]")
            response = bot_agent.run(bot_input_prompt)
            response_text = response.content
            console.print(f"[bold blue][BOT][/bold blue] Raw Response: {response_text}")
            log_debug_interaction(
                DEBUG_OUTPUT_FILE_PATH,
                "BOT",
                content=str(response_text),
                mode="a",
            )

            if "<message>" in response_text:
                try:
                    response_text = (
                        response_text.split("<message>")[1]
                        .split("</message>")[0]
                        .strip()
                    )
                except Exception as e:
                    console.print(f"[bold red]Error extracting message:[/bold red] {e}")

            last_bot_message = response_text
            chat_history.append({"user": query, "bot": response_text})
            if len(chat_history) > 5:
                chat_history.pop(0)

            if "<selected_object>" in response_text:
                try:
                    response_message = response_text
                    coords = None
                    obj_tag = None
                    room_name = None
                    if "<answer>" in response_text:
                        response_message = (
                            response_text.split("<answer>")[1]
                            .split("</answer>")[0]
                            .strip()
                        )
                    if "<object_tag>" in response_text:
                        obj_tag = (
                            response_text.split("<object_tag>")[1]
                            .split("</object_tag>")[0]
                            .strip()
                        )
                    if "<room_name>" in response_text:
                        room_name = (
                            response_text.split("<room_name>")[1]
                            .split("</room_name>")[0]
                            .strip()
                        )
                    try:
                        if "<target_coordinates>" in response_text:
                            coords_str = (
                                response_text.split("<target_coordinates>")[1]
                                .split("</target_coordinates>")[0]
                                .strip()
                            )
                            coords = [float(x.strip()) for x in coords_str.split(",")]

                            console.print(
                                f"\n[bold green][NAVIGATION][/bold green] Destination confirmed: [cyan]{obj_tag}[/cyan] in [yellow]{room_name}[/yellow]"
                            )
                            console.print(
                                f"[bold green][NAVIGATION][/bold green] Target Coordinates: [cyan]{coords}[/cyan]"
                            )
                    except TypeError:
                        console.print(
                            f"[bold red]Error parsing target coordinates:[/bold red] {coords_str}"
                        )
                        coords = None

                    if coords:
                        navigator.move_to_coordinate(tuple(coords))
                except Exception as e:
                    console.print(
                        f"[bold red]Error parsing selected object tag:[/bold red] {e}"
                    )
            else:
                response_message = response_text

            console.print(
                f"\n[bold green]Assistant:[/bold green] {response_message}\n{'-'*30}"
            )

        except KeyboardInterrupt:
            navigator.stop()
            break
        except Exception as e:
            console.print(f"[bold red]Error during chat loop:[/bold red] {e}")
            traceback.print_exc()

    navigator.stop()
    navigator.join()
    console.print("[bold green]Program finished.[/bold green]")

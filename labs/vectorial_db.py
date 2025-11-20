from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.rankers.jina import JinaRanker
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.document_stores.types import DuplicatePolicy
from semantic_text_splitter import TextSplitter
from haystack.dataclasses import ChatMessage
from joblib import Parallel, delayed
from haystack.utils import Secret
from dotenv import load_dotenv
from haystack import Document
from openai import OpenAI
import numpy as np
import traceback
import pickle
import json
import gzip
import cv2
import os


from conceptgraph.slam.slam_classes import MapObjectList
from prompts import AGENT_PROMPT_V2


def create_qdrant_document_store(
    url: str = "http://localhost:6333",
    index: str = "default",
    embedding_dim: int = 1536,
    recreate_index: bool = False,
    return_embedding: bool = True,
    wait_result_from_api: bool = True,
) -> QdrantDocumentStore:
    """
    Create and return a QdrantDocumentStore instance.

    :param url: The Qdrant instance URL.
    :type url: str
    :param index: The name of the index collection to use.
    :type index: str
    :param embedding_dim: The dimension of the embeddings.
    :type embedding_dim: int
    :param recreate_index: Whether to recreate the index if it exists.
    :type recreate_index: bool
    :param return_embedding: Whether to return embeddings with documents.
    :type return_embedding: bool
    :param wait_result_from_api: Whether to wait for the API result.
    :type wait_result_from_api: bool
    :raises ValueError: If the QdrantDocumentStore cannot be created.
    :return: An instance of QdrantDocumentStore.
    :rtype: QdrantDocumentStore
    """
    try:
        document_store = QdrantDocumentStore(
            url=url,
            index=index,
            embedding_dim=embedding_dim,
            recreate_index=recreate_index,
            return_embedding=return_embedding,
            wait_result_from_api=wait_result_from_api,
        )
        return document_store
    except (TypeError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to create QdrantDocumentStore: {e}")
    except (ImportError, AttributeError) as e:
        traceback.print_exc()
        raise ValueError(
            f"Import or attribute error while creating QdrantDocumentStore: {e}"
        )


def get_openai_embedding(text: str) -> list:
    """
    Generate embedding for text using OpenAI API via OpenRouter.

    :param text: The text to embed.
    :type text: str
    :raises ValueError: If embedding generation fails.
    :return: The embedding vector as a list of floats.
    :rtype: list
    """
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        embedding = client.embeddings.create(
            model="openai/text-embedding-3-small", input=text, encoding_format="float"
        )
        return embedding.data[0].embedding
    except (ValueError, KeyError, IndexError, AttributeError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to generate embedding: {e}")


def process_document_objects(obj_key: str, obj: dict) -> Document | None:
    """
    Processes a single object to generate a Document with embedding.

    :param obj_key: The key of the object in the dictionary.
    :type obj_key: str
    :param obj: The object dictionary.
    :type obj: dict
    :raises ValueError: If the object is not a dictionary or embedding is missing.
    :return: A Document instance with embedding or None if object_tag or object_caption is invalid.
    :rtype: Document | None
    """
    if not isinstance(obj, dict):
        raise ValueError(f"Object '{obj_key}' is not a dictionary.")

    object_tag = obj.get("object_tag", "")
    object_caption = obj.get("object_caption", "")

    if not isinstance(object_tag, str) or not object_tag.strip():
        return None
    if not isinstance(object_caption, str) or not object_caption.strip():
        return None

    text_to_embed = f"{object_tag}: {object_caption}"
    metadata = {k: v for k, v in obj.items()}

    embedding_vector = get_openai_embedding(text_to_embed)
    doc = Document(
        content=text_to_embed,
        meta=metadata,
    )
    doc.embedding = embedding_vector
    return doc


def sanitize_metadata(data: dict) -> dict:
    """
    Recursively removes non-primitive and callable values from a dictionary.

    :param data: The dictionary to sanitize.
    :type data: dict
    :return: The sanitized dictionary.
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


def create_qdrant_data_from_objects(objects: dict) -> int:
    """
    Creates Qdrant documents from either a dictionary of objects or a MapObjectList.

    :param objects: Either a dictionary where each key is an object identifier or a MapObjectList instance.
    :type objects: dict
    :raises ValueError: If embedding writing or counting fails or if the input type is unsupported.
    :return: The number of documents in the store after insertion.
    :rtype: int
    """
    try:
        document_store = create_qdrant_document_store()
        documents_with_embeddings = []

        if type(objects) is MapObjectList:
            for obj in objects:
                object_id: str | None = obj.get("id", None)
                object_tag: str = obj.get("class_name", "")
                object_caption: str = obj.get("consolidated_caption", "")
                bbox = obj.get("bbox", None)

                if object_id is not None:
                    object_id = str(object_id)

                if bbox is None:
                    bbox_extent = None
                    bbox_center = None
                    bbox_volume = None
                else:
                    bbox_extent = list(bbox.extent)
                    bbox_center = list(bbox.center)
                    bbox_volume = bbox.volume()

                if not isinstance(object_tag, str) or not object_tag.strip():
                    continue
                if not isinstance(object_caption, str) or not object_caption.strip():
                    continue

                text_to_embed: str = f"{object_tag}: {object_caption}"

                metadata = {
                    "id": object_id,
                    "object_tag": object_tag,
                    "object_caption": object_caption,
                    "bbox_extent": bbox_extent,
                    "bbox_center": bbox_center,
                    "bbox_volume": bbox_volume,
                }

                metadata = sanitize_metadata(metadata)

                embedding_vector = get_openai_embedding(text_to_embed)
                doc = Document(
                    content=text_to_embed,
                    meta=metadata,
                )
                doc.embedding = embedding_vector
                documents_with_embeddings.append(doc)
        elif isinstance(objects, dict):
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(process_document_objects)(obj_key, obj)
                for obj_key, obj in objects.items()
            )
            documents_with_embeddings.extend(
                [doc for doc in results if doc is not None]
            )
        else:
            raise ValueError("Input must be a dict or a MapObjectList object.")

        if not documents_with_embeddings:
            raise ValueError(
                "No documents with embeddings were created from the input objects."
            )

        document_store.write_documents(
            documents_with_embeddings, policy=DuplicatePolicy.SKIP
        )

        count = document_store.count_documents()
        print(f"Number of documents in the store: {count}")
        return count
    except (ImportError, AttributeError, TypeError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Error in Qdrant data creation from objects: {e}")
    except (RuntimeError, OSError) as e:
        traceback.print_exc()
        raise ValueError(
            f"Runtime or OS error in Qdrant data creation from objects: {e}"
        )


def create_qdrant_data_from_texts(texts: list[str]) -> int:
    """
    Splits input texts into semantic chunks using semantic-text-splitter and generates embeddings for each chunk.

    :param texts: A list of strings to be split and written to the document store.
    :type texts: list[str]
    :raises ValueError: If text splitting embedding writing or counting fails.
    :return: The number of documents in the store after insertion.
    :rtype: int
    """
    try:
        document_store = create_qdrant_document_store()

        splitter = TextSplitter(1000)

        all_chunks = []
        for text in texts:
            chunks = splitter.chunks(text)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No semantic chunks were generated from the input texts.")

        documents_with_embeddings = []
        for chunk in all_chunks:
            embedding_vector = get_openai_embedding(chunk)
            doc = Document(content=chunk)
            doc.embedding = embedding_vector
            documents_with_embeddings.append(doc)

        if not documents_with_embeddings:
            raise ValueError("No documents with embeddings were created.")

        document_store.write_documents(
            documents_with_embeddings, policy=DuplicatePolicy.SKIP
        )

        count = document_store.count_documents()
        print(f"Number of documents in the store: {count}")
        return count
    except (ImportError, AttributeError, TypeError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(
            f"Error in Qdrant data creation with semantic-text-splitter: {e}"
        )
    except (RuntimeError, OSError) as e:
        traceback.print_exc()
        raise ValueError(
            f"Runtime or OS error in Qdrant data creation with semantic-text-splitter: {e}"
        )


def query_relevant_chunks(
    query: str,
    top_k: int = 5,
    confidence_threshold: float = 0.1,
    rerank: bool = True,
    rerank_top_k: int = 10,
) -> list[tuple[str, dict, float]]:
    """
    Retrieves the most relevant document chunks for a given query using OpenAI embeddings and QdrantDocumentStore.

    :param query: The search query string to embed and search for relevant chunks.
    :type query: str
    :param top_k: The maximum number of relevant chunks to retrieve from vector search.
    :type top_k: int
    :param confidence_threshold: The minimum confidence score required for a chunk to be considered relevant.
    :type confidence_threshold: float
    :param rerank: Whether to apply reranking with JinaRanker.
    :type rerank: bool
    :param rerank_top_k: The number of top documents to keep after reranking.
    :type rerank_top_k: int
    :raises ValueError: If the retrieval process fails or embedding fails.
    :return: A list of tuples each containing a relevant document chunk its metadata and its confidence score.
    :rtype: list[tuple[str, dict, float]]
    """
    try:
        document_store = create_qdrant_document_store()

        all_documents = document_store.filter_documents(filters=None)
        if not all_documents:
            return []

        query_embedding = get_openai_embedding(query)

        if not hasattr(document_store, "_query_by_embedding"):
            raise AttributeError(
                "QdrantDocumentStore does not have a _query_by_embedding method."
            )

        retrieved_documents = document_store._query_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        relevant_docs = []
        reranked_chunks = []
        if rerank and retrieved_documents and os.getenv("JINA_API_KEY") is not None:
            try:
                print("Reranking chunks with Jina AI via Haystack...")
                reranker = JinaRanker(
                    api_key=Secret.from_env_var("JINA_API_KEY"),
                    model="jina-reranker-v2-base-multilingual",
                    top_k=rerank_top_k,
                )

                retrieved_documents = reranker.run(
                    query=query, documents=retrieved_documents
                )
            except (ImportError, ValueError, KeyError, IndexError) as e:
                traceback.print_exc()
                print(f"Error during reranking: {str(e)}")
                print("Using original chunks without reranking")
                raise

        relevant_docs = (
            retrieved_documents.get("documents", [])
            if isinstance(retrieved_documents, dict)
            else retrieved_documents
        )

        for doc in relevant_docs:
            score = getattr(doc, "score", None)
            if score is None and isinstance(doc, dict):
                score = doc.get("score", None)
            content = getattr(doc, "content", None)
            if content is None and isinstance(doc, dict):
                content = doc.get("content", None)
            meta = getattr(doc, "meta", None)
            if meta is None and isinstance(doc, dict):
                meta = doc.get("meta", {})
            if meta is None:
                meta = {}
            if score is None:
                score = 1.0
            if content is not None and score >= confidence_threshold:
                reranked_chunks.append((content, meta, float(score)))

        return reranked_chunks

    except (ImportError, AttributeError, TypeError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Error during query and retrieval of relevant chunks: {e}")
    except (RuntimeError, OSError) as e:
        traceback.print_exc()
        raise ValueError(
            f"Runtime or OS error during query and retrieval of relevant chunks: {e}"
        )


def infer_with_llm_model(
    llm_model: OpenAIChatGenerator,
    messages: list[ChatMessage],
) -> str:
    """
    Performs inference using the provided OpenAIChatGenerator model and the messages list.

    :param llm_model: The OpenAIChatGenerator instance for text generation.
    :type llm_model: OpenAIChatGenerator
    :param messages: The list of chat messages to send to the model.
    :type messages: list[ChatMessage]
    :raises ValueError: If inference fails due to input or model errors.
    :return: The generated response from the language model.
    :rtype: str
    """
    try:
        if not isinstance(messages, list):
            raise ValueError("messages must be a list of ChatMessage objects.")

        result = llm_model.run(messages)

        if isinstance(result, dict) and "replies" in result and result["replies"]:
            response = result["replies"][0]
            if hasattr(response, "text"):
                return response.text
            return response
        elif isinstance(result, str):
            return result
        else:
            raise ValueError(
                "Unexpected response format from OpenAIChatGenerator inference."
            )
    except (ValueError, AttributeError, KeyError, TypeError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to perform inference with OpenAIChatGenerator: {e}")
    except (RuntimeError, OSError) as e:
        traceback.print_exc()
        raise ValueError(
            f"Runtime or OS error during OpenAIChatGenerator inference: {e}"
        )


def read_objects_json(file_path: str = "objects.json") -> str:
    """
    Reads the contents of the specified objects.json file and returns it as a string.

    :param file_path: The path to the objects.json file.
    :type file_path: str
    :raises FileNotFoundError: If the file does not exist at the specified path.
    :raises OSError: If there is an error opening or reading the file.
    :return: The contents of the objects.json file as a string.
    :rtype: str
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        return content
    except (FileNotFoundError, OSError):
        traceback.print_exc()
        raise


def load_pkl_gz_result(result_path: str) -> MapObjectList:
    """
    Loads the result file and returns objects background objects and class colors.

    :param result_path: Path to the gzipped pickle result file.
    :type result_path: str
    :raises ValueError: If the loaded results are not a dictionary.
    :return: A MapObjectList instance containing the objects.
    :rtype: MapObjectList
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

    objects = MapObjectList()
    objects.load_serializable(results["objects"])
    return objects


def load_voronoi_map(filepath: str) -> tuple[dict, np.ndarray, dict, int, int] | None:
    """
    Loads Voronoi map data from a compressed pickle file.

    :param filepath: Path to the compressed pickle file to load.
    :type filepath: str
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the pickle data is invalid or corrupted.
    :raises OSError: If the file cannot be read.
    :return: Tuple containing merged_regions region_mask merged_colors width height or None if failed.
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


if __name__ == "__main__":
    load_dotenv()

    INFERENCE = False
    OBJECTS_PATH = "r_mapping_6_stride15\\pcd_r_mapping_6_stride15.pkl.gz"

    MAP_PATH = "voronoi_map.pkl.gz"
    USER_LOCATION = (155, 60)

    merged_regions, region_mask, merged_colors, width, height = load_voronoi_map(
        MAP_PATH
    )
    voronoi_map = reconstruct_voronoi_image(
        merged_regions, region_mask, merged_colors, width, height
    )
    cv2.imwrite("voronoi_map.png", voronoi_map)

    MAX_OBJECTS = -1

    OBJECTS_SOURCE = "pkl.gz" if OBJECTS_PATH.endswith(".pkl.gz") else "json"
    if not INFERENCE:
        print("Loading objects...")
        if OBJECTS_SOURCE == "pkl.gz":
            objects = load_pkl_gz_result(OBJECTS_PATH)
        elif OBJECTS_SOURCE == "json":
            objects = read_objects_json(OBJECTS_PATH)

        if MAX_OBJECTS != -1:
            if isinstance(objects, MapObjectList):
                objects = MapObjectList(objects[:MAX_OBJECTS])
            elif isinstance(objects, dict):
                objects = dict(list(objects.items())[:MAX_OBJECTS])

        print("Creating Qdrant data...")
        create_qdrant_data_from_objects(objects)
    else:
        print("Loading LLM model...")
        llm_model = OpenAIChatGenerator(
            api_key=Secret.from_env_var("OPENROUTER_API_KEY"),
            api_base_url=os.getenv("OPENROUTER_API_BASE_URL"),
            model="x-ai/grok-4.1-fast",
        )
        query = input("Please provide your query: ")

        print("Querying relevant chunks...")
        relevant_chunks = query_relevant_chunks(query, 100, 0.001, True, 10)

        print(relevant_chunks)
        print("--------------------------------")

        context = json.dumps(
            [chunk[1] for chunk in relevant_chunks] if relevant_chunks else "{}",
            indent=2,
            ensure_ascii=False,
        )

        prompt = AGENT_PROMPT_V2 + context

        messages = [ChatMessage.from_system(prompt)]
        messages.append(ChatMessage.from_user(query))

        selected_object = None
        while True:
            response = infer_with_llm_model(llm_model, messages)
            print(response)

            messages.append(ChatMessage.from_assistant(response))
            if "<end_conversation>" in response:
                break
            elif "<selected_object>" in response:
                selected_object = (
                    response.split("<selected_object>")[1]
                    .split("</selected_object>")[0]
                    .replace("'", '"')
                )
                selected_object = json.loads(selected_object)
                break

            user_follow_up = input("Please provide your follow-up: ")
            messages.append(ChatMessage.from_user(user_follow_up))

        if selected_object is not None:
            print("--------------------------------")
            print(selected_object)
            obj_id = selected_object["id"]
            retrieved_object = list(
                filter(lambda x: "id" in x[1] and x[1]["id"] == obj_id, relevant_chunks)
            )
            print(retrieved_object)

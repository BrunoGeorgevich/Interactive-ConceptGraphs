from flashrank import Ranker, RerankRequest
from typing import List, Optional, Tuple
from joblib import Parallel, delayed
from functools import lru_cache
from datetime import datetime
import numpy as np
import traceback
import uuid
import json
import os

from agno.document import Document as AgnoDocument
from agno.embedder.openai import OpenAIEmbedder
from agno.vectordb.search import SearchType
from agno.vectordb.qdrant import Qdrant
from qdrant_client import QdrantClient

from conceptgraph.interaction.schemas import SystemConfig
from conceptgraph.interaction.utils import VectorDbError


def _process_info_to_doc(info: dict) -> Optional[AgnoDocument]:
    info_type = info.get("type", "")
    description = info.get("description", None)
    class_name = info.get("class_name", None)
    room_name = info.get("room_name", None)
    coordinates = info.get("coordinates", None)

    text_to_embed = f"{info_type.capitalize()} -> "

    if class_name:
        text_to_embed += f"{class_name} :"
    if description:
        text_to_embed += f" {description}"
    if room_name and room_name.strip().lower() != "unknown":
        text_to_embed += f" Located in {room_name}"
    if coordinates and isinstance(coordinates, (list, tuple)) and len(coordinates) == 3:
        text_to_embed += f" at coordinates {coordinates}"

    info_id = info.get("id", uuid.uuid4())
    info["id"] = str(info_id)
    info["timestamp"] = info.get(
        "timestamp", datetime.now().strftime("%H:%M:%S - %d of %B of %Y")
    )

    return AgnoDocument(content=text_to_embed, meta_data=info, id=str(info_id))


def _process_object_to_doc(obj_key: str, obj: dict) -> Optional[AgnoDocument]:
    """
    Helper function to convert an object dictionary into an AgnoDocument.
    Kept as a standalone function for pickling compatibility.

    :param obj_key: Unique key for the object.
    :type obj_key: str
    :param obj: Object data dictionary.
    :type obj: dict
    :return: AgnoDocument or None.
    :rtype: Optional[AgnoDocument]
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

    # if not object_tag.strip() or not object_caption.strip():
    #     return None

    room_context = f" Located in {obj.get('room_name', 'an unknown area')}."

    metadata = {}
    exclude_keys = {
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
    }

    for k, v in obj.items():
        if k not in exclude_keys:
            if isinstance(v, (str, int, float, bool)) or v is None:
                metadata[k] = v
            elif isinstance(v, (list, tuple)):

                metadata[k] = [x for x in v if isinstance(x, (str, int, float, bool))]

    positions = obj.get("pcd_np", None)
    if positions is None:
        positions = obj.get("bbox_np", None)

    if positions is not None:
        metadata["centroid"] = np.mean(positions, axis=0).tolist()

    text_to_embed = f"{object_tag}: {object_caption}{room_context}. Centroid at {metadata.get('centroid', 'unknown coordinates')}."

    class_id_values = obj.get("class_id", [])
    if isinstance(class_id_values, (list, tuple)) and class_id_values:
        try:
            counts = {}
            for val in class_id_values:
                if isinstance(val, int):
                    counts[val] = counts.get(val, 0) + 1
            if counts:
                metadata["class_id"] = max(counts, key=counts.get)
        except Exception:
            metadata["class_id"] = None
    elif isinstance(class_id_values, int):
        metadata["class_id"] = class_id_values

    metadata["id"] = str(obj.get("id", obj_key))

    return AgnoDocument(
        content=text_to_embed, meta_data=metadata, id=str(obj.get("id", obj_key))
    )


def _insert_single_doc(vector_db: Qdrant, doc: AgnoDocument) -> None:
    """Helper for parallel insertion."""
    try:
        vector_db.insert([doc])
    except Exception:
        pass


class SemanticMemoryEngine:
    """
    Manages semantic memory using Qdrant and RAG processes.
    """

    def __init__(self, config: SystemConfig) -> None:
        """
        Initializes the memory engine.

        :param config: System configuration.
        :type config: SystemConfig
        """
        self.config = config
        self.collection_name = f"House_{config.house_id:02d}_{config.prefix}"
        self.ad_collection_name = f"House_{config.house_id:02d}_{config.prefix}_AD"
        self.embedder = self._get_embedder(config.prefix)
        self.vector_db: Optional[Qdrant] = None

        try:
            self.vector_db = Qdrant(
                collection=self.collection_name,
                url=config.qdrant_url,
                embedder=self.embedder,
                distance="cosine",
                search_type=SearchType.hybrid,
            )
            self.additional_knowledge_db = Qdrant(
                collection=self.ad_collection_name,
                url=config.qdrant_url,
                embedder=self.embedder,
                distance="cosine",
                search_type=SearchType.hybrid,
            )
        except Exception as e:
            raise VectorDbError(f"Failed to initialize Qdrant: {e}") from e

    @lru_cache(maxsize=1)
    def _get_embedder(self, prefix: str) -> OpenAIEmbedder:
        """
        Returns configured embedder.

        :param prefix: Prefix configuration.
        :type prefix: str
        :return: OpenAIEmbedder instance.
        :rtype: OpenAIEmbedder
        """
        if prefix == "offline":
            return OpenAIEmbedder(
                id="text-embedding-qwen3-embedding-0.6b",
                api_key="",
                base_url="http://localhost:1234/v1",
                dimensions=1024,
            )
        else:
            return OpenAIEmbedder(
                id="qwen/qwen3-embedding-8b",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                dimensions=4096,
            )

    def ensure_collection_ready(
        self, vector_db: Qdrant, objects: List[dict] = []
    ) -> None:
        """
        Ensures the collection is populated. If empty or force_recreate is True, repopulates it.

        :param vector_db: Qdrant vector database instance.
        :type vector_db: Qdrant
        :param objects: List of enriched object dictionaries.
        :type objects: List[dict]
        :raises VectorDbError: If population fails.
        """
        if self.config.force_recreate_table:
            self._delete_collection(vector_db.collection)

        if not self._check_collection_valid(vector_db.collection):
            self._delete_collection(vector_db.collection)
            vector_db.create()
            if len(objects) > 0:
                self._populate_from_objects(objects)

    def _delete_collection(self, collection_name: str) -> None:
        """Deletes the current collection.

        :param collection_name: Name of the collection to delete.
        :type collection_name: str
        """
        try:
            client = QdrantClient(url=self.config.qdrant_url)
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
        except Exception as e:
            raise VectorDbError(f"Failed to delete collection: {e}") from e

    def _check_collection_valid(self, collection_name: str) -> bool:
        """Checks if collection exists and has correct dimension.

        :param collection_name: Name of the collection.
        :type collection_name: str
        :return: True if valid, False otherwise.
        :rtype: bool
        """
        try:
            client = QdrantClient(url=self.config.qdrant_url)
            if not client.collection_exists(collection_name):
                return False

            count_res = client.count(collection_name, exact=True)
            if count_res.count == 0:
                return False

            vector_config = client.get_collection(collection_name).config.params.vectors

            size = (
                vector_config.size
                if hasattr(vector_config, "size")
                else vector_config["dense"].size
            )

            return size == self.embedder.dimensions
        except Exception:
            return False

    def _populate_from_objects(self, objects: List[dict]) -> None:
        """Populates the vector database from a list of objects."""
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(_process_object_to_doc)(str(idx), obj)
            for idx, obj in enumerate(objects)
        )
        docs_to_insert = [d for d in results if d is not None]

        if not docs_to_insert:
            raise VectorDbError("No valid documents created from objects.")

        Parallel(n_jobs=-1, backend="threading")(
            delayed(_insert_single_doc)(self.vector_db, doc) for doc in docs_to_insert
        )

    def _rerank_with_flashrank(
        self, query: str, documents: List[AgnoDocument], top_k: int = 10
    ) -> List[AgnoDocument]:
        """
        Reranks documents using FlashRank.

        :param query: Query string.
        :type query: str
        :param documents: Documents to rerank.
        :type documents: List[AgnoDocument]
        :param top_k: Top K results.
        :type top_k: int
        :return: Reranked documents.
        :rtype: List[AgnoDocument]
        """
        if not documents:
            return []
        try:
            ranker = Ranker()
            passages = [
                {"id": idx, "text": doc.content} for idx, doc in enumerate(documents)
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            response = ranker.rerank(rerank_request)

            for idx, doc in enumerate(documents):
                doc.meta_data["previous_score"] = getattr(doc, "reranking_score", 0.0)

                doc.reranking_score = float(response[idx]["score"])

            return sorted(documents, key=lambda d: d.reranking_score, reverse=True)[
                :top_k
            ]
        except Exception:
            traceback.print_exc()
            return documents[:top_k]

    def add_additional_information(
        self,
        additional_info: List[dict],
    ) -> None:
        """
        Adds additional information to the additional knowledge database.

        :param additional_info: List of additional information dictionaries.
        :type additional_info: List[dict]
        :raises VectorDbError: If insertion fails.
        """
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(_process_info_to_doc)(info) for info in additional_info
        )
        docs_to_insert = [d for d in results if d is not None]

        if not docs_to_insert:
            print("No valid documents created from additional information.")
        else:
            Parallel(n_jobs=-1, backend="threading")(
                delayed(_insert_single_doc)(self.additional_knowledge_db, doc)
                for doc in docs_to_insert
            )

    def query_relevant_chunks(
        self,
        queries: List[str],
        rerank_query: Optional[str] = None,
        top_k: int = 30,
        confidence_threshold: float = 0.1,
        rerank_top_k: int = 10,
    ) -> List[Tuple[str, dict, float]]:
        """
        Queries the vector database for relevant chunks.

        :param queries: List of query strings.
        :type queries: List[str]
        :param rerank_query: Optional query for reranking.
        :type rerank_query: Optional[str]
        :param top_k: Initial top K retrieval.
        :type top_k: int
        :param confidence_threshold: Score threshold.
        :type confidence_threshold: float
        :param rerank_top_k: Top K after reranking.
        :type rerank_top_k: int
        :return: List of (Content, Metadata, Score).
        :rtype: List[Tuple[str, dict, float]]
        """

        def _retrieve_single(vector_db: Qdrant, q: str, top_k: int) -> list:
            """
            Retrieves search results for a single query using the configured search type.

            :param q: Query string.
            :type q: str
            :param top_k: Number of top results to retrieve.
            :type top_k: int
            :raises ValueError: If the search type is unsupported.
            :return: List of search result documents.
            :rtype: list
            """
            try:
                filters = None
                if vector_db.search_type == SearchType.vector:
                    results = vector_db._run_vector_search_sync(q, top_k, filters)
                elif vector_db.search_type == SearchType.keyword:
                    results = vector_db._run_keyword_search_sync(q, top_k, filters)
                elif vector_db.search_type == SearchType.hybrid:
                    results = vector_db._run_hybrid_search_sync(q, top_k, filters)
                else:
                    raise ValueError(
                        f"Unsupported search type: {vector_db.search_type}"
                    )

                search_results = vector_db._build_search_results(results, q)
                for sr, r in zip(search_results, results):
                    sr.reranking_score = r.score
                return search_results
            except (ValueError, AttributeError, TypeError) as e:
                traceback.print_exc()
                raise ValueError(f"Failed to retrieve search results: {e}") from e

        if self.config.use_additional_knowledge:
            results_batches = Parallel(n_jobs=-1, backend="threading")(
                delayed(_retrieve_single)(db, q, top_k)
                for q in queries
                for db in [self.vector_db, self.additional_knowledge_db]
            )
        else:
            results_batches = Parallel(n_jobs=-1, backend="threading")(
                delayed(_retrieve_single)(self.vector_db, q, top_k) for q in queries
            )

        flat_results = [doc for batch in results_batches for doc in batch]

        unique_docs = {}
        for doc in flat_results:
            doc_id = doc.meta_data.get("id")
            score = getattr(doc, "reranking_score", None)
            doc_type = doc.meta_data.get("type", None)
            if doc_id and (score or doc_type):
                if doc_id in unique_docs:
                    if doc_type:
                        unique_docs[doc_id] = doc
                    elif (
                        score
                        and doc_id in unique_docs
                        and score > unique_docs[doc_id].reranking_score
                    ):
                        unique_docs[doc_id] = doc
                else:
                    unique_docs[doc_id] = doc

        results = list(unique_docs.values())

        # if rerank_query:
        #     results = self._rerank_with_flashrank(
        #         rerank_query, results, top_k=rerank_top_k
        #     )

        final_chunks = []
        for doc in results:

            score = doc.reranking_score
            doc.meta_data["score"] = score

            if score >= confidence_threshold:
                final_chunks.append((doc.content, doc.meta_data, score))

        return final_chunks

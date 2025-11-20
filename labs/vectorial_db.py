from joblib import Parallel, delayed
from dotenv import load_dotenv
from typing import Any
import numpy as np
import traceback
import pickle
import json
import gzip
import cv2
import os

# Agno Imports
from agno.document import Document as AgnoDocument
from agno.embedder.openai import OpenAIEmbedder
from agno.models.openrouter import OpenRouter
from agno.vectordb.search import SearchType
from agno.vectordb.qdrant import Qdrant
from agno.agent import Agent

# FlashRank Import
from flashrank import Ranker, RerankRequest

# Custom Imports (Mantenha seus arquivos locais)
from conceptgraph.slam.slam_classes import MapObjectList
from prompts import AGENT_PROMPT_V2, INTENTION_INTERPRETATION_PROMPT

# --- Configurações Globais ---


def get_embedder():
    """
    Configura o Embedder para usar a API do OpenRouter (compatível com OpenAI).
    """
    return OpenAIEmbedder(
        id="qwen/qwen3-embedding-8b",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        dimensions=4096,
    )


def create_qdrant_vector_db(
    collection: str = "SemanticMap",
    url: str = "http://localhost:6333",
) -> Qdrant:
    """
    Cria e retorna uma instância de Agno Qdrant VectorDb.
    """
    try:
        vector_db = Qdrant(
            collection=collection,
            url=url,
            embedder=get_embedder(),
            distance="cosine",
            search_type=SearchType.hybrid,
        )
        vector_db.create()
        return vector_db
    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Failed to create Qdrant VectorDb: {e}")


def rerank_with_flashrank(
    query: str, documents: list[AgnoDocument], top_k: int = 10
) -> list[AgnoDocument]:
    """
    Rerank the provided AgnoDocument list using the FlashRank model.

    :param query: The query string to use for reranking.
    :type query: str
    :param documents: List of AgnoDocument objects to rerank.
    :type documents: list[AgnoDocument]
    :param top_k: Number of top documents to return after reranking.
    :type top_k: int
    :raises ValueError: If reranking fails due to invalid input or processing errors.
    :return: List of reranked AgnoDocument objects, each with updated rerank_score in meta_data.
    :rtype: list[AgnoDocument]
    """
    if not isinstance(query, str) or not isinstance(documents, list):
        raise ValueError(
            "Invalid input: query must be a string and documents must be a list."
        )

    if not documents:
        return []

    try:
        ranker = Ranker()
        docs_to_rerank = [
            {"id": idx, "text": doc.content} for idx, doc in enumerate(documents)
        ]
        rerank_request = RerankRequest(query=query, passages=docs_to_rerank)
        response = ranker.rerank(rerank_request)

        for i, doc in enumerate(documents):
            try:
                doc.meta_data["rerank_score"] = float(response[i]["score"])
                doc.reranking_score = float(response[i]["score"])
            except (KeyError, IndexError, TypeError, ValueError) as e:
                traceback.print_exc()
                raise ValueError(
                    f"Failed to update rerank_score for document at index {i}: {str(e)}"
                ) from e

        reranked_docs = sorted(
            documents,
            key=lambda d: d.reranking_score,
            reverse=True,
        )[:top_k]

        return reranked_docs

    except (ValueError, KeyError, IndexError, TypeError) as e:
        traceback.print_exc()
        raise ValueError(f"Error during FlashRank reranking: {e}") from e


def sanitize_metadata(data: dict) -> dict:
    """
    Remove valores não primitivos recursivamente para garantir compatibilidade com JSON.
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
    Processa um objeto e retorna um AgnoDocument.
    """
    if not isinstance(obj, dict):
        pass

    object_tag = obj.get("object_tag", "")
    if not object_tag and "class_name" in obj:
        object_tag = obj.get("class_name", "")

    object_caption = obj.get("object_caption", "")
    if not object_caption and "consolidated_caption" in obj:
        object_caption = obj.get("consolidated_caption", "")

    object_caption = (
        object_caption.replace("'", '"')
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        object_caption = json.loads(object_caption).get("consolidated_caption", "")
    except json.decoder.JSONDecodeError:
        print("Caption is not valid JSON, using raw text.")

    if not isinstance(object_tag, str) or not object_tag.strip():
        return None
    if not isinstance(object_caption, str) or not object_caption.strip():
        return None

    text_to_embed = f"{object_tag}: {object_caption}"

    # Prepara Metadata
    metadata = {k: v for k, v in obj.items() if k not in ["embedding", "bbox"]}

    # Trata bbox especificamente se existir
    if "bbox" in obj and obj["bbox"] is not None:
        bbox = obj["bbox"]
        metadata["bbox_extent"] = list(bbox.extent)
        metadata["bbox_center"] = list(bbox.center)
        metadata["bbox_volume"] = bbox.volume()

    metadata = sanitize_metadata(metadata)

    # Cria documento Agno (ATENÇÃO: parametro é meta_data)
    return AgnoDocument(
        content=text_to_embed, meta_data=metadata, id=str(obj.get("id", obj_key))
    )


def retrieve(vector_db: Qdrant, query: str, limit: int = 100) -> list[Any]:
    """
    Retrieve data from the Qdrant database.

    :param query: The query to search in the database.
    :type query: str
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
        if sr.content != r.payload["content"]:
            raise ValueError(
                "Content mismatch: {} != {}".format(sr.content, r.payload["content"])
            )
        sr.reranking_score = r.score

    return search_results


def create_qdrant_data_from_objects(objects: dict) -> int:
    """
    Cria documentos no Qdrant usando Agno.
    """
    try:
        vector_db = create_qdrant_vector_db()

        # Opcional: Se quiser recriar a coleção do zero a cada run:
        # vector_db.delete()
        # vector_db.create()

        docs_to_insert = []

        if isinstance(objects, MapObjectList):
            for i, obj in enumerate(objects):
                doc = process_object_to_doc(str(i), obj)
                if doc:
                    docs_to_insert.append(doc)
        elif isinstance(objects, dict):
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(process_object_to_doc)(k, v) for k, v in objects.items()
            )
            docs_to_insert = [d for d in results if d is not None]
        else:
            raise ValueError("Input must be a dict or a MapObjectList object.")

        if not docs_to_insert:
            raise ValueError("No documents created from input objects.")

        print(f"Inserting {len(docs_to_insert)} documents into Qdrant (Agno)...")

        def insert_single_doc(doc):
            """Insert a single AgnoDocument into the vector database.

            :param doc: Document to insert.
            :type doc: AgnoDocument
            :raises ValueError: If document insertion fails due to invalid data.
            :raises TypeError: If document is of invalid type.
            :raises KeyError: If required metadata keys are missing.
            """
            try:
                vector_db.insert([doc])
            except (ValueError, TypeError, KeyError) as e:
                traceback.print_exc()
                raise ValueError(
                    f"Failed to insert single document id={getattr(doc, 'id', 'unknown')}: {str(e)}"
                ) from e

        Parallel(n_jobs=-1, backend="threading")(
            delayed(insert_single_doc)(doc) for doc in docs_to_insert
        )

        return len(docs_to_insert)

    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Error in Qdrant data creation: {e}")


def query_relevant_chunks(
    queries: str | list[str],
    rerank_query: str | None = None,
    top_k: int = 10,
    confidence_threshold: float = 0.1,
    rerank: bool = True,
    rerank_top_k: int = 10,
) -> list[tuple[str, dict, float]]:
    """
    Recupera chunks usando Agno VectorDb e reranqueia com FlashRank.
    """
    try:
        vector_db = create_qdrant_vector_db()

        if isinstance(queries, str):
            queries = [queries]

        results = []
        for q in queries:
            retrieved_docs = retrieve(vector_db, q, limit=top_k)
            results.extend(retrieved_docs)

        unique_docs = {}
        for doc in results:
            doc_id = getattr(doc, "id", None)
            score = getattr(doc, "reranking_score", None)
            if doc_id is None:
                continue
            if doc_id not in unique_docs or (
                score is not None
                and (
                    unique_docs[doc_id].reranking_score is None
                    or score > unique_docs[doc_id].reranking_score
                )
            ):
                unique_docs[doc_id] = doc
        results = list(unique_docs.values())

        if not results:
            return []

        # Reranking com FlashRank
        if rerank:
            results = rerank_with_flashrank(rerank_query, results, top_k=rerank_top_k)

        final_chunks = []
        for doc in results:
            # Prioriza o score do rerank, senão tenta pegar do vector search (se disponível)
            score = doc.meta_data.get("rerank_score", None)

            # Se não tiver score (ex: busca vetorial pura sem métrica exposta), assume 1.0
            if score is None:
                score = 1.0
            else:
                score = float(score)

            if score >= confidence_threshold:
                final_chunks.append((doc.content, doc.meta_data, score))

        return final_chunks

    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Error during query retrieval: {e}")


# --- Funções de Mapa (Inalteradas) ---


def load_voronoi_map(filepath: str) -> tuple[dict, np.ndarray, dict, int, int] | None:
    try:
        if not filepath.endswith(".pkl.gz"):
            filepath += ".pkl.gz" if not filepath.endswith(".pkl") else ".gz"

        with gzip.open(filepath, "rb") as f:
            voronoi_data = pickle.load(f)

        return (
            voronoi_data["merged_regions"],
            voronoi_data["region_mask"],
            voronoi_data["merged_colors"],
            voronoi_data["width"],
            voronoi_data["height"],
        )
    except Exception as e:
        print(f"Error loading map: {e}")
        return None


def reconstruct_voronoi_image(
    merged_regions, region_mask, merged_colors, width, height
) -> np.ndarray:
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
                cls = data.get("dominant_class", "unknown")
                cv2.putText(
                    reconstructed_image,
                    cls,
                    (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 0),
                    2,
                )  # Borda
                cv2.putText(
                    reconstructed_image,
                    cls,
                    (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                )

        return reconstructed_image
    except Exception:
        return np.zeros((height, width, 3), dtype=np.uint8)


def read_objects_json(file_path: str = "objects.json") -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pkl_gz_result(result_path: str) -> MapObjectList:
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)
    objects = MapObjectList()
    objects.load_serializable(results["objects"])
    return objects


# --- Main Execution ---

if __name__ == "__main__":
    load_dotenv()

    # Configuração do Agente Agno
    llm_model_id = "x-ai/grok-4.1-fast"

    # Instancia o agente (Stateful)
    agent = Agent(
        model=OpenRouter(
            id=llm_model_id,
            api_key=os.getenv("OPENROUTER_API_KEY"),
        ),
        markdown=True,
        description="You are a helpful spatial assistant.",
    )

    INFERENCE = True
    OBJECTS_PATH = "r_mapping_6_stride15/pcd_r_mapping_6_stride15.pkl.gz"
    MAP_PATH = "voronoi_map.pkl.gz"

    # Mapa Voronoi
    map_data = load_voronoi_map(MAP_PATH)
    if map_data:
        merged_regions, region_mask, merged_colors, width, height = map_data
        voronoi_map = reconstruct_voronoi_image(
            merged_regions, region_mask, merged_colors, width, height
        )
        cv2.imwrite("voronoi_map.png", voronoi_map)

    MAX_OBJECTS = -1

    if not INFERENCE:
        print("Loading objects...")
        if OBJECTS_PATH.endswith(".pkl.gz"):
            objects = load_pkl_gz_result(OBJECTS_PATH)
        else:
            objects = read_objects_json(OBJECTS_PATH)

        if MAX_OBJECTS != -1:
            if isinstance(objects, MapObjectList):
                objects = MapObjectList(objects[:MAX_OBJECTS])
            elif isinstance(objects, dict):
                objects = dict(list(objects.items())[:MAX_OBJECTS])

        print("Creating Qdrant data (Agno + OpenRouter Embeddings)...")
        create_qdrant_data_from_objects(objects)

    else:
        query = input("Please provide your query: ")

        agent.instructions = INTENTION_INTERPRETATION_PROMPT
        result = json.loads(agent.run(query).content)
        queries = result.get("rag_queries", [])
        rerank_query = result.get("rerank_query", query)

        print("Querying relevant chunks...")
        relevant_chunks = query_relevant_chunks(
            queries,
            rerank_query,
            top_k=10,
            confidence_threshold=0.1,
            rerank=True,
            rerank_top_k=10,
        )

        print(f"Found {len(relevant_chunks)} relevant chunks after FlashRank.")
        print("--------------------------------")

        context_json = json.dumps(
            [chunk[1] for chunk in relevant_chunks] if relevant_chunks else {},
            indent=2,
            ensure_ascii=False,
        )

        # Injeta o contexto no Prompt
        system_prompt = AGENT_PROMPT_V2 + "\n\nContext:\n" + context_json

        print("Starting Chat...")

        # Reinicia o agente com as instruções contendo o contexto RAG
        agent = Agent(
            model=OpenRouter(
                id=llm_model_id,
                api_key=os.getenv("OPENROUTER_API_KEY"),
            ),
            instructions=system_prompt,
            markdown=True,
        )

        # Primeira chamada
        response_gen = agent.run(query)
        response_text = response_gen.content
        print(response_text)

        selected_object = None

        # Loop de conversação
        while True:
            if "<end_conversation>" in response_text:
                break
            elif "<selected_object>" in response_text:
                try:
                    raw_json = (
                        response_text.split("<selected_object>")[1]
                        .split("</selected_object>")[0]
                        .replace("'", '"')
                    )
                    selected_object = json.loads(raw_json)
                    break
                except Exception as e:
                    print(f"Error parsing selected object: {e}")

            user_follow_up = input("Please provide your follow-up: ")

            # O agente mantém a memória da conversa automaticamente
            response_gen = agent.run(user_follow_up)
            response_text = response_gen.content
            print(response_text)

        if selected_object is not None:
            print("--------------------------------")
            print("Selected Object:", selected_object)
            obj_id = str(selected_object.get("id"))

            retrieved_object = list(
                filter(lambda x: str(x[1].get("id")) == obj_id, relevant_chunks)
            )
            print("Retrieved Original Metadata:", retrieved_object)

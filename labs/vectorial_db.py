from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.embedders import OpenAITextEmbedder
from haystack.document_stores.types import DuplicatePolicy
from semantic_text_splitter import TextSplitter
from haystack.dataclasses import ChatMessage
from joblib import Parallel, delayed
from haystack.utils import Secret
from dotenv import load_dotenv
from haystack import Document
from textwrap import dedent
import traceback
import pickle
import json
import gzip
import os

# LOCAL IMPORTS
from conceptgraph.slam.slam_classes import MapObjectList


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

    :param url: The Qdrant instance URL (e.g., "http://localhost:6333").
    :type url: str
    :param index: The name of the index (collection) to use.
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


def process_document_objects(
    obj_key: str, obj: dict, embedder: OpenAITextEmbedder
) -> Document | None:
    """
    Processes a single object to generate a Document with embedding.

    :param obj_key: The key of the object in the dictionary.
    :type obj_key: str
    :param obj: The object dictionary.
    :type obj: dict
    :param embedder: The OpenAITextEmbedder instance.
    :type embedder: OpenAITextEmbedder
    :raises ValueError: If the object is not a dictionary or embedding is missing.
    :return: A Document instance with embedding, or None if object_tag or object_caption is invalid.
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

    embedding_result = embedder.run(text_to_embed)
    if "embedding" not in embedding_result:
        raise ValueError("Embedding not found in OpenAITextEmbedder result.")
    doc = Document(
        content=text_to_embed,
        meta=metadata,
    )
    doc.embedding = embedding_result["embedding"]
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
        # Ignore all other types (including methods, objects, etc.)
    return sanitized


def create_qdrant_data_from_objects(objects: dict) -> int:
    """
    Creates Qdrant documents from either a dictionary of objects or a MapObjectList.
    For a dictionary, generates an embedding for the string 'object_tag: object_caption' using OpenAITextEmbedder,
    and writes the resulting documents to QdrantDocumentStore. For a MapObjectList, processes each object to extract
    a representative string (using 'class_name' and 'consolidated_caption' if available), generates embeddings,
    and writes to the store. All other fields are added as metadata to the document, and the complete object is included
    in the metadata under the key 'full_object'.

    :param objects: Either a dictionary where each key is an object identifier and the value is a dictionary containing
                   'object_tag', 'object_caption', and other metadata fields, or a MapObjectList instance.
    :type objects: dict
    :raises ValueError: If embedding, writing, or counting fails, or if the input type is unsupported.
    :return: The number of documents in the store after insertion.
    :rtype: int

    This function generates an embedding for the string 'object_tag: object_caption' (for dict) or
    'class_name: consolidated_caption' (for MapObjectList) of each object. All other fields are preserved as metadata,
    and the complete object is included in the metadata.
    """
    try:
        document_store = create_qdrant_document_store()
        embedder = OpenAITextEmbedder(model="text-embedding-3-small")
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

                # Only keep the most relevant fields, similar to objects.json
                metadata = {
                    "id": object_id,
                    "object_tag": object_tag,
                    "object_caption": object_caption,
                    "bbox_extent": bbox_extent,
                    "bbox_center": bbox_center,
                    "bbox_volume": bbox_volume,
                }

                # Remove any non-primitive or callable values from metadata to ensure serialization

                metadata = sanitize_metadata(metadata)

                embedding_result = embedder.run(text_to_embed)
                if "embedding" not in embedding_result:
                    raise ValueError(
                        "Embedding not found in OpenAITextEmbedder result."
                    )
                doc = Document(
                    content=text_to_embed,
                    meta=metadata,
                )
                doc.embedding = embedding_result["embedding"]
                documents_with_embeddings.append(doc)
        elif isinstance(objects, dict):
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(process_document_objects)(obj_key, obj, embedder)
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
        raise ValueError(
            f"Error in Qdrant data creation from objects with OpenAITextEmbedder: {e}"
        )
    except (RuntimeError, OSError) as e:
        traceback.print_exc()
        raise ValueError(
            f"Runtime or OS error in Qdrant data creation from objects with OpenAITextEmbedder: {e}"
        )


def create_qdrant_data_from_texts(texts: list[str]) -> int:
    """
    Splits input texts into semantic chunks using semantic-text-splitter, generates embeddings for each chunk
    using OpenAITextEmbedder, and writes the resulting documents to QdrantDocumentStore.

    :param texts: A list of strings to be split and written to the document store.
    :type texts: list[str]
    :raises ValueError: If text splitting, embedding, writing, or counting fails.
    :return: The number of documents in the store after insertion.
    :rtype: int

    This function uses semantic-text-splitter's TextSplitter to split texts into semantically meaningful chunks.
    See: https://pypi.org/project/semantic-text-splitter/
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

        embedder = OpenAITextEmbedder(model="text-embedding-3-small")
        documents_with_embeddings = []
        for chunk in all_chunks:
            embedding_result = embedder.run(chunk)
            if "embedding" not in embedding_result:
                raise ValueError("Embedding not found in OpenAITextEmbedder result.")
            doc = Document(content=chunk)
            doc.embedding = embedding_result["embedding"]
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
            f"Error in Qdrant data creation with semantic-text-splitter and OpenAITextEmbedder: {e}"
        )
    except (RuntimeError, OSError) as e:
        traceback.print_exc()
        raise ValueError(
            f"Runtime or OS error in Qdrant data creation with semantic-text-splitter and OpenAITextEmbedder: {e}"
        )


def query_relevant_chunks(
    query: str, top_k: int = 5, confidence_threshold: float = 0.1
) -> list[tuple[str, dict, float]]:
    """
    Retrieves the most relevant document chunks for a given query using OpenAITextEmbedder and QdrantDocumentStore's _query_by_embedding,
    filtering results by a minimum confidence threshold. Each returned chunk is a tuple containing the text, its metadata, and its confidence score.

    :param query: The search query string to embed and search for relevant chunks.
    :type query: str
    :param top_k: The maximum number of relevant chunks to retrieve.
    :type top_k: int
    :param confidence_threshold: The minimum confidence score required for a chunk to be considered relevant (range: 0.0 to 1.0).
    :type confidence_threshold: float
    :raises ValueError: If the retrieval process fails, embedding fails, or the pipeline cannot be constructed.
    :return: A list of tuples, each containing a relevant document chunk as a string, its metadata as a dict, and its confidence score as a float, filtered by the confidence threshold.
    :rtype: list[tuple[str, dict, float]]

    This function assumes that the QdrantDocumentStore is already populated and accessible.
    The OpenAITextEmbedder will use the OPENAI_API_KEY environment variable or a configured API key.
    For more details, see: https://docs.haystack.deepset.ai/docs/openaitextembedder
    """
    try:
        document_store = create_qdrant_document_store()

        all_documents = document_store.filter_documents(filters=None)
        if not all_documents:
            return []

        embedder = OpenAITextEmbedder(model="text-embedding-3-small")
        query_embedding_result = embedder.run(query)
        query_embedding = query_embedding_result["embedding"]

        if not hasattr(document_store, "_query_by_embedding"):
            raise AttributeError(
                "QdrantDocumentStore does not have a _query_by_embedding method."
            )

        retrieved_documents = document_store._query_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # Support both dict and list return types
        documents = (
            retrieved_documents.get("documents", [])
            if isinstance(retrieved_documents, dict)
            else retrieved_documents
        )

        if not documents:
            return []

        # Filter by confidence threshold if available, and return (content, metadata, score) tuples
        relevant_chunks = []
        for doc in documents:
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
            # If no score is available, include with score 1.0 by default
            if score is None:
                score = 1.0
            if content is not None and score >= confidence_threshold:
                relevant_chunks.append((content, meta, float(score)))
        return relevant_chunks

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

    For more details, see: https://docs.haystack.deepset.ai/docs/OpenAIChatGenerator
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
    Loads the result file and returns objects, background objects, and class colors.

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


if __name__ == "__main__":
    load_dotenv()

    INFERENCE = True
    # OBJECTS_PATH = "D:\\Documentos\\Datasets\\Replica\\room0\\exps\\r_mapping_stride11\\pcd_r_mapping_stride11.pkl.gz"
    # "D:\\Documentos\\Datasets\\Robot@VirtualHome\\Home01\\CustomWandering3\\exps\\r_mapping_stride13\\pcd_r_mapping_stride13.pkl.gz"
    OBJECTS_PATH = "D:\\Documentos\\Datasets\\Robot@VirtualHome\\Home01\\CustomWandering3\\exps\\r_mapping_stride14\\pcd_r_mapping_stride14.pkl.gz"
    # OBJECTS_PATH = "objects.json"
    MAX_OBJECTS = -1  # -1 to get all objects

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
                # Retain only the first MAX_OBJECTS key-value pairs in the dictionary
                objects = dict(list(objects.items())[:MAX_OBJECTS])

        print("Creating Qdrant data...")
        create_qdrant_data_from_objects(objects)
    else:
        print("Loading LLM model...")
        llm_model = OpenAIChatGenerator(
            api_key=Secret.from_token(os.getenv("GLAMA_API_KEY")),
            api_base_url=os.getenv("GLAMA_API_BASE_URL"),
            # model="gemini-2.0-flash-001",
            model="gemini-2.5-flash-preview",
            # model="gpt-4.1-mini",
        )
        query = input("Please provide your query: ")
        # query = "i'm tired."
        # query = "I want to see television."

        print("Querying relevant chunks...")
        relevant_chunks = query_relevant_chunks(query, 40, 0.1)
        print(relevant_chunks)
        print("--------------------------------")

        context = json.dumps(
            [chunk[1] for chunk in relevant_chunks] if relevant_chunks else "{}",
            indent=2,
            ensure_ascii=False,
        )

        prompt = dedent(
            f"""
            You are a skilled personal assistant capable of analyzing information about a home, such as its list of objects.
            Your main mission is to analyze a database of objects, interpret the user's intent from the provided query, and accurately identify which objects are most relevant to fulfill the presented request.

            The output MUST be in the same language as the user's query. Identify the language of the query and ensure that all blocks and responses are written in that language. This is mandatory for every response. The output must be defined with the specified tags in the prompt structure, do not modify that. The JSON parts must NOT be surrounded by ```json ... ``` tags. Present all JSON data directly without any code block formatting.

            Your response must strictly follow the structure below, always including the <language>, <user_intention>, <think>, and <relevant_objects> blocks in the output, regardless of the scenario:

            <language>
            Identify and state the language used in the user's query (for example: English, Portuguese, Spanish, etc.).
            </language>

            <user_intention>
            Analyze the user's query to identify and clearly articulate their main intent. Determine whether the request is conceptual (not requiring physical interaction with objects) or if it involves locating or interacting with specific objects. Consider multiple interpretation levels: direct requests ("I want to watch TV"), implicit needs ("I'm tired" might suggest a place to rest), or abstract concepts ("I need information" might relate to devices or books). Evaluate if the user is seeking an object directly or something functionally related to objects in the database. Pay attention to emotional states or needs expressed that might indicate specific object requirements. If the query mentions people or activities, connect these to relevant objects that would support those interactions. Provide a comprehensive analysis that captures both explicit statements and implicit needs to accurately identify the most relevant objects.
            </user_intention>

            <think>
            Analyze all provided objects, correlating each one with the user's intent identified in the <user_intention> block and with the user's query. Consider attributes such as 'object_tag', 'object_caption', 'bbox_extent', 'bbox_center', and 'bbox_volume' to identify relationships, similarities, or relevant distinctions between the objects and the intent expressed in the query. Highlight possible ambiguities, overlaps, or information gaps that may impact the selection of the most suitable object. It is important to analyze that there are objects related to the action, but that are not capable of performing the action themselves, for example, the door handle is related to the action of opening the door, but it cannot perform the action by itself. Exclude these objects from the list if there are more relevant objects.
            </think>

            <relevant_objects>
            Filter and list only the objects that are relevant to the user's query. Group these objects based on their spatial proximity using "bbox_center" and "bbox_extent" to avoid repetition. For each group, select the most representative object and include its dictionary with the attributes 'id', 'object_tag', 'object_caption', 'bbox_center', 'bbox_extent', 'bbox_volume' from the actual object collected from the context, and briefly justify its relevance based on the analyzed attributes. Use the following structure for each representative object:
            {{
                'id': ...,
                'justification': Brief justification for this object's relevance, including mention if it represents a group of similar objects,
                'object_tag': ...,
                'object_caption': ...,
                'bbox_center': ...,
                'bbox_extent': ...,
                'bbox_volume': ...
            }}
            
            </relevant_objects>

            <selected_object>
            If there is an object sufficient to meet the user's need, return the dictionary of the selected object. If multiple objects are relevant, analyze their position in the environment and if they are close, return the dictionary of the most relevant object. The dictionary MUST contain the attributes 'id', 'object_tag', 'object_caption', 'bbox_center', 'bbox_extent', 'bbox_volume' from the actual object collected from the context. These attributes are mandatory and must always be included. An answer must be provided to the user following the same language as the user's query. This answer must be concise and to the point, telling to the user what and why the selected object is the most relevant to the user's query. Do not describe the object in an unusual way, if you are going to talk about its features, try to sound natural. Avoid talking about geometric shapes. The answer must be natural for a human. Do not generate false information, only return data that exists in the collected object. Use the following structure:
            {{
                'id': ...,
                'answer': ...,
                'object_tag': ...,
                'object_caption': ...,
                'bbox_center': ...,
                'bbox_extent': ...,
                'bbox_volume': ...
            }}
            </selected_object>

            <follow_up>
            The follow up question must make sense with the user's query and with the user's intent identified in the <user_intention> block. It should be elaborated in order to help the user select the most suitable object.
            The output must be in the same language as the user's query. If more information is needed to provide options to the user, the output should be a question requesting this information without listing options, focusing on filtering the current options. This question must be strongly connected to the user's intention.
            
            EXTREMELY IMPORTANT: All interactions with the user MUST be completely natural and conversational, as if speaking with another human. NEVER use technical terms, coordinates, geometric descriptions, or any language that only computers or robots would understand. Humans don't think in terms of coordinates, dimensions, or geometric shapes - they understand relative positions (like "next to", "in front of", "behind"), visual descriptions, and everyday language. Always describe objects and locations in ways that are intuitive and easily visualized by humans. For example, instead of saying "the object at coordinates [2.3, 1.5, 0.8]", say "the lamp on the side table next to the couch". Instead of "a rectangular prism with dimensions 0.5x0.3x0.2", say "a small box". Use language that any person would naturally understand in everyday conversation.
            
            When sufficient information is available to present options, the output must have a final question asking the user to select one of the options. The output must have the following structure:
            {{
                [A simple introduction to the follow up question before presenting the options]
                [The word for "Option" in the user's language] 1: [Object 1 described in natural, human-friendly terms]
                [The word for "Option" in the user's language] 2: [Object 2 described in natural, human-friendly terms]
                ...
                [The word for "Option" in the user's language] N: [Object N described in natural, human-friendly terms]
                [Final question asking the user to select one of the options in a conversational way]
            }}
            If more information is needed before presenting options, use this structure instead:
            {{
                [A clear and natural question that a human would understand, using everyday language and avoiding any technical terminology, requesting specific information to help filter the available options, strongly connected to the user's intention]
            }}
            </follow_up>

            <requested_information>
            If the user's intent is to obtain information about something (like asking where something is located), provide a detailed response based on the analysis of relevant objects in the environment. Use the spatial information (bbox_center, bbox_extent) of the objects to determine locations and provide clear directions or descriptions to the user. Focus on contextual relationships between objects (e.g., "next to the bookshelf," "in front of the window") rather than raw coordinates. Prioritize information that directly answers the user's query, emphasizing object functionality, appearance, and relative position. Avoid providing technical details like coordinates, center points, or volumetric measurements to the user. If the request relates to a specific object, provide relevant information about that object's features and purpose. Be extremely precise and concise with your answer. Do not fabricate information - only use data that exists in the collected objects. The response must be informative, accurate, directly address the user's query, and be in the same language as the user's query. The output must have the following structure:
            {{
                "answer": [A precise and concise answer to the user's query that directly addresses what they want. If unable to provide the requested information, clearly state why and offer alternative assistance. Never leave the user without a clear response or with vague information. If coordinates are provided, include a natural way to inform the user that they will be guided to the location. The interaction with the user must be natural and human-like, avoiding explicit coordinates or technical characteristics that would not be understood by a typical human. Use contextual descriptions like 'near the window' or 'next to the bookshelf' rather than numerical positions. Ensure the language is consistent with the user's query and maintains a conversational tone.],
                "coordinates": [The coordinates that the user should go. If the user's request is more conceptual and does not require him to go to a specific place, the answer must be 'null'. If the user's intention is related to go to somewhere or to see something, it must be provided a coordinates to go.]
            }}
            </requested_information>

            <end_conversation>
            This block should be used when the conversation needs to be concluded. If the user indicates they want to end the conversation or if the requested information is sufficient to complete the interaction, include this block in your response. If the conversation should simply end without additional content, leave this block empty. If a final message to the user is needed, include that message within this block. The output must be in the same language as the user's query and should provide a natural conclusion to the conversation without suggesting any follow-up questions.
            </end_conversation>

            After the above blocks, follow these rules for the final answer (always in the language of the query):
            1. If there is no relevant object, return: <no_object>No object found. Please provide more details about what you are looking for.</no_object>
            2. If there are multiple potentially relevant objects, return a follow-up question in the format <follow_up>...</follow_up>, presenting clear options that allow the user to differentiate between the possible objects (for example, highlighting differences in 'object_tag', 'object_caption', or other relevant attributes).
            3. If any of the collected objects is sufficient, return <selected_obj>object dictionary</selected_obj>.
            4. If the user's intent is to obtain information about something (like locations, descriptions, etc.), return <requested_information>detailed response based on object analysis</requested_information>.
            5. If the user indicates they want to end the conversation or if the requested information is sufficient to complete the interaction, include <end_conversation>...</end_conversation> after the appropriate response block.
            6. The output must be ONLY ONE of: <selected_object>, <follow_up>, <requested_information>, or a combination of <requested_information> or <selected_object> followed by <end_conversation> if the conversation should end.
            7. The follow up question must detail the possible objects that can be selected, allowing the user to answer by choosing one of the options.

            Do not include explanations, justifications, or any other text besides the <language>, <think>, <relevant_objects> blocks and the final answer (<no_object>, <follow_up>, <selected_obj>, <requested_information>, or <end_conversation>).

            IMPORTANT: The JSON parts must NOT be surrounded by ```json ... ``` tags. Present all JSON data directly without any code block formatting.

            EXTREMELY IMPORTANT: Remember that you are communicating with a human who does not understand technical language, coordinates, or geometric descriptions. All your responses must be in natural, conversational language that any person would understand. Never mention coordinates, dimensions, or technical specifications in your direct communication with the user. Always translate technical information into everyday language and descriptions that relate to how humans naturally perceive and navigate their environment.

            Structure of an object in the database:
            {{
                'id': ..., # Unique identifier for the object with the following format: "id": "bc720df9-3082-415b-a124-351894aa1b61"
                'object_tag': ..., # Object tag with the following format: "object_tag":"nightstand"
                'object_caption': ..., # Object caption with the following format: "object_caption": "object_caption":"A small rectangular dark brown wooden nightstand with three drawers and silver handles."
                'bbox_extent': [..., ..., ...], # Bounding box extent with the following format: "bbox_extent": [0.0, 0.0, 0.0]
                'bbox_center': [..., ..., ...], # Bounding box center with the following format: "bbox_center": [0.0, 0.0, 0.0]
                'bbox_volume': ... # Bounding box volume with the following format: "bbox_volume": 0.0
            }}
            
            Context:
            {context}
        """
        )

        # Create messages list here rather than in the function
        messages = [ChatMessage.from_system(prompt)]
        messages.append(ChatMessage.from_user(query))

        selected_object = None
        # Loop until we get a response containing <selected_object>
        while True:
            # Get response using the updated function
            response = infer_with_llm_model(llm_model, messages)
            print(response)

            # Add the assistant's response to the messages list
            messages.append(ChatMessage.from_assistant(response))
            # Check if we have a selected object in the response
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

            # Allow the user to provide a follow-up
            user_follow_up = input("Please provide your follow-up: ")
            messages.append(ChatMessage.from_user(user_follow_up))

        if selected_object is not None:
            print("--------------------------------")
            print(selected_object)
            obj_id = selected_object["id"]
            retrieved_object = list(
                filter(lambda x: x[1]["id"] == obj_id, relevant_chunks)
            )
            print(retrieved_object)

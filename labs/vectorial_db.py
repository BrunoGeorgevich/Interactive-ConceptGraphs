from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.embedders import OpenAITextEmbedder
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.generators import OpenAIGenerator
from semantic_text_splitter import TextSplitter
from joblib import Parallel, delayed
from haystack.utils import Secret
from dotenv import load_dotenv
from haystack import Document
from textwrap import dedent
import traceback
import json
import os


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


def create_qdrant_data_from_objects(objects: dict) -> int:
    """
    For each object in the input dictionary, generates an embedding for the string
    'object_tag: object_caption' using OpenAITextEmbedder, and writes the resulting
    documents to QdrantDocumentStore. All other fields are added as metadata to the document,
    and the complete object is included in the metadata under the key 'full_object'.

    :param objects: A dictionary where each key is an object identifier and the value is a dictionary containing
                   'object_tag', 'object_caption', and other metadata fields.
    :type objects: dict
    :raises ValueError: If embedding, writing, or counting fails.
    :return: The number of documents in the store after insertion.
    :rtype: int

    This function generates an embedding for the string 'object_tag: object_caption' of each object.
    All other fields are preserved as metadata, and the complete object is included in the metadata.
    """
    try:
        document_store = create_qdrant_document_store()
        embedder = OpenAITextEmbedder(model="text-embedding-3-small")

        documents_with_embeddings = []

        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(process_document_objects)(obj_key, obj, embedder)
            for obj_key, obj in objects.items()
        )
        documents_with_embeddings.extend([doc for doc in results if doc is not None])

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
    llm_model: OpenAIGenerator,
    relevant_chunks: list[tuple[str, dict, float]],
    prompt: str,
) -> str:
    """
    Performs inference using the provided OpenAIGenerator model, relevant context, and prompt.

    :param llm_model: The OpenAIGenerator instance for text generation.
    :type llm_model: OpenAIGenerator
    :param relevant_chunks: The context or relevant text chunks to provide to the model, as a list of (text, metadata, confidence) tuples.
    :type relevant_chunks: list[tuple[str, dict, float]]
    :param prompt: The prompt to send to the model.
    :type prompt: str
    :raises ValueError: If inference fails due to input or model errors.
    :return: The generated response from the language model.
    :rtype: str

    For more details, see: https://docs.haystack.deepset.ai/docs/openaigenerator
    """
    try:
        if not isinstance(relevant_chunks, list) or not all(
            isinstance(chunk, tuple)
            and len(chunk) == 3
            and isinstance(chunk[0], str)
            and isinstance(chunk[1], dict)
            and isinstance(chunk[2], float)
            for chunk in relevant_chunks
        ):
            raise ValueError(
                "relevant_chunks must be a list of (str, dict, float) tuples."
            )
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string.")

        # Compose the full prompt with context (only the text part)
        context = (
            "\n".join(chunk[0] for chunk in relevant_chunks) if relevant_chunks else ""
        )
        full_prompt = f'{prompt}\n\nContexto:"\n{context}\n"' if context else prompt

        # Run inference using the OpenAIGenerator
        result = llm_model.run(full_prompt)
        if isinstance(result, dict) and "replies" in result and result["replies"]:
            return result["replies"][0]
        elif isinstance(result, str):
            return result
        else:
            raise ValueError(
                "Unexpected response format from OpenAIGenerator inference."
            )
    except (ValueError, AttributeError, KeyError, TypeError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to perform inference with OpenAIGenerator: {e}")
    except (RuntimeError, OSError) as e:
        traceback.print_exc()
        raise ValueError(f"Runtime or OS error during OpenAIGenerator inference: {e}")


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


if __name__ == "__main__":
    load_dotenv()

    # objects = read_objects_json()
    # create_qdrant_data_from_objects(objects)
    llm_model = OpenAIGenerator(
        api_key=Secret.from_token(os.getenv("GLAMA_API_KEY")),
        api_base_url=os.getenv("GLAMA_API_BASE_URL"),
        model="gemini-2.5-flash-preview",
    )

    query = "I'm hungry"

    prompt = dedent(
        f"""
        You are a highly specialized research assistant with deep expertise in data analysis and understanding complex contexts.
        Your main mission is to analyze a database of objects, interpret the user's intent from the provided query, and accurately identify which objects are most relevant to the presented need.

        The output MUST be in the same language as the user's query. Identify the language of the query and ensure that all blocks and responses are written in that language. This is mandatory for every response.

        Your response must strictly follow the structure below, always including the <language>, <user_intention>, <think>, and <relevant_object> blocks in the output, regardless of the scenario:

        <language>
        Identify and state the language used in the user's query (for example: English, Portuguese, Spanish, etc.).
        </language>

        <user_intention>
        Analyze the user's query and clearly and objectively describe the user's main intent based on what was requested. Use this analysis to support the next steps.
        </user_intention>

        <think>
        Analyze all provided objects, correlating each one with the user's intent identified in the <user_intention> block and with the user's query. Consider attributes such as 'object_tag', 'object_caption', 'bbox_extent', 'bbox_center', and 'bbox_volume' to identify relationships, similarities, or relevant distinctions between the objects and the intent expressed in the query. Highlight possible ambiguities, overlaps, or information gaps that may impact the selection of the most suitable object. It is important to analyze that there are objects related to the action, but that are not capable of performing the action themselves, for example, the door handle is related to the action of opening the door, but it cannot perform the action by itself. Exclude these objects from the list if there are more relevant objects.
        </think>

        <relevant_object>
        Filter and list only the objects that are relevant to the user's query, briefly justifying the relevance of each one based on the analyzed attributes.
        </relevant_object>
        
        <selected_object>
        If there is an object sufficient to meet the user's need, return the dictionary of the selected object. If multiple objects are relevant, analyze their position in the environment and if they are close, return the dictionary of the most relevant object. The dictionary must contain only the attributes 'id', 'object_tag', 'object_caption', 'bbox_extent', 'bbox_center', and 'bbox_volume'. With the following structure:
        {{
            'id': ...,
            'object_tag': ...,
            'object_caption': ...,
            'bbox_extent': [..., ..., ...],
            'bbox_center': [..., ..., ...],
            'bbox_volume': ...
        }}
        </selected_object>
        
        <follow_up>
        The follow up question must make sense with the user's query and with the user's intent identified in the <user_intention> block. It should be elaborated in order to help the user select the most suitable object.
        </follow_up>

        After the above blocks, follow these rules for the final answer (always in the language of the query):
        1. If there is no relevant object, return: <no_object>No object found. Please provide more details about what you are looking for.</no_object>
        2. If there are multiple potentially relevant objects, return a follow-up question in the format <follow_up>...</follow_up>, presenting clear options that allow the user to differentiate between the possible objects (for example, highlighting differences in 'object_tag', 'object_caption', or other relevant attributes).
        3. If any of the collected objects is sufficient, return <selected_obj>object dictionary</selected_obj>.

        Do not include explanations, justifications, or any other text besides the <language>, <think>, <relevant_object> blocks and the final answer (<no_object>, <follow_up>, or <selected_obj>).

        Structure of an object in the database:
        {{
            'id': ...,
            'object_tag': ...,
            'object_caption': ...,
            'bbox_extent': [..., ..., ...],
            'bbox_center': [..., ..., ...],
            'bbox_volume': ...
        }}

        Query: {query}
        """
    )

    # create_qdrant_data(texts)
    relevant_chunks = query_relevant_chunks(query, 10, 0.1)
    print(relevant_chunks)
    print("--------------------------------")
    response = infer_with_llm_model(llm_model, relevant_chunks, prompt)
    print(response)

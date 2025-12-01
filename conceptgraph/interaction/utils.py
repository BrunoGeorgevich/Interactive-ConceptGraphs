import traceback
import time


class InteractionError(Exception):
    """Base exception for interaction module errors."""

    pass


class MapLoadError(InteractionError):
    """Raised when map data cannot be loaded or computed."""

    pass


class VectorDbError(InteractionError):
    """Raised when vector database operations fail."""

    pass


def log_debug_interaction(
    file_path: str,
    stage: str,
    system_prompt: str = "",
    user_input: str = "",
    content: str = "",
    rag_docs: list[dict] = [],
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
    :param rag_docs: List of retrieved documents for RAG context.
    :type rag_docs: list[dict]
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
            if rag_docs:
                f.write("\n--- RAG CONTEXT DOCUMENTS ---\n")
                for i, doc in enumerate(rag_docs):
                    f.write(f"Document {i + 1}:\n")
                    for key, value in doc.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
            if content:
                f.write("\n--- MODEL RESPONSE / OUTPUT ---\n")
                f.write(f"{content}\n")
            f.write(f"{separator}\n")
    except (IOError, OSError):
        traceback.print_exc()

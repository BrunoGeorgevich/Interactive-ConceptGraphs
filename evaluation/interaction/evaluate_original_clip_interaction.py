from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.nn.functional as F
from dotenv import load_dotenv
from openai import OpenAI
from glob import glob
import numpy as np
import traceback
import open_clip
import pickle
import torch
import json
import gzip
import csv
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from conceptgraph.slam.slam_classes import MapObjectList
from evaluation.prompts import ORIGINAL_JUDGE_PROMPT


class ConceptGraphRetriever:
    """
    Retriever class for querying objects from ConceptGraph SLAM results.

    This class loads pre-processed SLAM results and provides methods to retrieve
    the most relevant objects based on CLIP similarity to text queries.
    """

    def __init__(self, result_path: str, device: str = "cuda"):
        """
        Initializes the ConceptGraphRetriever with SLAM results and CLIP model.

        :param result_path: Path to the gzipped pickle result file.
        :type result_path: str
        :param device: Device to use for CLIP model computations.
        :type device: str
        :raises FileNotFoundError: If the result file does not exist.
        :raises RuntimeError: If loading the result or initializing CLIP fails.
        :return: None
        :rtype: None
        """
        self.result_path = result_path
        self.device = device
        self.objects = None
        self.clip_model = None
        self.clip_tokenizer = None
        self._load_result()
        self._initialize_clip()

    def _load_result(self) -> None:
        """
        Loads the SLAM result file and extracts object data.

        :raises FileNotFoundError: If the result file does not exist.
        :raises RuntimeError: If the result file cannot be loaded or parsed.
        :return: None
        :rtype: None
        """
        if not os.path.exists(self.result_path):
            raise FileNotFoundError(f"Result file not found: {self.result_path}")

        try:
            with gzip.open(self.result_path, "rb") as f:
                results = pickle.load(f)

            if not isinstance(results, dict):
                raise RuntimeError("Results should be a dictionary.")

            self.objects = MapObjectList()
            self.objects.load_serializable(results["objects"])
        except (OSError, pickle.PickleError, KeyError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to load result file: {e}")

    def _initialize_clip(self) -> None:
        """
        Initializes the CLIP model and tokenizer.

        :raises RuntimeError: If CLIP model initialization fails.
        :return: None
        :rtype: None
        """
        try:
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-H-14", "laion2b_s32b_b79k"
            )
            self.clip_model = self.clip_model.to(self.device)
            self.clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        except (RuntimeError, OSError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize CLIP model: {e}")

    def retrieve_top_k_objects(
        self, query: str, k: int = 3, similarity_threshold: float = 0.23
    ) -> list:
        """
        Retrieves the top-k most relevant objects for a given text query.

        :param query: The text query to search for.
        :type query: str
        :param k: Number of top objects to retrieve.
        :type k: int
        :param similarity_threshold: Similarity threshold for considering relevant objects.
        :type similarity_threshold: float
        :raises ValueError: If query is empty or k is less than 1.
        :raises RuntimeError: If object retrieval fails.
        :return: List of dictionaries containing object information and similarity scores.
        :rtype: list
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if k < 1:
            raise ValueError("k must be at least 1.")

        try:
            text_queries_tokenized = self.clip_tokenizer([query]).to(self.device)
            with torch.no_grad():
                text_query_ft = self.clip_model.encode_text(text_queries_tokenized)
                text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
                text_query_ft = text_query_ft.squeeze()

            objects_clip_fts = self.objects.get_stacked_values_torch("clip_ft")
            objects_clip_fts = objects_clip_fts.to(self.device)

            similarities = F.cosine_similarity(
                text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
            )

            top_k_indices = torch.topk(similarities, min(k, len(self.objects))).indices
            top_k_indices = top_k_indices.cpu().numpy()

            results = []
            for idx in top_k_indices:
                if similarities[idx].cpu().item() < similarity_threshold:
                    continue
                obj = self.objects[idx]
                obj_class = self._get_most_common_class(obj)
                bbox_center = obj["bbox"].center if obj.get("bbox") else [0, 0, 0]

                results.append(
                    {
                        "index": int(idx),
                        "class_name": obj.get("class_name", "unknown"),
                        "most_common_class": obj_class,
                        "similarity": float(similarities[idx].cpu().item()),
                        "location": list(bbox_center),
                    }
                )

            return results

        except (RuntimeError, KeyError, AttributeError, TypeError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to retrieve objects: {e}")

    def _get_most_common_class(self, obj: dict) -> str:
        """
        Gets the most common class for an object based on class_id.

        :param obj: Object dictionary containing class information.
        :type obj: dict
        :return: Most common class name or 'unknown'.
        :rtype: str
        """
        try:
            obj_classes = np.asarray(obj.get("class_id", []))
            if len(obj_classes) == 0:
                return obj.get("class_name", "unknown")
            values, counts = np.unique(obj_classes, return_counts=True)
            return str(values[np.argmax(counts)])
        except (ValueError, TypeError, AttributeError):
            return obj.get("class_name", "unknown")


def get_result_path(
    database_path: str, home_id: int, processing_type: str = "original"
) -> str:
    """
    Constructs the path to the SLAM result file for a given home.

    :param database_path: Base path to the database.
    :type database_path: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param processing_type: Type of processing (default: 'original').
    :type processing_type: str
    :return: Path to the result file.
    :rtype: str
    """
    return os.path.join(
        database_path,
        "outputs",
        f"Home{home_id:02d}",
        "Wandering",
        "exps",
        f"{processing_type}_house_{home_id}_map",
        f"pcd_{processing_type}_house_{home_id}_map.pkl.gz",
    )


def print_progress_bar(current: int, total: int, prefix: str = "Progress") -> None:
    """
    Prints a progress bar to the console.

    :param current: Current progress count.
    :type current: int
    :param total: Total count for completion.
    :type total: int
    :param prefix: Prefix text for the progress bar.
    :type prefix: str
    :return: None
    :rtype: None
    """
    bar_length: int = 40
    if total == 0:
        return
    filled_length: int = int(bar_length * current // total)
    bar: str = "=" * filled_length + "-" * (bar_length - filled_length)
    percent: float = (current / total) * 100
    print(f"\r{prefix}: |{bar}| {percent:.1f}% ({current}/{total})", end="", flush=True)
    if current == total:
        print()


def get_question_output_path(
    output_base_dir: str, home_id: int, question_type: str, question_index: int
) -> str:
    """
    Generates the output file path for a specific question's evaluation result.

    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param question_type: Type of question.
    :type question_type: str
    :param question_index: Index of the question within its type.
    :type question_index: int
    :return: Path to the JSON file for the question's evaluation result.
    :rtype: str
    """
    home_dir = os.path.join(output_base_dir, f"Home{home_id:02d}", question_type)
    os.makedirs(home_dir, exist_ok=True)
    return os.path.join(home_dir, f"question_{question_index:04d}.json")


def is_question_already_processed(
    output_base_dir: str, home_id: int, question_type: str, question_index: int
) -> bool:
    """
    Checks if a question has already been processed.

    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param question_type: Type of question.
    :type question_type: str
    :param question_index: Index of the question within its type.
    :type question_index: int
    :return: True if the question result file exists, False otherwise.
    :rtype: bool
    """
    output_path = get_question_output_path(
        output_base_dir, home_id, question_type, question_index
    )
    return os.path.exists(output_path)


def load_question_result(
    output_base_dir: str, home_id: int, question_type: str, question_index: int
) -> dict | None:
    """
    Loads a previously saved question evaluation result from disk.

    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param question_type: Type of question.
    :type question_type: str
    :param question_index: Index of the question within its type.
    :type question_index: int
    :return: Dictionary containing the question result or None if loading fails.
    :rtype: dict | None
    """
    output_path = get_question_output_path(
        output_base_dir, home_id, question_type, question_index
    )
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_question_result(
    output_base_dir: str,
    home_id: int,
    question_type: str,
    question_index: int,
    result: dict,
) -> None:
    """
    Saves a question evaluation result to disk as a JSON file.

    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param question_type: Type of question.
    :type question_type: str
    :param question_index: Index of the question within its type.
    :type result: dict
    :raises RuntimeError: If writing the file fails.
    :return: None
    :rtype: None
    """
    output_path = get_question_output_path(
        output_base_dir, home_id, question_type, question_index
    )
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except OSError as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to save question result: {e}")


def save_type_metrics(
    output_base_dir: str, home_id: int, question_type: str, metrics: dict
) -> None:
    """
    Saves the aggregated metrics for a specific question type within a home.

    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param question_type: Type of question.
    :type question_type: str
    :param metrics: Dictionary containing the calculated metrics.
    :type metrics: dict
    :raises RuntimeError: If writing the file fails.
    :return: None
    :rtype: None
    """
    home_dir = os.path.join(output_base_dir, f"Home{home_id:02d}", question_type)
    os.makedirs(home_dir, exist_ok=True)
    output_path = os.path.join(home_dir, f"metrics_{question_type}.json")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    except OSError as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to save type metrics: {e}")


def save_type_results_csv(
    output_base_dir: str, home_id: int, question_type: str, results: list
) -> None:
    """
    Saves the detailed results for a specific question type as a CSV file.

    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param question_type: Type of question.
    :type question_type: str
    :param results: List of dictionaries containing the detailed results.
    :type results: list
    :raises RuntimeError: If writing the file fails.
    :return: None
    :rtype: None
    """
    if not results:
        return

    home_dir = os.path.join(output_base_dir, f"Home{home_id:02d}", question_type)
    os.makedirs(home_dir, exist_ok=True)
    output_path = os.path.join(home_dir, f"results_{question_type}.csv")

    try:
        fieldnames = list(results[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                flat_row = {}
                for key, value in row.items():
                    if isinstance(value, (dict, list)):
                        flat_row[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        flat_row[key] = value
                writer.writerow(flat_row)
    except OSError as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to save type results CSV: {e}")


def evaluate_with_judge(
    input_data: dict,
    openai_client: OpenAI,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict:
    """
    Evaluates the retrieval result using the semantic judge LLM.

    :param input_data: Dictionary containing the question data and retrieved objects.
    :type input_data: dict
    :param openai_client: OpenAI client for API calls.
    :type openai_client: OpenAI
    :param model_id: Model identifier for the LLM.
    :type model_id: str
    :param temperature: Temperature parameter for generation.
    :type temperature: float
    :param top_p: Top-p parameter for generation.
    :type top_p: float
    :param max_tokens: Maximum tokens for the response.
    :type max_tokens: int
    :raises RuntimeError: If the LLM call fails after retries.
    :return: Dictionary with evaluation verdict and reasoning.
    :rtype: dict
    """
    judge_input = json.dumps(input_data, indent=2, ensure_ascii=False)
    system_prompt = ORIGINAL_JUDGE_PROMPT + "\n<RETRIEVAL_MODE>\nCLIP\n</RETRIEVAL_MODE>"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": judge_input},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content.strip()

            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                if "True" in content:
                    return {
                        "verdict": "True",
                        "reasoning": response.choices[0].message.reasoning.strip(),
                    }
                elif "Partial" in content:
                    return {
                        "verdict": "Partial",
                        "reasoning": response.choices[0].message.reasoning.strip(),
                    }
                elif "False" in content:
                    return {
                        "verdict": "False",
                        "reasoning": response.choices[0].message.reasoning.strip(),
                    }
                return {
                    "verdict": "Unknown",
                    "reasoning": response.choices[0].message.reasoning.strip(),
                }

        except (OSError, RuntimeError, TimeoutError) as e:
            traceback.print_exc()
            if attempt == max_retries - 1:
                raise RuntimeError(f"LLM call failed after {max_retries} attempts: {e}")

    raise RuntimeError("LLM evaluation failed unexpectedly.")


def extract_query_from_question(question: dict) -> str:
    """
    Extracts the query string from a question dictionary.

    :param question: Question dictionary containing either 'query' or 'messages'.
    :type question: dict
    :raises ValueError: If no valid query can be extracted.
    :return: The extracted query string.
    :rtype: str
    """
    if "query" in question:
        return question["query"]
    elif "messages" in question:
        return question["messages"]
    raise ValueError("No valid query found in question.")


def process_single_question(
    question: dict,
    retriever: ConceptGraphRetriever,
    openai_client: OpenAI,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    similarity_threshold: float = 0.23,
) -> dict:
    """
    Processes a single evaluation question using the ConceptGraphRetriever.

    :param question: The question data containing either 'query' or 'messages'.
    :type question: dict
    :param retriever: ConceptGraphRetriever instance for object retrieval.
    :type retriever: ConceptGraphRetriever
    :param openai_client: OpenAI client for judge evaluation.
    :type openai_client: OpenAI
    :param model_id: Model identifier for the judge LLM.
    :type model_id: str
    :param temperature: Temperature parameter for generation.
    :type temperature: float
    :param top_p: Top-p parameter for generation.
    :type top_p: float
    :param max_tokens: Maximum tokens for the response.
    :type max_tokens: int
    :param similarity_threshold: Similarity threshold for considering relevant objects.
    :type similarity_threshold: float
    :raises ValueError: If the question format is invalid.
    :return: The complete evaluation result dictionary.
    :rtype: dict
    """
    try:
        query = extract_query_from_question(question)
    except ValueError as e:
        traceback.print_exc()
        raise ValueError(f"Invalid question format: {e}")

    is_follow_up = isinstance(query, list)
    try:
        if isinstance(query, list):
            top_objects = []
            for q in query:
                if q["role"] != "user":
                    continue
                objs = retriever.retrieve_top_k_objects(
                    q["content"].replace("transitioningway", "transitioning"), k=3, similarity_threshold=similarity_threshold
                )
                top_objects.append(objs)
        elif isinstance(query, str):
            top_objects = retriever.retrieve_top_k_objects(
                query.replace("transitioningway", "transitioning"), k=3, similarity_threshold=similarity_threshold
            )
        else:
            raise ValueError("Query must be a string or a list of strings.")
    except (RuntimeError, ValueError):
        traceback.print_exc()
        top_objects = []

    if is_follow_up:
        input_data = {
            "messages": query,
            "obtained_messages": [],
            "is_follow_up": True,
        }

        for i, objs in enumerate(top_objects):
            input_data["obtained_messages"].append(
                {
                    "role": "user",
                    "content": query[i * 2]["content"],
                }
            )
            most_relevant = objs[0] if objs else "None"
            top_3_classes = [obj.get("class_name", "None") for obj in objs]

            if len(top_3_classes) < 3:
                top_3_classes += ["None"] * (3 - len(top_3_classes))

            input_data["obtained_messages"].append(
                {
                    "role": "robot",
                    "most_relevant_object": (
                        most_relevant["class_name"]
                        if most_relevant and "class_name" in most_relevant
                        else "None"
                    ),
                    "top_3_classes": top_3_classes,
                }
            )
    else:
        most_relevant = top_objects[0] if top_objects else "None"
        top_3_classes = [obj.get("class_name", "None") for obj in top_objects]

        if len(top_3_classes) < 3:
            top_3_classes += ["None"] * (3 - len(top_3_classes))

        input_data = {
            "query": query,
            "expected_answer": question.get("expected_answer", ""),
            "most_relevant_object": (
                most_relevant["class_name"]
                if most_relevant and "class_name" in most_relevant
                else "None"
            ),
            "top_3_classes": top_3_classes,
            "is_follow_up": is_follow_up,
        }

    judge_result = evaluate_with_judge(
        input_data=input_data,
        openai_client=openai_client,
        model_id=model_id,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    result = {
        **input_data,
        "evaluation": judge_result,
    }

    return result


def extract_question_type_from_path(questions_path: str) -> str:
    """
    Extracts the question type from the file path.

    :param questions_path: Path to the questions JSON file.
    :type questions_path: str
    :return: The question type extracted from the filename.
    :rtype: str
    """
    filename = os.path.basename(questions_path)
    return filename.replace(".json", "")


def calculate_metrics(results: list) -> dict:
    """
    Calculates evaluation metrics from a list of question results.

    :param results: List of evaluation result dictionaries.
    :type results: list
    :return: Dictionary containing calculated metrics.
    :rtype: dict
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "true_count": 0,
            "partial_count": 0,
            "false_count": 0,
            "true_rate": 0.0,
            "partial_rate": 0.0,
            "false_rate": 0.0,
            "success_rate": 0.0,
        }

    true_count = sum(
        1 for r in results if r.get("evaluation", {}).get("verdict") == "True"
    )
    partial_count = sum(
        1 for r in results if r.get("evaluation", {}).get("verdict") == "Partial"
    )
    false_count = sum(
        1 for r in results if r.get("evaluation", {}).get("verdict") == "False"
    )

    return {
        "total": total,
        "true_count": true_count,
        "partial_count": partial_count,
        "false_count": false_count,
        "true_rate": true_count / total,
        "partial_rate": partial_count / total,
        "false_rate": false_count / total,
        "success_rate": (true_count + partial_count * 0.5 + false_count * 0.0) / total,
    }


def save_summary_csv(output_base_dir: str, all_metrics: list) -> None:
    """
    Saves the summary metrics for all homes to a CSV file.

    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param all_metrics: List of dictionaries containing metrics for each home.
    :type all_metrics: list
    :raises RuntimeError: If writing the CSV file fails.
    :return: None
    :rtype: None
    """
    summary_path = os.path.join(
        output_base_dir, "summary_original_interaction_eval.csv"
    )
    try:
        if not all_metrics:
            return
        fieldnames = list(all_metrics[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
    except OSError as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to save summary CSV: {e}")


def evaluate_home(
    home_id: int,
    dataset_base_path: str,
    output_base_dir: str,
    openai_client: OpenAI,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    processing_type: str = "original",
) -> dict:
    """
    Evaluates all questions for a single home using ConceptGraph retrieval.

    :param home_id: Identifier for the home.
    :type home_id: int
    :param dataset_base_path: Base path to the dataset.
    :type dataset_base_path: str
    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param openai_client: OpenAI client for API calls.
    :type openai_client: OpenAI
    :param model_id: Model identifier for the LLM.
    :type model_id: str
    :param temperature: Temperature parameter for generation.
    :type temperature: float
    :param top_p: Top-p parameter for generation.
    :type top_p: float
    :param max_tokens: Maximum tokens for the response.
    :type max_tokens: int
    :param processing_type: Type of processing for result files.
    :type processing_type: str
    :return: Dictionary containing aggregated metrics for the home.
    :rtype: dict
    """
    home_path = os.path.join(dataset_base_path, f"Home{home_id:02d}")
    questions_files = glob(os.path.join(home_path, "evaluation_questions", "*.json"))

    result_path = get_result_path(dataset_base_path, home_id, processing_type)
    retriever = None

    home_metrics = {"home_id": home_id}
    all_results = []

    for questions_path in questions_files:
        question_type = extract_question_type_from_path(questions_path)

        try:
            with open(questions_path, "r", encoding="utf-8") as f:
                questions_data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            traceback.print_exc()
            print(f"[Home {home_id:02d}] Failed to load questions: {e}")
            continue

        questions = questions_data.get("samples", [])
        type_results = []

        for q_idx, question in enumerate(questions):
            if is_question_already_processed(
                output_base_dir, home_id, question_type, q_idx
            ):
                cached_result = load_question_result(
                    output_base_dir, home_id, question_type, q_idx
                )
                if cached_result:
                    type_results.append(cached_result)
                    all_results.append(cached_result)
                    continue

            if retriever is None:
                try:
                    retriever = ConceptGraphRetriever(result_path)
                except (FileNotFoundError, RuntimeError) as e:
                    traceback.print_exc()
                    print(f"[Home {home_id:02d}] Failed to initialize retriever: {e}")
                    return {"home_id": home_id, "error": str(e)}

            try:
                result = process_single_question(
                    question=question,
                    retriever=retriever,
                    openai_client=openai_client,
                    model_id=model_id,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                save_question_result(
                    output_base_dir, home_id, question_type, q_idx, result
                )
                type_results.append(result)
                all_results.append(result)
            except (ValueError, RuntimeError) as e:
                traceback.print_exc()
                print(f"[Home {home_id:02d}] Question {q_idx} failed: {e}")

        type_metrics = calculate_metrics(type_results)
        save_type_metrics(output_base_dir, home_id, question_type, type_metrics)
        save_type_results_csv(output_base_dir, home_id, question_type, type_results)

        home_metrics[f"{question_type}_total"] = type_metrics["total"]
        home_metrics[f"{question_type}_true"] = type_metrics["true_count"]
        home_metrics[f"{question_type}_partial"] = type_metrics["partial_count"]
        home_metrics[f"{question_type}_false"] = type_metrics["false_count"]
        home_metrics[f"{question_type}_success_rate"] = type_metrics["success_rate"]

    overall_metrics = calculate_metrics(all_results)
    home_metrics["overall_total"] = overall_metrics["total"]
    home_metrics["overall_true"] = overall_metrics["true_count"]
    home_metrics["overall_partial"] = overall_metrics["partial_count"]
    home_metrics["overall_false"] = overall_metrics["false_count"]
    home_metrics["overall_success_rate"] = overall_metrics["success_rate"]

    return home_metrics


def worker_evaluate_home(
    home_id: int,
    dataset_base_path: str,
    output_base_dir: str,
    api_key: str,
    base_url: str,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    processing_type: str = "original",
) -> dict | None:
    """
    Worker function for parallel home evaluation.

    :param home_id: Identifier for the home.
    :type home_id: int
    :param dataset_base_path: Base path to the dataset.
    :type dataset_base_path: str
    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param api_key: API key for OpenAI client.
    :type api_key: str
    :param base_url: Base URL for OpenAI API.
    :type base_url: str
    :param model_id: Model identifier for the LLM.
    :type model_id: str
    :param temperature: Temperature parameter for generation.
    :type temperature: float
    :param top_p: Top-p parameter for generation.
    :type top_p: float
    :param max_tokens: Maximum tokens for the response.
    :type max_tokens: int
    :param timeout: Timeout for API calls.
    :type timeout: float
    :param processing_type: Type of processing for result files.
    :type processing_type: str
    :return: Dictionary containing home metrics or None on failure.
    :rtype: dict | None
    """
    try:
        openai_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

        return evaluate_home(
            home_id=home_id,
            dataset_base_path=dataset_base_path,
            output_base_dir=output_base_dir,
            openai_client=openai_client,
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            processing_type=processing_type,
        )
    except (RuntimeError, OSError, ValueError) as e:
        traceback.print_exc()
        print(f"[Home {home_id:02d}] Worker failed: {e}")
        return None


def run_parallel_evaluation(
    dataset_base_path: str,
    output_base_dir: str,
    home_ids: list,
    api_key: str,
    base_url: str,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    max_workers: int = 4,
    processing_type: str = "original",
) -> None:
    """
    Runs the evaluation for all homes in parallel using threads.

    :param dataset_base_path: Base path to the dataset.
    :type dataset_base_path: str
    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param home_ids: List of home identifiers.
    :type home_ids: list
    :param api_key: API key for OpenAI client.
    :type api_key: str
    :param base_url: Base URL for OpenAI API.
    :type base_url: str
    :param model_id: Model identifier for the LLM.
    :type model_id: str
    :param temperature: Temperature parameter for generation.
    :type temperature: float
    :param top_p: Top-p parameter for generation.
    :type top_p: float
    :param max_tokens: Maximum tokens for the response.
    :type max_tokens: int
    :param timeout: Timeout for API calls.
    :type timeout: float
    :param max_workers: Maximum number of parallel workers.
    :type max_workers: int
    :param processing_type: Type of processing for result files.
    :type processing_type: str
    :return: None
    :rtype: None
    """
    os.makedirs(output_base_dir, exist_ok=True)
    all_metrics = []

    print(f"Starting parallel evaluation with {max_workers} workers...")
    print(f"Processing {len(home_ids)} homes: {home_ids}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                worker_evaluate_home,
                home_id,
                dataset_base_path,
                output_base_dir,
                api_key,
                base_url,
                model_id,
                temperature,
                top_p,
                max_tokens,
                timeout,
                processing_type,
            ): home_id
            for home_id in home_ids
        }

        completed = 0
        total = len(futures)
        print_progress_bar(0, total)

        for future in as_completed(futures):
            home_id = futures[future]
            try:
                result = future.result()
                if result:
                    all_metrics.append(result)
                    print(f"\n[Home {home_id:02d}] Completed successfully.")
                else:
                    print(f"\n[Home {home_id:02d}] Completed with errors.")
            except (RuntimeError, OSError, ValueError) as e:
                traceback.print_exc()
                print(f"\n[Home {home_id:02d}] Failed: {e}")

            completed += 1
            print_progress_bar(completed, total)

    all_metrics.sort(key=lambda x: x.get("home_id", 0))
    save_summary_csv(output_base_dir, all_metrics)
    print(f"\nEvaluation complete. Summary saved to {output_base_dir}")


if __name__ == "__main__":
    load_dotenv()

    DATASET_BASE_PATH: str = "THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET"
    OUTPUT_BASE_DIR: str = os.path.join(
        DATASET_BASE_PATH, "original_interaction_eval_results"
    )
    PROCESSING_TYPE: str = "original"
    OPENROUTER_MODEL_ID: str = "openai/gpt-oss-120b"
    OPENROUTER_TEMPERATURE: float = 0.0
    OPENROUTER_TOP_P: float = 0.1
    OPENROUTER_MAX_TOKENS: int = 64000
    OPENROUTER_TIMEOUT: float = 60.0
    HOME_IDS: list = list(range(1, 31))
    MAX_WORKERS: int = 10

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    base_url = "https://openrouter.ai/api/v1"

    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        exit(1)

    run_parallel_evaluation(
        dataset_base_path=DATASET_BASE_PATH,
        output_base_dir=OUTPUT_BASE_DIR,
        home_ids=HOME_IDS,
        api_key=api_key,
        base_url=base_url,
        model_id=OPENROUTER_MODEL_ID,
        temperature=OPENROUTER_TEMPERATURE,
        top_p=OPENROUTER_TOP_P,
        max_tokens=OPENROUTER_MAX_TOKENS,
        timeout=OPENROUTER_TIMEOUT,
        max_workers=MAX_WORKERS,
        processing_type=PROCESSING_TYPE,
    )

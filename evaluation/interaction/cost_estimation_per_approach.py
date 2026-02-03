from dotenv import load_dotenv
from glob import glob
import traceback
import sys
import os
import json
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from conceptgraph.interaction.original_llm_interaction import OriginalLLMInteraction
from conceptgraph.interaction.system import InteractionSystem
from conceptgraph.interaction.schemas import SystemConfig
from conceptgraph.inference.cost_estimator import CostEstimator


def get_result_path(
    database_path: str, home_id: int, processing_type: str = "original"
) -> str:
    """
    Constructs the result path for a given home and processing type.

    :param database_path: Base path to the database.
    :type database_path: str
    :param home_id: The home identifier.
    :type home_id: int
    :param processing_type: The processing type (default: "original").
    :type processing_type: str
    :return: Full path to the result file.
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


def extract_query_from_question(question: dict) -> str | list:
    """
    Extracts the query from a question dictionary.

    :param question: The question dictionary.
    :type question: dict
    :return: The extracted query string or list.
    :rtype: str | list
    """
    return question.get("question", question.get("queries", ""))


def load_questions_from_file(file_path: str, max_questions: int = 5) -> list[dict]:
    """
    Loads questions from a JSON file, limiting to a maximum number.

    :param file_path: Path to the questions JSON file.
    :type file_path: str
    :param max_questions: Maximum number of questions to load (default: 5).
    :type max_questions: int
    :raises FileNotFoundError: If the file does not exist.
    :raises json.JSONDecodeError: If the file is not valid JSON.
    :return: List of question dictionaries.
    :rtype: list[dict]
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
            return questions["samples"][:max_questions]
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load questions from {file_path}: {exc}")


def estimate_original_llm_costs(
    dataset_base_path: str,
    home_id: int,
    llm_model_id: str,
    estimator: CostEstimator,
    max_questions_per_file: int = 5,
) -> dict[str, float]:
    """
    Estimates costs for the original LLM interaction approach.

    :param dataset_base_path: Base path to the dataset.
    :type dataset_base_path: str
    :param home_id: The home identifier.
    :type home_id: int
    :param llm_model_id: The LLM model identifier.
    :type llm_model_id: str
    :param estimator: The cost estimator instance.
    :type estimator: CostEstimator
    :param max_questions_per_file: Maximum questions to process per file.
    :type max_questions_per_file: int
    :raises RuntimeError: If estimation fails.
    :return: Dictionary with total cost and cost per question.
    :rtype: dict[str, float]
    """
    estimator.clear_executions()
    result_path = get_result_path(dataset_base_path, home_id, "original")

    try:
        llm_interaction = OriginalLLMInteraction(result_path, llm_model_id)
        home_path = os.path.join(dataset_base_path, f"Home{home_id:02d}")
        questions_files = glob(
            os.path.join(home_path, "evaluation_questions", "*_questions.json")
        )

        total_questions = 0
        for questions_file in questions_files:
            questions = load_questions_from_file(questions_file, max_questions_per_file)
            total_questions += len(questions)

            for question in questions:
                query = extract_query_from_question(question)
                try:
                    llm_interaction.process_query(query)
                except (ValueError, RuntimeError) as exc:
                    print(f"Error processing query: {exc}")
                    traceback.print_exc()

        total_cost = estimator.estimate_cost()
        cost_per_question = total_cost / total_questions if total_questions > 0 else 0.0

        return {
            "total_cost": float(f"{total_cost:.8f}"),
            "cost_per_question": float(f"{cost_per_question:.8f}"),
            "total_questions": total_questions,
        }
    except (OSError, RuntimeError, ZeroDivisionError) as exc:
        traceback.print_exc()
        raise RuntimeError(
            f"Failed to estimate original LLM costs for home {home_id}: {exc}"
        )


def estimate_interaction_system_costs(
    dataset_base_path: str,
    home_id: int,
    estimator: CostEstimator,
    use_additional_knowledge: bool = False,
    max_questions_per_file: int = 5,
) -> dict[str, float]:
    """
    Estimates costs for the smart wheelchair system approach.

    :param dataset_base_path: Base path to the dataset.
    :type dataset_base_path: str
    :param home_id: The home identifier.
    :type home_id: int
    :param estimator: The cost estimator instance.
    :type estimator: CostEstimator
    :param use_additional_knowledge: Whether to use additional knowledge.
    :type use_additional_knowledge: bool
    :param max_questions_per_file: Maximum questions to process per file.
    :type max_questions_per_file: int
    :raises RuntimeError: If estimation fails.
    :return: Dictionary with total cost and cost per question.
    :rtype: dict[str, float]
    """
    estimator.clear_executions()
    home_path = os.path.join(dataset_base_path, f"Home{home_id:02d}")

    try:
        config = SystemConfig(
            house_id=home_id,
            dataset_base_path=dataset_base_path,
            use_additional_knowledge=use_additional_knowledge,
        )
        system = InteractionSystem(config)

        questions_files = glob(
            os.path.join(home_path, "evaluation_questions", "*_questions.json")
        )

        total_questions = 0
        for questions_file in questions_files:
            questions = load_questions_from_file(questions_file, max_questions_per_file)
            total_questions += len(questions)

            for question in questions:
                query = question.get("question", "")
                try:
                    system.process_query(query)
                except (ValueError, RuntimeError) as exc:
                    print(f"Error processing query: {exc}")
                    traceback.print_exc()

        total_cost = estimator.estimate_cost()
        cost_per_question = total_cost / total_questions if total_questions > 0 else 0.0

        return {
            "total_cost": float(f"{total_cost:.8f}"),
            "cost_per_question": float(f"{cost_per_question:.8f}"),
            "total_questions": total_questions,
        }
    except (OSError, RuntimeError, ZeroDivisionError) as exc:
        traceback.print_exc()
        raise RuntimeError(
            f"Failed to estimate smart wheelchair costs for home {home_id}: {exc}"
        )


def export_costs_to_csv(
    costs_per_approach: dict[str, dict[int, dict]], output_path: str
) -> None:
    """
    Exports estimated costs to a CSV file.

    :param costs_per_approach: Nested dictionary with approach as key and home costs as value.
    :type costs_per_approach: dict[str, dict[int, dict]]
    :param output_path: Path to the output CSV file.
    :type output_path: str
    :raises OSError: If the file cannot be written.
    :return: None
    :rtype: None
    """
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            header = [
                "Approach",
                "HomeID",
                "TotalQuestions",
                "TotalCost",
                "CostPerQuestion",
            ]
            writer.writerow(header)

            for approach, homes in costs_per_approach.items():
                for home_id, costs in homes.items():
                    row = [
                        approach,
                        home_id,
                        costs["total_questions"],
                        f"{costs['total_cost']:.8f}",
                        f"{costs['cost_per_question']:.8f}",
                    ]
                    writer.writerow(row)
    except (OSError, PermissionError) as exc:
        traceback.print_exc()
        raise OSError(f"Failed to write CSV file at {output_path}: {exc}")


def compute_average_costs_per_approach(
    costs_per_approach: dict[str, dict[int, dict]],
) -> dict[str, dict[str, float]]:
    """
    Computes average costs per approach across all homes.

    :param costs_per_approach: Nested dictionary with approach as key and home costs as value.
    :type costs_per_approach: dict[str, dict[int, dict]]
    :raises ZeroDivisionError: If there are no homes to average.
    :return: Dictionary mapping approach to average costs.
    :rtype: dict[str, dict[str, float]]
    """
    average_costs: dict[str, dict[str, float]] = {}

    for approach, homes in costs_per_approach.items():
        total_cost_sum = 0.0
        cost_per_question_sum = 0.0
        total_questions_sum = 0
        num_homes = len(homes)

        for costs in homes.values():
            total_cost_sum += costs["total_cost"]
            cost_per_question_sum += costs["cost_per_question"]
            total_questions_sum += costs["total_questions"]

        try:
            average_costs[approach] = {
                "avg_total_cost": float(f"{total_cost_sum / num_homes:.8f}"),
                "avg_cost_per_question": float(
                    f"{cost_per_question_sum / num_homes:.8f}"
                ),
                "avg_total_questions": float(f"{total_questions_sum / num_homes:.2f}"),
            }
        except ZeroDivisionError as exc:
            traceback.print_exc()
            print(f"No homes to average for approach {approach}: {exc}")

    return average_costs


if __name__ == "__main__":
    load_dotenv()

    DATASET_BASE_PATH = r"D:\Documentos\Datasets\Robot@VirtualHomeLarge"
    LLM_MODEL_ID = "openai/gpt-4o"
    HOME_ID = 1
    MAX_QUESTIONS_PER_FILE = 3

    estimations_dir = "estimations/interaction"
    try:
        os.makedirs(estimations_dir, exist_ok=True)
    except (OSError, PermissionError) as exc:
        traceback.print_exc()
        raise OSError(f"Failed to create estimations directory: {exc}")

    costs_csv_path = os.path.join(estimations_dir, "interaction_costs.csv")
    average_costs_json_path = os.path.join(
        estimations_dir, "average_interaction_costs.json"
    )

    estimator = CostEstimator()
    estimator.register_model(
        "openai/gpt-4o", {"input_cost": 2.5 / 1_000_000, "output_cost": 10 / 1_000_000}
    )
    estimator.register_model(
        "gemini-2.5-flash-lite",
        {"input_cost": 0.3 / 1_000_000, "output_cost": 0.4 / 1_000_000},
    )
    estimator.register_model(
        "google/gemini-2.5-flash-lite",
        {"input_cost": 0.3 / 1_000_000, "output_cost": 0.4 / 1_000_000},
    )
    estimator.register_model(
        "openai/gpt-oss-120b:nitro",
        {"input_cost": 0.35 / 1_000_000, "output_cost": 0.75 / 1_000_000},
    )

    costs_per_approach: dict[str, dict[int, dict]] = {
        "original_llm": {},
        "interaction_system": {},
    }

    print(f"Processing Home {HOME_ID:02d}...")

    try:
        costs_per_approach["original_llm"][HOME_ID] = estimate_original_llm_costs(
            DATASET_BASE_PATH, HOME_ID, LLM_MODEL_ID, estimator, MAX_QUESTIONS_PER_FILE
        )
        print(f"  Original LLM: {costs_per_approach['original_llm'][HOME_ID]}")
    except RuntimeError as exc:
        print(f"Failed to estimate original LLM costs for home {HOME_ID}: {exc}")
        traceback.print_exc()

    try:
        costs_per_approach["interaction_system"][HOME_ID] = (
            estimate_interaction_system_costs(
                DATASET_BASE_PATH,
                HOME_ID,
                estimator,
                use_additional_knowledge=False,
                max_questions_per_file=MAX_QUESTIONS_PER_FILE,
            )
        )
        print(f"  Interaction System: {costs_per_approach['interaction_system'][HOME_ID]}")
    except RuntimeError as exc:
        print(f"Failed to estimate Interaction System costs for home {HOME_ID}: {exc}")
        traceback.print_exc()

    export_costs_to_csv(costs_per_approach, costs_csv_path)

    average_costs = compute_average_costs_per_approach(costs_per_approach)

    print(f"\nAverage costs per approach: {json.dumps(average_costs, indent=4)}")

    try:
        with open(average_costs_json_path, "w", encoding="utf-8") as f:
            json.dump(average_costs, f, indent=4)
    except (OSError, PermissionError) as exc:
        traceback.print_exc()
        raise OSError(
            f"Failed to write average costs to {average_costs_json_path}: {exc}"
        )

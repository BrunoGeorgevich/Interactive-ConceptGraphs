from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from glob import glob
import traceback
import json
import csv
import sys
import re
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


from conceptgraph.interaction.system import SmartWheelchairSystem
from conceptgraph.interaction.schemas import SystemConfig
from evaluation.prompts import SEMANTIC_JUDGE_PROMPT


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


def extract_final_message(response: str) -> str:
    """
    Extracts the final user-facing message from the system's response, handling all possible output types
    as defined by the INTENTION_INTERPRETATION_PROMPT and AGENT_PROMPT_V3 prompts.

    :param response: The raw response string (XML or JSON) from the system.
    :type response: str
    :raises ValueError: If the response format is not recognized or no message can be extracted.
    :return: The extracted final message for the user.
    :rtype: str
    """
    try:
        if not response or not isinstance(response, str):
            return ""

        response = response.strip()
        if response.startswith("{"):
            try:
                data = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                traceback.print_exc()
                raise ValueError("Failed to parse JSON response.")

            if "direct_response" in data:
                return str(data["direct_response"])
            if "intent_explanation" in data:
                return str(data["intent_explanation"])
            if "rerank_query" in data and data["rerank_query"]:
                return str(data["rerank_query"])
            if "state" in data:
                return f"State: {data['state']}"
            return response

        tag_patterns = [
            (r"<selected_object>.*?<answer>(.*?)</answer>.*?</selected_object>", True),
            (
                r"<possible_objects>.*?<message>(.*?)</message>.*?</possible_objects>",
                True,
            ),
            (r"<follow_up>.*?<question>(.*?)</question>.*?</follow_up>", True),
            (
                r"<propositive_failure>.*?<question>(.*?)</question>.*?</propositive_failure>",
                True,
            ),
            (r"<no_object>.*?<message>(.*?)</message>.*?</no_object>", True),
        ]

        for pattern, dotall in tag_patterns:
            flags = re.DOTALL if dotall else 0
            match = re.search(pattern, response, flags)
            if match:
                return match.group(1).strip()

        return response

    except (ValueError, json.JSONDecodeError, TypeError, AttributeError) as e:
        traceback.print_exc()
        raise ValueError(
            f"Failed to extract final message: {str(e)} -> Response: {response}"
        )


def get_question_output_path(
    output_base_dir: str, home_id: int, question_type: str, question_index: int
) -> str:
    """
    Generates the output file path for a specific question's evaluation result.

    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param question_type: Type of question (e.g., 'basic', 'indirect', 'adversarial', 'follow_up').
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
    Checks if a question has already been processed by verifying the existence of its result file.

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
        traceback.print_exc()
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
    :type question_index: int
    :param result: Dictionary containing the evaluation result to save.
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
            json.dump(result, f, indent=4, ensure_ascii=False)
    except OSError as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to save question result to {output_path}: {e}")


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
            json.dump(metrics, f, indent=4, ensure_ascii=False)
    except OSError as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to save type metrics to {output_path}: {e}")


def save_type_results_csv(
    output_base_dir: str, home_id: int, question_type: str, results: list[dict]
) -> None:
    """
    Saves the detailed results for a specific question type within a home as a CSV file.

    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param home_id: Identifier for the home.
    :type home_id: int
    :param question_type: Type of question.
    :type question_type: str
    :param results: List of dictionaries containing the detailed results.
    :type results: list[dict]
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
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL
            )
            writer.writeheader()
            writer.writerows(results)
    except OSError as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to save type results CSV to {output_path}: {e}")


def evaluate_with_judge(
    input_data: dict,
    openai_client: OpenAI,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict:
    """
    Evaluates the system's response using the semantic judge LLM.

    :param input_data: Dictionary containing the question data and obtained response.
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

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SEMANTIC_JUDGE_PROMPT},
                    {"role": "user", "content": judge_input},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            verdict_raw = response.choices[0].message.content.strip()
            verdict = verdict_raw.lower()

            if "true" in verdict:
                verdict = "True"
            elif "partial" in verdict:
                verdict = "Partial"
            elif "false" in verdict:
                verdict = "False"
            else:
                verdict = "Unknown"

            return {
                "verdict": verdict,
                "raw_response": verdict_raw,
                "reasoning": getattr(
                    response.choices[0].message, "reasoning", "No reasoning provided."
                ),
            }

        except (OSError, AttributeError, TypeError, ValueError) as e:
            traceback.print_exc()
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Judge LLM call failed after {max_retries} retries: {e}"
                )


def process_single_question(
    question: dict,
    system: SmartWheelchairSystem,
    openai_client: OpenAI,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict:
    """
    Processes a single evaluation question using the SmartWheelchairSystem and evaluates with the judge.

    :param question: The question data containing either 'query' or 'messages'.
    :type question: dict
    :param system: Instance of SmartWheelchairSystem for processing queries.
    :type system: SmartWheelchairSystem
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
    :raises ValueError: If the question format is invalid.
    :return: The complete evaluation result dictionary.
    :rtype: dict
    """
    system.clear_active_memory()

    is_follow_up = True
    if "query" in question:
        is_follow_up = False
    elif "messages" not in question:
        raise ValueError(
            f"Question must contain either 'query' or 'messages': {question}"
        )

    if is_follow_up:
        messages = question.get("messages", None)
        if not messages:
            raise ValueError(
                f"'messages' field is empty or not found in follow-up question: {question}"
            )
        obtained_messages = []
        for msg in messages:
            if msg["role"] == "user":
                obtained_messages.append(msg)
                response = extract_final_message(
                    system.process_query(query=msg["content"]).text_response
                )
                obtained_messages.append({"role": "robot", "content": response})

        input_data = {
            "messages": messages,
            "obtained_messages": obtained_messages,
            "is_follow_up": is_follow_up,
        }
    else:
        query = question.get("query", "").replace("transitioningway", "transitioning")
        expected_answer = question.get("expected_answer", "").replace(
            "transitioningway", "transitioning"
        )

        response = extract_final_message(
            system.process_query(
                query=query,
                user_pose=(0.0, 0.0, 0.0),
            ).text_response
        )

        input_data = {
            "query": query,
            "expected_answer": expected_answer,
            "obtained_response": response,
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
        "success_rate": (true_count + partial_count) / total,
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
    summary_path = os.path.join(output_base_dir, "summary_interaction_eval.csv")
    try:
        if not all_metrics:
            return

        fieldnames = list(all_metrics[0].keys())

        with open(summary_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, delimiter=";", quoting=csv.QUOTE_ALL
            )
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
    use_additional_knowledge: bool = False,
) -> dict:
    """
    Evaluates all questions for a single home sequentially.

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
    :return: Dictionary containing aggregated metrics for the home.
    :rtype: dict
    :param use_additional_knowledge: Whether to use additional knowledge in evaluation.
    :type use_additional_knowledge: bool
    """
    home_path = os.path.join(dataset_base_path, f"Home{home_id:02d}")
    questions_files = glob(os.path.join(home_path, "evaluation_questions", "*.json"))

    config = SystemConfig(
        house_id=home_id,
        dataset_base_path=dataset_base_path,
        prefix="online",
        qdrant_url="http://localhost:6333",
        force_recreate_table=False,
        use_additional_knowledge=use_additional_knowledge,
        local_data_dir="data",
        debug_input_path=os.path.join("data", "input_debug.txt"),
        debug_output_path=os.path.join("data", "output_debug.txt"),
    )
    system = SmartWheelchairSystem(config)

    home_metrics = {
        "home_id": home_id,
    }

    all_results = []

    for questions_path in questions_files:
        question_type = extract_question_type_from_path(questions_path)

        try:
            with open(questions_path, "r", encoding="utf-8") as f:
                questions_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            traceback.print_exc()
            continue

        samples = questions_data.get("samples", [])
        type_results = []
        total_samples = len(samples)

        print_progress_bar(0, total_samples, f"Home {home_id:02d} - {question_type}")

        for idx, question in enumerate(samples):
            if is_question_already_processed(
                output_base_dir, home_id, question_type, idx
            ):
                result = load_question_result(
                    output_base_dir, home_id, question_type, idx
                )
                if result:
                    type_results.append(result)
                    print_progress_bar(
                        idx + 1, total_samples, f"Home {home_id:02d} - {question_type}"
                    )
                    continue

            try:
                result = process_single_question(
                    question=question,
                    system=system,
                    openai_client=openai_client,
                    model_id=model_id,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                save_question_result(
                    output_base_dir, home_id, question_type, idx, result
                )
                type_results.append(result)
            except (ValueError, KeyError, RuntimeError):
                traceback.print_exc()

            print_progress_bar(
                idx + 1, total_samples, f"Home {home_id:02d} - {question_type}"
            )

        type_metrics = calculate_metrics(type_results)

        # --- NEW CODE BLOCK: Save intermediate metrics and results ---
        save_type_metrics(
            output_base_dir=output_base_dir,
            home_id=home_id,
            question_type=question_type,
            metrics=type_metrics,
        )
        save_type_results_csv(
            output_base_dir=output_base_dir,
            home_id=home_id,
            question_type=question_type,
            results=type_results,
        )
        # -------------------------------------------------------------

        home_metrics[f"{question_type}_total"] = type_metrics["total"]
        home_metrics[f"{question_type}_true"] = type_metrics["true_count"]
        home_metrics[f"{question_type}_partial"] = type_metrics["partial_count"]
        home_metrics[f"{question_type}_false"] = type_metrics["false_count"]
        home_metrics[f"{question_type}_success_rate"] = type_metrics["success_rate"]

        all_results.extend(type_results)

    overall_metrics = calculate_metrics(all_results)
    home_metrics["overall_total"] = overall_metrics["total"]
    home_metrics["overall_true"] = overall_metrics["true_count"]
    home_metrics["overall_partial"] = overall_metrics["partial_count"]
    home_metrics["overall_false"] = overall_metrics["false_count"]
    home_metrics["overall_success_rate"] = overall_metrics["success_rate"]

    return home_metrics


def run_full_interaction_evaluation(
    dataset_base_path: str,
    output_base_dir: str,
    home_ids: list,
    openai_client: OpenAI,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> None:
    """
    Runs the full interaction evaluation for all specified homes sequentially.

    :param dataset_base_path: Base path to the dataset.
    :type dataset_base_path: str
    :param output_base_dir: Base directory for evaluation results.
    :type output_base_dir: str
    :param home_ids: List of home identifiers to evaluate.
    :type home_ids: list
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
    :return: None
    :rtype: None
    """
    os.makedirs(output_base_dir, exist_ok=True)

    all_home_metrics = []
    total_homes = len(home_ids)

    print_progress_bar(0, total_homes, "Overall Evaluation")

    for home_idx, home_id in enumerate(home_ids[:1]):
        try:
            home_metrics = evaluate_home(
                home_id=home_id,
                dataset_base_path=dataset_base_path,
                output_base_dir=output_base_dir,
                openai_client=openai_client,
                model_id=model_id,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            all_home_metrics.append(home_metrics)

        except (RuntimeError, KeyError, ValueError):
            traceback.print_exc()
            continue
        finally:
            print_progress_bar(home_idx + 1, total_homes, "Overall Evaluation")

    save_summary_csv(output_base_dir, all_home_metrics)


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
    use_additional_knowledge: bool = False,
) -> dict | None:
    """
    Função Worker que roda em um processo separado.
    Instancia seu próprio cliente OpenAI para evitar problemas de Pickling/Concorrência.
    """
    try:
        # Instancia o cliente DENTRO do processo worker
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

        print(f"[Start] Iniciando avaliação da Home {home_id:02d} (PID: {os.getpid()})")

        metrics = evaluate_home(
            home_id=home_id,
            dataset_base_path=dataset_base_path,
            output_base_dir=output_base_dir,
            openai_client=client,
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            use_additional_knowledge=use_additional_knowledge,
        )

        print(f"[Done] Finalizada Home {home_id:02d}")
        return metrics

    except Exception:
        print(f"[Error] Falha crítica na Home {home_id:02d}")
        traceback.print_exc()
        return None


def run_parallel_interaction_evaluation(
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
    use_additional_knowledge: bool = False,
) -> None:
    """
    Executa a avaliação completa utilizando multiprocessamento.
    """
    os.makedirs(output_base_dir, exist_ok=True)
    all_home_metrics = []

    print(
        f"Iniciando avaliação paralela com {max_workers} workers para {len(home_ids)} casas."
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Mapeia cada execução futura para o ID da casa
        future_to_home = {
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
                use_additional_knowledge,
            ): home_id
            for home_id in home_ids
        }

        completed_count = 0
        total_homes = len(home_ids)

        for future in as_completed(future_to_home):
            home_id = future_to_home[future]
            try:
                metrics = future.result()
                if metrics:
                    all_home_metrics.append(metrics)
            except Exception as exc:
                print(f"Home {home_id} gerou uma exceção: {exc}")

            completed_count += 1
            print(f"Progresso Geral: {completed_count}/{total_homes} casas concluídas.")

    # Ordena as métricas por Home ID antes de salvar, para manter o CSV organizado
    all_home_metrics.sort(key=lambda x: x["home_id"])

    print("Salvando resumo final...")
    save_summary_csv(output_base_dir, all_home_metrics)
    print("Avaliação completa.")


if __name__ == "__main__":
    load_dotenv()
    DATASET_BASE_PATH: str = THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET

    # Configurações da API
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL_ID: str = "openai/gpt-oss-120b"
    OPENROUTER_TEMPERATURE: float = 0.0
    OPENROUTER_TOP_P: float = 0.1
    OPENROUTER_MAX_TOKENS: int = 64000
    OPENROUTER_TIMEOUT: float = 60.0

    HOME_IDS: list = list(range(1, 31))

    # Defina o número de workers.
    # Se usar GPU para inferência local, mantenha baixo (ex: 1 ou 2).
    # Se for apenas CPU/API, pode usar cpu_count() ou um valor como 4 a 8.
    MAX_WORKERS = 60
    USE_ADDITIONAL_KNOWLEDGE = True
    OUTPUT_BASE_DIR: str = os.path.join(DATASET_BASE_PATH, "interaction_eval_results" if not USE_ADDITIONAL_KNOWLEDGE else "interaction_eval_results_with_ak")

    run_parallel_interaction_evaluation(
        dataset_base_path=DATASET_BASE_PATH,
        output_base_dir=OUTPUT_BASE_DIR,
        home_ids=HOME_IDS,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model_id=OPENROUTER_MODEL_ID,
        temperature=OPENROUTER_TEMPERATURE,
        top_p=OPENROUTER_TOP_P,
        max_tokens=OPENROUTER_MAX_TOKENS,
        timeout=OPENROUTER_TIMEOUT,
        max_workers=MAX_WORKERS,
        use_additional_knowledge=USE_ADDITIONAL_KNOWLEDGE,
    )
from agno.agent import RunResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from enum import Enum
import traceback
import json
import sys
import csv
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from evaluation.prompts import (
    SEMANTIC_JUDGE_PROMPT_TEMPORAL,
    ORIGINAL_JUDGE_PROMPT_TEMPORAL,
)
from conceptgraph.interaction.system import InteractionSystem
from conceptgraph.interaction.schemas import SystemConfig
from conceptgraph.interaction.original_llm_interaction import OriginalLLMInteraction
from evaluation.interaction.evaluate_original_clip_interaction import (
    ConceptGraphRetriever,
    get_result_path as get_clip_path,
)


class EvalMode(Enum):
    """
    Enumeration for evaluation modes.
    """

    SYSTEM = "system"
    CLIP = "clip"
    LLM = "llm"


def extract_final_message(response: str) -> str:
    """
    Extracts clean text from XML/JSON system responses.

    :param response: The raw response string to process.
    :type response: str
    :return: The extracted clean message text.
    :rtype: str
    """
    import re

    try:
        if not response:
            return ""
        if response.startswith("{"):
            data = json.loads(response)
            return str(
                data.get("direct_response")
                or data.get("intent_explanation")
                or data.get("state")
                or response
            )

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
        ]
        for pattern, dotall in tag_patterns:
            match = re.search(pattern, response, re.DOTALL if dotall else 0)
            if match:
                return match.group(1).strip()
        return response
    except (json.JSONDecodeError, ValueError, AttributeError):
        traceback.print_exc()
        return str(response)


def evaluate_with_judge(
    input_data: dict,
    openai_client: OpenAI,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    mode: EvalMode,
) -> dict:
    """
    Uses an LLM to judge the temporal consistency.

    :param input_data: The evaluation input containing task and messages.
    :type input_data: dict
    :param openai_client: The OpenAI client instance.
    :type openai_client: OpenAI
    :param model_id: The model identifier for the judge.
    :type model_id: str
    :param temperature: The sampling temperature.
    :type temperature: float
    :param top_p: The nucleus sampling parameter.
    :type top_p: float
    :param max_tokens: Maximum tokens for the response.
    :type max_tokens: int
    :param mode: The evaluation mode to determine which prompt to use.
    :type mode: EvalMode
    :return: Dictionary containing verdict and reasoning.
    :rtype: dict
    """
    if mode == EvalMode.SYSTEM:
        system_prompt = SEMANTIC_JUDGE_PROMPT_TEMPORAL
    else:
        system_prompt = ORIGINAL_JUDGE_PROMPT_TEMPORAL

    judge_input = json.dumps(input_data, indent=2, ensure_ascii=False)

    for _ in range(3):
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

            content = response.choices[0].message.content
            if content:
                content = content.strip()

            verdict = "Unknown"
            if "true" in content.lower():
                verdict = "True"
            elif "partial" in content.lower():
                verdict = "Partial"
            elif "false" in content.lower():
                verdict = "False"

            return {
                "verdict": verdict,
                "raw_response": content,
                "reasoning": getattr(response.choices[0].message, "reasoning", ""),
            }
        except (ConnectionError, TimeoutError, RuntimeError):
            traceback.print_exc()
            continue

    return {"verdict": "Error", "raw_response": "Judge failed"}


def calculate_metrics(results: list) -> dict:
    """
    Calculates evaluation metrics from results.

    :param results: List of evaluation results.
    :type results: list
    :return: Dictionary containing metrics.
    :rtype: dict
    """
    total = len(results)
    if total == 0:
        return {"total": 0, "success_rate": 0.0}

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
        "success_rate": (true_count + partial_count * 0.5 + false_count * 0.0) / total,
    }


def process_interaction_system(system: InteractionSystem, messages: list) -> list:
    """
    Process logic for the proposed Interaction System.

    :param system: The interaction system instance.
    :type system: InteractionSystem
    :param messages: List of conversation messages.
    :type messages: list
    :return: List of obtained messages with responses.
    :rtype: list
    """
    system.clear_active_memory()
    obtained_messages = []

    for msg in messages:
        if msg["role"] == "user":
            obtained_messages.append(msg)
            result = system.process_query(query=msg["content"])
            response_text = extract_final_message(result.text_response)
            obtained_messages.append({"role": "robot", "content": response_text})

    return obtained_messages


def process_clip_system(retriever: ConceptGraphRetriever, messages: list) -> list:
    """
    Process logic for Original CLIP retrieval system.

    :param retriever: The CLIP-based retriever instance.
    :type retriever: ConceptGraphRetriever
    :param messages: List of conversation messages.
    :type messages: list
    :return: List of obtained messages with responses.
    :rtype: list
    """
    obtained_messages = []

    for msg in messages:
        if msg["role"] == "user":
            obtained_messages.append(msg)

            try:
                query = msg["content"].replace("transitioningway", "transitioning")
                top_objs = retriever.retrieve_top_k_objects(query, k=1)

                if top_objs:
                    response_text = ", ".join(
                        top_obj.get("class_name", "object")
                        for top_obj in top_objs
                        if top_obj
                    )
                else:
                    response_text = "None"
            except (KeyError, IndexError, TypeError):
                traceback.print_exc()
                response_text = "Error in retrieval."

            obtained_messages.append({"role": "robot", "content": response_text})

    return obtained_messages


def process_llm_system(llm_interaction: OriginalLLMInteraction, messages: list) -> list:
    """
    Process logic for Original LLM system.

    :param llm_interaction: The LLM interaction instance.
    :type llm_interaction: OriginalLLMInteraction
    :param messages: List of conversation messages.
    :type messages: list
    :return: List of obtained messages with responses.
    :rtype: list
    """
    obtained_messages = []

    for msg in messages:
        if msg["role"] == "user":
            obtained_messages.append(msg)
            query = msg["content"]

            try:
                top_1, _, response_text = llm_interaction.process_query(query)

                if top_1:
                    response_text = top_1
                else:
                    response_text = "None"
            except (RuntimeError, ValueError):
                traceback.print_exc()
                response_text = "Error processing query."

            obtained_messages.append({"role": "robot", "content": str(response_text)})

    return obtained_messages


def evaluate_home_temporal(
    home_id: int,
    dataset_base_path: str,
    output_base_dir: str,
    mode: EvalMode,
    openai_client: OpenAI,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    llm_model_id: str = "gpt-4o",
) -> dict:
    """
    Evaluates temporal consistency for a specific home and mode.

    :param home_id: The home identifier.
    :type home_id: int
    :param dataset_base_path: Base path to the dataset.
    :type dataset_base_path: str
    :param output_base_dir: Base directory for output files.
    :type output_base_dir: str
    :param mode: The evaluation mode to use.
    :type mode: EvalMode
    :param openai_client: The OpenAI client instance.
    :type openai_client: OpenAI
    :param model_id: The judge model identifier.
    :type model_id: str
    :param temperature: The sampling temperature.
    :type temperature: float
    :param top_p: The nucleus sampling parameter.
    :type top_p: float
    :param max_tokens: Maximum tokens for responses.
    :type max_tokens: int
    :param llm_model_id: The LLM model identifier for baseline.
    :type llm_model_id: str
    :return: Dictionary containing evaluation results and metrics.
    :rtype: dict
    """
    home_str = f"Home{home_id:02d}"
    home_path = os.path.join(dataset_base_path, home_str)
    question_path = os.path.join(
        home_path, "evaluation_questions", "temporal_consistency.json"
    )

    system_output_dir = os.path.join(output_base_dir, mode.value, home_str)
    os.makedirs(system_output_dir, exist_ok=True)

    if not os.path.exists(question_path):
        print(f"[{home_str}] No temporal_consistency.json found.")
        return {}

    system_engine: (
        InteractionSystem | ConceptGraphRetriever | OriginalLLMInteraction | None
    ) = None
    try:
        if mode == EvalMode.SYSTEM:
            config = SystemConfig(
                house_id=home_id,
                dataset_base_path=dataset_base_path,
                prefix="online",
                qdrant_url="http://localhost:6333",
                use_additional_knowledge=True,
                local_data_dir="data",
                debug_input_path=os.path.join("data", "input_debug.txt"),
                debug_output_path=os.path.join("data", "output_debug.txt"),
            )
            system_engine = InteractionSystem(config)

        elif mode == EvalMode.CLIP:
            result_path = get_clip_path(dataset_base_path, home_id)
            system_engine = ConceptGraphRetriever(result_path)

        elif mode == EvalMode.LLM:
            result_path = get_clip_path(dataset_base_path, home_id)
            system_engine = OriginalLLMInteraction(result_path, llm_model_id)

    except (FileNotFoundError, RuntimeError, ValueError):
        traceback.print_exc()
        return {
            "home_id": home_id,
            "mode": mode.value,
            "error": "System initialization failed",
        }

    with open(question_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        samples = data.get("samples", [])

    results = []

    for idx, sample in enumerate(samples):
        out_file = os.path.join(system_output_dir, f"q_{idx:03d}.json")
        if os.path.exists(out_file):
            with open(out_file, "r") as f:
                results.append(json.load(f))
            continue

        messages = sample.get("messages", [])
        if not messages:
            continue

        try:
            if mode == EvalMode.SYSTEM and isinstance(system_engine, InteractionSystem):
                obtained_msgs = process_interaction_system(system_engine, messages)
            elif mode == EvalMode.CLIP and isinstance(
                system_engine, ConceptGraphRetriever
            ):
                obtained_msgs = process_clip_system(system_engine, messages)
            elif mode == EvalMode.LLM and isinstance(
                system_engine, OriginalLLMInteraction
            ):
                obtained_msgs = process_llm_system(system_engine, messages)
            else:
                obtained_msgs = [
                    {"role": "system", "content": "Invalid system engine type"}
                ]
        except (RuntimeError, ValueError, TypeError):
            traceback.print_exc()
            obtained_msgs = [{"role": "system", "content": "Processing error occurred"}]

        judge_payload = {
            "task": "temporal_consistency",
            "messages": messages,
            "obtained_messages": obtained_msgs,
        }

        eval_result = evaluate_with_judge(
            judge_payload, openai_client, model_id, temperature, top_p, max_tokens, mode
        )

        full_result = {**judge_payload, "evaluation": eval_result}

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(full_result, f, indent=2, ensure_ascii=False)

        results.append(full_result)

    metrics = calculate_metrics(results)

    with open(os.path.join(system_output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {"home_id": home_id, "mode": mode.value, **metrics}


def worker_wrapper(
    home_id: int,
    dataset_path: str,
    out_dir: str,
    mode: EvalMode,
    api_key: str,
    base_url: str,
    j_model: str,
    temp: float,
    top_p: float,
    max_tok: int,
    timeout: float,
    llm_id: str,
) -> dict:
    """
    Safe wrapper for threading/multiprocessing.

    :param home_id: The home identifier.
    :type home_id: int
    :param dataset_path: Path to the dataset.
    :type dataset_path: str
    :param out_dir: Output directory.
    :type out_dir: str
    :param mode: Evaluation mode.
    :type mode: EvalMode
    :param api_key: API key for OpenAI.
    :type api_key: str
    :param base_url: Base URL for API.
    :type base_url: str
    :param j_model: Judge model identifier.
    :type j_model: str
    :param temp: Temperature parameter.
    :type temp: float
    :param top_p: Top-p parameter.
    :type top_p: float
    :param max_tok: Maximum tokens.
    :type max_tok: int
    :param timeout: Request timeout.
    :type timeout: float
    :param llm_id: LLM model identifier.
    :type llm_id: str
    :return: Evaluation results dictionary.
    :rtype: dict
    """
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    return evaluate_home_temporal(
        home_id,
        dataset_path,
        out_dir,
        mode,
        client,
        j_model,
        temp,
        top_p,
        max_tok,
        llm_id,
    )


if __name__ == "__main__":
    load_dotenv()

    DATASET_BASE_PATH = "THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET"
    OUTPUT_BASE_DIR = os.path.join(
        DATASET_BASE_PATH, "interaction_eval_temporal_consistency"
    )

    JUDGE_MODEL_ID = "openai/gpt-oss-120b"
    LLM_BASELINE_MODEL = "google/gemini-2.5-flash-lite"

    API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    BASE_URL = "https://openrouter.ai/api/v1"

    HOME_IDS = list(range(1, 31))
    MAX_WORKERS = 21

    print("Starting Temporal Consistency Evaluation for ALL MODES")

    all_metrics = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for hid in HOME_IDS:
            for mode in [EvalMode.SYSTEM, EvalMode.CLIP, EvalMode.LLM]:
                future = executor.submit(
                    worker_wrapper,
                    hid,
                    DATASET_BASE_PATH,
                    OUTPUT_BASE_DIR,
                    mode,
                    API_KEY,
                    BASE_URL,
                    JUDGE_MODEL_ID,
                    0.0,
                    0.1,
                    16000,
                    60.0,
                    LLM_BASELINE_MODEL,
                )
                futures[future] = (hid, mode)

        for future in as_completed(futures):
            hid, mode = futures[future]
            try:
                res = future.result()
                if res:
                    all_metrics.append(res)
                    print(
                        f"Home {res.get('home_id')} [{mode.value}]: Success Rate {res.get('success_rate', 0):.2f}"
                    )
            except (RuntimeError, ValueError):
                traceback.print_exc()
                print(f"Home {hid} [{mode.value}]: Failed")

    all_metrics.sort(key=lambda x: (x.get("mode", ""), x.get("home_id", 0)))

    for mode in [EvalMode.SYSTEM, EvalMode.CLIP, EvalMode.LLM]:
        mode_metrics = [m for m in all_metrics if m.get("mode") == mode.value]
        if mode_metrics:
            summary_path = os.path.join(OUTPUT_BASE_DIR, f"summary_{mode.value}.csv")
            keys = mode_metrics[0].keys()
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(mode_metrics)
            print(f"Summary for {mode.value} saved at {summary_path}")

    print("Done. All evaluations completed.")

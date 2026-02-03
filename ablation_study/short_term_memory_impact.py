from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
import traceback
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.interaction.evaluate_interaction import worker_evaluate_home
from evaluation.prompts import SEMANTIC_JUDGE_PROMPT_TEMPORAL


def run_worker(task: dict[str, int | str | bool]) -> None:
    """
    Executes worker_evaluate_home with the provided task parameters.

    :param task: Dictionary containing parameters for worker_evaluate_home.
    :type task: dict[str, int | str | bool]
    :raises RuntimeError: If worker_evaluate_home fails due to ValueError or KeyError.
    :return: None
    :rtype: None
    """
    try:
        kwargs = task.copy()
        judge_prompt = kwargs.pop("judge_prompt")
        if judge_prompt is not None:
            worker_evaluate_home(**kwargs, judge_prompt=judge_prompt)
        else:
            worker_evaluate_home(**kwargs)
    except (ValueError, KeyError) as exc:
        traceback.print_exc()
        raise RuntimeError(f"Worker failed for home_id {task.get('home_id')}: {exc}")


if __name__ == "__main__":
    load_dotenv()

    DATABASE_PATH: str = "THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET"
    OUTPUT_DIR: str = (
        rf"{DATABASE_PATH}\results\ablation_study\short_term_memory_impact"
    )

    HOME_IDS: list[int] = list(range(1, 31))
    HOME_IDS = random.sample(HOME_IDS, 1)

    configs: list[tuple[bool, str]] = [
        (True, "with_short_term_memory"),
        (False, "without_short_term_memory"),
    ]

    tasks: list[dict[str, int | str | bool]] = []
    for enable_short_term_memory, config_suffix in configs:
        output_dir: str = os.path.join(OUTPUT_DIR, config_suffix)
        for home_id in HOME_IDS:
            tasks.append(
                {
                    "home_id": home_id,
                    "dataset_base_path": DATABASE_PATH,
                    "output_base_dir": output_dir,
                    "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                    "base_url": "https://openrouter.ai/api/v1",
                    "model_id": "openai/gpt-oss-120b",
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "max_tokens": 64000,
                    "timeout": 60.0,
                    "use_additional_knowledge": True,
                    "allowed_question_types": [
                        "follow_up_questions",
                    ],
                    "enable_short_term_memory": enable_short_term_memory,
                    "judge_prompt": None,
                }
            )
            tasks.append(
                {
                    "home_id": home_id,
                    "dataset_base_path": DATABASE_PATH,
                    "output_base_dir": output_dir,
                    "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                    "base_url": "https://openrouter.ai/api/v1",
                    "model_id": "openai/gpt-oss-120b",
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "max_tokens": 64000,
                    "timeout": 60.0,
                    "use_additional_knowledge": True,
                    "allowed_question_types": [
                        "temporal_consistency",
                    ],
                    "enable_short_term_memory": enable_short_term_memory,
                    "judge_prompt": SEMANTIC_JUDGE_PROMPT_TEMPORAL,
                }
            )

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_worker, task) for task in tasks]
        for future in futures:
            try:
                future.result()
            except RuntimeError as exc:
                traceback.print_exc()
                raise RuntimeError(f"Parallel evaluation failed: {exc}")

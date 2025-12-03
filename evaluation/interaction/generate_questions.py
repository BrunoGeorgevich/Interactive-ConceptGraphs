from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from glob import glob
import pandas as pd
import traceback
import json
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from evaluation.prompts import (
    ADVERSARIAL_QUESTION_PROMPT,
    FOLLOW_UP_QUESTION_PROMPT,
    INDIRECT_QUESTION_PROMPT,
    BASIC_QUESTION_PROMPT,
    BASE_PROMPT,
)


def generate_context(virtual_objects: pd.DataFrame) -> str:
    """
    Generates a context string from the virtual objects DataFrame.

    :param virtual_objects: DataFrame containing virtual object data.
    :type virtual_objects: pd.DataFrame
    :return: Context string summarizing the virtual objects.
    :rtype: str
    """
    context = []
    for _, row in virtual_objects.iterrows():
        context.append(f"{row['id']} at {row['room']}")
    context = list(set(context))
    return " - " + "\n - ".join(context)


def parse_virtual_object(data_path: str) -> pd.DataFrame:
    """
    Parses the virtual object data from a CSV file.

    :param data_path: Path to the CSV file containing virtual object data.
    :type data_path: str
    :return: DataFrame containing the parsed virtual object data.
    :rtype: pd.DataFrame
    :raises FileNotFoundError: If the specified file does not exist.
    :raises RuntimeError: If the file is empty or cannot be parsed.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file {data_path} does not exist.")

    try:
        df = pd.read_csv(data_path, sep=";")
        if "globalPosition" in df.columns:
            df[["x", "y", "z"]] = (
                df["globalPosition"]
                .str.extract(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)")
                .astype(float)
            )
            df = df.drop(columns=["globalPosition"])

        if "rotation" in df.columns:
            df[["roll", "pitch", "yaw"]] = (
                df["rotation"]
                .str.extract(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)")
                .astype(float)
            )
            df = df.drop(columns=["rotation"])

        if "color" in df.columns:
            rgba_matches = df["color"].str.extract(
                r"RGBA\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)"
            )
            if not rgba_matches.isnull().all().all():
                df["R"] = (rgba_matches[0].astype(float) * 255).round().astype(int)
                df["G"] = (rgba_matches[1].astype(float) * 255).round().astype(int)
                df["B"] = (rgba_matches[2].astype(float) * 255).round().astype(int)
            df = df.drop(columns=["color"])

        if "roomType" in df.columns:
            df["roomType"] = df["roomType"].replace(
                {"Standard": "Transitioning", "Hall": "Transitioning"}
            )
        if "type" in df.columns:
            df = df[df["type"] != "Standard"]
        return df
    except pd.errors.EmptyDataError:
        traceback.print_exc()
        raise RuntimeError(f"The file {data_path} is empty.")
    except (KeyError, ValueError, TypeError) as e:
        traceback.print_exc()
        raise RuntimeError(f"An error occurred while reading the file {data_path}: {e}")


def call_openai_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    top_p: float = 0.1,
    max_tokens: int = 16000,
) -> str:
    """
    Calls the OpenAI-compatible API to generate a completion.

    :param client: OpenAI client instance.
    :type client: OpenAI
    :param model: Model identifier to use.
    :type model: str
    :param prompt: Prompt to send to the model.
    :type prompt: str
    :param temperature: Sampling temperature.
    :type temperature: float
    :param top_p: Top-p sampling parameter.
    :type top_p: float
    :param max_tokens: Maximum tokens in the response.
    :type max_tokens: int
    :return: Generated text response.
    :rtype: str
    :raises RuntimeError: If the API call fails.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except (OSError, AttributeError, TypeError, ValueError) as e:
        traceback.print_exc()
        raise RuntimeError(f"OpenAI API call failed: {e}")


def extract_json_from_response(response: str) -> dict:
    """
    Extracts JSON object from LLM response, handling markdown code blocks.

    :param response: Raw response string from the LLM.
    :type response: str
    :return: Parsed JSON dictionary.
    :rtype: dict
    :raises json.JSONDecodeError: If the response cannot be parsed as JSON.
    """
    response = (
        response.strip()
        .replace("```json", "")
        .replace("```", "")
        .replace(" '", ' "')
        .replace(":'", ':"')
    )
    return json.loads(response.strip())


def format_samples_to_text(samples: list, question_type: str) -> str:
    """
    Formats a list of samples to text based on question type.

    :param samples: List of sample dictionaries.
    :type samples: list
    :param question_type: Type of questions (basic, indirect, adversarial, follow_up).
    :type question_type: str
    :return: Formatted text string with all samples.
    :rtype: str
    """
    return json.dumps({"samples": samples}, indent=4)


def process_question_type(
    home: str,
    question_type: str,
    task_prompt: str,
    num_questions: int,
    openai_client: OpenAI,
    model_id: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    objects_to_remove: list,
    objects_to_replace: dict,
    force_rewrite: bool = False,
) -> dict:
    """
    Processes a single question type for a given home.

    :param home: Path to the home directory.
    :type home: str
    :param question_type: Type of question to generate.
    :type question_type: str
    :param task_prompt: The task-specific prompt template.
    :type task_prompt: str
    :param num_questions: Number of questions to generate.
    :type num_questions: int
    :param openai_client: OpenAI client instance.
    :type openai_client: OpenAI
    :param model_id: Model identifier to use.
    :type model_id: str
    :param temperature: Sampling temperature.
    :type temperature: float
    :param top_p: Top-p sampling parameter.
    :type top_p: float
    :param max_tokens: Maximum tokens in the response.
    :type max_tokens: int
    :param objects_to_remove: List of object types to filter out.
    :type objects_to_remove: list
    :param objects_to_replace: Dictionary mapping object types to replacements.
    :type objects_to_replace: dict
    :param force_rewrite: Whether to force rewriting existing files.
    :type force_rewrite: bool
    :return: Dictionary with status, home, question_type, and sample count.
    :rtype: dict
    """
    home_name = os.path.basename(home)
    output_dir = os.path.join(home, "evaluation_questions")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{question_type}_questions.json")

    if os.path.exists(output_file) and not force_rewrite:
        return {
            "status": "skipped",
            "home": home_name,
            "question_type": question_type,
            "message": f"Skipped: {output_file} already exists.",
        }

    try:
        virtual_objects = parse_virtual_object(os.path.join(home, "VirtualObjects.csv"))

        virtual_objects = virtual_objects[
            ~virtual_objects["type"].astype(str).str.lower().isin(objects_to_remove)
        ]
        virtual_objects["type"] = virtual_objects["type"].replace(objects_to_replace)

        context = generate_context(virtual_objects)

        prompt = BASE_PROMPT
        prompt = prompt.replace("{task}", task_prompt)
        prompt = prompt.replace("{num_questions}", str(num_questions))
        prompt = prompt.replace("{context}", context)

        parsed_samples: list[dict] = []

        llm_response = call_openai_llm(
            client=openai_client,
            model=model_id,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        json_data = extract_json_from_response(llm_response)
        parsed_samples = json_data.get("samples", [])

        with open(output_file, "w", encoding="utf-8") as f:
            formatted_output = format_samples_to_text(parsed_samples, question_type)
            formatted_output = formatted_output.replace(
                " Hall", " Transitioning"
            ).replace(" hall", " transitioning")
            f.write(formatted_output)

        return {
            "status": "success",
            "home": home_name,
            "question_type": question_type,
            "samples_count": len(parsed_samples),
        }

    except (FileNotFoundError, RuntimeError, json.JSONDecodeError, OSError) as e:
        traceback.print_exc()
        return {
            "status": "error",
            "home": home_name,
            "question_type": question_type,
            "error": str(e),
        }


if __name__ == "__main__":
    load_dotenv()
    DATA_FOLDER: str = os.path.join(
        "D:", "Documentos", "Datasets", "Robot@VirtualHomeLarge"
    )
    QUESTIONS_PROMPT_DICT: dict[str, tuple] = {
        "basic": (BASIC_QUESTION_PROMPT, 30),
        "indirect": (INDIRECT_QUESTION_PROMPT, 30),
        "adversarial": (ADVERSARIAL_QUESTION_PROMPT, 30),
        "follow_up": (FOLLOW_UP_QUESTION_PROMPT, 10),
    }
    OPENROUTER_MODEL_ID: str = "openai/gpt-oss-120b"
    OPENROUTER_TEMPERATURE: float = 0.0
    OPENROUTER_TOP_P: float = 0.1
    OPENROUTER_MAX_TOKENS: int = 16000
    OPENROUTER_TIMEOUT: float = 60.0
    MAX_WORKERS: int = 100
    OBJECTS_TO_REMOVE: list[str] = [
        "wall",
        "floor",
        "ceiling",
        "door",
        "decoration",
    ]
    OBJECTS_TO_REPLACE: dict[str, str] = {
        "Standard": "Shelf or Cabinet",
        "KitchenFurniture": "Shelf or Cabinet",
        "Drying": "Towel Drying Rack",
        "Burner": "Stove or Cooktop",
    }
    FORCE_REWRITE: bool = True

    openai_client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        timeout=OPENROUTER_TIMEOUT,
    )

    homes = glob(os.path.join(DATA_FOLDER, "Home*"))

    tasks = []
    for home in homes:
        for question_type, (
            task_prompt,
            num_questions,
        ) in QUESTIONS_PROMPT_DICT.items():
            tasks.append((home, question_type, task_prompt, num_questions))

    print(f"Starting generation with {MAX_WORKERS} workers for {len(tasks)} tasks...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_question_type,
                home,
                question_type,
                task_prompt,
                num_questions,
                openai_client,
                OPENROUTER_MODEL_ID,
                OPENROUTER_TEMPERATURE,
                OPENROUTER_TOP_P,
                OPENROUTER_MAX_TOKENS,
                OBJECTS_TO_REMOVE,
                OBJECTS_TO_REPLACE,
                FORCE_REWRITE,
            ): (home, question_type)
            for home, question_type, task_prompt, num_questions in tasks
        }

        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "success":
                print(
                    f"[SUCCESS] {result['home']} - {result['question_type']}: "
                    f"{result['samples_count']} samples"
                )
            elif result["status"] == "skipped":
                print(
                    f"[SKIPPED] {result['home']} - {result['question_type']}: "
                    f"{result['message']}"
                )
            else:
                print(
                    f"[ERROR] {result['home']} - {result['question_type']}: "
                    f"{result.get('error', 'Unknown error')}"
                )

    print("Generation complete.")

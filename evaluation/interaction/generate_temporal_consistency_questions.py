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

from evaluation.prompts import TEMPORAL_CONSISTENCY_QUESTION_PROMPT, BASE_PROMPT


def generate_context(virtual_objects: pd.DataFrame) -> str:
    """
    Generate a formatted context string from virtual objects DataFrame.

    :param virtual_objects: DataFrame containing virtual objects with 'id' and 'room' columns
    :type virtual_objects: pd.DataFrame
    :return: Formatted context string with unique object-room combinations
    :rtype: str
    """
    context = []
    for _, row in virtual_objects.iterrows():
        context.append(f"{row['id'].split('_')[0]} at {row['room'].replace('_', ' ')}")
    context = list(set(context))
    return " - " + "\n - ".join(context)


def parse_virtual_object(data_path: str) -> pd.DataFrame:
    """
    Parse virtual objects CSV file and apply necessary transformations.

    :param data_path: Path to the CSV file containing virtual objects
    :type data_path: str
    :raises FileNotFoundError: If the specified file does not exist
    :raises RuntimeError: If there is an error reading or parsing the CSV file
    :return: Parsed and transformed DataFrame
    :rtype: pd.DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file {data_path} does not exist.")
    try:
        df = pd.read_csv(data_path, sep=";")
        if "roomType" in df.columns:
            df["roomType"] = df["roomType"].replace(
                {"Standard": "Transitioning", "Hall": "Transitioning"}
            )
        if "type" in df.columns:
            df = df[df["type"] != "Standard"]
        return df
    except (ValueError, KeyError, IOError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Error reading {data_path}: {e}")


def call_openai_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """
    Call OpenAI API with specified parameters.

    :param client: OpenAI client instance
    :type client: OpenAI
    :param model: Model identifier to use
    :type model: str
    :param prompt: Prompt text to send to the model
    :type prompt: str
    :param temperature: Temperature parameter for response randomness
    :type temperature: float
    :param top_p: Top-p parameter for nucleus sampling
    :type top_p: float
    :param max_tokens: Maximum number of tokens to generate
    :type max_tokens: int
    :raises RuntimeError: If the API call fails
    :return: Response content from the model
    :rtype: str
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
    except (ConnectionError, TimeoutError, ValueError) as e:
        traceback.print_exc()
        raise RuntimeError(f"OpenAI API call failed: {e}")


def extract_json_from_response(response: str) -> dict:
    """
    Extract and parse JSON data from LLM response string.

    :param response: Raw response string from LLM
    :type response: str
    :raises ValueError: If JSON parsing fails
    :return: Parsed JSON data as dictionary
    :rtype: dict
    """
    response = (
        response.strip()
        .replace("```json", "")
        .replace("```", "")
        .replace(" '", ' "')
        .replace(":'", ':"')
    )
    return json.loads(response.strip())


def format_samples_to_text(samples: list) -> str:
    """
    Format samples list into a JSON string.

    :param samples: List of sample dictionaries
    :type samples: list
    :return: Formatted JSON string
    :rtype: str
    """
    return json.dumps({"samples": samples}, indent=4, ensure_ascii=False)


def process_temporal_consistency(
    home: str,
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
    Process temporal consistency question generation for a single home.

    :param home: Path to the home directory
    :type home: str
    :param num_questions: Number of questions to generate
    :type num_questions: int
    :param openai_client: OpenAI client instance
    :type openai_client: OpenAI
    :param model_id: Model identifier to use
    :type model_id: str
    :param temperature: Temperature parameter for response randomness
    :type temperature: float
    :param top_p: Top-p parameter for nucleus sampling
    :type top_p: float
    :param max_tokens: Maximum number of tokens to generate
    :type max_tokens: int
    :param objects_to_remove: List of object types to filter out
    :type objects_to_remove: list
    :param objects_to_replace: Dictionary mapping object types to replacements
    :type objects_to_replace: dict
    :param force_rewrite: Whether to overwrite existing files
    :type force_rewrite: bool
    :return: Dictionary containing processing status and results
    :rtype: dict
    """

    question_type = "temporal_consistency"
    home_name = os.path.basename(home)
    output_dir = os.path.join(home, "evaluation_questions")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{question_type}.json")

    if os.path.exists(output_file) and not force_rewrite:
        return {
            "status": "skipped",
            "home": home_name,
            "question_type": question_type,
            "message": f"Skipped: {output_file} already exists.",
        }

    try:
        vo_path = os.path.join(home, "VirtualObjects.csv")
        virtual_objects = parse_virtual_object(vo_path)

        virtual_objects = virtual_objects[
            ~virtual_objects["type"].astype(str).str.lower().isin(objects_to_remove)
        ]
        virtual_objects["type"] = virtual_objects["type"].replace(objects_to_replace)

        context = generate_context(virtual_objects)

        prompt = BASE_PROMPT
        prompt = prompt.replace("{task}", TEMPORAL_CONSISTENCY_QUESTION_PROMPT)
        prompt = prompt.replace("{num_questions}", str(num_questions))
        prompt = prompt.replace("{context}", context)

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
            formatted_output = format_samples_to_text(parsed_samples)
            formatted_output = (
                formatted_output.replace(" Hall", " Transitioning")
                .replace(" hall", " transitioning")
                .replace("KitchenLivingRoom", "Kitchen")
                .replace("Corridor", "transitioning")
                .replace("Transitioningway", "Transitioning")
                .replace("transitioningway", "transitioning")
            )
            f.write(formatted_output)

        return {
            "status": "success",
            "home": home_name,
            "question_type": question_type,
            "samples_count": len(parsed_samples),
        }

    except (FileNotFoundError, RuntimeError, ValueError, KeyError, IOError) as e:
        traceback.print_exc()
        return {
            "status": "error",
            "home": home_name,
            "question_type": question_type,
            "error": str(e),
        }


if __name__ == "__main__":
    load_dotenv()

    DATA_FOLDER: str = "THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET"
    NUM_QUESTIONS: int = 10
    OPENROUTER_MODEL_ID: str = "openai/gpt-oss-120b"
    OPENROUTER_TEMPERATURE: float = 0.0
    OPENROUTER_TOP_P: float = 0.1
    OPENROUTER_MAX_TOKENS: int = 16000
    OPENROUTER_TIMEOUT: float = 60.0
    MAX_WORKERS: int = 20
    FORCE_REWRITE: bool = True

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

    openai_client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        timeout=OPENROUTER_TIMEOUT,
    )

    homes = glob(os.path.join(DATA_FOLDER, "Home*"))
    print(f"Starting Temporal Consistency generation for {len(homes)} homes...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_temporal_consistency,
                home,
                NUM_QUESTIONS,
                openai_client,
                OPENROUTER_MODEL_ID,
                OPENROUTER_TEMPERATURE,
                OPENROUTER_TOP_P,
                OPENROUTER_MAX_TOKENS,
                OBJECTS_TO_REMOVE,
                OBJECTS_TO_REPLACE,
                FORCE_REWRITE,
            ): home
            for home in homes
        }

        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "success":
                print(f"[SUCCESS] {result['home']}: {result['samples_count']} samples")
            elif result["status"] == "skipped":
                print(f"[SKIPPED] {result['home']}: {result['message']}")
            else:
                print(f"[ERROR] {result['home']}: {result.get('error')}")

    print("Generation complete.")

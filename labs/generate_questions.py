from llama_index.llms.openai_like import OpenAILike
from dotenv import load_dotenv
from glob import glob
import pandas as pd
import os

from prompts import (
    BASE_PROMPT_2,
    BASIC_QUESTION_PROMPT,
    ADVERSARIAL_QUESTION_PROMPT,
    INDIRECT_QUESTION_PROMPT,
    FOLLOW_UP_QUESTION_PROMPT,
)

# TODO: Elaborar um conjunto de interações rotineiras de uma pessoa com cadeira de rodas
#       Salientar as interações específicas do público alvo (exercicios de mobilidade)
#       Gerar os mapas das casas
# TODO: Considerar um dia de interação (dia todo)


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
        context.append(f"{row['type']} at {row['roomType']}")
    return " - " + "\n - ".join(context)


def parse_virtual_object(data_path: str) -> pd.DataFrame:
    """
    Parses the virtual object data from a CSV file.

    :param data_path: Path to the CSV file containing virtual object data.
    :type data_path: str
    :return: DataFrame containing the parsed virtual object data.
    :rtype: pd.DataFrame
    :raises FileNotFoundError: If the specified file does not exist.
    :raises pd.errors.EmptyDataError: If the file is empty.
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
        raise pd.errors.EmptyDataError(f"The file {data_path} is empty.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file {data_path}: {e}")


if __name__ == "__main__":
    load_dotenv()
    DATA_FOLDER = os.path.join("D:", "Documentos", "Datasets", "Robot@VirtualHome")
    QUESTIONS_TYPE = "follow_up"  # "basic", "adversarial", "indirect", "follow_up"
    QUESTIONS_PROMPT_DICT = {
        "basic": BASIC_QUESTION_PROMPT,
        "adversarial": ADVERSARIAL_QUESTION_PROMPT,
        "indirect": INDIRECT_QUESTION_PROMPT,
        "follow_up": FOLLOW_UP_QUESTION_PROMPT,
    }
    NUM_QUESTIONS = 10

    prompt = BASE_PROMPT_2.replace("{num_questions}", str(NUM_QUESTIONS))
    previous_outputs = []
    llm = OpenAILike(
        model="google/gemini-2.5-pro",
        api_base=os.getenv("OPENROUTER_API_BASE_URL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    prompt = prompt.replace("{task}", QUESTIONS_PROMPT_DICT[QUESTIONS_TYPE])
    homes = glob(os.path.join(DATA_FOLDER, "Home*"))

    home = homes[0]
    home_name = os.path.basename(home)
    output_dir = os.path.join("output", home_name)
    os.makedirs(output_dir, exist_ok=True)

    virtual_objects = parse_virtual_object(
        os.path.join(home, "Wandering", "VirtualObjects.csv")
    )

    virtual_objects = virtual_objects[~virtual_objects["type"].isin(["Wall", "Door"])]

    object_types = virtual_objects["type"].value_counts()
    context = generate_context(virtual_objects)
    prompt = prompt.replace("{context}", context)

    # for _ in range(NUM_QUESTIONS):
    #     if len(previous_outputs) > 0:
    #         previous_outputs_str = " - " + "\n - ".join(
    #             [el[0] for el in previous_outputs]
    #         )
    #         parsed_prompt = prompt.replace("{previous_outputs}", previous_outputs_str)
    #     else:
    #         parsed_prompt = prompt.replace(
    #             "{previous_outputs}", "No previous outputs available."
    #         )

    #     llm_response = llm.complete(parsed_prompt)
    #     question, answer = llm_response.text.split(";", 1)
    #     previous_outputs.append((question.strip(), answer.strip()))
    #     print(f"LLM Response: {llm_response}")

    llm_response = llm.complete(prompt)
    questions_and_answers = llm_response.text.strip().split("\n")
    for qa in questions_and_answers:
        if ";" in qa:
            question, answer = qa.split(";", 1)
            previous_outputs.append((question.strip(), answer.strip()))
        else:
            print(f"Skipping malformed line: {qa}")

    output_file = os.path.join(output_dir, f"{QUESTIONS_TYPE}_questions.txt")
    with open(output_file, "w") as f:
        for question, answer in previous_outputs:
            f.write(f"Q: {question}\nA: {answer}\n\n")

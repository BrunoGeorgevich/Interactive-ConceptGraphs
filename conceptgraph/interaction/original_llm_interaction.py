from agno.models.openrouter import OpenRouter
from agno.agent import Agent, RunResponse
from dotenv import load_dotenv
from typing import Any
import open3d as o3d
import traceback
import pickle
import gzip
import json
import os
import re

from conceptgraph.inference.cost_estimator import CostEstimator
from conceptgraph.interaction.prompts import ORIGINAL_LLM_PROMPT


def convert_object_to_json_entry(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Converts a single object from the pkl.gz format to JSON entry format.

    :param obj: Dictionary containing object data with keys like 'id', 'bbox', 'class_name', 'caption', etc.
    :type obj: dict[str, Any]
    :raises KeyError: If required keys are missing in the object dictionary.
    :raises ValueError: If bbox data is invalid or cannot be processed.
    :return: Dictionary formatted as JSON entry with id, bbox_extent, bbox_center, object_tag, and caption.
    :rtype: dict[str, Any]
    """
    try:
        obj_id = str(obj.get("id", -1))
        class_name = obj.get("class_name", "unknown")
        caption = obj.get("consolidated_caption", "")

        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(obj["bbox_np"])
        )

        if bbox is None:
            raise ValueError(f"Invalid bbox data for object {obj_id}: bbox is None")

        if not hasattr(bbox, "extent") or not hasattr(bbox, "center"):
            raise ValueError(
                f"bbox object missing 'extent' or 'center' attribute for object {obj_id}"
            )

        bbox_extent = [round(val, 2) for val in bbox.extent]
        bbox_center = [round(val, 2) for val in bbox.center]
        bbox_volume = round(bbox_extent[0] * bbox_extent[1] * bbox_extent[2], 2)

        return {
            "id": obj_id,
            "bbox_extent": bbox_extent,
            "bbox_center": bbox_center,
            "bbox_volume": bbox_volume,
            "object_tag": class_name,
            "caption": caption,
        }

    except (KeyError, ValueError, IndexError, AttributeError) as err:
        traceback.print_exc()
        raise RuntimeError(f"Error converting object to JSON entry: {err}")


def load_scene_graph_from_pkl(pkl_path: str) -> list[dict[str, Any]]:
    """
    Loads scene graph data from pkl.gz file and converts to JSON format.

    :param pkl_path: Path to the .pkl.gz file containing scene graph data.
    :type pkl_path: str
    :raises FileNotFoundError: If the pkl.gz file does not exist.
    :raises IOError: If file cannot be read or decompressed.
    :raises ValueError: If pkl data structure is invalid.
    :return: List of dictionaries, each representing an object in JSON format.
    :rtype: list[dict[str, Any]]
    """
    try:
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"File not found: {pkl_path}")

        with gzip.open(pkl_path, "rb") as f:
            raw_results = pickle.load(f)

        raw_objects = raw_results.get("objects", [])

        if not isinstance(raw_objects, list):
            raise ValueError("Expected 'objects' to be a list in pkl data")

        json_entries = []
        for obj in raw_objects:
            try:
                json_entry = convert_object_to_json_entry(obj)
                json_entries.append(json_entry)
            except RuntimeError:
                continue

        return json_entries

    except (FileNotFoundError, IOError, ValueError, KeyError) as err:
        traceback.print_exc()
        raise RuntimeError(f"Error loading scene graph from pkl: {err}")


def format_scene_graph_as_json_string(scene_objects: list[dict[str, Any]]) -> str:
    """
    Formats scene graph objects as a JSON string for LLM input.

    :param scene_objects: List of object dictionaries in JSON format.
    :type scene_objects: list[dict[str, Any]]
    :raises TypeError: If scene_objects is not a list.
    :return: JSON formatted string representation of the scene graph.
    :rtype: str
    """
    try:
        if not isinstance(scene_objects, list):
            raise TypeError("scene_objects must be a list")

        return json.dumps(scene_objects, indent=2, ensure_ascii=False)

    except (TypeError, ValueError) as err:
        traceback.print_exc()
        raise RuntimeError(f"Error formatting scene graph as JSON string: {err}")


def prepare_llm_input(scene_json_string: str) -> str:
    """
    Prepares the full LLM input combining system prompt and scene data.

    :param scene_json_string: JSON string representation of the scene graph.
    :type scene_json_string: str
    :raises TypeError: If scene_json_string is not a string.
    :return: Complete input string for LLM with system prompt and scene data.
    :rtype: str
    """
    try:
        if not isinstance(scene_json_string, str):
            raise TypeError("scene_json_string must be a string")

        return f"{ORIGINAL_LLM_PROMPT}\n\n{scene_json_string}"

    except TypeError as err:
        traceback.print_exc()
        raise RuntimeError(f"Error preparing LLM input: {err}")


def process_pkl_to_llm_format(pkl_path: str) -> str:
    """
    End-to-end conversion from pkl.gz file to LLM-ready input format.

    :param pkl_path: Path to the .pkl.gz file containing scene graph data.
    :type pkl_path: str
    :raises RuntimeError: If any step in the conversion process fails.
    :return: Complete formatted string ready for LLM input.
    :rtype: str
    """
    try:
        scene_objects = load_scene_graph_from_pkl(pkl_path)
        scene_json_string = format_scene_graph_as_json_string(scene_objects)
        llm_input = prepare_llm_input(scene_json_string)
        return llm_input

    except RuntimeError as err:
        traceback.print_exc()
        raise RuntimeError(f"Error processing pkl to LLM format: {err}")


def parse_json(input_string: str) -> dict:
    """
    Extracts and parses a JSON object from a string, handling Markdown code blocks.

    :param input_string: The raw string output from the LLM containing JSON.
    :type input_string: str
    :return: The parsed JSON content as a dictionary.
    :rtype: dict
    :raises TypeError: If the input is not a string.
    :raises ValueError: If no valid JSON object is found or if parsing fails.
    """
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string")

    try:
        json_match = re.search(r"(\{.*\})", input_string, re.DOTALL)

        if not json_match:
            raise ValueError("No JSON object found using regex extraction.")

        json_str = json_match.group(1)

        return json.loads(json_str)

    except (json.JSONDecodeError, ValueError):
        traceback.print_exc()
        print(f"Failed to parse JSON content. Content extracted: {input_string}")
        return {}


class OriginalLLMInteraction:
    """
    Handles interaction with LLM using the original ConceptGraphs paper format.
    """

    def __init__(self, pkl_path: str, model_id: str) -> None:
        """
        Initializes the interaction handler with scene graph data.

        :param pkl_path: Path to the .pkl.gz file containing scene graph data.
        :type pkl_path: str
        :param model_id: Model identifier for OpenRouter.
        :type model_id: str
        :raises RuntimeError: If initialization fails.
        """
        try:
            self.scene_objects = load_scene_graph_from_pkl(pkl_path)
            self.scene_json_string = format_scene_graph_as_json_string(
                self.scene_objects
            )
            self.agent = Agent(
                model=OpenRouter(
                    id=model_id,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    max_tokens=8192,
                    system_prompt=prepare_llm_input(self.scene_json_string),
                    reasoning_effort="minimal",
                ),
                markdown=True,
            )

        except RuntimeError as err:
            traceback.print_exc()
            raise RuntimeError(f"Error initializing OriginalLLMInteraction: {err}")

    def get_system_prompt(self) -> str:
        """
        Returns the system prompt for LLM.

        :return: System prompt string from ORIGINAL_LLM_PROMPT.
        :rtype: str
        """
        return ORIGINAL_LLM_PROMPT

    def get_scene_json(self) -> str:
        """
        Returns the scene graph as JSON string.

        :return: JSON formatted scene graph string.
        :rtype: str
        """
        return self.scene_json_string

    def retrieve_object_by_id(self, object_id: str) -> dict[str, Any]:
        """
        Retrieves an object from the scene graph by its ID.

        :param object_id: The ID of the object to retrieve.
        :type object_id: str
        :raises ValueError: If the object with the specified ID is not found.
        :return: The object dictionary if found.
        :rtype: dict[str, Any]
        """
        for obj in self.scene_objects:
            if obj.get("id") == object_id:
                return obj
        raise ValueError(f"Object with ID {object_id} not found in scene graph.")

    def process_query(self, query: str) -> tuple[str | None, list[str | None], RunResponse]:
        """
        Processes a user query using the LLM agent.

        :param query: User query string.
        :type query: str
        :raises RuntimeError: If query processing fails.
        :return: Tuple containing most relevant object class, top 3 object classes, and LLM response.
        :rtype: tuple[str | None, list[str | None], RunResponse]
        """
        try:
            response = self.agent.run(query)

            try:
                CostEstimator().register_execution("interaction", response)
            except (TypeError, ValueError, RuntimeError):
                traceback.print_exc()

            parsed_response = parse_json(response.content)

            if "final_relevant_objects" not in parsed_response:
                print(
                    "No 'final_relevant_objects' found in the response: ",
                    response.content,
                )

            most_relevant = None
            top_3_classes = []

            for idx, obj in enumerate(
                parsed_response.get("final_relevant_objects", [])
            ):
                try:
                    scene_object = self.retrieve_object_by_id(obj)
                    if idx == 0:
                        most_relevant = scene_object.get("object_tag", None)
                    top_3_classes.append(scene_object.get("object_tag", None))
                    if len(top_3_classes) == 3:
                        break
                except ValueError as ve:
                    traceback.print_exc()
                    print(ve)

            return (most_relevant, top_3_classes, response)

        except RuntimeError as err:
            traceback.print_exc()
            raise RuntimeError(f"Error processing query: {err}")


if __name__ == "__main__":
    load_dotenv()
    DATABASE_PATH = r"D:\Documentos\Datasets\Robot@VirtualHomeLarge"
    HOME_ID = 1
    pkl_file_path = os.path.join(
        DATABASE_PATH,
        "outputs",
        f"Home{HOME_ID:02d}",
        "Wandering",
        "exps",
        f"original_house_{HOME_ID}_map",
        f"pcd_original_house_{HOME_ID}_map.pkl.gz",
    )

    llm_interaction = OriginalLLMInteraction(
        pkl_file_path, model_id="openai/gpt-4o"
    )

    try:
        most_relevant, top_3_classes, response = llm_interaction.process_query(
            "I want to bake a cake"
        )
        print("Most Relevant Object ID:", most_relevant)
        print("Top 3 Object Classes:", top_3_classes)
        print("Full LLM Response:", response.content)
    except RuntimeError as e:
        print(f"An error occurred: {e}")

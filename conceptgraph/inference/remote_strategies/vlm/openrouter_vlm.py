from agno.models.openrouter import OpenRouter
from agno.agent import Agent, RunResponse
from agno.media import Image as AgnoImage
import traceback
import logging
import json
import re

from conceptgraph.utils.vlm import extract_list_of_tuples, vlm_extract_object_captions
from conceptgraph.inference.interfaces import IVLM


class OpenrouterVLM(IVLM):
    """
    Provides Vision-Language Model (VLM) inference using OpenRouter with Remote VLM.
    Handles scene understanding, object captioning, spatial reasoning, and query answering.
    """

    # This class implements a remote VLM interface using OpenRouter as the backend.

    def __init__(
        self,
        model_id: str = "google/gemini-2.5-flash-lite",
        api_key: str | None = None,
    ) -> None:
        """
        Initialize OpenrouterVLM.

        :param model_id: Model identifier for OpenRouter.
        :type model_id: str
        :param api_key: Optional API key for OpenRouter.
        :type api_key: str | None
        """
        self.model_id = model_id
        self.api_key = api_key
        self.agent: Agent | None = None
        self.__prompts = {
            "relations": "",
            "captions": "",
            "room_class": "",
            "consolidate": "",
            "class_env": "",
        }

    def load_model(self) -> None:
        """
        Initialize the VLM agent with OpenRouter backend.

        :raises RuntimeError: If agent initialization fails.
        :return: None
        :rtype: None
        """
        if not self.is_loaded():
            logging.info(f"Loading OpenrouterVLM agent with model: {self.model_id}")
            try:
                self.agent = Agent(
                    model=OpenRouter(id=self.model_id, api_key=self.api_key),
                    markdown=True,
                )
            except (ValueError, RuntimeError, ConnectionError) as e:
                traceback.print_exc()
                raise RuntimeError(f"Failed to load OpenrouterVLM agent: {e}")

    def unload_model(self) -> None:
        """
        Unload the VLM agent.

        :return: None
        :rtype: None
        """
        if self.is_loaded():
            logging.info("Unloading OpenrouterVLM agent.")
            del self.agent
            self.agent = None

    def is_loaded(self) -> bool:
        """
        Check if the VLM agent is loaded.

        :return: True if loaded, False otherwise.
        :rtype: bool
        """
        return self.agent is not None

    def get_type(self) -> str:
        """
        Get the type of the VLM.

        :return: The string "remote".
        :rtype: str
        """
        return "remote"

    def get_relations(self, annotated_image_path: str, labels: list[str]) -> list:
        """
        Extract spatial relations between labeled objects in an image.

        :param annotated_image_path: Path to the annotated image.
        :type annotated_image_path: str
        :param labels: List of object labels (e.g., ["1: chair", "2: table"]).
        :type labels: list[str]
        :return: List of spatial relation tuples.
        :rtype: list
        :raises RuntimeError: If relation extraction fails.
        """
        self.load_model()
        try:
            prompt = f"{self.__prompts['relations']}\n\n{', '.join(labels)}"
            response: RunResponse = Agent(
                model=OpenRouter(id=self.model_id, api_key=self.api_key),
                markdown=True,
            ).run(prompt, images=[AgnoImage(filepath=annotated_image_path)])
            relations = extract_list_of_tuples(response.content)
            return relations
        except (AttributeError, ValueError, RuntimeError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Remote relation extraction failed: {e}")

    def get_captions(self, annotated_image_path: str, labels: list[str]) -> list:
        """
        Generate captions for each labeled object in an image.

        :param annotated_image_path: Path to the annotated image.
        :type annotated_image_path: str
        :param labels: List of object labels.
        :type labels: list[str]
        :return: List of captions corresponding to each object.
        :rtype: list
        :raises RuntimeError: If caption generation fails.
        """
        self.load_model()
        try:
            prompt = f"{self.__prompts['captions']}\n\n{', '.join(labels)}"
            response: RunResponse = Agent(
                model=OpenRouter(id=self.model_id, api_key=self.api_key),
                markdown=True,
            ).run(prompt, images=[AgnoImage(filepath=annotated_image_path)])
            captions = vlm_extract_object_captions(response.content)
            return captions
        except (AttributeError, ValueError, RuntimeError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Remote caption generation failed: {e}")

    def get_room_data(self, image_path: str, context: list) -> dict:
        """
        Classify the room or environment type from an image.

        :param image_path: Path to the image.
        :type image_path: str
        :param context: Additional contextual information (currently unused).
        :type context: list
        :return: Dictionary with room classification data.
        :rtype: dict
        :raises RuntimeError: If room classification fails.
        """
        self.load_model()
        try:
            response: RunResponse = Agent(
                model=OpenRouter(id=self.model_id, api_key=self.api_key),
                markdown=True,
            ).run(self.__prompts["room_class"], images=[AgnoImage(filepath=image_path)])
            json_match = re.search(
                r"```json\s*(\{.*?\})\s*```", response.content, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(1)
                room_data = json.loads(json_str)
            else:
                room_data = json.loads(response.content)
            return room_data
        except (AttributeError, ValueError, json.JSONDecodeError, RuntimeError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Remote room classification failed: {e}")

    def consolidate_captions(self, captions: list) -> str:
        """
        Consolidate multiple captions into a single coherent description.

        :param captions: List of captions to consolidate.
        :type captions: list
        :return: Consolidated caption.
        :rtype: str
        :raises RuntimeError: If consolidation fails.
        """
        self.load_model()
        try:
            captions_strs = [
                c["caption"] if isinstance(c, dict) and "caption" in c else str(c)
                for c in captions
            ]
            prompt = f"{self.__prompts['consolidate']}\n\n{', '.join(captions_strs)}"
            response: RunResponse = Agent(
                model=OpenRouter(
                    id="openai/gpt-oss-120b:nitro",
                    reasoning_effort="low",
                    api_key=self.api_key,
                ),
                markdown=True,
            ).run(prompt)
            return response.content
        except (AttributeError, ValueError, RuntimeError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Remote caption consolidation failed: {e}")

    def set_prompts(self, prompt_dict: dict) -> None:
        """
        Sets or updates the system prompts used for VLM/LLM inference.

        :param prompt_dict: Dictionary containing prompt configurations.
        :type prompt_dict: dict
        """
        self.__prompts = prompt_dict

    def classify_environment(self, image_path: str) -> str:
        """
        Classifies the overall environment type from an image.

        :param image_path: Path to the image file.
        :type image_path: str
        :return: Classified environment type as a string.
        :rtype: str
        """
        self.load_model()
        try:
            max_retries = 5
            last_exception = None

            for attempt in range(max_retries):
                try:
                    response: RunResponse = self.agent.run(
                        self.__prompts["class_env"],
                        images=[AgnoImage(filepath=image_path)],
                    )
                    json_match = re.search(
                        r"```json\s*(\{.*?\})\s*```", response.content, re.DOTALL
                    )
                    if json_match:
                        json_str = json_match.group(1)
                        class_env = json.loads(json_str)
                    else:
                        class_env = json.loads(response.content)

                    if "class" in class_env:
                        return class_env["class"]
                    else:
                        raise ValueError(
                            "No 'class' key found in environment classification response."
                        )
                except (
                    AttributeError,
                    ValueError,
                    json.JSONDecodeError,
                    RuntimeError,
                ) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"Environment classification attempt {attempt + 1} failed: {e} -> Content: {response.content if 'response' in locals() else 'N/A'}. Retrying..."
                        )
                        continue
                    else:
                        break

            traceback.print_exc()
            raise RuntimeError(
                f"Environment classification failed after {max_retries} attempts: {last_exception}"
            )
        except (AttributeError, ValueError, RuntimeError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Environment classification failed: {e}")

    def query_map(self, query: str, map_context: str, tools: list) -> str:
        """
        Answer user queries about the map using LLM inference.

        :param query: User question.
        :type query: str
        :param map_context: Context information about the map.
        :type map_context: str
        :param tools: Available tools for query resolution.
        :type tools: list
        :return: Answer to the query.
        :rtype: str
        :raises RuntimeError: If query processing fails.
        """
        self.load_model()
        try:
            prompt = f"Context: {map_context}\n\nQuery: {query}"
            response: RunResponse = self.agent.run(prompt)
            return response.content
        except (AttributeError, ValueError, RuntimeError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Remote query processing failed: {e}")

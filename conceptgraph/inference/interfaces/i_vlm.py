from abc import abstractmethod

from .i_inference_strategy import IInferenceStrategy


class IVLM(IInferenceStrategy):
    """
    Interface for Vision-Language Model (VLM) and Large Language Model (LLM) services.
    Extends IInferenceStrategy to provide mapping and inference capabilities for scene understanding.
    """

    @abstractmethod
    def get_relations(self, annotated_image_path: str, labels: list[str]) -> list:
        """
        Extracts spatial relations between objects in a scene.

        :param annotated_image_path: Path to the annotated image file.
        :type annotated_image_path: str
        :param labels: List of object labels.
        :type labels: list[str]
        :return: List of spatial relations.
        :rtype: list
        """
        pass

    @abstractmethod
    def get_captions(self, annotated_image_path: str, labels: list[str]) -> list:
        """
        Extracts captions for detected objects.

        :param annotated_image_path: Path to the annotated image file.
        :type annotated_image_path: str
        :param labels: List of object labels.
        :type labels: list[str]
        :return: List of captions for each object.
        :rtype: list
        """
        pass

    @abstractmethod
    def get_room_data(self, image_path: str, context: list) -> dict:
        """
        Classifies the environment or room type.

        :param image_path: Path to the image file.
        :type image_path: str
        :param context: Contextual information about the scene.
        :type context: list
        :return: Dictionary containing room classification data.
        :rtype: dict
        """
        pass

    @abstractmethod
    def consolidate_captions(self, captions: list) -> str:
        """
        Consolidates multiple captions into a single coherent description.

        :param captions: List of captions to consolidate.
        :type captions: list
        :return: Consolidated caption.
        :rtype: str
        """
        pass

    @abstractmethod
    def set_prompts(self, prompt_dict: dict) -> None:
        """
        Sets or updates the system prompts used for VLM/LLM inference.

        :param prompt_dict: Dictionary containing prompt configurations.
        :type prompt_dict: dict
        """
        pass

    @abstractmethod
    def classify_environment(self, image_path: str) -> str:
        """
        Classifies the overall environment type from an image.

        :param image_path: Path to the image file.
        :type image_path: str
        :return: Classified environment type as a string.
        :rtype: str
        """
        pass

    @abstractmethod
    def query_map(self, query: str, map_context: str, tools: list) -> str:
        """
        Answers user queries about the map using LLM inference.

        :param query: User question or query.
        :type query: str
        :param map_context: Context information about the map.
        :type map_context: str
        :param tools: List of available tools for query resolution.
        :type tools: list
        :return: Answer to the user query.
        :rtype: str
        """
        pass

from supervision import Detections
from abc import abstractmethod
import numpy as np

from .i_inference_strategy import IInferenceStrategy


class IObjectDetector(IInferenceStrategy):
    """
    Interface for object detection strategies.
    Extends IInferenceStrategy to include class configuration and detection capabilities.
    """

    @abstractmethod
    def set_classes(self, classes: list[str]):
        """
        Sets the classes that the detector should search for.

        :param classes: List of class names to detect.
        :type classes: list[str]
        """
        pass

    @abstractmethod
    def detect(self, image_path: str, image_np: np.ndarray) -> Detections:
        """
        Executes object detection and returns detections without masks.

        :param image_path: Path to the image file.
        :type image_path: str
        :param image_np: Image as a numpy array.
        :type image_np: np.ndarray
        :return: Detection results.
        :rtype: Detections
        """
        pass

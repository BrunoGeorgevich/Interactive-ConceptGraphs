from supervision import Detections
from abc import abstractmethod
import numpy as np

from .i_inference_strategy import IInferenceStrategy


class IFeatureExtractor(IInferenceStrategy):
    """
    Interface for feature extraction strategies.
    Extends IInferenceStrategy to extract visual and textual features from detections.
    """

    @abstractmethod
    def extract_features(
        self, image_np: np.ndarray, detections: Detections, classes: list[str]
    ) -> tuple:
        """
        Extracts features from image detections.

        :param image_np: Image as a numpy array.
        :type image_np: np.ndarray
        :param detections: Detection results.
        :type detections: Detections
        :param classes: List of class names.
        :type classes: list[str]
        :return: Tuple containing image crops, image features, and text features as tensors.
        :rtype: tuple
        """
        pass

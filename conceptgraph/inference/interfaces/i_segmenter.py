from abc import abstractmethod
import numpy as np
import torch

from .i_inference_strategy import IInferenceStrategy


class ISegmenter(IInferenceStrategy):
    """
    Interface for segmentation strategies.
    Extends IInferenceStrategy to provide mask generation from bounding boxes.
    """

    @abstractmethod
    def segment(
        self, image_path: str, image_np: np.ndarray, boxes: torch.Tensor, classes: list[str]
    ) -> torch.Tensor:
        """
        Receives bounding boxes and returns segmentation masks.

        :param image_path: Path to the image file.
        :type image_path: str
        :param image_np: Image as a numpy array.
        :type image_np: np.ndarray
        :param boxes: Bounding boxes as a tensor.
        :type boxes: torch.Tensor
        :param classes: List of class names corresponding to detection class IDs.
        :type classes: list[str]
        :return: Segmentation masks as a tensor.
        :rtype: torch.Tensor
        """
        pass

from ultralytics import SAM
import numpy as np
import traceback
import logging
import torch

from conceptgraph.inference.interfaces import ISegmenter


class SAMSegmenterStrategy(ISegmenter):
    """
    SAM (Segment Anything Model) segmentation strategy.
    Encapsulates preprocessing, segmentation, and postprocessing for SAM models.
    """

    def __init__(self, checkpoint_path: str, device: str) -> None:
        """
        Initialize SAMSegmenterStrategy.

        :param checkpoint_path: Path to the SAM model checkpoint.
        :type checkpoint_path: str
        :param device: Device to load the model on ("cuda" or "cpu").
        :type device: str
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model: SAM | None = None

    def load_model(self) -> None:
        """
        Load the SAM model if not already loaded.

        :raises RuntimeError: If model loading fails.
        """
        if not self.is_loaded():
            logging.info(f"Loading SAM segmenter model from: {self.checkpoint_path}")
            try:
                self.model = SAM(self.checkpoint_path).to(self.device)
            except (OSError, ValueError) as e:
                traceback.print_exc()
                raise RuntimeError(f"Failed to load SAM model: {e}")

    def unload_model(self) -> None:
        """
        Unload the SAM model and clear CUDA cache if applicable.

        """
        if self.is_loaded():
            logging.info("Unloading SAM segmenter model.")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        """
        Check if the SAM model is loaded.

        :return: True if loaded, False otherwise.
        :rtype: bool
        """
        return self.model is not None

    def get_type(self) -> str:
        """
        Get the type of the segmenter.

        :return: The string "local".
        :rtype: str
        """
        return "local"

    def _preprocess(
        self, image_path: str, image_np: np.ndarray, boxes: torch.Tensor
    ) -> tuple[str, torch.Tensor]:
        """
        Preprocess the image and boxes before segmentation.

        :param image_path: Path to the image file.
        :type image_path: str
        :param image_np: Image as a numpy array.
        :type image_np: np.ndarray
        :param boxes: Bounding boxes as a tensor.
        :type boxes: torch.Tensor
        :return: Preprocessed image path and boxes.
        :rtype: tuple[str, torch.Tensor]
        """
        return image_path, boxes

    def _process(self, image_path: str, boxes: torch.Tensor) -> torch.Tensor:
        """
        Run SAM inference on the image with bounding boxes.

        :param image_path: Path to the image file.
        :type image_path: str
        :param boxes: Bounding boxes as a tensor.
        :type boxes: torch.Tensor
        :return: Segmentation masks.
        :rtype: torch.Tensor
        """
        result = self.model(image_path, bboxes=boxes, verbose=False)[0]
        return result.masks.data

    def _postprocess(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Postprocess segmentation masks.

        :param masks: Raw segmentation masks.
        :type masks: torch.Tensor
        :return: Postprocessed masks.
        :rtype: torch.Tensor
        """
        return masks

    def segment(
        self,
        image_path: str,
        image_np: np.ndarray,
        boxes: torch.Tensor,
        classes: list[str],
    ) -> torch.Tensor:
        """
        Perform segmentation on an image given bounding boxes.
        Encapsulates the full preprocessing, processing, and postprocessing pipeline.

        :param image_path: Path to the image file.
        :type image_path: str
        :param image_np: Image as a numpy array.
        :type image_np: np.ndarray
        :param boxes: Bounding boxes as a tensor (Nx4 format).
        :type boxes: torch.Tensor
        :param classes: List of class names corresponding to detection class IDs.
        :type classes: list[str]
        :return: Segmentation masks as a tensor.
        :rtype: torch.Tensor
        :raises RuntimeError: If segmentation fails or model is not loaded.
        """
        self.load_model()
        try:
            preprocessed_path, preprocessed_boxes = self._preprocess(
                image_path, image_np, boxes
            )
            masks = self._process(preprocessed_path, preprocessed_boxes)
            return self._postprocess(masks)
        except (AttributeError, TypeError, IndexError, ValueError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Segmentation failed: {e}")

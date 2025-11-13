from supervision import Detections
from ultralytics import YOLO
import numpy as np
import traceback
import logging
import torch

from conceptgraph.inference.interfaces import IObjectDetector


class YOLODetectorStrategy(IObjectDetector):
    """
    YOLO-based object detection strategy.
    Encapsulates preprocessing, detection, and postprocessing for YOLO models.
    """

    def __init__(self, checkpoint_path: str, device: str) -> None:
        """
        Initialize YOLODetectorStrategy.

        :param checkpoint_path: Path to the YOLO model checkpoint.
        :type checkpoint_path: str
        :param device: Device to load the model on ("cuda" or "cpu").
        :type device: str
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model: YOLO | None = None

    def load_model(self) -> None:
        """
        Load the YOLO model if not already loaded.

        :raises RuntimeError: If model loading fails.
        :return: None
        :rtype: None
        """
        if not self.is_loaded():
            logging.info(f"Loading YOLO detector model from: {self.checkpoint_path}")
            try:
                self.model = YOLO(self.checkpoint_path).to(self.device)
            except (OSError, ValueError) as e:
                traceback.print_exc()
                raise RuntimeError(f"Failed to load YOLO model: {e}")

    def unload_model(self) -> None:
        """
        Unload the YOLO model and clear CUDA cache if applicable.

        :return: None
        :rtype: None
        """
        if self.is_loaded():
            logging.info("Unloading YOLO detector model.")
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        """
        Check if the YOLO model is loaded.

        :return: True if loaded, False otherwise.
        :rtype: bool
        """
        return self.model is not None

    def get_type(self) -> str:
        """
        Get the type of the detector.

        :return: The string "local".
        :rtype: str
        """
        return "local"

    def set_classes(self, classes: list[str]) -> None:
        """
        Set the classes for the YOLO model.

        :param classes: List of class names.
        :type classes: list[str]
        :raises RuntimeError: If setting classes fails.
        :return: None
        :rtype: None
        """
        self.load_model()
        try:
            self.model.set_classes(classes)
        except (AttributeError, TypeError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Failed to set classes: {e}")

    def _preprocess(self, image_path: str, image_np: np.ndarray) -> str:
        """
        Preprocess the image before detection.

        :param image_path: Path to the image file.
        :type image_path: str
        :param image_np: Image as a numpy array.
        :type image_np: np.ndarray
        :return: Preprocessed image path.
        :rtype: str
        """
        return image_path

    def _process(self, image_path: str) -> Detections:
        """
        Run YOLO inference on the image.

        :param image_path: Path to the image file.
        :type image_path: str
        :return: Raw detection results.
        :rtype: Detections
        """
        result = self.model(image_path, verbose=False)[0]
        return Detections.from_ultralytics(result)

    def _postprocess(self, detections: Detections) -> Detections:
        """
        Postprocess detection results.

        :param detections: Raw detection results.
        :type detections: Detections
        :return: Postprocessed detections.
        :rtype: Detections
        """
        return detections

    def detect(self, image_path: str, image_np: np.ndarray) -> Detections:
        """
        Perform object detection on an image.
        Encapsulates the full preprocessing, processing, and postprocessing pipeline.

        :param image_path: Path to the image file.
        :type image_path: str
        :param image_np: Image as a numpy array.
        :type image_np: np.ndarray
        :return: Detections object containing bounding boxes, confidences, and class IDs.
        :rtype: Detections
        :raises RuntimeError: If detection fails or model is not loaded.
        """
        self.load_model()
        try:
            preprocessed_path = self._preprocess(image_path, image_np)
            detections = self._process(preprocessed_path)
            return self._postprocess(detections)
        except (AttributeError, TypeError, IndexError, ValueError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Detection failed: {e}")

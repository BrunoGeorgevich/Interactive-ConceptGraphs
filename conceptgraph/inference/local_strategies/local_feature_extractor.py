from supervision import Detections
import numpy as np
import traceback
import logging
import warnings
import torch
import os
warnings.filterwarnings(
    "ignore", category=UserWarning, module="huggingface_hub"
)

from conceptgraph.inference.interfaces import IFeatureExtractor


class LocalFeatureExtractor(IFeatureExtractor):
    """
    Provides feature extraction using CLIP (Contrastive Language-Image Pre-training).
    Extracts visual and textual features from image regions defined by detections.
    """

    def __init__(
        self, model_name: str, pretrained: str | None = None, device: str = "cpu"
    ) -> None:
        """
        Initialize LocalFeatureExtractor.

        :param model_name: Name of the CLIP model to use.
        :type model_name: str
        :param pretrained: Pretrained weights identifier.
        :type pretrained: str
        :param device: Device to load the model on ("cuda" or "cpu").
        :type device: str
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def load_model(self) -> None:
        """
        Load the CLIP model if not already loaded.

        :raises RuntimeError: If model loading fails.
        :return: None
        :rtype: None
        """
        model_cache_dir = "models"

        if os.path.exists(model_cache_dir):
            cache_exists = any(
                entry.lower().startswith(
                    f"models--laion--CLIP-{self.model_name}-{self.pretrained.replace('_', '-')}".lower()
                )
                for entry in os.listdir(model_cache_dir)
                if os.path.isdir(os.path.join(model_cache_dir, entry))
            )
            if cache_exists:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import open_clip

        if not self.is_loaded():
            logging.info(
                f"Loading LocalFeatureExtractor model: {self.model_name} ({self.pretrained})"
            )
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained=self.pretrained,
                    device=self.device,
                    cache_dir="models",
                )
                self.tokenizer = open_clip.get_tokenizer(
                    self.model_name, cache_dir="models"
                )
            except (OSError, ValueError, RuntimeError) as e:
                traceback.print_exc()
                raise RuntimeError(f"Failed to load CLIP model: {e}")

    def unload_model(self) -> None:
        """
        Unload the CLIP model and clear CUDA cache if applicable.

        :return: None
        :rtype: None
        """
        if self.is_loaded():
            logging.info("Unloading LocalFeatureExtractor model.")
            del self.model
            del self.preprocess
            del self.tokenizer
            self.model = None
            self.preprocess = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        """
        Check if the CLIP model is loaded.

        :return: True if loaded, False otherwise.
        :rtype: bool
        """
        return self.model is not None

    def get_type(self) -> str:
        """
        Get the type of the feature extractor.

        :return: The string "local".
        :rtype: str
        """
        return "local"

    def extract_features(
        self, image_np: np.ndarray, detections: Detections, classes: list[str]
    ) -> tuple:
        """
        Extract CLIP features from detected objects in an image.

        :param image_np: Image as a numpy array.
        :type image_np: np.ndarray
        :param detections: Detections object with bounding boxes and masks.
        :type detections: Detections
        :param classes: List of class names corresponding to detection class IDs.
        :type classes: list[str]
        :return: Tuple of (image_crops, image_features, text_features).
        :rtype: tuple
        :raises RuntimeError: If feature extraction fails or model is not loaded.
        """
        from conceptgraph.utils.model_utils import compute_clip_features_batched

        self.load_model()
        try:
            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image=image_np,
                detections=detections,
                clip_model=self.model,
                clip_preprocess=self.preprocess,
                clip_tokenizer=self.tokenizer,
                classes=classes,
                device=self.device,
            )
            return image_crops, image_feats, text_feats
        except (AttributeError, TypeError, ValueError, RuntimeError) as e:
            traceback.print_exc()
            raise RuntimeError(f"Feature extraction failed: {e}")

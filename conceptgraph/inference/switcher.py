from dotenv import load_dotenv
from pathlib import Path
import traceback
import numpy as np
import logging
import httpx
import torch
import agno
import cv2
import os

from conceptgraph.inference.local_strategies.detector.yolo_detector_strategy import (
    YOLODetectorStrategy,
)
from conceptgraph.inference.local_strategies.segmenter.sam_segmenter_strategy import (
    SAMSegmenterStrategy,
)
from conceptgraph.inference.remote_strategies.detector.gemini_detector_strategy import (
    GeminiDetectorStrategy,
)
from conceptgraph.inference.remote_strategies.segmenter.gemini_segmenter_strategy import (
    GeminiSegmenterStrategy,
)
from conceptgraph.inference.local_strategies import LocalFeatureExtractor, LMStudioVLM
from conceptgraph.inference.remote_strategies import OpenrouterVLM
from conceptgraph.utils.prompts import (
    SYSTEM_PROMPT_ONLY_TOP,
    SYSTEM_PROMPT_CAPTIONS,
    SYSTEM_PROMPT_ROOM_CLASS,
    SYSTEM_PROMPT_CONSOLIDATE_CAPTIONS,
    ENVIRONMENT_CLASSIFIER,
)
from conceptgraph.inference.interfaces import (
    IObjectDetector,
    ISegmenter,
    IFeatureExtractor,
    IVLM,
)


class SystemContext:
    """
    Stores the current system state including connectivity and environment profile.
    """

    def __init__(self) -> None:
        """
        Initialize the system context with default values.

        :return: None
        :rtype: None
        """
        self.connectivity_status: str = "offline"
        self.environment_profile: str = "indoor"
        self.update()

    def update(self) -> None:
        """
        Update the current system state by checking network connectivity.

        :return: None
        :rtype: None
        """
        self.connectivity_status = self._check_network()

    def set_environment_profile(self, profile: str) -> None:
        """
        Set the environment profile based on classification.

        :param profile: The environment profile ('indoor' or 'outdoor').
        :type profile: str
        :raises ValueError: If profile is not 'indoor' or 'outdoor'.
        :return: None
        :rtype: None
        """
        if profile not in {"indoor", "outdoor"}:
            raise ValueError(
                f"Invalid profile: {profile}. Must be 'indoor' or 'outdoor'"
            )
        self.environment_profile = profile
        logging.info(f"Environment profile set to: {profile}")

    def _check_network(self, timeout: float = 2.0) -> str:
        """
        Check internet connectivity by attempting to reach a known endpoint.

        :param timeout: Maximum time to wait for response in seconds.
        :type timeout: float
        :return: Connectivity status as 'online' or 'offline'.
        :rtype: str
        """
        try:
            httpx.get("http://www.google.com", timeout=timeout)
            logging.debug("Network check: Online")
            return "online"
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
            logging.debug("Network check: Offline")
            return "offline"


class StrategySwitcher:
    """
    Manages dynamic switching between preferred and fallback models based on connectivity.
    """

    def __init__(
        self,
        preferred_factories: dict[str, callable],
        fallback_factories: dict[str, callable],
        shared_model: IFeatureExtractor,
        vlm_prompts: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the strategy switcher with factory functions for preferred and fallback models.

        :param preferred_factories: Dictionary with keys 'det', 'seg', 'vlm' containing factory functions for preferred models.
        :type preferred_factories: dict[str, callable]
        :param fallback_factories: Dictionary with keys 'det', 'seg', 'vlm' containing factory functions for fallback models.
        :type fallback_factories: dict[str, callable]
        :param shared_model: Shared CLIP model used across all contexts.
        :type shared_model: IFeatureExtractor
        :param vlm_prompts: Optional dictionary of prompts to set for VLMs.
        :type vlm_prompts: dict[str, str] | None
        :raises ValueError: If required keys are missing from either dictionary.
        :return: None
        :rtype: None
        """
        required_keys: set[str] = {"det", "seg", "vlm"}

        if not required_keys.issubset(preferred_factories.keys()):
            raise ValueError(f"preferred_factories must contain keys: {required_keys}")
        if not required_keys.issubset(fallback_factories.keys()):
            raise ValueError(f"fallback_factories must contain keys: {required_keys}")

        self.preferred_factories: dict[str, callable] = preferred_factories
        self.fallback_factories: dict[str, callable] = fallback_factories
        self.preferred_models: dict[
            str, IObjectDetector | ISegmenter | IFeatureExtractor | IVLM
        ] = {}
        self.fallback_models: dict[
            str, IObjectDetector | ISegmenter | IFeatureExtractor | IVLM
        ] = {}
        self.active_models: dict[
            str, IObjectDetector | ISegmenter | IFeatureExtractor | IVLM
        ] = {}
        self.vlm_prompts: dict[str, str] = vlm_prompts.copy() if vlm_prompts else {}
        self.shared_clip: IFeatureExtractor = shared_model
        self.system_context: SystemContext = SystemContext()
        self.is_offline_mode: bool = False

    def set_vlm_prompts(self, prompts: dict[str, str]) -> None:
        """
        Set custom prompts for all VLMs managed by the switcher.

        :param prompts: Dictionary of prompt names and their string values.
        :type prompts: dict[str, str]
        :return: None
        :rtype: None
        """
        self.vlm_prompts = prompts.copy() if prompts else {}
        for model_dict in (
            self.preferred_models,
            self.fallback_models,
            self.active_models,
        ):
            vlm = model_dict.get("vlm")
            if vlm and hasattr(vlm, "set_prompts"):
                vlm.set_prompts(self.vlm_prompts)

    def classify_environment(self, image_path: str) -> str:
        """
        Classify the environment as indoor or outdoor using VLM.

        :param image_path: Path to the image to classify.
        :type image_path: str
        :raises RuntimeError: If environment classification fails.
        :return: Environment classification ('indoor' or 'outdoor').
        :rtype: str
        """
        try:
            environment_class = self.execute_with_fallback(
                "vlm", "classify_environment", image_path
            )

            if isinstance(environment_class, str) and environment_class in {
                "indoor",
                "outdoor",
            }:
                self.system_context.set_environment_profile(environment_class)
                return environment_class
            else:
                raise ValueError(
                    f"Invalid environment class returned: {environment_class}"
                )

        except (RuntimeError, ValueError, TypeError) as classification_error:
            traceback.print_exc()
            raise RuntimeError(
                f"Environment classification failed: {classification_error}"
            )

    def get_classes_for_environment(self) -> list[str]:
        """
        Get appropriate object classes based on current environment profile.

        :raises RuntimeError: If class file cannot be found or read.
        :return: List of class names for current environment.
        :rtype: list[str]
        """
        if self.system_context.environment_profile == "indoor":
            class_file = "conceptgraph/indoor_classes.txt"
        else:
            class_file = "conceptgraph/outdoor_classes.txt"

        try:
            with open(class_file, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
            return classes
        except (FileNotFoundError, OSError):
            traceback.print_exc()
            raise RuntimeError(f"Cannot load class file: {class_file}")

    def get_model(
        self, model_key: str
    ) -> IObjectDetector | ISegmenter | IFeatureExtractor | IVLM:
        """
        Get the currently active model for the specified key.

        :param model_key: The model key ('det', 'seg', 'clip', 'vlm').
        :type model_key: str
        :raises ValueError: If the model key is invalid.
        :return: The active model instance.
        :rtype: IObjectDetector | ISegmenter | IFeatureExtractor | IVLM
        """
        if model_key == "clip":
            return self.shared_clip

        if model_key not in {"det", "seg", "vlm"}:
            raise ValueError(
                f"Invalid model key: {model_key}. Must be one of: det, seg, clip, vlm"
            )

        if model_key not in self.active_models:
            self._instantiate_preferred_model(model_key)

        return self.active_models[model_key]

    def execute_with_fallback(self, model_key: str, method_name: str, *args, **kwargs):
        """
        Execute a method on the specified model with automatic fallback on failure.

        :param model_key: The model key ('det', 'seg', 'clip', 'vlm').
        :type model_key: str
        :param method_name: Name of the method to execute on the model.
        :type method_name: str
        :param args: Positional arguments to pass to the method.
        :param kwargs: Keyword arguments to pass to the method.
        :raises ValueError: If the model key is invalid.
        :raises AttributeError: If the method does not exist on the model.
        :raises RuntimeError: If both preferred and fallback models fail.
        :return: The result of the method execution.
        """
        if model_key == "clip":
            try:
                method = getattr(self.shared_clip, method_name)
                result = method(*args, **kwargs)
                return result
            except (AttributeError, RuntimeError, ValueError, TypeError) as model_error:
                traceback.print_exc()
                raise RuntimeError(
                    f"Shared CLIP model execution failed for '{method_name}': {model_error}"
                )

        if model_key not in {"det", "seg", "vlm"}:
            raise ValueError(
                f"Invalid model key: {model_key}. Must be one of: det, seg, clip, vlm"
            )

        self._check_and_update_connectivity()

        if model_key not in self.active_models:
            self._instantiate_preferred_model(model_key)

        model = self.active_models[model_key]

        try:
            method = getattr(model, method_name)
            result = method(*args, **kwargs)
            return result

        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.RequestError,
            ConnectionError,
            agno.exceptions.ModelProviderError,
        ):
            logging.warning(
                f"Preferred model '{model_key}' failed due to connectivity issue. Switching to fallback."
            )

            self._switch_model_to_fallback(model_key)

            fallback_model = self.active_models[model_key]

            try:
                method = getattr(fallback_model, method_name)
                result = method(*args, **kwargs)
                return result

            except (
                AttributeError,
                RuntimeError,
                ValueError,
                TypeError,
            ) as fallback_error:
                traceback.print_exc()
                raise RuntimeError(
                    f"Both preferred and fallback models failed for '{model_key}'. Fallback error: {fallback_error}"
                )

        except (AttributeError, RuntimeError, ValueError, TypeError) as model_error:
            traceback.print_exc()
            raise RuntimeError(
                f"Model execution failed for '{model_key}.{method_name}': {model_error}"
            )

    def _instantiate_preferred_model(self, model_key: str) -> None:
        """
        Instantiate the preferred model for the given key if not already instantiated.

        :param model_key: The model key to instantiate.
        :type model_key: str
        :return: None
        :rtype: None
        """
        if model_key not in self.preferred_models:
            logging.info(f"Instantiating preferred model for '{model_key}'")
            try:
                self.preferred_models[model_key] = self.preferred_factories[model_key]()

                if (
                    model_key == "vlm"
                    and self.vlm_prompts
                    and hasattr(self.preferred_models[model_key], "set_prompts")
                ):
                    self.preferred_models[model_key].set_prompts(self.vlm_prompts)

            except (RuntimeError, ValueError, TypeError) as factory_error:
                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to instantiate preferred model for '{model_key}': {factory_error}"
                )

        self.active_models[model_key] = self.preferred_models[model_key]

    def _check_and_update_connectivity(self) -> None:
        """
        Check current connectivity and update model state accordingly.

        :return: None
        :rtype: None
        """
        current_status: str = self._check_network()

        if current_status == "online" and self.is_offline_mode:
            logging.info("Connectivity restored. Switching back to online models.")
            self._switch_back_to_online()
        elif current_status == "offline" and not self.is_offline_mode:
            logging.info("Connectivity lost. Will use offline models when needed.")
            self.is_offline_mode = True

    def _check_network(self, timeout: float = 2.0) -> str:
        """
        Check internet connectivity by attempting to reach a known endpoint.

        :param timeout: Maximum time to wait for response in seconds.
        :type timeout: float
        :return: Connectivity status as 'online' or 'offline'.
        :rtype: str
        """
        try:
            httpx.get("http://www.google.com", timeout=timeout)
            return "online"
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
            return "offline"

    def _switch_model_to_fallback(self, model_key: str) -> None:
        """
        Switch a specific model to its fallback counterpart, instantiating it lazily.

        :param model_key: The model key to switch to fallback.
        :type model_key: str
        :return: None
        :rtype: None
        """
        if model_key not in self.fallback_models:
            logging.info(f"Instantiating fallback model for '{model_key}'")
            try:
                self.fallback_models[model_key] = self.fallback_factories[model_key]()

                if (
                    model_key == "vlm"
                    and self.vlm_prompts
                    and hasattr(self.fallback_models[model_key], "set_prompts")
                ):
                    self.fallback_models[model_key].set_prompts(self.vlm_prompts)

            except (RuntimeError, ValueError, TypeError) as factory_error:
                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to instantiate fallback model for '{model_key}': {factory_error}"
                )

        self.active_models[model_key] = self.fallback_models[model_key]
        self.is_offline_mode = True
        logging.info(f"Switched model '{model_key}' to fallback")

    def _switch_back_to_online(self) -> None:
        """
        Switch all active models back to preferred online models and unload fallback models.

        :return: None
        :rtype: None
        """
        for key in list(self.fallback_models.keys()):
            if (
                key in self.active_models
                and self.active_models[key] == self.fallback_models[key]
            ):
                if key not in self.preferred_models:
                    self._instantiate_preferred_model(key)
                else:
                    self.active_models[key] = self.preferred_models[key]

                fallback_model = self.fallback_models[key]
                if hasattr(fallback_model, "unload_model"):
                    try:
                        fallback_model.unload_model()
                        logging.info(f"Unloaded fallback model for '{key}'")
                    except (RuntimeError, ValueError, TypeError) as unload_error:
                        traceback.print_exc()
                        logging.error(
                            f"Failed to unload fallback model '{key}': {unload_error}"
                        )

                del self.fallback_models[key]

        self.is_offline_mode = False
        logging.info("Switched back to online models")

    def is_using_fallback(self) -> bool:
        """
        Check if the switcher is currently in offline mode.

        :return: True if using offline mode, False otherwise.
        :rtype: bool
        """
        return self.is_offline_mode

    def reset_to_preferred(self) -> None:
        """
        Reset active models to preferred models.

        :return: None
        :rtype: None
        """
        logging.info("Resetting to preferred models")
        self.active_models.clear()
        self.is_offline_mode = False

    def unload_all_models(self) -> None:
        """
        Unload all active models from memory.

        :return: None
        :rtype: None
        """
        for key, model in self.active_models.items():
            if hasattr(model, "unload_model"):
                try:
                    model.unload_model()
                    logging.debug(f"Unloaded model: {key}")
                except (RuntimeError, ValueError, TypeError) as unload_error:
                    traceback.print_exc()
                    logging.error(f"Failed to unload model '{key}': {unload_error}")

        for key, model in self.fallback_models.items():
            if hasattr(model, "unload_model"):
                try:
                    model.unload_model()
                    logging.debug(f"Unloaded fallback model: {key}")
                except (RuntimeError, ValueError, TypeError) as unload_error:
                    traceback.print_exc()
                    logging.error(
                        f"Failed to unload fallback model '{key}': {unload_error}"
                    )

        for key, model in self.preferred_models.items():
            if hasattr(model, "unload_model"):
                try:
                    model.unload_model()
                    logging.debug(f"Unloaded preferred model: {key}")
                except (RuntimeError, ValueError, TypeError) as unload_error:
                    traceback.print_exc()
                    logging.error(
                        f"Failed to unload preferred model '{key}': {unload_error}"
                    )

        if hasattr(self.shared_clip, "unload_model"):
            try:
                self.shared_clip.unload_model()
                logging.debug("Unloaded shared CLIP model")
            except (RuntimeError, ValueError, TypeError) as unload_error:
                traceback.print_exc()
                logging.error(f"Failed to unload shared CLIP model: {unload_error}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    load_dotenv()

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    test_image_path: str = "assets/test_image_2.png"
    output_dir: Path = Path("test_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Initializing shared CLIP model...")
    shared_clip: LocalFeatureExtractor = LocalFeatureExtractor(
        model_name="ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        device=device,
    )

    logging.info("Preparing preferred (remote) model factories...")
    preferred_factories: dict = {
        "det": lambda: GeminiDetectorStrategy(api_key=os.getenv("OPENROUTER_API_KEY")),
        "seg": lambda: GeminiSegmenterStrategy(api_key=os.getenv("GEMINI_API_KEY")),
        "vlm": lambda: OpenrouterVLM(
            model_id="google/gemini-2.5-flash-lite",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        ),
    }

    logging.info("Preparing fallback (local) model factories...")
    fallback_factories: dict = {
        "det": lambda: YOLODetectorStrategy(
            checkpoint_path="models/yolov8x-worldv2.pt",
            device=device,
        ),
        "seg": lambda: SAMSegmenterStrategy(
            checkpoint_path="models/sam2.1_l.pt",
            device=device,
        ),
        "vlm": lambda: LMStudioVLM(
            model_id="qwen/qwen3-vl-8b",
            device=device,
            api_key=None,
        ),
    }

    vlm_prompts: dict[str, str] = {
        "relations": SYSTEM_PROMPT_ONLY_TOP,
        "captions": SYSTEM_PROMPT_CAPTIONS,
        "room_class": SYSTEM_PROMPT_ROOM_CLASS,
        "consolidate": SYSTEM_PROMPT_CONSOLIDATE_CAPTIONS,
        "class_env": ENVIRONMENT_CLASSIFIER,
    }

    switcher: StrategySwitcher = StrategySwitcher(
        preferred_factories=preferred_factories,
        fallback_factories=fallback_factories,
        shared_model=shared_clip,
        vlm_prompts=vlm_prompts,
    )

    logging.info("=" * 50)
    logging.info("STEP 0: Environment Classification")
    logging.info("=" * 50)

    if not Path(test_image_path).exists():
        logging.error(f"Test image not found: {test_image_path}")
        switcher.unload_all_models()
    else:
        try:
            environment_type = switcher.classify_environment(test_image_path)
            logging.info(f"Environment classified as: {environment_type}")

            test_classes = switcher.get_classes_for_environment()
            logging.info(
                f"Loaded {len(test_classes)} classes for {environment_type} environment"
            )

        except (RuntimeError, ValueError, TypeError) as classification_error:
            traceback.print_exc()
            logging.error(f"Environment classification failed: {classification_error}")
            switcher.unload_all_models()
            raise

        logging.info("=" * 50)
        logging.info("STEP 1: Object Detection")
        logging.info("=" * 50)

        image: np.ndarray | None = cv2.imread(test_image_path)

        if image is None:
            logging.error("Failed to load test image")
            switcher.unload_all_models()
        else:
            image_rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height: int
            width: int
            height, width = image_rgb.shape[:2]
            logging.info(f"Image loaded: {width}x{height}")

            try:
                detector = switcher.get_model("det")
                detector.set_classes(test_classes)

                detections = switcher.execute_with_fallback(
                    "det", "detect", test_image_path, image_rgb
                )
                logging.info(f"Detected {len(detections)} objects")

                if len(detections) == 0:
                    logging.warning("No objects detected. Exiting pipeline.")
                    switcher.unload_all_models()
                else:
                    logging.info("=" * 50)
                    logging.info("STEP 2: Segmentation")
                    logging.info("=" * 50)

                    boxes_tensor = torch.from_numpy(detections.xyxy).to(device)
                    classes: list[str] = [
                        test_classes[class_id] for class_id in detections.class_id
                    ]

                    masks = switcher.execute_with_fallback(
                        "seg",
                        "segment",
                        test_image_path,
                        image_rgb,
                        boxes_tensor,
                        classes,
                    )
                    detections.mask = masks.cpu().numpy()
                    logging.info(f"Generated {len(masks)} segmentation masks")

                    logging.info("=" * 50)
                    logging.info("STEP 3: Feature Extraction")
                    logging.info("=" * 50)

                    feature_extractor = switcher.get_model("clip")
                    crops, image_feats, text_feats = feature_extractor.extract_features(
                        image_rgb, detections, test_classes
                    )

                    image_feats_array = np.array(image_feats)
                    text_feats_array = np.array(text_feats)
                    logging.info(
                        f"Extracted features: image_feats={image_feats_array.shape}, text_feats={text_feats_array.shape}"
                    )

                    logging.info("=" * 50)
                    logging.info("STEP 4: VLM Inference")
                    logging.info("=" * 50)

                    vlm_labels: list[str] = [
                        f"{i+1}: {test_classes[class_id]}"
                        for i, class_id in enumerate(detections.class_id)
                    ]

                    vlm_image_path_annotated: Path = output_dir / "vlm_annotated.png"
                    cv2.imwrite(
                        str(vlm_image_path_annotated),
                        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                    )

                    logging.info("Getting room classification...")
                    room_data = switcher.execute_with_fallback(
                        "vlm", "get_room_data", test_image_path, []
                    )
                    logging.info(f"Room data: {room_data}")

                    logging.info("Getting object captions...")
                    captions = switcher.execute_with_fallback(
                        "vlm", "get_captions", str(vlm_image_path_annotated), vlm_labels
                    )
                    logging.info(f"Captions: {captions}")

                    logging.info("Getting spatial relations...")
                    relations = switcher.execute_with_fallback(
                        "vlm",
                        "get_relations",
                        str(vlm_image_path_annotated),
                        vlm_labels,
                    )
                    logging.info(f"Relations: {relations}")

                    logging.info("=" * 50)
                    logging.info("PIPELINE COMPLETED SUCCESSFULLY")
                    logging.info("=" * 50)
                    logging.info(f"Total objects detected: {len(detections)}")
                    logging.info(
                        f"Environment profile: {switcher.system_context.environment_profile}"
                    )
                    logging.info(f"Using fallback: {switcher.is_using_fallback()}")

            except (RuntimeError, ValueError, TypeError) as execution_error:
                traceback.print_exc()
                logging.error(f"Pipeline execution failed: {execution_error}")

            finally:
                switcher.unload_all_models()

    logging.info("StrategySwitcher test completed")

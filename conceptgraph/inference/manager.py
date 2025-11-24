from pathlib import Path, WindowsPath
from dotenv import load_dotenv
import supervision as sv
from PIL import Image
import numpy as np
import traceback
import logging
import torch
import json
import uuid
import cv2
import re
import os

from conceptgraph.inference.remote_strategies.detector.gemini_detector_strategy import (
    GeminiDetectorStrategy,
)
from conceptgraph.inference.local_strategies.detector.yolo_detector_strategy import (
    YOLODetectorStrategy,
)
from conceptgraph.inference.remote_strategies.segmenter.gemini_segmenter_strategy import (
    GeminiSegmenterStrategy,
)
from conceptgraph.inference.local_strategies.segmenter.sam_segmenter_strategy import (
    SAMSegmenterStrategy,
)
from conceptgraph.inference.components.system_resource_logger import (
    SystemResourceLogger,
)
from conceptgraph.inference.local_strategies import LocalFeatureExtractor, LMStudioVLM
from conceptgraph.inference.remote_strategies import OpenrouterVLM
from conceptgraph.inference.switcher import StrategySwitcher
from conceptgraph.slam.utils import to_serializable
from conceptgraph.utils.general_utils import (
    annotate_for_vlm,
    plot_edges_from_vlm,
    ObjectClasses,
    filter_detections,
)

from conceptgraph.inference.prompts import (
    SYSTEM_PROMPT_ONLY_TOP,
    SYSTEM_PROMPT_CAPTIONS,
    SYSTEM_PROMPT_ROOM_CLASS,
    SYSTEM_PROMPT_CONSOLIDATE_CAPTIONS,
    ENVIRONMENT_CLASSIFIER,
)


class AdaptiveInferenceManager:
    """
    Manages adaptive inference execution with automatic model and environment switching.
    """

    def __init__(
        self,
        yolo_checkpoint: str = "models/yolov8x-worldv2.pt",
        sam_checkpoint: str = "models/sam2.1_l.pt",
        clip_model_name: str = "ViT-H-14",
        clip_pretrained: str = "laion2b_s32b_b79k",
        vlm_local_model_id: str = "qwen3-vl-8b-instruct",
        vlm_remote_model_id: str = "google/gemini-2.5-flash-lite",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        openrouter_api_key: str | None = None,
        gemini_api_key: str | None = None,
        output_dir: str | WindowsPath | None = None,
        save_frame_outputs: bool = True,
        resource_log_interval: float = 0.1,
        configuration: str = "online",
    ) -> None:
        """
        Initialize the adaptive inference manager with model configurations.

        :param yolo_checkpoint: Path to YOLO model checkpoint.
        :type yolo_checkpoint: str
        :param sam_checkpoint: Path to SAM model checkpoint.
        :type sam_checkpoint: str
        :param clip_model_name: Name of the CLIP model.
        :type clip_model_name: str
        :param clip_pretrained: Pretrained dataset for CLIP model.
        :type clip_pretrained: str
        :param vlm_local_model_id: Model ID for local VLM.
        :type vlm_local_model_id: str
        :param vlm_remote_model_id: Model ID for remote VLM.
        :type vlm_remote_model_id: str
        :param device: Device to run models on.
        :type device: str
        :param openrouter_api_key: API key for OpenRouter service.
        :type openrouter_api_key: str | None
        :param gemini_api_key: API key for Gemini service.
        :type gemini_api_key: str | None
        :param output_dir: Directory to save outputs.
        :type output_dir: str | None
        :param save_frame_outputs: Whether to save frame outputs.
        :type save_frame_outputs: bool
        :param resource_log_interval: Interval in seconds for resource logging.
        :type resource_log_interval: float
        :param configuration: Inference configuration (e.g., "online" or "offline").
        :type configuration: str
        :raises ValueError: If required parameters are invalid.
        :return: None
        :rtype: None
        """
        if resource_log_interval <= 0:
            raise ValueError("resource_log_interval must be a positive number")

        self.device: str = device
        self.save_frame_outputs: bool = save_frame_outputs
        self.resource_log_interval: float = resource_log_interval
        self.configuration: str = configuration

        self.openrouter_api_key: str | None = openrouter_api_key or os.getenv(
            "OPENROUTER_API_KEY"
        )
        self.gemini_api_key: str | None = gemini_api_key or os.getenv("GEMINI_API_KEY")

        if isinstance(output_dir, WindowsPath):
            output_dir = str(output_dir)
        self.output_dir: Path = Path(output_dir or "test_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.yolo_checkpoint: str = yolo_checkpoint
        self.sam_checkpoint: str = sam_checkpoint
        self.clip_model_name: str = clip_model_name
        self.clip_pretrained: str = clip_pretrained
        self.vlm_local_model_id: str = vlm_local_model_id
        self.vlm_remote_model_id: str = vlm_remote_model_id

        logging.info("Initializing shared CLIP model...")
        self.shared_clip: LocalFeatureExtractor = LocalFeatureExtractor(
            model_name=self.clip_model_name,
            pretrained=self.clip_pretrained,
            device=self.device,
        )

        vlm_prompts: dict[str, str] = {
            "relations": SYSTEM_PROMPT_ONLY_TOP,
            "captions": SYSTEM_PROMPT_CAPTIONS,
            "room_class": SYSTEM_PROMPT_ROOM_CLASS,
            "consolidate": SYSTEM_PROMPT_CONSOLIDATE_CAPTIONS,
            "class_env": ENVIRONMENT_CLASSIFIER,
        }

        self.switcher: StrategySwitcher = self._initialize_switcher(vlm_prompts)

        self.resource_logger: SystemResourceLogger | None = None
        self.frame_counter: int = self._get_next_available_frame_number()
        self.environment_classified: bool = False

    def _get_next_available_frame_number(self) -> int:
        """
        Get the next available frame number by checking existing frame directories.

        :return: The next available frame number.
        :rtype: int
        """
        frame_number: int = 0
        while (self.output_dir / f"frame_{frame_number:05d}").exists():
            frame_number += 1
        return frame_number

    def _initialize_switcher(self, vlm_prompts: dict[str, str]) -> StrategySwitcher:
        """
        Initialize the strategy switcher with preferred and fallback models.

        :param vlm_prompts: Dictionary of VLM prompts.
        :type vlm_prompts: dict[str, str]
        :raises RuntimeError: If model initialization fails.
        :return: Configured StrategySwitcher instance.
        :rtype: StrategySwitcher
        """
        try:
            logging.info("Preparing preferred (remote) model factories...")
            preferred_factories: dict = {
                "det": lambda: GeminiDetectorStrategy(api_key=self.openrouter_api_key),
                "seg": lambda: GeminiSegmenterStrategy(api_key=self.gemini_api_key),
                "vlm": lambda: OpenrouterVLM(
                    model_id=self.vlm_remote_model_id,
                    api_key=self.openrouter_api_key,
                ),
            }

            logging.info("Preparing fallback (local) model factories...")
            fallback_factories: dict = {
                "det": lambda: YOLODetectorStrategy(
                    checkpoint_path=self.yolo_checkpoint,
                    device=self.device,
                ),
                "seg": lambda: SAMSegmenterStrategy(
                    checkpoint_path=self.sam_checkpoint,
                    device=self.device,
                ),
                "vlm": lambda: LMStudioVLM(
                    model_id=self.vlm_local_model_id,
                    device=self.device,
                    api_key=None,
                ),
            }

            if self.configuration == "offline":
                temp = preferred_factories
                preferred_factories = fallback_factories
                fallback_factories = temp
            elif self.configuration == "improved":
                preferred_factories: dict = {
                    "det": lambda: YOLODetectorStrategy(
                        checkpoint_path=self.yolo_checkpoint,
                        device=self.device,
                    ),
                    "seg": lambda: SAMSegmenterStrategy(
                        checkpoint_path=self.sam_checkpoint,
                        device=self.device,
                    ),
                    "vlm": lambda: OpenrouterVLM(
                        model_id=self.vlm_remote_model_id,
                        api_key=self.openrouter_api_key,
                    ),
                }

            switcher: StrategySwitcher = StrategySwitcher(
                preferred_factories,
                fallback_factories,
                shared_model=self.shared_clip,
                vlm_prompts=vlm_prompts,
            )

            logging.info("StrategySwitcher initialized successfully")
            return switcher

        except (RuntimeError, ValueError, TypeError) as init_error:
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize StrategySwitcher: {init_error}")

    def start_resource_logging(self) -> None:
        """
        Start resource logging in the background.

        :raises RuntimeError: If resource logger is already running.
        :return: None
        :rtype: None
        """
        if self.resource_logger is not None and self.resource_logger.running:
            raise RuntimeError("Resource logger is already running")

        resource_log_path: str = str(self.output_dir / "manager" / "resource_log.csv")
        os.makedirs(os.path.dirname(resource_log_path), exist_ok=True)
        self.resource_logger = SystemResourceLogger(
            sample_interval=self.resource_log_interval,
            output_path=resource_log_path,
        )
        self.resource_logger.start()
        logging.info("Resource logging started")

    def stop_resource_logging(self) -> None:
        """
        Stop resource logging and generate plots.

        :return: None
        :rtype: None
        """
        if self.resource_logger is not None:
            self.resource_logger._cleanup()
            logging.info("Generating resource usage plots...")
            try:
                self.resource_logger.plot_resources(
                    output_image_path=str(self.output_dir / "resource_plot.png")
                )
            except (RuntimeError, ValueError, TypeError) as plot_error:
                traceback.print_exc()
                logging.error(f"Failed to generate resource plots: {plot_error}")
            self.resource_logger = None
            logging.info("Resource logging stopped")

    def set_baseline_collection_flag(self, flag: bool) -> None:
        """
        Set the baseline collection flag for resource logging.

        :param flag: True if collecting baseline data, False otherwise.
        :type flag: bool
        :return: None
        :rtype: None
        """
        if self.resource_logger is not None:
            self.resource_logger.set_baseline_collection_flag(flag)

    def associate_masks_to_detections(
        self,
        detections: sv.Detections,
        masks: np.ndarray,
    ) -> sv.Detections:
        """
        Associate segmentation masks to detection bounding boxes based on centroid proximity.

        :param detections: Detection results containing bounding boxes, class IDs, and confidences.
        :type detections: sv.Detections
        :param masks: Array of segmentation masks.
        :type masks: np.ndarray
        :raises ValueError: If detections or masks are empty or invalid.
        :return: Detections object with masks associated to bounding boxes.
        :rtype: sv.Detections
        """
        if len(detections) == 0:
            raise ValueError("Detections cannot be empty")
        if len(masks) == 0:
            raise ValueError("Masks cannot be empty")

        detection_centers: list[tuple[float, float]] = []
        for bbox in detections.xyxy:
            center_x: float = (bbox[0] + bbox[2]) / 2.0
            center_y: float = (bbox[1] + bbox[3]) / 2.0
            detection_centers.append((center_x, center_y))

        mask_centers: list[tuple[float, float]] = []
        for mask in masks:
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) == 0:
                mask_centers.append((0.0, 0.0))
                continue
            center_x: float = float(np.mean(x_coords))
            center_y: float = float(np.mean(y_coords))
            mask_centers.append((center_x, center_y))

        associated_masks: list[np.ndarray] = []
        used_mask_indices: set[int] = set()

        for det_center in detection_centers:
            min_distance: float = float("inf")
            closest_mask_idx: int = -1

            for mask_idx, mask_center in enumerate(mask_centers):
                if mask_idx in used_mask_indices:
                    continue

                distance: float = np.sqrt(
                    (det_center[0] - mask_center[0]) ** 2
                    + (det_center[1] - mask_center[1]) ** 2
                )

                if distance < min_distance:
                    min_distance = distance
                    closest_mask_idx = mask_idx

            if closest_mask_idx != -1:
                associated_masks.append(masks[closest_mask_idx])
                used_mask_indices.add(closest_mask_idx)
            else:
                associated_masks.append(np.zeros_like(masks[0]))

        associated_masks = np.array(associated_masks)

        detections = sv.Detections(
            xyxy=detections.xyxy,
            confidence=detections.confidence,
            class_id=detections.class_id,
            mask=associated_masks,
        )

        return detections

    def process_frame(
        self,
        image_path: str,
        image_np: np.ndarray,
    ) -> dict:
        """
        Process a single frame through the complete inference pipeline.

        :param image_path: Path to the image file.
        :type image_path: str
        :param image_np: Image as a numpy array in RGB format.
        :type image_np: np.ndarray
        :raises RuntimeError: If processing fails at any step.
        :return: Dictionary containing detection results, features, and VLM outputs.
        :rtype: dict
        """
        self.prepare_results(image_path)

        try:
            logging.info(f"Processing frame {self.frame_counter}: {image_path}")

            self.classify_environment(image_path)
            detections = self.detect_objects(image_path, image_np)

            if len(detections) == 0:
                return self.results

            masks = self.segment_objects(image_path, image_np, detections)
            detections = self.associate_masks_to_detections(detections, masks)

            self.perform_vlm_inference(image_path, image_np, detections)
            self.extract_features(image_np, detections)

            logging.info(f"Frame {self.frame_counter} processed successfully")
            self.frame_counter += 1
            return self.results

        except (RuntimeError, ValueError, TypeError) as e:
            traceback.print_exc()
            raise RuntimeError(
                f"Failed to process frame {self.frame_counter}: {e}"
            ) from e

    def get_detection_classes(self, detections: sv.Detections) -> list[str]:
        """
        Get the list of detection classes based on the classified environment.

        :param detections: Detection results containing class IDs.
        :type detections: sv.Detections
        :raises RuntimeError: If environment is not classified.
        :return: List of class names for detection.
        :rtype: list[str]
        """
        _, obj_classes = self.switcher.get_classes_for_environment(False)

        labels = [
            f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(detections.class_id)
        ]

        return labels

    def perform_vlm_inference(
        self,
        image_path: str,
        image_np: np.ndarray,
        detections: sv.Detections,
        save_results: bool = False,
    ) -> tuple[list, list, np.ndarray | None, list, dict]:
        """
        Perform VLM inference to extract room data, object captions, and spatial relations.

        :param image_path: Path to the image file.
        :type image_path: str
        :param image_np: Image as a numpy array in RGB format.
        :type image_np: np.ndarray
        :param detections: Detection results with masks.
        :type detections: sv.Detections
        :raises RuntimeError: If VLM inference fails.
        :return: Tuple containing labels, edges, edge_image, captions, and room_data.
        :rtype: tuple[list, list, np.ndarray | None, list, dict]
        """

        logging.info("Step 3: VLM Inference")

        classes, obj_classes = self.switcher.get_classes_for_environment(False)

        labels = [
            f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(detections.class_id)
        ]

        detections, labels = filter_detections(
            image=image_np,
            detections=detections,
            classes=obj_classes,
            top_x_detections=150000,
            confidence_threshold=0.5,
            given_labels=labels,
        )

        edges: list = []
        edge_image: np.ndarray | None = None
        captions: list = []
        room_data: dict = {"room_class": "None", "room_description": "None"}

        vlm_annotated_path, vlm_annotated_image, sorted_indices = (
            self._save_vlm_annotated_image(
                image_np,
                detections,
                obj_classes,
                labels,
                self.frame_output_dir,
            )
        )

        vlm_image_input: str = (
            str(vlm_annotated_path) if vlm_annotated_path else image_path
        )

        label_list: list[str] = []
        for label in labels:
            label_num: str = str(label.split(" ")[-1])
            label_name: str = re.sub(r"\s*\d+$", "", label).strip()
            full_label: str = f"{label_num}: {label_name}"
            label_list.append(full_label)

        logging.info("Getting object captions...")
        captions = self.switcher.execute_with_fallback(
            "vlm",
            "get_captions",
            vlm_image_input,
            label_list,
        )

        logging.info("Getting spatial relations...")
        edges = self.switcher.execute_with_fallback(
            "vlm",
            "get_relations",
            vlm_image_input,
            label_list,
        )

        logging.info("Getting room classification...")
        room_data = self.switcher.execute_with_fallback(
            "vlm",
            "get_room_data",
            image_path,
            [],
        )

        self.results["vlm_outputs"] = {
            "room_data": room_data,
            "captions": captions,
            "relations": edges,
        }

        if (self.save_frame_outputs and self.frame_output_dir) or (
            save_results and self.frame_output_dir
        ):
            edge_image = plot_edges_from_vlm(
                vlm_annotated_image,
                edges,
                detections,
                obj_classes,
                labels,
                sorted_indices,
            )
            vlm_image_path = self.frame_output_dir / "04_vlm_edges.png"
            image_pil = Image.fromarray(edge_image)
            image_pil.save(vlm_image_path)
            self._save_vlm_outputs(room_data, captions, edges, self.frame_output_dir)
        else:
            os.remove(vlm_annotated_path)

        return labels, edges, edge_image, captions, room_data

    def consolidate_captions(
        self,
        obj: dict,
        save_results: bool = False,
    ) -> str:
        """
        Consolidate multiple captions for the same object into a single, clear caption using VLM.

        :param obj: Dictionary containing object captions.
        :type obj: dict
        :param save_results: Whether to save the consolidated caption to a file.
        :type save_results: bool
        :raises ValueError: If captions list is empty or invalid.
        :return: A single consolidated caption string.
        :rtype: str
        """
        obj_captions = obj["captions"]
        obj_class_name = obj.get("class_name", None)

        if obj_class_name is None:
            logging.warning("Object class name is missing")
            return ""

        obj_captions = [
            caption
            for caption in obj_captions
            if caption is not None
            and ("caption" in caption)
            and (caption["caption"] is not None)
        ]

        if len(obj_captions) == 0:
            obj["consolidated_caption"] = "No captions available."
            return "No captions available."

        captions_text: str = "\n".join(
            [
                f" - {cap['caption']}"
                for cap in obj_captions
                if cap.get("caption") is not None
            ]
        )

        captions_text += f"\nClass: {obj_class_name}"

        if not captions_text.strip():
            logging.warning("No valid captions to consolidate")
            return ""

        consolidated_caption: str = ""
        try:
            logging.info("Consolidating captions...")
            consolidated_caption = self.switcher.execute_with_fallback(
                "vlm",
                "consolidate_captions",
                obj_captions,
            )
            obj["consolidated_caption"] = consolidated_caption
            logging.info(f"Consolidated Caption: {consolidated_caption}")
        except (ValueError, TypeError, KeyError) as e:
            traceback.print_exc()
            logging.error(f"Failed to consolidate captions: {str(e)}")
            consolidated_caption = ""
            obj["consolidated_caption"] = consolidated_caption

        if (self.save_frame_outputs and self.frame_output_dir) or (
            save_results and self.frame_output_dir
        ):
            obj_id = obj.get("id", None)
            if obj_id is None:
                obj["id"] = str(uuid.uuid4())
                obj_id = obj["id"]

            with open(
                self.frame_output_dir / f"{obj_id}.json",
                "w",
                encoding="utf-8",
            ) as f:
                obj_data = obj.copy()
                del obj_data["pcd"]
                del obj_data["bbox"]
                del obj_data["contain_number"]
                del obj_data["mask"]
                del obj_data["xyxy"]
                json.dump(
                    obj_data, f, ensure_ascii=False, indent=2, default=to_serializable
                )

        return consolidated_caption

    def extract_features(self, image_np, detections, save_results: bool = False):
        logging.info("Step 4: Feature Extraction")
        classes: list[str] = self.switcher.get_classes_for_environment()
        crops, image_feats, text_feats = self.switcher.execute_with_fallback(
            "clip",
            "extract_features",
            image_np,
            detections,
            classes,
        )

        image_feats_array = np.array(image_feats)
        text_feats_array = np.array(text_feats)
        logging.info(
            f"Extracted features: image_feats={image_feats_array.shape}, text_feats={text_feats_array.shape}"
        )

        self.results["features"] = {
            "crops": crops,
            "image_features": image_feats_array,
            "text_features": text_feats_array,
        }

        if (self.save_frame_outputs and self.frame_output_dir) or (
            save_results and self.frame_output_dir
        ):
            self._save_crops(crops, detections, self.frame_output_dir)

        return crops, image_feats, text_feats

    def segment_objects(
        self, image_path, image_np, detections, save_results: bool = False
    ):
        logging.info("Step 2: Segmentation")
        classes: list[str] = self.switcher.get_classes_for_environment()
        boxes_tensor = torch.tensor(detections.xyxy, device=self.device)
        classes_detected = [classes[i] for i in detections.class_id]

        masks = self.switcher.execute_with_fallback(
            "seg", "segment", image_path, image_np, boxes_tensor, classes_detected
        )

        masks = masks.cpu().numpy()
        detections.mask = masks
        logging.info(f"Generated {len(masks)} segmentation masks")
        self.results["masks"] = masks

        if (self.save_frame_outputs and self.frame_output_dir) or (
            save_results and self.frame_output_dir
        ):
            self._save_segmentation_visualization(
                image_np, detections, self.frame_output_dir
            )

        return masks

    def detect_objects(self, image_path, image_np, save_results: bool = False):
        logging.info("Step 1: Object Detection")
        classes: list[str] = self.switcher.get_classes_for_environment()
        detector = self.switcher.get_model("det")
        detector.set_classes(classes)

        detections = self.switcher.execute_with_fallback(
            "det", "detect", image_path, image_np
        )

        detections = detections[detections.confidence > 0.4]

        logging.info(f"Detected {len(detections)} objects")
        self.results["detections"] = detections
        self.results["used_fallback"] = self.switcher.is_using_fallback()

        if len(detections) == 0:
            logging.warning("No objects detected in frame")
            if (self.save_frame_outputs and self.frame_output_dir) or (
                save_results and self.frame_output_dir
            ):
                self._save_empty_frame_info(self.frame_output_dir, self.results)

        if (self.save_frame_outputs and self.frame_output_dir) or (
            save_results and self.frame_output_dir
        ):
            self._save_detection_visualization(
                image_np, detections, classes, self.frame_output_dir
            )

        return detections

    def set_frame_output_dir(
        self, frame_idx: int | None = None, output_dir_name: str | None = None
    ):
        self.frame_output_dir: Path | None = None

        if frame_idx is not None:
            self.frame_counter = frame_idx

        if output_dir_name is not None:
            self.frame_output_dir = self.output_dir / "manager" / output_dir_name
            self.frame_output_dir.mkdir(parents=True, exist_ok=True)
        elif self.save_frame_outputs:
            self.frame_output_dir = (
                self.output_dir / "manager" / f"frame_{self.frame_counter:05d}"
            )
            self.frame_output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_results(self, image_path, frame_idx: int | None = None):
        self.set_frame_output_dir(frame_idx)

        self.results: dict = {
            "frame_number": self.frame_counter,
            "image_path": image_path,
            "environment_profile": None,
            "detections": None,
            "masks": None,
            "features": None,
            "vlm_outputs": None,
            "used_fallback": False,
        }

    def classify_environment(self, image_path):
        logging.info("Step 0: Environment Classification")
        environment_type = self.switcher.classify_environment(image_path)
        self.environment_classified = True
        logging.info(f"Environment classified as: {environment_type}")
        self.results["environment_profile"] = environment_type

        if self.save_frame_outputs and self.frame_output_dir:
            env_data: dict = {
                "environment_type": environment_type,
                "frame_number": self.frame_counter,
            }
            with open(
                self.frame_output_dir / "00_environment.json", "w", encoding="utf-8"
            ) as f:
                json.dump(env_data, f, ensure_ascii=False, indent=2)

    def _save_empty_frame_info(self, output_dir: Path, results: dict) -> None:
        """
        Save information for frames with no detections.

        :param output_dir: Directory to save the frame information.
        :type output_dir: Path
        :param results: Results dictionary.
        :type results: dict
        :return: None
        :rtype: None
        """
        empty_data: dict = {
            "frame_number": results["frame_number"],
            "detections": 0,
            "message": "No objects detected",
        }
        with open(output_dir / "frame_info.json", "w", encoding="utf-8") as f:
            json.dump(empty_data, f, ensure_ascii=False, indent=2)

    def _save_detection_visualization(
        self,
        image_rgb: np.ndarray,
        detections: object,
        classes: list[str],
        output_dir: Path,
    ) -> None:
        """
        Save detection visualization with bounding boxes and labels.

        :param image_rgb: Image as a numpy array (RGB format).
        :type image_rgb: np.ndarray
        :param detections: Detection results from the detector.
        :type detections: object
        :param classes: List of class names.
        :type classes: list[str]
        :param output_dir: Directory to save the visualization.
        :type output_dir: Path
        :return: None
        :rtype: None
        """
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels: list[str] = [
            f"{classes[class_id]} {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated_image: np.ndarray = box_annotator.annotate(
            scene=image_rgb.copy(), detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        image_pil = Image.fromarray(annotated_image)
        image_pil.save(output_dir / "01_detections.png")

        detection_data: dict = {
            "num_detections": len(detections),
            "used_fallback": self.switcher.is_using_fallback(),
            "detections": [
                {
                    "class": classes[class_id],
                    "confidence": float(conf),
                    "bbox": bbox.tolist(),
                }
                for class_id, conf, bbox in zip(
                    detections.class_id, detections.confidence, detections.xyxy
                )
            ],
        }

        with open(output_dir / "01_detections.json", "w", encoding="utf-8") as f:
            json.dump(detection_data, f, ensure_ascii=False, indent=2)

    def _save_segmentation_visualization(
        self,
        image_rgb: np.ndarray,
        detections: object,
        output_dir: Path,
    ) -> None:
        """
        Save segmentation visualization with masks.

        :param image_rgb: Image as a numpy array (RGB format).
        :type image_rgb: np.ndarray
        :param detections: Detection results with masks.
        :type detections: object
        :param output_dir: Directory to save the visualization.
        :type output_dir: Path
        :return: None
        :rtype: None
        """
        if detections.mask is None or len(detections.mask) == 0:
            logging.warning("No masks available for segmentation visualization")
            segmentation_data: dict = {
                "num_masks": 0,
                "used_fallback": self.switcher.is_using_fallback(),
            }
            with open(output_dir / "02_segmentation.json", "w", encoding="utf-8") as f:
                json.dump(segmentation_data, f, ensure_ascii=False, indent=2)
            return

        num_detections: int = len(detections.xyxy)
        num_masks: int = len(detections.mask)

        if num_masks != num_detections:
            logging.warning(
                f"Mask count mismatch: {num_masks} masks for {num_detections} detections. Truncating to match."
            )
            min_count: int = min(num_masks, num_detections)
            detections.mask = detections.mask[:min_count]
            detections.xyxy = detections.xyxy[:min_count]
            detections.class_id = detections.class_id[:min_count]
            detections.confidence = detections.confidence[:min_count]

        mask_annotator = sv.MaskAnnotator()
        masked_image: np.ndarray = mask_annotator.annotate(
            scene=image_rgb.copy(), detections=detections
        )

        image_pil = Image.fromarray(masked_image)
        image_pil.save(output_dir / "02_segmentation.png")

        segmentation_data: dict = {
            "num_masks": len(detections.mask),
            "used_fallback": self.switcher.is_using_fallback(),
        }

        with open(output_dir / "02_segmentation.json", "w", encoding="utf-8") as f:
            json.dump(segmentation_data, f, ensure_ascii=False, indent=2)

    def _save_crops(
        self,
        crops: list,
        detections: object,
        output_dir: Path,
    ) -> None:
        """
        Save individual crops of detected objects.

        :param crops: List of cropped images.
        :type crops: list
        :param detections: Detection results.
        :type detections: object
        :param output_dir: Directory to save the crops.
        :type output_dir: Path
        :return: None
        :rtype: None
        """
        crops_dir: Path = output_dir / "04_crops"
        crops_dir.mkdir(exist_ok=True)

        classes: list[str] = self.switcher.get_classes_for_environment()

        for i, crop in enumerate(crops):
            class_name: str = classes[detections.class_id[i]]
            crop_filename: str = f"crop_{i:03d}_{class_name}.png"

            if isinstance(crop, np.ndarray):
                crop_rgb: np.ndarray = (
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    if len(crop.shape) == 3
                    else crop
                )
                crop_pil = Image.fromarray(crop_rgb)
            elif isinstance(crop, Image.Image):
                crop_pil = crop
            else:
                continue

            crop_pil.save(crops_dir / crop_filename)

    def _save_vlm_annotated_image(
        self,
        image_rgb: np.ndarray,
        detections: object,
        obj_classes: ObjectClasses,
        vlm_labels: list[str],
        output_dir: Path,
    ) -> tuple[Path, np.ndarray, list[int]]:
        """
        Save annotated image with numbered labels for VLM input.

        :param image_rgb: Image as a numpy array (RGB format).
        :type image_rgb: np.ndarray
        :param detections: Detection results.
        :type detections: object
        :param obj_classes: ObjectClasses instance for class name mapping.
        :type obj_classes: ObjectClasses
        :param vlm_labels: List of numbered labels for each detection.
        :type vlm_labels: list[str]
        :param output_dir: Directory to save the annotated image.
        :type output_dir: Path
        :return: Tuple containing the path to the saved image, the annotated image array, and sorted indices.
        :rtype: tuple[Path, np.ndarray, list[int]]
        """
        vlm_annotated_image, sorted_indices = annotate_for_vlm(
            image_rgb, detections, obj_classes, vlm_labels
        )

        os.makedirs(output_dir, exist_ok=True)
        vlm_image_path: Path = output_dir / "03_vlm_annotated.png"
        image_pil = Image.fromarray(vlm_annotated_image)
        image_pil.save(vlm_image_path)

        return vlm_image_path, vlm_annotated_image, sorted_indices

    def _save_vlm_outputs(
        self,
        room_data: dict | list,
        captions: dict | list,
        relations: dict | list,
        output_dir: Path,
    ) -> None:
        """
        Save VLM outputs to JSON files.

        :param room_data: Room classification data.
        :type room_data: dict | list
        :param captions: Object captions.
        :type captions: dict | list
        :param relations: Spatial relations.
        :type relations: dict | list
        :param output_dir: Directory to save the outputs.
        :type output_dir: Path
        :return: None
        :rtype: None
        """
        with open(output_dir / "03a_room_data.json", "w", encoding="utf-8") as f:
            json.dump(room_data, f, ensure_ascii=False, indent=2)

        with open(output_dir / "03b_captions.json", "w", encoding="utf-8") as f:
            json.dump(captions, f, ensure_ascii=False, indent=2)

        with open(output_dir / "03c_relations.json", "w", encoding="utf-8") as f:
            json.dump(relations, f, ensure_ascii=False, indent=2)

        vlm_data: dict = {
            "used_fallback": self.switcher.is_using_fallback(),
        }
        with open(output_dir / "03_vlm_info.json", "w", encoding="utf-8") as f:
            json.dump(vlm_data, f, ensure_ascii=False, indent=2)

    def unload_all_models(self) -> None:
        """
        Unload all loaded models from memory.

        :return: None
        :rtype: None
        """
        logging.info("Unloading all models...")
        self.switcher.unload_all_models()
        logging.info("All models unloaded")

    def get_fallback_status(self) -> bool:
        """
        Check if the manager is currently using fallback models.

        :return: True if using fallback models, False otherwise.
        :rtype: bool
        """
        return self.switcher.is_using_fallback()

    def __enter__(self) -> "AdaptiveInferenceManager":
        """
        Start resource logging upon entering the context manager.

        :return: The AdaptiveInferenceManager instance.
        :rtype: AdaptiveInferenceManager
        """
        self.start_resource_logging()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: object | None,
        tb: object | None,
    ) -> None:
        """
        Stop resource logging and unload models upon exiting the context manager.

        :param exc_type: Exception type, if any.
        :type exc_type: type | None
        :param exc_value: Exception value, if any.
        :type exc_value: object | None
        :param tb: Traceback object, if any.
        :type tb: object | None
        :return: None
        :rtype: None
        """
        self.stop_resource_logging()
        self.unload_all_models()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    test_image_path: str = "assets/test_image_4.jpg"
    output_directory: str = "test_output"

    if not Path(test_image_path).exists():
        logging.error(f"Test image not found: {test_image_path}")
        exit(1)

    try:
        image: np.ndarray = cv2.imread(test_image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {test_image_path}")

        image_rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        logging.info("Initializing AdaptiveInferenceManager...")
        with AdaptiveInferenceManager(
            output_dir=output_directory,
            save_frame_outputs=True,
            resource_log_interval=0.05,
        ) as manager:
            logging.info("Processing test frame...")
            results: dict = manager.process_frame(test_image_path, image_rgb)

            logging.info("Processing complete. Results:")
            logging.info(f"  Frame number: {results['frame_number']}")
            logging.info(f"  Environment: {results['environment_profile']}")
            logging.info(
                f"  Detections: {len(results['detections']) if results['detections'] else 0}"
            )
            logging.info(f"  Used fallback: {results['used_fallback']}")

            if results["vlm_outputs"]:
                logging.info("  VLM outputs available")

        logging.info(
            f"Test completed successfully. Check output at: {output_directory}"
        )

    except (RuntimeError, ValueError, TypeError, OSError) as error:
        traceback.print_exc()
        logging.error(f"Test failed: {error}")
        exit(1)

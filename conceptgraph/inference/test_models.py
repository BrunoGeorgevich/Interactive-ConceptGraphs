from dotenv import load_dotenv
from time import perf_counter
import supervision as sv
from pathlib import Path
from PIL import Image
import numpy as np
import traceback
import logging
import torch
import json
import time
import sys
import cv2
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


class ModelConfig:
    """
    Configuration class for model selection and file paths.
    Stores paths to model checkpoints, test data, and output directories.
    """

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    YOLO_CHECKPOINT: str = "models/yolov8x-worldv2.pt"
    SAM_CHECKPOINT: str = "models/sam2.1_l.pt"

    CLIP_MODEL_NAME: str = "ViT-H-14"
    CLIP_PRETRAINED: str = "laion2b_s32b_b79k"

    VLM_LOCAL_MODEL_ID: str = "qwen/qwen3-vl-8b"
    VLM_REMOTE_MODEL_ID: str = "google/gemini-2.5-flash-lite"

    TEST_IMAGE_PATH: str = "assets/test_image.png"
    try:
        with open("conceptgraph/scannet200_classes.txt", "r", encoding="utf-8") as f:
            TEST_CLASSES: list[str] = [line.strip() for line in f if line.strip()]
    except (FileNotFoundError, OSError) as e:
        traceback.print_exc()
        raise RuntimeError(
            "Failed to load class names from scannet200_classes.txt. Ensure the file exists and is readable."
        ) from e

    OUTPUT_DIR: str = "test_output"


def create_output_directory(output_dir: str) -> Path:
    """
    Create the output directory if it does not exist.

    :param output_dir: The path to the output directory.
    :type output_dir: str
    :return: The Path object of the created or existing directory.
    :rtype: Path
    """
    output_path: Path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_image(
    image: np.ndarray | Image.Image, filename: str, output_dir: Path
) -> None:
    """
    Save a numpy array or PIL Image as an image file.

    :param image: The image as a numpy array or PIL Image.
    :type image: np.ndarray | Image.Image
    :param filename: The filename for the saved image.
    :type filename: str
    :param output_dir: The directory to save the image in.
    :type output_dir: Path
    :raises ValueError: If the image type is unsupported or saving fails.
    :return: None
    :rtype: None
    """
    try:
        if isinstance(image, Image.Image):
            image_to_save = image
        elif isinstance(image, np.ndarray):
            image_rgb: np.ndarray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if len(image.shape) == 3
                else image
            )
            image_to_save = Image.fromarray(image_rgb)
        else:
            raise ValueError(
                "Unsupported image type for saving. Must be np.ndarray or PIL.Image.Image."
            )
        image_to_save.save(output_dir / filename)
        logging.info(f"Saved image: {filename}")
    except (OSError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Failed to save image '{filename}': {e}") from e


def save_json_data(data: dict | list, filename: str, output_dir: Path) -> None:
    """
    Save data as a JSON file.

    :param data: The data to save (dictionary or list).
    :type data: dict | list
    :param filename: The filename for the saved JSON.
    :type filename: str
    :param output_dir: The directory to save the JSON in.
    :type output_dir: Path
    :return: None
    :rtype: None
    """
    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved JSON: {filename}")


def create_switcher(config: ModelConfig) -> StrategySwitcher:
    """
    Create and configure the strategy switcher with preferred and fallback models.

    :param config: The model configuration.
    :type config: ModelConfig
    :raises RuntimeError: If model instantiation fails.
    :return: Configured StrategySwitcher instance.
    :rtype: StrategySwitcher
    """
    try:
        logging.info("Initializing shared CLIP model...")
        shared_clip: LocalFeatureExtractor = LocalFeatureExtractor(
            model_name=config.CLIP_MODEL_NAME,
            pretrained=config.CLIP_PRETRAINED,
            device=config.DEVICE,
        )

        logging.info("Preparing preferred (remote) model factories...")
        preferred_factories: dict = {
            "det": lambda: GeminiDetectorStrategy(
                api_key=os.getenv("OPENROUTER_API_KEY")
            ),
            "seg": lambda: GeminiSegmenterStrategy(api_key=os.getenv("GEMINI_API_KEY")),
            "vlm": lambda: OpenrouterVLM(
                model_id=config.VLM_REMOTE_MODEL_ID,
                api_key=os.getenv("OPENROUTER_API_KEY"),
            ),
        }

        logging.info("Preparing fallback (local) model factories...")
        fallback_factories: dict = {
            "det": lambda: YOLODetectorStrategy(
                checkpoint_path=config.YOLO_CHECKPOINT,
                device=config.DEVICE,
            ),
            "seg": lambda: SAMSegmenterStrategy(
                checkpoint_path=config.SAM_CHECKPOINT,
                device=config.DEVICE,
            ),
            "vlm": lambda: LMStudioVLM(
                model_id=config.VLM_LOCAL_MODEL_ID,
                device=config.DEVICE,
                api_key=None,
            ),
        }

        switcher: StrategySwitcher = StrategySwitcher(
            preferred_factories=preferred_factories,
            fallback_factories=fallback_factories,
            shared_model=shared_clip,
        )

        logging.info("StrategySwitcher initialized successfully")
        return switcher

    except (RuntimeError, ValueError, TypeError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to create StrategySwitcher: {e}") from e


def run_detection_pipeline(
    switcher: StrategySwitcher,
    config: ModelConfig,
    output_dir: Path,
) -> None:
    """
    Run the detection pipeline using the strategy switcher with automatic fallback.

    :param switcher: The strategy switcher instance.
    :type switcher: StrategySwitcher
    :param config: The model configuration.
    :type config: ModelConfig
    :param output_dir: The output directory.
    :type output_dir: Path
    :raises RuntimeError: If pipeline execution fails.
    :return: None
    :rtype: None
    """
    logging.info(f"Loading test image: {config.TEST_IMAGE_PATH}")
    image_path: Path = Path(config.TEST_IMAGE_PATH)

    if not image_path.exists():
        logging.error(f"Test image not found: {image_path}")
        return

    image: np.ndarray | None = cv2.imread(str(image_path))
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return

    image_rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height: int
    width: int
    height, width = image_rgb.shape[:2]
    logging.info(f"Image loaded: {width}x{height}")

    logging.info("=" * 50)
    logging.info("STEP 1: Object Detection")
    logging.info("=" * 50)

    try:
        detector = switcher.get_model("det")
        detector.set_classes(config.TEST_CLASSES)

        detections = switcher.execute_with_fallback(
            "det", "detect", str(image_path), image_rgb
        )

        logging.info(f"Detected {len(detections)} objects")
        logging.info(f"Using fallback: {switcher.is_using_fallback()}")

        if len(detections) == 0:
            logging.warning("No objects detected. Exiting pipeline.")
            return

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels: list[str] = [
            f"{config.TEST_CLASSES[class_id]} {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated_image: np.ndarray = box_annotator.annotate(
            scene=image_rgb.copy(), detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        save_image(annotated_image, "01_detections.png", output_dir)

        detection_data: dict = {
            "num_detections": len(detections),
            "used_fallback": switcher.is_using_fallback(),
            "detections": [
                {
                    "class": config.TEST_CLASSES[class_id],
                    "confidence": float(conf),
                    "bbox": bbox.tolist(),
                }
                for class_id, conf, bbox in zip(
                    detections.class_id, detections.confidence, detections.xyxy
                )
            ],
        }
        save_json_data(detection_data, "01_detections.json", output_dir)

    except RuntimeError as e:
        traceback.print_exc()
        raise RuntimeError(f"Detection step failed: {e}") from e

    logging.info("=" * 50)
    logging.info("STEP 2: Segmentation")
    logging.info("=" * 50)

    try:
        boxes_tensor = torch.from_numpy(detections.xyxy).to(config.DEVICE)
        classes: list[str] = [
            config.TEST_CLASSES[class_id] for class_id in detections.class_id
        ]

        masks = switcher.execute_with_fallback(
            "seg", "segment", str(image_path), image_rgb, boxes_tensor, classes
        )

        detections.mask = masks.cpu().numpy()

        logging.info(f"Generated {len(masks)} segmentation masks")
        logging.info(f"Using fallback: {switcher.is_using_fallback()}")

        mask_annotator = sv.MaskAnnotator()
        masked_image: np.ndarray = mask_annotator.annotate(
            scene=image_rgb.copy(), detections=detections
        )

        save_image(masked_image, "02_segmentation.png", output_dir)

        segmentation_data: dict = {
            "num_masks": len(masks),
            "used_fallback": switcher.is_using_fallback(),
        }
        save_json_data(segmentation_data, "02_segmentation.json", output_dir)

    except RuntimeError as e:
        traceback.print_exc()
        raise RuntimeError(f"Segmentation step failed: {e}") from e

    logging.info("=" * 50)
    logging.info("STEP 3: Feature Extraction")
    logging.info("=" * 50)

    try:
        crops, image_feats, text_feats = switcher.execute_with_fallback(
            "clip",
            "extract_features",
            image_rgb,
            detections,
            config.TEST_CLASSES,
        )

        image_feats = np.array(image_feats)
        text_feats = np.array(text_feats)

        logging.info(
            f"Extracted features: image_feats={image_feats.shape}, text_feats={text_feats.shape}"
        )

        crops_dir: Path = output_dir / "03_crops"
        crops_dir.mkdir(exist_ok=True)

        for i, crop in enumerate(crops):
            class_name: str = config.TEST_CLASSES[detections.class_id[i]]
            crop_filename: str = f"crop_{i:03d}_{class_name}.png"
            save_image(crop, crop_filename, crops_dir)

        image_feats_stats = {
            "mean": float(image_feats.mean()) if image_feats.size > 0 else 0.0,
            "std": float(image_feats.std()) if image_feats.size > 0 else 0.0,
            "min": float(image_feats.min()) if image_feats.size > 0 else 0.0,
            "max": float(image_feats.max()) if image_feats.size > 0 else 0.0,
        }
        text_feats_stats = {
            "mean": float(text_feats.mean()) if text_feats.size > 0 else 0.0,
            "std": float(text_feats.std()) if text_feats.size > 0 else 0.0,
            "min": float(text_feats.min()) if text_feats.size > 0 else 0.0,
            "max": float(text_feats.max()) if text_feats.size > 0 else 0.0,
        }
        feature_data: dict = {
            "num_crops": len(crops),
            "image_features_shape": list(image_feats.shape),
            "text_features_shape": list(text_feats.shape),
            "image_features_stats": image_feats_stats,
            "text_features_stats": text_feats_stats,
        }
        save_json_data(feature_data, "03_features.json", output_dir)

    except RuntimeError as e:
        traceback.print_exc()
        raise RuntimeError(f"Feature extraction step failed: {e}") from e

    logging.info("=" * 50)
    logging.info("STEP 4: VLM Inference")
    logging.info("=" * 50)

    try:
        vlm_labels: list[str] = [
            f"{i+1}: {config.TEST_CLASSES[class_id]}"
            for i, class_id in enumerate(detections.class_id)
        ]

        vlm_annotated_image: np.ndarray = box_annotator.annotate(
            scene=image_rgb.copy(), detections=detections
        )
        vlm_annotated_image = label_annotator.annotate(
            scene=vlm_annotated_image, detections=detections, labels=vlm_labels
        )

        vlm_image_path: Path = output_dir / "04_vlm_annotated.png"
        save_image(vlm_annotated_image, "04_vlm_annotated.png", output_dir)

        logging.info("Getting room classification...")
        room_data: dict | list = switcher.execute_with_fallback(
            "vlm", "get_room_data", str(image_path), []
        )
        save_json_data(room_data, "04a_room_data.json", output_dir)

        logging.info("Getting object captions...")
        captions: dict | list = switcher.execute_with_fallback(
            "vlm", "get_captions", str(vlm_image_path), vlm_labels
        )
        save_json_data(captions, "04b_captions.json", output_dir)

        logging.info("Getting spatial relations...")
        relations: dict | list = switcher.execute_with_fallback(
            "vlm", "get_relations", str(vlm_image_path), vlm_labels
        )
        save_json_data(relations, "04c_relations.json", output_dir)

        if captions:
            logging.info("Consolidating captions...")
            vlm_model = switcher.get_model("vlm")
            consolidated: str = vlm_model.consolidate_captions(captions)
            with open(
                output_dir / "04d_consolidated_caption.txt", "w", encoding="utf-8"
            ) as f:
                f.write(consolidated)
            logging.info("Consolidated caption saved")

        vlm_data: dict = {
            "used_fallback": switcher.is_using_fallback(),
        }
        save_json_data(vlm_data, "04_vlm_info.json", output_dir)

    except RuntimeError as e:
        traceback.print_exc()
        raise RuntimeError(f"VLM inference step failed: {e}") from e

    logging.info("=" * 50)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY")
    logging.info("=" * 50)
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"Total objects detected: {len(detections)}")
    logging.info(f"Final fallback status: {switcher.is_using_fallback()}")


def main() -> None:
    """
    Main function to run the test pipeline with strategy switcher.

    :raises SystemExit: If an error occurs during pipeline execution.
    :return: None
    :rtype: None
    """
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    torch.set_grad_enabled(False)

    config: ModelConfig = ModelConfig()
    output_dir: Path = create_output_directory(config.OUTPUT_DIR)

    logging.info("=" * 50)
    logging.info("MODEL CONFIGURATION")
    logging.info("=" * 50)
    logging.info(f"Device: {config.DEVICE}")
    logging.info(f"Test Image: {config.TEST_IMAGE_PATH}")
    logging.info(f"Output Directory: {output_dir}")
    logging.info("Using StrategySwitcher with automatic fallback")
    logging.info("=" * 50)

    switcher: StrategySwitcher | None = None

    try:
        resource_log_path: str = str(output_dir / "resource_log.csv")

        with SystemResourceLogger(
            sample_interval=0.1, output_path=resource_log_path
        ) as logger:
            logging.info("Collecting baseline metrics (5 seconds)...")
            time.sleep(5)
            logger.set_baseline_collection_flag(False)

            logging.info("Initializing StrategySwitcher...")
            switcher = create_switcher(config)

            run_detection_pipeline(
                switcher=switcher,
                config=config,
                output_dir=output_dir,
            )

            logger.set_baseline_collection_flag(True)
            logging.info("Collecting final baseline metrics (5 seconds)...")
            time.sleep(5)

        logging.info("Generating resource usage plots...")
        logger.plot_resources(output_image_path=str(output_dir / "resource_plot.png"))

    except (ValueError, FileNotFoundError, OSError, RuntimeError) as e:
        logging.error(f"Error during pipeline execution: {e}")
        traceback.print_exc()
        sys.exit(1)

    finally:
        if switcher is not None:
            logging.info("Unloading all models...")
            switcher.unload_all_models()


if __name__ == "__main__":
    whole_it = perf_counter()
    main()
    logging.info(f"Total execution time: {perf_counter() - whole_it:.2f} seconds")

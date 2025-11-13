from agno.agent import Agent, RunResponse
from agno.models.openrouter import OpenRouter
from agno.media import Image as AgnoImage
from supervision import Detections
import numpy as np
import traceback
import logging
import json


from conceptgraph.inference.interfaces import IObjectDetector

REMOTE_DETECTION_PROMPT = """Analyze this image and detect all instances of the following objects: {classes_str}.

For each detected object, provide:
1. The class name (must be one of: {classes_str})
2. Bounding box coordinates in pixels (x1, y1, x2, y2) where (x1,y1) is top-left and (x2,y2) is bottom-right
3. Confidence score (0.0 to 1.0)

Image dimensions: {width}x{height} pixels

Return the results as a JSON array with this exact format:
```json
[
  {{"class": "class_name", "bbox": [x1, y1, x2, y2], "confidence": 0.95}},
  ...
]
```

Only include objects you can clearly see in the image."""


class GeminiDetectorStrategy(IObjectDetector):
    """
    Gemini-based object detection strategy via OpenRouter.

    This class encapsulates preprocessing, detection, and postprocessing for Gemini VLM,
    including conversion of bounding boxes from normalized coordinates to original image size.
    """

    def __init__(
        self,
        model_id: str = "google/gemini-2.5-flash-lite",
        api_key: str | None = None,
    ) -> None:
        """
        Initialize GeminiDetectorStrategy.

        :param model_id: Model identifier for OpenRouter.
        :type model_id: str
        :param api_key: Optional API key for OpenRouter.
        :type api_key: str | None
        """
        self.model_id = model_id
        self.api_key = api_key
        self.agent: Agent | None = None
        self.classes: list[str] = []

    def load_model(self) -> None:
        """
        Initialize the VLM agent with OpenRouter backend.

        :raises RuntimeError: If agent initialization fails.
        :return: None
        :rtype: None
        """
        if not self.is_loaded():
            logging.info(f"Loading Gemini detector agent with model: {self.model_id}")
            try:
                self.agent = Agent(
                    model=OpenRouter(
                        id=self.model_id, api_key=self.api_key, max_tokens=8192
                    ),
                    markdown=True,
                )
            except (ValueError, RuntimeError, ConnectionError) as e:
                traceback.print_exc()
                raise RuntimeError(f"Failed to load Gemini detector agent: {e}")

    def unload_model(self) -> None:
        """
        Unload the VLM agent.

        :return: None
        :rtype: None
        """
        if self.is_loaded():
            logging.info("Unloading Gemini detector agent.")
            del self.agent
            self.agent = None

    def is_loaded(self) -> bool:
        """
        Check if the VLM agent is loaded.

        :return: True if loaded, False otherwise.
        :rtype: bool
        """
        return self.agent is not None

    def get_type(self) -> str:
        """
        Get the type of the detector.

        :return: The string "remote".
        :rtype: str
        """
        return "remote"

    def set_classes(self, classes: list[str]) -> None:
        """
        Set the classes for detection.

        :param classes: List of class names to detect.
        :type classes: list[str]
        :return: None
        :rtype: None
        """
        self.classes = classes
        logging.info(f"Gemini detector classes set to: {classes}")

    def _preprocess(
        self, image_path: str, image_np: np.ndarray
    ) -> tuple[str, int, int]:
        """
        Preprocess the image before detection.

        :param image_path: Path to the image file.
        :type image_path: str
        :param image_np: Image as a numpy array.
        :type image_np: np.ndarray
        :return: Image path, width, and height.
        :rtype: tuple[str, int, int]
        """
        height, width = image_np.shape[:2]
        return image_path, width, height

    def _process(self, image_path: str, width: int, height: int) -> str:
        """
        Run Gemini VLM inference on the image.

        :param image_path: Path to the image file.
        :type image_path: str
        :param width: Image width.
        :type width: int
        :param height: Image height.
        :type height: int
        :return: Raw response content from VLM.
        :rtype: str
        """
        classes_str = ", ".join(self.classes)
        prompt = REMOTE_DETECTION_PROMPT.format(
            classes_str=classes_str, width=width, height=height
        )
        response: RunResponse = self.agent.run(
            prompt, images=[AgnoImage(filepath=image_path)]
        )
        return response.content

    def _parse_json(self, response: str) -> list[dict]:
        """
        Parses a JSON string, handling code blocks that start with ```json.

        :param response: The JSON string, possibly wrapped in a code block.
        :type response: str
        :raises json.JSONDecodeError: If the JSON is invalid and not recoverable.
        :return: The parsed JSON as a list of dictionaries, or an empty list if parsing fails.
        :rtype: list[dict]
        """
        try:
            resp = response.strip()
            if resp.startswith("```json"):
                resp = resp[7:]
                if resp.endswith("```"):
                    resp = resp[:-3]
                resp = resp.strip()
            elif resp.startswith("```"):
                resp = resp[3:]
                if resp.endswith("```"):
                    resp = resp[:-3]
                resp = resp.strip()
            return json.loads(resp)
        except json.JSONDecodeError:
            traceback.print_exc()
            print(f"Failed to parse JSON from response: {response}")
            return []

    def _convert_bboxes_to_original_size(
        self,
        detections_data: list[dict],
        width: int,
        height: int,
    ) -> list[dict]:
        """
        Converts bounding boxes from normalized coordinates (0-1000) to original image size.

        :param detections_data: List of detection dictionaries with 'bbox' key.
        :type detections_data: list[dict]
        :param width: Original image width.
        :type width: int
        :param height: Original image height.
        :type height: int
        :raises ValueError: If any bounding box is not a list of four numbers or is malformed.
        :return: List of detections with bounding boxes converted to original image size.
        :rtype: list[dict]
        """
        try:

            def scale_bbox(bbox: list[int | float] | None) -> list[float] | None:
                if bbox is None:
                    return None
                if not (
                    isinstance(bbox, list)
                    and len(bbox) == 4
                    and all(isinstance(coord, (int, float)) for coord in bbox)
                ):
                    raise ValueError(
                        f"Each bounding box must be a list of four numbers or None: {bbox}"
                    )
                y1 = bbox[0] / 1000 * height
                x1 = bbox[1] / 1000 * width
                y2 = bbox[2] / 1000 * height
                x2 = bbox[3] / 1000 * width
                return [x1, y1, x2, y2]

            converted = []
            for det in detections_data:
                if not isinstance(det, dict):
                    raise ValueError(f"Each detection must be a dictionary: {det}")
                new_det = det.copy()
                new_det["bbox"] = scale_bbox(det.get("bbox"))
                converted.append(new_det)
            return converted
        except (TypeError, ValueError) as e:
            traceback.print_exc()
            raise ValueError(f"Failed to convert bounding boxes: {e}") from e

    def _postprocess(
        self, response_content: str, width: int, height: int
    ) -> Detections:
        """
        Postprocess VLM response into Detections object, converting bounding boxes to original image size.

        :param response_content: Raw response content from VLM.
        :type response_content: str
        :param width: Original image width.
        :type width: int
        :param height: Original image height.
        :type height: int
        :return: Detections object.
        :rtype: Detections
        """
        detections_data = self._parse_json(response_content)
        if not detections_data:
            return Detections.empty()

        detections_data = self._convert_bboxes_to_original_size(
            detections_data, width, height
        )

        xyxy = []
        confidences = []
        class_ids = []

        for det in detections_data:
            class_name = det.get("class")
            bbox = det.get("bbox")
            confidence = det.get("confidence")
            if (
                class_name in self.classes
                and bbox is not None
                and confidence is not None
            ):
                class_id = self.classes.index(class_name)
                xyxy.append(bbox)
                confidences.append(confidence)
                class_ids.append(class_id)

        if not xyxy:
            return Detections.empty()

        return Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(class_ids, dtype=np.int32),
        )

    def detect(self, image_path: str, image_np: np.ndarray) -> Detections:
        """
        Perform object detection using VLM.
        Encapsulates the full preprocessing, processing, and postprocessing pipeline.

        :param image_path: Path to the image file.
        :type image_path: str
        :param image_np: Image as a numpy array.
        :type image_np: np.ndarray
        :return: Detections object with bounding boxes, confidences, and class IDs.
        :rtype: Detections
        :raises RuntimeError: If detection fails.
        """
        self.load_model()
        max_retries = 5
        for attempt in range(max_retries):
            try:
                preprocessed_path, width, height = self._preprocess(
                    image_path, image_np
                )
                response_content = self._process(preprocessed_path, width, height)
                return self._postprocess(response_content, width, height)
            except (
                AttributeError,
                ValueError,
                json.JSONDecodeError,
                RuntimeError,
            ) as e:
                traceback.print_exc()
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Gemini detection failed after {max_retries} attempts: {e}"
                    )
                logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
        raise RuntimeError(f"Gemini detection failed after {max_retries} attempts.")

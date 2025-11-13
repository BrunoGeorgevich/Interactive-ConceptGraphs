from google.genai.types import HttpOptions
from google.genai import types
from google import genai
from PIL import Image
import numpy as np
import traceback
import logging
import base64
import torch
import json
import io
import os

from conceptgraph.inference.interfaces import ISegmenter


REMOTE_SEGMENTATION_PROMPT = """Give the segmentation masks for the {classes} items. It must return exactly {num_detections} segmentation masks.

Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key "box_2d", the segmentation mask in key "mask", and the text label in the key "label". Use descriptive labels.
"""


def parse_json(json_output: str) -> str:
    """
    Parse a JSON string possibly wrapped in markdown fencing.

    :param json_output: The raw JSON output string.
    :type json_output: str
    :return: The cleaned JSON string.
    :rtype: str
    """
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1 :])
            output = json_output.split("```")[0]
            return output
    return json_output


class GeminiSegmenterStrategy(ISegmenter):
    """
    Implements a Gemini-based segmentation strategy using the Google Gemini API.

    This class provides methods for loading and unloading the Gemini client, preprocessing input data,
    running segmentation inference, and postprocessing the results to produce segmentation masks.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the GeminiSegmenterStrategy instance.

        :param model_id: The identifier for the Gemini model.
        :type model_id: str
        :param api_key: The API key for Gemini, or None if not required.
        :type api_key: str | None
        :return: None
        :rtype: None
        """
        self.model_id = model_id
        self.api_key = api_key
        self.client: genai.Client | None = None

    def load_model(self) -> None:
        """
        Load and initialize the Gemini client.

        :raises RuntimeError: If the client fails to initialize.
        :return: None
        :rtype: None
        """
        if not self.is_loaded():
            logging.info(f"Loading Gemini segmenter client with model: {self.model_id}")
            try:
                if self.api_key is not None:
                    os.environ["GOOGLE_API_KEY"] = self.api_key
                self.client = genai.Client(http_options=HttpOptions(timeout=30 * 1000))
            except (ValueError, RuntimeError, OSError) as e:
                traceback.print_exc()
                raise RuntimeError(f"Failed to load Gemini segmenter client: {e}")

    def unload_model(self) -> None:
        """
        Unload the Gemini client and release associated resources.

        :return: None
        :rtype: None
        """
        if self.is_loaded():
            logging.info("Unloading Gemini segmenter client.")
            del self.client
            self.client = None

    def is_loaded(self) -> bool:
        """
        Determine whether the Gemini client is currently loaded.

        :return: True if the client is loaded, False otherwise.
        :rtype: bool
        """
        return self.client is not None

    def get_type(self) -> str:
        """
        Retrieve the type of the segmenter.

        :return: The string "remote".
        :rtype: str
        """
        return "remote"

    def _preprocess(
        self, image_path: str, image_np: np.ndarray, boxes: torch.Tensor
    ) -> tuple[str, np.ndarray, int, int]:
        """
        Preprocess the image and bounding boxes for segmentation.

        :param image_path: The file path to the image.
        :type image_path: str
        :param image_np: The image as a numpy array.
        :type image_np: np.ndarray
        :param boxes: The bounding boxes as a torch tensor.
        :type boxes: torch.Tensor
        :return: A tuple containing the image path, bounding boxes as a numpy array, image width, and image height.
        :rtype: tuple[str, np.ndarray, int, int]
        """
        height, width = image_np.shape[:2]
        boxes_np = boxes.cpu().numpy()
        return image_path, boxes_np, width, height

    def _process(
        self,
        image_path: str,
        classes: list[str],
        num_detections: int,
    ) -> list[dict]:
        """
        Run segmentation inference using Gemini API.

        :param image_path: The file path to the image.
        :type image_path: str
        :param classes: List of class names corresponding to detection class IDs.
        :type classes: list[str]
        :param num_detections: The number of detections to segment.
        :type num_detections: int
        :raises RuntimeError: If the Gemini API call fails.
        :return: A list of dictionaries containing mask data for each bounding box.
        :rtype: list[dict]
        """
        prompt = REMOTE_SEGMENTATION_PROMPT.format(
            classes=", ".join(classes), num_detections=num_detections
        )
        try:
            im = Image.open(image_path)
            im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
            response = self.client.models.generate_content(
                model=self.model_id, contents=[prompt, im], config=config
            )
            items = json.loads(parse_json(response.text))
            return items
        except (
            OSError,
            AttributeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as e:
            traceback.print_exc()
            raise RuntimeError(f"Gemini segmentation API call failed: {e}")

    def _postprocess(
        self, mask_data_list: list[dict], width: int, height: int
    ) -> torch.Tensor:
        """
        Convert mask data into a tensor of segmentation masks.

        :param mask_data_list: A list of dictionaries containing mask data.
        :type mask_data_list: list[dict]
        :param width: The width of the image.
        :type width: int
        :param height: The height of the image.
        :type height: int
        :raises RuntimeError: If mask decoding fails.
        :return: A torch tensor containing the segmentation masks.
        :rtype: torch.Tensor
        """
        masks = []
        for mask_info in mask_data_list:
            mask = np.zeros((height, width), dtype=np.uint8)
            try:
                box = mask_info.get("box_2d", [])
                if len(box) == 4:
                    y0 = int(box[0] / 1000 * height)
                    x0 = int(box[1] / 1000 * width)
                    y1 = int(box[2] / 1000 * height)
                    x1 = int(box[3] / 1000 * width)
                    if y0 >= y1 or x0 >= x1:
                        masks.append(mask)
                        continue
                else:
                    masks.append(mask)
                    continue
                png_str = mask_info.get("mask", "")
                if not png_str.startswith("data:image/png;base64,"):
                    masks.append(mask)
                    continue
                png_str = png_str.removeprefix("data:image/png;base64,")
                mask_data = base64.b64decode(png_str)
                mask_img = Image.open(io.BytesIO(mask_data))
                mask_img = mask_img.resize(
                    (x1 - x0, y1 - y0), Image.Resampling.BILINEAR
                )
                mask_array = np.array(mask_img)
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        if mask_array[y - y0, x - x0] > 128:
                            mask[y, x] = 255
            except (OSError, ValueError, KeyError, base64.binascii.Error) as e:
                traceback.print_exc()
                raise RuntimeError(f"Failed to decode mask: {e}")
            masks.append(mask)
        masks_array = np.array(masks, dtype=bool)
        return torch.from_numpy(masks_array)

    def segment(
        self,
        image_path: str,
        image_np: np.ndarray,
        boxes: torch.Tensor,
        classes: list[str],
    ) -> torch.Tensor:
        """
        Perform segmentation using the Gemini client.

        This method runs the full pipeline: preprocessing, segmentation inference, and postprocessing.
        Ensures that the number of segmentations matches the number of detections; otherwise, retries the execution.

        :param image_path: The file path to the image.
        :type image_path: str
        :param image_np: The image as a numpy array.
        :type image_np: np.ndarray
        :param boxes: The bounding boxes as a torch tensor (Nx4 format).
        :type boxes: torch.Tensor
        :param classes: List of class names corresponding to detection class IDs.
        :type classes: list[str]
        :raises RuntimeError: If segmentation fails or the number of segmentations does not match detections after retries.
        :return: A torch tensor containing the segmentation masks.
        :rtype: torch.Tensor
        """
        self.load_model()
        max_retries = 5
        for attempt in range(max_retries):
            try:
                preprocessed_path, boxes_np, width, height = self._preprocess(
                    image_path, image_np, boxes
                )
                num_detections = len(boxes_np)
                mask_data_list = self._process(
                    preprocessed_path, classes, num_detections
                )
                if len(mask_data_list) == len(boxes_np):
                    return self._postprocess(mask_data_list, width, height)
                logging.warning(
                    f"Attempt {attempt + 1}: Number of segmentations ({len(mask_data_list)}) does not match number of detections ({len(boxes_np)}). Retrying..."
                )
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                traceback.print_exc()
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Gemini segmentation failed after {max_retries} attempts: {e}"
                    )
        raise RuntimeError(
            f"Gemini segmentation failed: Number of segmentations did not match number of detections after {max_retries} attempts."
        )

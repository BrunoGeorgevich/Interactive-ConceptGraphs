from conceptgraph.utils.prompts import (
    SYSTEM_PROMPT_ONLY_TOP,
    SYSTEM_PROMPT_CAPTIONS,
    SYSTEM_PROMPT_CONSOLIDATE_CAPTIONS,
    SYSTEM_PROMPT_ROOM_CLASS,
)
from openai import OpenAI
from PIL import Image
import traceback
import base64
import json
import ast
import os
import re


def get_vlm_openai_like_client(model: str, api_key: str, base_url: str) -> OpenAI:
    """
    Create and return an OpenAI client for VLM (Vision-Language Model) usage.

    :param model: The model name to use for the client.
    :type model: str
    :param api_key: The API key for OpenAI authentication.
    :type api_key: str
    :param base_url: The base URL for the OpenAI API.
    :type base_url: str
    :return: An OpenAI client instance with the model attribute set.
    :rtype: OpenAI

    This function initializes an OpenAI client and attaches the model name as an attribute.
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    setattr(client, "model", model)
    return client


def encode_image_for_openai(
    image_path: str, resize: bool = False, target_size: int = 512
) -> str:
    """
    Encode an image file as a base64 string for OpenAI API usage, with optional resizing.

    :param image_path: Path to the image file to encode.
    :type image_path: str
    :param resize: Whether to resize the image before encoding.
    :type resize: bool
    :param target_size: Target size for the largest dimension if resizing is enabled.
    :type target_size: int
    :return: Base64-encoded string of the image.
    :rtype: str
    :raises FileNotFoundError: If the image file does not exist.

    This function checks for the existence of the image, optionally resizes it while maintaining aspect ratio,
    and encodes it as a base64 string for use with the OpenAI API.
    """
    print(f"Checking if image exists at path: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not resize:
        # No resizing: encode the original image directly.
        print(f"Opening image from path: {image_path}")
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
            print("Image encoded in base64 format.")
        return encoded_image

    # Resize the image to the target size, maintaining aspect ratio.
    print(f"Opening image from path: {image_path}")
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        print(f"Original image dimensions: {original_width} x {original_height}")

        if original_width > original_height:
            scale = target_size / original_width
            new_width = target_size
            new_height = int(original_height * scale)
        else:
            scale = target_size / original_height
            new_height = target_size
            new_width = int(original_width * scale)

        print(f"Resized image dimensions: {new_width} x {new_height}")

        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        print("Image resized successfully.")

        # Save the resized image temporarily for encoding.
        with open("temp_resized_image.jpg", "wb") as temp_file:
            img_resized.save(temp_file, format="JPEG")
            print("Resized image saved temporarily for encoding.")

        # Encode the resized image as base64.
        with open("temp_resized_image.jpg", "rb") as temp_file:
            encoded_image = base64.b64encode(temp_file.read()).decode("utf-8")
            print("Image encoded in base64 format.")

        # Remove the temporary file.
        os.remove("temp_resized_image.jpg")
        print("Temporary file removed.")

    return encoded_image


def consolidate_captions(client: OpenAI, captions: list) -> str:
    """
    Consolidate multiple captions for the same object into a single, clear caption using the OpenAI API.

    :param client: OpenAI client with the model attribute set.
    :type client: OpenAI
    :param captions: List of caption dictionaries, each with a 'caption' key.
    :type captions: list
    :return: A single consolidated caption string.
    :rtype: str

    This function formats the input captions into a prompt, sends it to the OpenAI API,
    and parses the response to extract the consolidated caption.
    """
    # Format the captions into a single string for the prompt.
    captions_text = "\n".join(
        [f"{cap['caption']}" for cap in captions if cap["caption"] is not None]
    )
    user_query = (
        f"Here are several captions for the same object:\n{captions_text}\n\n"
        "Please consolidate these into a single, clear caption that accurately describes the object."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CONSOLIDATE_CAPTIONS},
        {"role": "user", "content": user_query},
    ]

    consolidated_caption = ""
    try:
        response = client.chat.completions.create(
            model=f"{client.model}",
            messages=messages,
            response_format={"type": "json_object"},
            timeout=20,
        )
        consolidated_caption_json = response.choices[0].message.content.strip()
        consolidated_caption = json.loads(consolidated_caption_json).get(
            "consolidated_caption", ""
        )
        print(f"Consolidated Caption: {consolidated_caption}")

    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"Failed to parse consolidated caption from OpenAI response: {str(e)}")
        traceback.print_exc()
        consolidated_caption = ""
    except (OSError, TypeError) as e:
        print(f"Error during OpenAI API call for caption consolidation: {str(e)}")
        traceback.print_exc()
        consolidated_caption = ""

    return consolidated_caption


def extract_list_of_tuples(text: str) -> list:
    """
    Extract a list of tuples from a string, typically from VLM output.

    :param text: Text containing a list of tuples (e.g., "[('a', 'b'), ...]").
    :type text: str
    :return: Extracted list of tuples, or an empty list if extraction fails.
    :rtype: list

    This function searches for a list pattern in the text and attempts to parse it as a Python list of tuples.
    """
    # Replace newlines for uniformity and search for a list pattern.
    text = text.replace("\n", " ")
    pattern = r"\[.*?\]"

    match = re.search(pattern, text)
    if match:
        list_str = match.group(0)
        try:
            result = ast.literal_eval(list_str)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            print("Found string cannot be converted to a list of tuples.")
            traceback.print_exc()
            return []
    else:
        print("No list of tuples found in the text.")
        return []


def vlm_extract_object_captions(text: str) -> list:
    """
    Extract a list of object captions from a string containing a JSON-like list of objects.

    :param text: Text containing a list of objects with captions (e.g., "[{'caption': ...}, ...]").
    :type text: str
    :return: Extracted list of caption objects, or an empty list if extraction fails.
    :rtype: list

    This function attempts to parse a list of dictionaries from the text, handling both well-formed and partially-formed lists.
    """
    text = text.replace("\n", " ")
    pattern = r"\[(.*?)\]"

    match = re.search(pattern, text)
    if match:
        list_str = match.group(0)
        try:
            result = ast.literal_eval(list_str)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            # If the whole string conversion fails, process each element individually.
            elements = re.findall(r"{.*?}", list_str)
            result = []
            for element in elements:
                try:
                    obj = ast.literal_eval(element)
                    if isinstance(obj, dict):
                        result.append(obj)
                except (ValueError, SyntaxError):
                    print(f"Error processing element: {element}")
                    traceback.print_exc()
            return result
    else:
        print("No list of objects found in the text.")
        return []


def get_obj_rel_from_image(client: OpenAI, image_path: str, label_list: list) -> list:
    """
    Extract object relationships from an image using GPT-4V via the OpenAI API.

    :param client: OpenAI client with the model attribute set.
    :type client: OpenAI
    :param image_path: Path to the image file to analyze.
    :type image_path: str
    :param label_list: List of object labels present in the image.
    :type label_list: list
    :return: List of tuples describing object relationships, or an empty list if extraction fails.
    :rtype: list

    This function encodes the image, constructs a prompt, sends it to the OpenAI API,
    and parses the response to extract object relationships.
    """
    # Encode the image as base64 for API usage.
    base64_image = encode_image_for_openai(image_path)

    user_query = (
        f"Here is the list of labels for the annotations of the objects in the image: {label_list}. "
        "Please describe the spatial relationships between the objects in the image."
    )

    vlm_answer = []
    try:
        response = client.chat.completions.create(
            model=f"{client.model}",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ONLY_TOP},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                },
                {"role": "user", "content": user_query},
            ],
            timeout=20,
        )

        vlm_answer_str = response.choices[0].message.content
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")

        vlm_answer = extract_list_of_tuples(vlm_answer_str)

    except (AttributeError, KeyError, TypeError) as e:
        print(f"Error extracting object relationships from OpenAI response: {str(e)}")
        traceback.print_exc()
        print("Setting vlm_answer to an empty list.")
        vlm_answer = []
    print(f"Line 68, user_query: {user_query}")
    print(f"Line 97, vlm_answer: {vlm_answer}")

    return vlm_answer


def get_obj_captions_from_image(
    client: OpenAI, image_path: str, label_list: list
) -> list:
    """
    Extract object captions from an image using GPT-4V via the OpenAI API.

    :param client: OpenAI client with the model attribute set.
    :type client: OpenAI
    :param image_path: Path to the image file to analyze.
    :type image_path: str
    :param label_list: List of object labels present in the image.
    :type label_list: list
    :return: List of dictionaries containing object captions, or an empty list if extraction fails.
    :rtype: list

    This function encodes the image, constructs a prompt, sends it to the OpenAI API,
    and parses the response to extract object captions for each object in the image.
    """
    # Encode the image as base64 for API usage.
    base64_image = encode_image_for_openai(image_path)

    user_query = (
        f"Here is the list of labels for the annotations of the objects in the image: {label_list}. "
        "Please accurately caption the objects in the image."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CAPTIONS},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        },
        {"role": "user", "content": user_query},
    ]

    vlm_answer_captions = []
    try:
        response = client.chat.completions.create(
            model=f"{client.model}", messages=messages, timeout=20
        )

        vlm_answer_str = response.choices[0].message.content
        print(f"Line 113, vlm_answer_str: {vlm_answer_str}")

        vlm_answer_captions = vlm_extract_object_captions(vlm_answer_str)

    except (AttributeError, KeyError, TypeError) as e:
        print(f"Error extracting object captions from OpenAI response: {str(e)}")
        traceback.print_exc()
        print("Setting vlm_answer to an empty list.")
        vlm_answer_captions = []
    print(f"Line 68, user_query: {user_query}")
    print(f"Line 97, vlm_answer: {vlm_answer_captions}")

    return vlm_answer_captions


def get_room_data_from_image(
    client: OpenAI, image_path: str, room_data_list: list
) -> str:
    """
    Extract room class from an image using GPT-4V via the OpenAI API.

    :param client: OpenAI client with the model attribute set.
    :type client: OpenAI
    :param image_path: Path to the image file to analyze.
    :type image_path: str
    :param room_data_list: List of room data dictionaries.
    :type room_data_list: list
    :return: Room data dictionary.
    :rtype: dict
    """
    base64_image = encode_image_for_openai(image_path)
    room_classes = [
        "kitchen",
        "bathroom",
        "bedroom",
        "living room",
        "office",
        "hallway",
        "laundry room",
        "transitioning",
    ]

    last_room_data = None

    if isinstance(room_data_list, list) and len(room_data_list) > 0:
        last_room_data = [
            {
                "room_class": room_data_list[-1]["room_class"],
                "room_description": room_data_list[-1]["room_description"],
            }
        ]

    user_query = f"Extract the room class and description from the image based on the provided room classes and last room data. Be extremely detailed in your analysis of colors, textures, materials, and spatial arrangements while remaining factually accurate. Focus on the visual and physical characteristics of the room rather than inferring its function. \n\n The list of possible room classes: {room_classes} \n\n Previous Room Data: {last_room_data}"

    response = client.chat.completions.create(
        model=f"{client.model}",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT_ROOM_CLASS,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            },
            {"role": "user", "content": user_query},
        ],
        response_format={"type": "json_object"},
        timeout=20,
    )

    room_data = response.choices[0].message.content
    try:
        if "```json" in room_data and "```" in room_data:
            json_content = room_data.split("```json")[1].split("```")[0].strip()
            room_data = json.loads(json_content)
        elif "```" in room_data:
            json_content = room_data.split("```")[1].split("```")[0].strip()
            room_data = json.loads(json_content)
        else:
            room_data = json.loads(room_data)

        print(f"Room class: {room_data['room_class']}")
        print(f"Room description: {room_data['room_description']}")
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        traceback.print_exc()
        print(f"Failed to parse room class JSON: {room_data}")
        room_data = {"room_class": "error", "room_description": e}
    return room_data

import traceback
import cv2

from conceptgraph.interaction.watershed_segmenter import (
    load_house_context,
    load_profile,
    process_house_segmentation,
    DEFAULT_PARAMS,
    CLASS_COLORS,
    TRAJECTORY_IMAGE_DIMMING,
    MAP_BINARY_THRESHOLD,
    MIN_CONTOUR_AREA,
    CROP_PADDING,
)


def save_watershed_image(
    house_id: int,
    prefix: str,
    base_path: str,
    output_filename: str = "watershed_segmented.png",
) -> None:
    """
    Loads the house context, runs watershed segmentation, and saves the segmented image.

    :param house_id: Identifier for the house.
    :type house_id: int
    :param prefix: Prefix for the map context.
    :type prefix: str
    :param base_path: Base path to the dataset.
    :type base_path: str
    :param output_filename: Output filename for the segmented image.
    :type output_filename: str
    :raises RuntimeError: If segmentation or saving fails.
    :return: None
    :rtype: None
    """
    try:
        context = load_house_context(
            house_id=house_id,
            base_path=base_path,
            prefixes=[prefix],
            map_binary_threshold=MAP_BINARY_THRESHOLD,
            min_contour_area=MIN_CONTOUR_AREA,
            crop_padding=CROP_PADDING,
        )
    except (OSError, IOError, KeyError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load house context: {e}")

    if prefix not in context:
        raise RuntimeError(f"Prefix '{prefix}' not found in loaded context.")

    try:
        params = load_profile(house_id, DEFAULT_PARAMS, base_path=base_path)
        watershed_response = process_house_segmentation(
            house_id=house_id,
            house_context=context,
            params=params,
            prefixes=[prefix],
            llm_agent=None,
            generate_descriptions=False,
            prompt_mask="",
            class_colors=CLASS_COLORS,
            trajectory_dimming=TRAJECTORY_IMAGE_DIMMING,
        )
    except (KeyError, TypeError, ValueError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to process watershed segmentation: {e}")

    try:
        reconstructed_image = watershed_response[prefix]["reconstructed_img"]
        watershed_image = watershed_response[prefix]["watershed_img"]
        cv2.imwrite(
            output_filename.replace("watershed", "reconstructed"), reconstructed_image
        )
        cv2.imwrite(output_filename, watershed_image)
        print(f"Segmented watershed image saved as '{output_filename}'.")
    except (OSError, IOError, ImportError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to save watershed image: {e}")


if __name__ == "__main__":
    HOUSE_ID = 1
    PREFIX = "online"
    BASE_PATH = "THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET"
    OUTPUT_FILENAME = "watershed_segmented.png"

    save_watershed_image(
        house_id=HOUSE_ID,
        prefix=PREFIX,
        base_path=BASE_PATH,
        output_filename=OUTPUT_FILENAME,
    )

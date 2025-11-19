from pathlib import Path
import traceback
import yaml
import cv2
import os


def process_map_image(
    image_path: str, output_image_path: str = "processed_map.png"
) -> dict:
    """
    Processes a map image by detecting its border, correcting rotation, rotating 90° counterclockwise, and cropping.
    Returns a dictionary with information required to recalculate the origin.

    :param image_path: Path to the input map image.
    :type image_path: str
    :param output_image_path: Path to save the processed image. Defaults to "processed_map.png".
    :type output_image_path: str
    :raises FileNotFoundError: If the image cannot be loaded from the given path.
    :raises ValueError: If no contours are found to crop the map.
    :return: Dictionary containing crop and transformation information.
    :rtype: dict
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        mask_background = cv2.inRange(image, 205, 205)
        map_mask = cv2.bitwise_not(mask_background)

        contours, _ = cv2.findContours(
            map_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("No contours found to crop the map.")

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        center, size, angle = rect
        width, height = size

        if width < height:
            width, height = height, width

        M_align = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(
            image,
            M_align,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=205,
        )

        rotated = cv2.rotate(aligned, cv2.ROTATE_90_COUNTERCLOCKWISE)

        mask_rotated = cv2.rotate(map_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        contours_rot, _ = cv2.findContours(
            mask_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        x, y, w, h = cv2.boundingRect(max(contours_rot, key=cv2.contourArea))

        cropped = rotated[y : y + h, x : x + w]

        output_path_full = os.path.join(os.path.dirname(image_path), output_image_path)
        cv2.imwrite(output_path_full, cropped)
        print(f"Mapa processado salvo em: {output_path_full}")

        info = {
            "crop_x": x,
            "crop_y": y,
            "crop_w": w,
            "crop_h": h,
            "original_width": image.shape[1],
            "original_height": image.shape[0],
            "rotation_angle": angle,
            "final_image": cropped,
        }
        return info
    except (cv2.error, FileNotFoundError, ValueError) as exc:
        traceback.print_exc()
        raise RuntimeError(f"Failed to process map image: {exc}") from exc


def update_map_yaml(
    original_yaml_path: str,
    crop_info: dict,
    new_yaml_path: str = "map_processed.yaml",
    resolution: float = 0.05,
) -> dict:
    """
    Updates the map YAML file with a new origin and other parameters after cropping and rotating the map image.

    :param original_yaml_path: Path to the original YAML file.
    :type original_yaml_path: str
    :param crop_info: Dictionary containing crop and transformation information.
    :type crop_info: dict
    :param new_yaml_path: Path to save the updated YAML file. Defaults to "map_processed.yaml".
    :type new_yaml_path: str
    :param resolution: Map resolution in meters per pixel. Defaults to 0.05.
    :type resolution: float
    :raises OSError: If the YAML file cannot be written.
    :return: Dictionary with updated YAML data.
    :rtype: dict
    """
    try:
        orig_w = crop_info["original_width"]
        orig_h = crop_info["original_height"]
        x = crop_info["crop_x"]
        y = crop_info["crop_y"]
        w = crop_info["crop_w"]
        h = crop_info["crop_h"]

        old_origin_x = -20.0
        old_origin_y = -20.0

        meters_per_pixel = resolution

        crop_bottom_left_x = x * meters_per_pixel
        crop_bottom_left_y = (orig_h - (y + h)) * meters_per_pixel

        world_x = old_origin_x + crop_bottom_left_x
        world_y = old_origin_y + crop_bottom_left_y

        new_origin_y = world_y
        new_origin_x = world_x

        new_data = {
            "image": "processed_map.png",
            "resolution": resolution,
            "origin": [round(new_origin_x, 6), round(new_origin_y, 6), 0.000000],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        }

        yaml_full_path = os.path.join(
            os.path.dirname(original_yaml_path), new_yaml_path
        )
        with open(yaml_full_path, "w") as f:
            yaml.safe_dump(new_data, f, sort_keys=False)

        print(f"Novo map.yaml gerado: {yaml_full_path}")
        print(f"   Novo origin: [{new_origin_x:.6f}, {new_origin_y:.6f}, 0.000000]")

        return new_data
    except (OSError, KeyError, TypeError) as exc:
        traceback.print_exc()
        raise RuntimeError(f"Failed to update map YAML: {exc}") from exc


def process_image(home_folder: str) -> None:
    """
    Main function to process map images and YAML files for folders named Home01, Home02, ..., Home29.

    :param home_folder: Path to the home folder containing map.pgm and map.yaml.
    :type home_folder: str
    :raises RuntimeError: If processing fails for a given home folder.
    :return: None
    :rtype: None
    """
    home_path = Path(home_folder)
    map_pgm = home_path / "map.pgm"
    map_yaml = home_path / "map.yaml"

    if not map_pgm.exists():
        print(f"{map_pgm} não encontrado!")
        return

    try:
        print(f"\nProcessando {home_path.name}...")
        info = process_map_image(str(map_pgm), "processed_map.png")
        update_map_yaml(str(map_yaml), info, "processed_map.yaml")
    except (RuntimeError,) as exc:
        print(f"Erro ao processar {home_path.name}: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    base_dir = r"D:\Documentos\Datasets\Robot@VirtualHomeLarge"
    for i in range(1, 31):
        folder = os.path.join(base_dir, f"Home{i:02d}")
        process_image(folder)

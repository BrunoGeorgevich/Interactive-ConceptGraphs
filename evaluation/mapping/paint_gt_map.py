import pandas as pd
import numpy as np
import traceback
import yaml
import cv2
import os

from labs.utils import (
    read_virtual_objects,
    unity_to_ros_coordinates,
    world_to_map,
    load_pkl_gz_result,
)


def virtual_objects_csv_path(database_path: str, home_id: int) -> str:
    """
    Constructs the file path for the VirtualObjects.csv file.

    :param database_path: The base path to the database.
    :type database_path: str
    :param home_id: The identifier of the home.
    :type home_id: int
    :raises ValueError: If database_path is empty or home_id is negative.
    :return: The absolute path to the CSV file.
    :rtype: str
    """
    if not database_path or home_id < 0:
        raise ValueError("Invalid database_path or home_id for CSV path construction.")
    return os.path.join(database_path, f"Home{home_id:02d}", "VirtualObjects.csv")


def processed_objects_pkl_path(
    database_path: str, home_id: int, processing_type: str
) -> str:
    """
    Constructs the file path for the processed objects .pkl.gz file.

    :param database_path: The base path to the database.
    :type database_path: str
    :param home_id: The identifier of the home.
    :type home_id: int
    :param processing_type: The type of processing.
    :type processing_type: str
    :raises ValueError: If database_path is empty or home_id is negative.
    :return: The absolute path to the processed objects file.
    :rtype: str
    """
    if not database_path or home_id < 0 or not processing_type:
        raise ValueError("Invalid arguments for processed objects path construction.")
    return os.path.join(
        database_path,
        "outputs",
        f"Home{home_id:02d}",
        "Wandering",
        "exps",
        f"{processing_type}_house_{home_id}_map",
        f"pcd_{processing_type}_house_{home_id}_map.pkl.gz",
    )


def map_image_path(database_path: str, home_id: int) -> str:
    """
    Constructs the file path for the generated map image.

    :param database_path: The base path to the database.
    :type database_path: str
    :param home_id: The identifier of the home.
    :type home_id: int
    :raises ValueError: If database_path is empty or home_id is negative.
    :return: The absolute path to the map image.
    :rtype: str
    """
    if not database_path or home_id < 0:
        raise ValueError(
            "Invalid database_path or home_id for map image path construction."
        )
    return os.path.join(database_path, f"Home{home_id:02d}", "generated_map.png")


def read_map_properties_yaml(map_path: str) -> dict:
    """
    Reads the map properties from a YAML file associated with the map image.

    :param map_path: The path to the map image file.
    :type map_path: str
    :raises RuntimeError: If the YAML file does not exist or cannot be parsed.
    :return: A dictionary containing map properties.
    :rtype: dict
    """
    map_properties_path = map_path.replace(".png", ".yaml")
    try:
        with open(map_properties_path, "r") as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        traceback.print_exc()
        raise RuntimeError(f"Error reading map properties YAML: {e}")


def get_color_for_class(class_name: str) -> tuple:
    """
    Returns a color tuple for a given class name.

    :param class_name: Name of the object class.
    :type class_name: str
    :return: RGB color tuple.
    :rtype: tuple
    """
    color_map = {
        "sofa": (255, 99, 71),
        "bed": (186, 85, 211),
        "table": (255, 215, 0),
        "chair": (60, 179, 113),
        "kitchen": (255, 99, 71),
        "bathroom": (135, 206, 250),
        "bedroom": (186, 85, 211),
        "living room": (60, 179, 113),
        "office": (255, 215, 0),
        "hallway": (255, 140, 0),
        "laundry room": (70, 130, 180),
        "transitioning": (128, 128, 128),
    }
    return color_map.get(class_name.lower(), (128, 128, 128))


def draw_objects_by_class_on_map(
    map_image: np.ndarray,
    objects: list | pd.DataFrame,
    origin: list | tuple,
    resolution: float,
    image_height: int,
    class_name: str = "",
    radius: int = 6,
    is_processed: bool = False,
) -> np.ndarray:
    """
    Draws objects filtered by class_name as colored circles on the map image.

    :param map_image: The map image as a NumPy array.
    :type map_image: np.ndarray
    :param objects: List or DataFrame of objects.
    :type objects: list | pd.DataFrame
    :param origin: Origin coordinates of the map.
    :type origin: list | tuple
    :param resolution: Map resolution in meters per pixel.
    :type resolution: float
    :param image_height: Height of the map image.
    :type image_height: int
    :param class_name: Class name to filter objects. If empty or '*', draws all objects.
    :type class_name: str
    :param radius: Radius of the circle to draw for each object.
    :type radius: int
    :param is_processed: Whether the objects are processed (True) or virtual (False).
    :type is_processed: bool
    :raises RuntimeError: If coordinate conversion fails.
    :return: Map image with filtered objects drawn.
    :rtype: np.ndarray
    """
    output_img = map_image.copy()
    if len(output_img.shape) == 2 or output_img.shape[2] == 1:
        output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
    filter_class = class_name.strip().lower()
    if isinstance(objects, pd.DataFrame):
        iterator = objects.iterrows()
    else:
        iterator = enumerate(objects)
    for _, row in iterator:
        try:
            if is_processed:
                obj_dict = (
                    row if isinstance(row, dict) else getattr(row, "__dict__", {})
                )
                obj_class = str(obj_dict.get("class_name", "Unknown"))
                pcd_np = obj_dict.get("pcd_np", np.array([]))
                if not hasattr(pcd_np, "mean"):
                    continue
                pos = pcd_np.mean(axis=0)
                ros_x, ros_y = float(pos[2]), -float(pos[0])
                # if (
                #     filter_class
                #     and filter_class != "*"
                #     and obj_class.lower() != filter_class
                # ):
                #     continue
            else:
                obj_class = str(row.get("type", "Unknown"))
                if (
                    filter_class
                    and filter_class != "*"
                    and obj_class.lower() != filter_class
                ):
                    continue
                ros_x, ros_y, _, _ = unity_to_ros_coordinates(
                    row["globalPosition"], row["rotation"], ""
                )
            pixel_x, pixel_y = world_to_map(
                ros_x, ros_y, origin, resolution, image_height
            )
            color = get_color_for_class(obj_class)
            cv2.circle(output_img, (int(pixel_x), int(pixel_y)), radius, color, -1)
        except (KeyError, TypeError, ValueError):
            traceback.print_exc()
            continue
    return output_img


def main_class_map(
    class_name: str = "*", home_id: int = 1, processing_type: str = "original"
) -> None:
    """
    Generates two map images: one with ground truth (virtual) objects painted, and another with processed objects painted, filtered by class_name.
    Allows clicking on the map to print the class names of virtual or processed objects at the clicked position.

    :param class_name: Class name to filter objects. If empty or '*', draws all objects.
    :type class_name: str
    :param home_id: Identifier of the home to process.
    :type home_id: int
    :param processing_type: The type of processing for processed objects.
    :type processing_type: str
    :raises RuntimeError: If any step fails.
    :return: None
    :rtype: None
    """
    try:
        database_path = "THIS PATH MUST POINT TO THE ROOT FOLDER OF YOUR DATASET"
        output_dir = os.path.join(database_path, "evaluation_results")
        os.makedirs(output_dir, exist_ok=True)

        virtual_objects = read_virtual_objects(
            virtual_objects_csv_path(database_path, home_id)
        )
        map_path = map_image_path(database_path, home_id)
        map_image = cv2.imread(map_path)
        if map_image is None:
            raise RuntimeError(f"Could not load map image: {map_path}")
        map_properties = read_map_properties_yaml(map_path)
        origin = map_properties["origin"]
        resolution = map_properties["resolution"]
        image_height = map_image.shape[0]

        processed_path = processed_objects_pkl_path(
            database_path, home_id, processing_type
        )
        processed_data = load_pkl_gz_result(processed_path)
        processed_objects = processed_data.get("objects", [])

        painted_virtual_map = draw_objects_by_class_on_map(
            map_image,
            virtual_objects,
            origin,
            resolution,
            image_height,
            class_name=class_name,
            is_processed=False,
        )

        painted_processed_map = draw_objects_by_class_on_map(
            map_image,
            processed_objects,
            origin,
            resolution,
            image_height,
            class_name=class_name,
            is_processed=True,
        )

        def mouse_callback_virtual_object(
            event: int, x: int, y: int, flags: int, param: None
        ) -> None:
            """
            Mouse callback to print class names of virtual objects at the clicked position.

            :param event: Mouse event type.
            :type event: int
            :param x: X coordinate of the mouse event.
            :type x: int
            :param y: Y coordinate of the mouse event.
            :type y: int
            :param flags: Event flags.
            :type flags: int
            :param param: Additional parameters.
            :type param: None
            :return: None
            :rtype: None
            """
            if event == cv2.EVENT_LBUTTONDOWN:
                try:
                    found_classes = []
                    for _, row in virtual_objects.iterrows():
                        try:
                            obj_class = str(row.get("type", "Unknown"))
                            ros_x, ros_y, _, _ = unity_to_ros_coordinates(
                                row["globalPosition"], row["rotation"], ""
                            )
                            pixel_x, pixel_y = world_to_map(
                                ros_x, ros_y, origin, resolution, image_height
                            )
                            dist = (
                                (int(pixel_x) - x) ** 2 + (int(pixel_y) - y) ** 2
                            ) ** 0.5
                            if dist <= 6:
                                found_classes.append(obj_class)
                        except (KeyError, TypeError, ValueError):
                            traceback.print_exc()
                            continue
                    if found_classes:
                        print(
                            f"Virtual objects at ({x}, {y}): {', '.join(found_classes)}"
                        )
                    else:
                        print(f"No virtual objects found at ({x}, {y})")
                except (KeyError, TypeError, ValueError):
                    traceback.print_exc()
                    print("Error finding objects at clicked position.")

        def mouse_callback_processed_object(
            event: int, x: int, y: int, flags: int, param: None
        ) -> None:
            """
            Mouse callback to print class names of processed objects at the clicked position.

            :param event: Mouse event type.
            :type event: int
            :param x: X coordinate of the mouse event.
            :type x: int
            :param y: Y coordinate of the mouse event.
            :type y: int
            :param flags: Event flags.
            :type flags: int
            :param param: Additional parameters.
            :type param: None
            :return: None
            :rtype: None
            """
            if event == cv2.EVENT_LBUTTONDOWN:
                try:
                    found_classes = []
                    for obj in processed_objects:
                        try:
                            obj_dict = (
                                obj
                                if isinstance(obj, dict)
                                else getattr(obj, "__dict__", {})
                            )
                            obj_class = str(obj_dict.get("class_name", "Unknown"))
                            obj_caption = str(obj_dict.get("consolidated_caption", ""))

                            obj_class += " (" + obj_caption + ")"
                            pcd_np = obj_dict.get("pcd_np", np.array([]))
                            if not hasattr(pcd_np, "mean"):
                                continue
                            pos = pcd_np.mean(axis=0)
                            ros_x, ros_y = float(pos[2]), -float(pos[0])
                            pixel_x, pixel_y = world_to_map(
                                ros_x, ros_y, origin, resolution, image_height
                            )
                            dist = (
                                (int(pixel_x) - x) ** 2 + (int(pixel_y) - y) ** 2
                            ) ** 0.5
                            if dist <= 6:
                                found_classes.append(obj_class)
                        except (KeyError, TypeError, ValueError):
                            traceback.print_exc()
                            continue
                    if found_classes:
                        print(
                            f"Processed objects at ({x}, {y}): {', '.join(found_classes)}"
                        )
                    else:
                        print(f"No processed objects found at ({x}, {y})")
                except (KeyError, TypeError, ValueError):
                    traceback.print_exc()
                    print("Error finding processed objects at clicked position.")

        cv2.namedWindow("Painted Virtual Objects Map")
        cv2.setMouseCallback(
            "Painted Virtual Objects Map", mouse_callback_virtual_object
        )
        cv2.namedWindow("Painted Processed Objects Map")
        cv2.setMouseCallback(
            "Painted Processed Objects Map", mouse_callback_processed_object
        )
        while True:
            cv2.imshow("Painted Virtual Objects Map", painted_virtual_map)
            cv2.imshow("Painted Processed Objects Map", painted_processed_map)
            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                break
        cv2.destroyAllWindows()

    except (RuntimeError, OSError, IOError, KeyError, ValueError) as e:
        traceback.print_exc()
        print(f"Failed to generate painted class maps: {e}")


if __name__ == "__main__":
    HOME_ID = 1
    CLASS_NAME = "Lighter"
    PROCESSING_TYPE = "original"
    main_class_map(CLASS_NAME, HOME_ID, PROCESSING_TYPE)

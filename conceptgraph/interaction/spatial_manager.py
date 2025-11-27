from typing import Tuple, List, Dict, Optional
from functools import lru_cache
import numpy as np
import pickle
import gzip
import json
import cv2
import os

from conceptgraph.interaction.schemas import SystemConfig
from conceptgraph.interaction.utils import MapLoadError


try:
    from watershed_segmenter import (
        load_house_context,
        process_house_segmentation,
        load_profile,
        CLASS_COLORS,
    )
except ImportError:
    from conceptgraph.interaction.watershed_segmenter import (
        load_house_context,
        process_house_segmentation,
        load_profile,
        CLASS_COLORS,
    )


class SceneGraphNode:
    """
    Represents a node in the scene graph corresponding to an object.
    """

    def __init__(self, obj_dict: dict) -> None:
        """
        Initializes a scene graph node from an object dictionary.

        :param obj_dict: Object data dictionary.
        :type obj_dict: dict
        """
        self.name: str = obj_dict.get("class_name", "unknown_object")
        self.room: str = obj_dict.get("room_name", "Unknown Area")

        raw_caption: str = obj_dict.get("consolidated_caption", "") or obj_dict.get(
            "object_caption", ""
        )
        caption_clean: str = (
            raw_caption.replace("'", '"')
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        try:
            loaded = json.loads(caption_clean)
            if isinstance(loaded, dict):
                caption_clean = loaded.get("consolidated_caption", caption_clean)
        except (json.JSONDecodeError, TypeError):
            pass

        self.caption: str = caption_clean
        self.centroid: Optional[np.ndarray] = None

        if "pcd_np" in obj_dict and len(obj_dict["pcd_np"]) > 0:
            self.centroid = np.mean(obj_dict["pcd_np"], axis=0)
        elif "bbox_np" in obj_dict:
            self.centroid = np.mean(obj_dict["bbox_np"], axis=0)


class SceneGraphManager:
    """
    Manages the scene graph structure for all objects in the environment.
    """

    def __init__(self, enriched_objects: List[dict]) -> None:
        """
        Initializes the scene graph manager.

        :param enriched_objects: List of enriched object dictionaries.
        :type enriched_objects: List[dict]
        """
        self.nodes: List[SceneGraphNode] = [
            SceneGraphNode(obj) for obj in enriched_objects
        ]
        self.nodes_by_room: Dict[str, List[SceneGraphNode]] = {}
        for node in self.nodes:
            if node.room not in self.nodes_by_room:
                self.nodes_by_room[node.room] = []
            self.nodes_by_room[node.room].append(node)

    @lru_cache(maxsize=128)
    def get_room_centers(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Computes and returns the centroids of each room.

        :return: Mapping of room names to their centroid coordinates.
        :rtype: Dict[str, Tuple[float, float, float]]
        """
        room_centers = {}
        for room, nodes in self.nodes_by_room.items():
            centroids = [node.centroid for node in nodes if node.centroid is not None]
            if centroids:
                mean_centroid = np.mean(centroids, axis=0)
                room_centers[room] = (
                    mean_centroid[0],
                    mean_centroid[1],
                    mean_centroid[2],
                )
            else:
                room_centers[room] = (0.0, 0.0, 0.0)
        return room_centers

    def get_text_representation(self, user_pos: Tuple[float, float, float]) -> str:
        """
        Generates a text representation of the scene graph sorted by distance.

        :param user_pos: User world coordinates (x, y, z).
        :type user_pos: Tuple[float, float, float]
        :return: XML-formatted scene graph string.
        :rtype: str
        """
        output = "<SCENE_GRAPH>\n"
        user_arr = np.array(user_pos)
        room_centers = self.get_room_centers()

        for room in sorted(self.nodes_by_room.keys()):
            room_items = []
            for node in self.nodes_by_room[room]:
                dist_val = float("inf")
                dist_str = "N/A"
                if node.centroid is not None:
                    dist_val = np.linalg.norm(node.centroid - user_arr)
                    dist_str = f"{dist_val:.2f}m"
                room_items.append((dist_val, node, dist_str))

            room_items.sort(key=lambda x: x[0])
            room_center = room_centers.get(room, (0.0, 0.0))
            output += f'<ROOM id="{room}" center="{room_center}"> '
            for _, node, dist_str in room_items:
                output += "<OBJECT> "
                output += f"Class: {node.name} | "
                output += f"Description: {node.caption} | "
                output += f"Distance: {dist_str} | "
                output += "</OBJECT> "
            output += "</ROOM>\n"

        output += "</SCENE_GRAPH>"
        return output


class SpatialContextManager:
    """
    Manages spatial data, including map segmentation, coordinate conversion, and scene graphs.
    """

    def __init__(self, config: SystemConfig) -> None:
        """
        Initializes the spatial context manager.

        :param config: System configuration.
        :type config: SystemConfig
        """
        self.config = config
        self.merged_regions: Dict = {}
        self.region_mask: Optional[np.ndarray] = None
        self.merged_colors: Dict = {}
        self.width: int = 0
        self.height: int = 0
        self.unique_names: Dict[int, str] = {}
        self.map_origin: Tuple[float, float] = (0.0, 0.0)
        self.map_resolution: float = 0.05
        self.scene_manager: Optional[SceneGraphManager] = None

        self._load_or_compute_map_data()

    def _load_or_compute_map_data(self) -> None:
        """
        Loads cached map data or recomputes it using Watershed segmentation.

        :raises MapLoadError: If loading or computation fails.
        """
        os.makedirs(self.config.local_data_dir, exist_ok=True)
        filename = (
            f"Home{self.config.house_id:02d}_{self.config.prefix}_watershed.pkl.gz"
        )
        local_path = os.path.join(self.config.local_data_dir, filename)

        loaded = False
        if os.path.exists(local_path):
            try:
                with gzip.open(local_path, "rb") as f:
                    data = pickle.load(f)
                self.merged_regions = data["merged_regions"]
                self.region_mask = data["region_mask"]
                self.merged_colors = data["merged_colors"]
                self.width = data["width"]
                self.height = data["height"]
                loaded = True
            except (OSError, pickle.UnpicklingError):
                pass

        if not loaded:
            try:
                context = load_house_context(
                    house_id=self.config.house_id,
                    base_path=self.config.dataset_base_path,
                    prefixes=[self.config.prefix],
                    map_binary_threshold=self.config.map_binary_threshold,
                    min_contour_area=self.config.min_contour_area,
                    crop_padding=self.config.crop_padding,
                )

                params = load_profile(
                    self.config.house_id, {}, base_path=self.config.dataset_base_path
                )
                results = process_house_segmentation(
                    house_id=self.config.house_id,
                    house_context=context,
                    params=params,
                    prefixes=[self.config.prefix],
                    llm_agent=None,
                    generate_descriptions=False,
                    prompt_mask="",
                    class_colors=CLASS_COLORS,
                    trajectory_dimming=self.config.trajectory_image_dimming,
                )

                res_data = results[self.config.prefix]
                self.merged_regions = res_data["merged_regions"]
                self.region_mask = res_data["region_mask"]
                self.merged_colors = res_data["merged_colors"]
                self.width = res_data["width"]
                self.height = res_data["height"]

                data_to_save = {
                    "merged_regions": self.merged_regions,
                    "region_mask": self.region_mask,
                    "merged_colors": self.merged_colors,
                    "width": self.width,
                    "height": self.height,
                }
                try:
                    with gzip.open(local_path, "wb") as f:
                        pickle.dump(data_to_save, f)
                except (OSError, pickle.PicklingError):
                    pass
            except Exception as e:
                raise MapLoadError(f"Failed to load or compute map data: {e}") from e

        self._generate_unique_room_names()
        self._load_map_metadata()

    def _generate_unique_room_names(self) -> None:
        """
        Generates unique room names for regions.

        """
        self.unique_names = {}
        class_counters = {}
        sorted_ids = sorted(list(self.merged_regions.keys()))

        for rid in sorted_ids:
            data = self.merged_regions[rid]
            base_class = data.get("dominant_class", "unknown")

            if base_class not in class_counters:
                class_counters[base_class] = 1
            else:
                class_counters[base_class] += 1

            unique_name = f"{base_class} {class_counters[base_class]}"
            self.unique_names[rid] = unique_name

    def _load_map_metadata(self) -> None:
        """
        Loads map origin and resolution from context.

        :raises MapLoadError: If context loading fails.
        """
        try:
            temp_context = load_house_context(
                house_id=self.config.house_id,
                base_path=self.config.dataset_base_path,
                prefixes=[self.config.prefix],
                map_binary_threshold=self.config.map_binary_threshold,
                min_contour_area=self.config.min_contour_area,
                crop_padding=self.config.crop_padding,
            )
            self.map_origin = temp_context[self.config.prefix]["origin"]
            self.map_resolution = temp_context[self.config.prefix]["resolution"]
        except Exception as e:
            raise MapLoadError(f"Failed to load map metadata: {e}") from e

    def world_to_map_coordinates(
        self, world_coords: Tuple[float, float, float]
    ) -> Tuple[int, int]:
        """
        Converts world coordinates to map pixel coordinates.

        :param world_coords: World coordinates (x, y, z).
        :type world_coords: Tuple[float, float, float]
        :return: Pixel coordinates (x, y).
        :rtype: Tuple[int, int]
        """
        world_x_map = world_coords[2]
        world_y_map = -world_coords[0]
        pixel_x = int((world_x_map - self.map_origin[0]) / self.map_resolution)
        pixel_y = int(
            self.height - ((world_y_map - self.map_origin[1]) / self.map_resolution)
        )
        return pixel_x, pixel_y

    def map_to_world_coordinates(
        self, pixel_coords: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Converts map pixel coordinates to world coordinates.

        :param pixel_coords: Pixel coordinates (x, y).
        :type pixel_coords: Tuple[int, int]
        :return: World coordinates (x, y, z).
        :rtype: Tuple[float, float, float]
        """
        px, py = pixel_coords
        world_y_map = ((self.height - py) * self.map_resolution) + self.map_origin[1]
        world_x_map = (px * self.map_resolution) + self.map_origin[0]

        raw_x = -world_y_map
        raw_y = 0.0
        raw_z = world_x_map
        return (raw_x, raw_y, raw_z)

    def get_room_name_at_location(self, user_pos: Tuple[float, float, float]) -> str:
        """
        Gets the name of the room at a specific world location.

        :param user_pos: World coordinates.
        :type user_pos: Tuple[float, float, float]
        :return: Room name.
        :rtype: str
        """
        px, py = self.world_to_map_coordinates(user_pos)
        if 0 <= px < self.width and 0 <= py < self.height:
            region_id = self.region_mask[py, px]
            if region_id < 0:
                return "Hallway/Unknown Area"
            return self.unique_names.get(int(region_id), "Unknown Room")
        return "Outside Map"

    def reconstruct_debug_image(self) -> np.ndarray:
        """
        Reconstructs the map image with regions colored.

        :return: RGB image array.
        :rtype: np.ndarray
        """
        reconstructed_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for mid, color in self.merged_colors.items():
            reconstructed_image[self.region_mask == int(mid)] = color

        for mid, data in self.merged_regions.items():
            pts = np.where(self.region_mask == int(mid))
            if len(pts[0]) > 0:
                cy, cx = int(np.mean(pts[0])), int(np.mean(pts[1]))
                cls_name = self.unique_names.get(
                    mid, data.get("dominant_class", "unknown")
                )
                num_char = len(cls_name)

                text_pos = (int(cx + (2 * num_char) - 30), cy)
                cv2.putText(
                    reconstructed_image,
                    cls_name,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    reconstructed_image,
                    cls_name,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return reconstructed_image

    def inject_objects_and_build_graph(
        self, raw_objects: List[dict]
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Injects objects into the map and initializes the Scene Graph Manager.

        :param raw_objects: List of raw object dictionaries.
        :type raw_objects: List[dict]
        :return: Tuple of (Debug Image, Enriched Objects).
        :rtype: Tuple[np.ndarray, List[dict]]
        """
        img_viz = self.reconstruct_debug_image()
        enriched_objects = []

        region_contours = {}
        region_centers = {}

        for rid in self.merged_regions.keys():
            mask_rid = (self.region_mask == int(rid)).astype(np.uint8)
            cnts, _ = cv2.findContours(
                mask_rid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if cnts:
                region_contours[rid] = cnts
                pts = np.where(self.region_mask == int(rid))
                if len(pts[0]) > 0:
                    region_centers[rid] = (int(np.mean(pts[1])), int(np.mean(pts[0])))

        for obj in raw_objects:
            obj_dict = obj if isinstance(obj, dict) else obj.__dict__
            class_name = obj_dict.get("class_name")

            if not class_name:
                enriched_objects.append(obj_dict)
                continue

            centroid = None
            if "pcd_np" in obj_dict and len(obj_dict["pcd_np"]) > 0:
                centroid = np.mean(obj_dict["pcd_np"], axis=0)
            elif "bbox_np" in obj_dict:
                centroid = np.mean(obj_dict["bbox_np"], axis=0)

            px, py = (-1, -1)
            if centroid is not None:
                px, py = self.world_to_map_coordinates(tuple(centroid))

            final_rid = -1
            if 0 <= px < self.width and 0 <= py < self.height:
                rid_at_pixel = self.region_mask[py, px]
                if rid_at_pixel in self.merged_regions:
                    final_rid = rid_at_pixel

            if final_rid == -1 and centroid is not None:
                min_dist = float("inf")
                point_float = (float(px), float(py))
                for rid, cnts in region_contours.items():
                    for cnt in cnts:
                        dist = abs(cv2.pointPolygonTest(cnt, point_float, True))
                        if dist < min_dist:
                            min_dist = dist
                            final_rid = rid

            room_name = self.unique_names.get(final_rid, "Unmapped Space")
            obj_dict["room_name"] = room_name
            obj_dict["region_center"] = region_centers.get(final_rid, (0, 0))
            obj_dict["region_id"] = int(final_rid)

            if 0 <= px < self.width and 0 <= py < self.height:
                cv2.circle(img_viz, (px, py), 3, (0, 0, 0), -1)
                cv2.circle(img_viz, (px, py), 2, (0, 255, 255), -1)
                text_pos = (px + 5, py - 5)
                cv2.putText(
                    img_viz,
                    class_name,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.2,
                    (255, 255, 255),
                    1,
                )

            enriched_objects.append(obj_dict)

        self.scene_manager = SceneGraphManager(enriched_objects)
        return img_viz, enriched_objects

import torch.nn.functional as F
import open3d as o3d
import numpy as np
import matplotlib
import open_clip
import argparse
import pickle
import torch
import copy
import json
import gzip
import os

from conceptgraph.utils.pointclouds import Pointclouds
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh


def create_ball_mesh(center: tuple, radius: float, color: tuple = (0, 1, 0)) -> o3d.geometry.TriangleMesh:
    """
    Creates a colored sphere mesh at a given position.

    :param center: (x, y, z) coordinates for the sphere center.
    :type center: tuple
    :param radius: Sphere radius.
    :type radius: float
    :param color: RGB color in [0, 1].
    :type color: tuple
    :return: The colored sphere mesh.
    :rtype: o3d.geometry.TriangleMesh
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argparse.ArgumentParser for command-line options.

    :return: Configured parser for this script.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--rgb_pcd_path", type=str, default=None)
    parser.add_argument("--edge_file", type=str, default=None)
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="If set, the CLIP model will not init for fast debugging.",
    )
    parser.add_argument("--merge_overlap_thresh", type=float, default=-1)
    parser.add_argument("--merge_visual_sim_thresh", type=float, default=-1)
    parser.add_argument("--merge_text_sim_thresh", type=float, default=-1)
    parser.add_argument("--obj_min_points", type=int, default=0)
    parser.add_argument("--obj_min_detections", type=int, default=0)
    return parser


def load_result(result_path: str) -> tuple:
    """
    Loads the result file and returns objects, background objects, and class colors.

    :param result_path: Path to the gzipped pickle result file.
    :type result_path: str
    :raises ValueError: If the loaded results are not a dictionary.
    :return: (objects, bg_objects, class_colors)
        - objects (MapObjectList): List of all objects.
        - bg_objects (MapObjectList | None): List of background objects or None if not present.
        - class_colors (dict): Mapping from class id to color.
    :rtype: tuple
    """
    potential_path = os.path.realpath(result_path)
    if potential_path != result_path:
        print(f"Resolved symlink for result_path: {result_path} -> \n{potential_path}")
        result_path = potential_path

    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if not isinstance(results, dict):
        raise ValueError("Results should be a dictionary! other types are not supported!")

    objects = MapObjectList()
    objects.load_serializable(results["objects"])
    bg_objects = MapObjectList()
    bg_objects.extend(obj for obj in objects if obj["is_background"])
    if len(bg_objects) == 0:
        bg_objects = None
    class_colors = results["class_colors"]

    return objects, bg_objects, class_colors


def main(args) -> None:
    """
    Main visualization routine. Loads data, sets up the Open3D visualizer, and registers
    keyboard callbacks for interactive exploration of the SLAM results.

    :param args: Parsed command-line arguments.
    :type args: argparse.Namespace
    :raises AssertionError: If neither result_path nor rgb_pcd_path is provided.
    :return: None
    :rtype: None
    """
    result_path = args.result_path
    rgb_pcd_path = args.rgb_pcd_path

    assert not (
        result_path is None and rgb_pcd_path is None
    ), "Either result_path or rgb_pcd_path must be provided."

    if rgb_pcd_path is not None:
        pointclouds = Pointclouds.load_pointcloud_from_h5(rgb_pcd_path)
        global_pcd = pointclouds.open3d(0, include_colors=True)
        if result_path is None:
            print("Only visualizing the pointcloud...")
            o3d.visualization.draw_geometries([global_pcd])
            exit()

    objects, bg_objects, class_colors = load_result(result_path)

    if args.edge_file is not None:
        scene_graph_geometries = []
        with open(args.edge_file, "r") as f:
            edges = json.load(f)

        classes = objects.get_most_common_class()
        colors = [class_colors[str(c)] for c in classes]
        obj_centers = []

        for obj, c in zip(objects, colors):
            pcd = obj["pcd"]
            bbox = obj["bbox"]
            points = np.asarray(pcd.points)
            center = np.mean(points, axis=0)
            extent = bbox.get_max_bound()
            extent = np.linalg.norm(extent)
            radius = 0.10
            obj_centers.append(center)
            ball = create_ball_mesh(center, radius, c)
            scene_graph_geometries.append(ball)

        for edge in edges:
            if edge["object_relation"] == "none of these":
                continue
            id1 = edge["object1"]["id"]
            id2 = edge["object2"]["id"]
            line_mesh = LineMesh(
                points=np.array([obj_centers[id1], obj_centers[id2]]),
                lines=np.array([[0, 1]]),
                colors=[1, 0, 0],
                radius=0.02,
            )
            scene_graph_geometries.extend(line_mesh.cylinder_segments)

    if not args.no_clip:
        print("Initializing CLIP model...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to("cuda")
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        print("Done initializing CLIP model.")

    cmap = matplotlib.colormaps.get_cmap("turbo")

    if bg_objects is not None:
        indices_bg = []
        for obj_idx, obj in enumerate(objects):
            if obj["is_background"]:
                indices_bg.append(obj_idx)

    for i in range(len(objects)):
        pcd = objects[i]["pcd"]
        # pcd = pcd.voxel_down_sample(0.05)
        objects[i]["pcd"] = pcd

    pcds = copy.deepcopy(objects.get_values("pcd"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))

    object_classes = []
    for i in range(len(objects)):
        obj = objects[i]
        pcd = pcds[i]
        obj_classes = np.asarray(obj["class_id"])
        values, counts = np.unique(obj_classes, return_counts=True)
        obj_class = values[np.argmax(counts)]
        object_classes.append(obj_class)

    vis = o3d.visualization.VisualizerWithKeyCallback()

    if result_path is not None:
        vis.create_window(
            window_name=f"Open3D - {os.path.basename(result_path)}",
            width=1280,
            height=720,
        )
    else:
        vis.create_window(window_name="Open3D", width=1280, height=720)

    for geometry in pcds + bboxes:
        vis.add_geometry(geometry)

    main.show_bg_pcd = True

    def toggle_bg_pcd(vis) -> None:
        """
        Toggle visibility of background objects in the visualizer.

        :param vis: The Open3D visualizer.
        :type vis: o3d.visualization.VisualizerWithKeyCallback
        :return: None
        :rtype: None
        """
        if bg_objects is None:
            print("No background objects found.")
            return

        for idx in indices_bg:
            if main.show_bg_pcd:
                vis.remove_geometry(pcds[idx], reset_bounding_box=False)
                vis.remove_geometry(bboxes[idx], reset_bounding_box=False)
            else:
                vis.add_geometry(pcds[idx], reset_bounding_box=False)
                vis.add_geometry(bboxes[idx], reset_bounding_box=False)

        main.show_bg_pcd = not main.show_bg_pcd

    main.show_global_pcd = False

    def toggle_global_pcd(vis) -> None:
        """
        Toggle visibility of the global (full) point cloud in the visualizer.

        :param vis: The Open3D visualizer.
        :type vis: o3d.visualization.VisualizerWithKeyCallback
        :return: None
        :rtype: None
        """
        if args.rgb_pcd_path is None:
            print("No RGB pcd path provided.")
            return

        if main.show_global_pcd:
            vis.remove_geometry(global_pcd, reset_bounding_box=False)
        else:
            vis.add_geometry(global_pcd, reset_bounding_box=False)

        main.show_global_pcd = not main.show_global_pcd

    main.show_scene_graph = False

    def toggle_scene_graph(vis) -> None:
        """
        Toggle visibility of scene graph (nodes and edges) in the visualizer.

        :param vis: The Open3D visualizer.
        :type vis: o3d.visualization.VisualizerWithKeyCallback
        :return: None
        :rtype: None
        """
        if args.edge_file is None:
            print("No edge file provided.")
            return

        if main.show_scene_graph:
            for geometry in scene_graph_geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in scene_graph_geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)

        main.show_scene_graph = not main.show_scene_graph

    def color_by_class(vis) -> None:
        """
        Color each object by its most common class.

        :param vis: The Open3D visualizer.
        :type vis: o3d.visualization.VisualizerWithKeyCallback
        :return: None
        :rtype: None
        """
        for i in range(len(objects)):
            pcd = pcds[i]
            obj_class = object_classes[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(class_colors[str(obj_class)], (len(pcd.points), 1))
            )
        for pcd in pcds:
            vis.update_geometry(pcd)

    def color_by_rgb(vis) -> None:
        """
        Restore original RGB colors for each object.

        :param vis: The Open3D visualizer.
        :type vis: o3d.visualization.VisualizerWithKeyCallback
        :return: None
        :rtype: None
        """
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = objects[i]["pcd"].colors
        for pcd in pcds:
            vis.update_geometry(pcd)

    def color_by_instance(vis) -> None:
        """
        Assign a unique color to each object instance.

        :param vis: The Open3D visualizer.
        :type vis: o3d.visualization.VisualizerWithKeyCallback
        :return: None
        :rtype: None
        """
        instance_colors = cmap(np.linspace(0, 1, len(pcds)))
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(instance_colors[i, :3], (len(pcd.points), 1))
            )
        for pcd in pcds:
            vis.update_geometry(pcd)

    def color_by_clip_sim(vis) -> None:
        """
        Color objects by similarity to a text query using CLIP.
        Prompts the user for a text query and colors objects according to their
        CLIP similarity to the query.

        :param vis: The Open3D visualizer.
        :type vis: o3d.visualization.VisualizerWithKeyCallback
        :return: None
        :rtype: None
        """
        if args.no_clip:
            print("CLIP model is not initialized.")
            return

        text_query = input("Enter your query: ")
        text_queries = [text_query]

        text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
        text_query_ft = clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()

        objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        similarities = F.cosine_similarity(
            text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
        )
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[
            ..., :3
        ]

        max_prob_object = objects[max_prob_idx]
        print(
            f"Most probable object is at index {max_prob_idx} with class name '{max_prob_object['class_name']}'"
        )
        print(f"location xyz: {max_prob_object['bbox'].center}")

        for i in range(len(objects)):
            pcd = pcds[i]
            map_colors = np.asarray(pcd.colors)
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    [
                        similarity_colors[i, 0].item(),
                        similarity_colors[i, 1].item(),
                        similarity_colors[i, 2].item(),
                    ],
                    (len(pcd.points), 1),
                )
            )
        for pcd in pcds:
            vis.update_geometry(pcd)

    def save_view_params(vis) -> None:
        """
        Save the current camera view parameters to a file (temp.json).

        :param vis: The Open3D visualizer.
        :type vis: o3d.visualization.VisualizerWithKeyCallback
        :return: None
        :rtype: None
        """
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("temp.json", param)

    vis.register_key_callback(ord("B"), toggle_bg_pcd)
    vis.register_key_callback(ord("S"), toggle_global_pcd)
    vis.register_key_callback(ord("C"), color_by_class)
    vis.register_key_callback(ord("R"), color_by_rgb)
    vis.register_key_callback(ord("F"), color_by_clip_sim)
    vis.register_key_callback(ord("I"), color_by_instance)
    vis.register_key_callback(ord("V"), save_view_params)
    vis.register_key_callback(ord("G"), toggle_scene_graph)

    vis.run()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)

# =========================
# Imports Section
# =========================

from open3d.io import read_pinhole_camera_parameters
from omegaconf import DictConfig
from collections import Counter
from dotenv import load_dotenv
import supervision as sv
from pathlib import Path
from tqdm import trange
from PIL import Image
import numpy as np
import traceback
import pickle
import shutil
import pprint
import torch
import hydra
import uuid
import gzip
import json
import glob
import cv2
import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from conceptgraph.utils.vlm import get_vlm_openai_like_client, consolidate_captions
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.inference.manager import AdaptiveInferenceManager
from conceptgraph.utils.logging_metrics import MappingTracker
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.slam.utils import to_serializable

from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun,
    orr_log_annotated_image,
    orr_log_camera,
    orr_log_depth_image,
    orr_log_edges,
    orr_log_objs_pcd_and_bbox,
    orr_log_rgb_image,
    orr_log_vlm_image,
)
from conceptgraph.utils.general_utils import (
    ObjectClasses,
    get_det_out_path,
    get_exp_out_path,
    get_vlm_annotated_image_path,
    load_saved_detections,
    measure_time,
    save_detection_results,
    save_room_data,
    save_edge_json,
    save_hydra_config,
    save_obj_json,
    save_objects_for_frame,
    save_pointcloud,
    should_exit_early,
    vis_render_image,
)
from conceptgraph.utils.vis import (
    OnlineObjectRenderer,
    vis_result_fast_on_depth,
    vis_result_fast,
    save_video_detections,
)
from conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    merge_objects,
    detections_to_obj_pcd_and_bbox,
    process_cfg,
    process_edges,
    processing_needed,
    resize_gobs,
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches,
)
from conceptgraph.utils.general_utils import (
    get_vis_out_path,
    cfg_to_dict,
    check_run_detections,
    matrix_to_translation_rotation,
    make_vlm_edges_and_captions,
)

# Disable gradient computation for efficiency (inference only)
torch.set_grad_enabled(False)

# =========================
# Main Function
# =========================


# TODO: Coletar mais semântica do ambiente a partir de regras pré-definidas
# Definir um motor de regras para coletar mais semântica do ambiente
# Classificar e segmentar os ambientes com o mapa geométrico:
# Inserir o mapa na LLM como uma estrutura de diretórios
# Inserir a posição atual do usuário no ambiente de forma não numérica
# Buscar datasets para validar as melhorias realizadas no ConceptGraph
# Coletar junto a Assoc. Port. de Paralisia Cerebral questões comuns
# Extrair mais semântica do ambiente a partir de regras pré-definidas
# TODO: Extrair o objeto mais relevante a partir da abordagem padrão do ConceptGraph
# TODO: Enviar e-mail para o percurso acadêmico a perguntar sobre os prazos


# @hydra.main(
#     version_base=None,
#     config_path="../hydra_configs/",
#     config_name="rerun_realtime_mapping",
# )
def run_mapping_process(
    cfg: DictConfig, selected_house: int | None = None, preffix: str | None = None
) -> None:
    load_dotenv()

    if selected_house is not None:
        cfg.selected_house = selected_house
    if preffix is not None:
        cfg.preffix = preffix
        cfg.detections_exp_suffix = f"{preffix}_house_{cfg.selected_house}_det"
        cfg.exp_suffix = f"{preffix}_house_{cfg.selected_house}_map"

    cfg.scene_id = f"Home{cfg.selected_house:02d}/Wandering"
    new_inference_system = cfg.preffix != "original"

    if not new_inference_system:
        from ultralytics import YOLO, SAM
        import open_clip

        cfg.sim_threshold = 1.2
        cfg.merge_overlap_thresh = 0.7
        cfg.merge_visual_sim_thresh = 0.7
        cfg.merge_text_sim_thresh = 0.7
        cfg.denoise_interval = 5
        cfg.filter_interval = 5
        cfg.merge_interval = 5

    # Initialize a tracker for mapping statistics
    tracker = MappingTracker()

    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)
    det_exp_path = get_exp_out_path(
        cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix, make_dir=False
    )
    rerun_file_path = exp_out_path / f"rerun_{cfg.exp_suffix}.rrd"

    # Initialize OptionalReRun for optional logging/visualization
    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)
    orr.init("realtime_mapping")
    # orr.save(rerun_file_path)

    # Initialize OptionalWandB for optional experiment tracking
    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(
        project="concept-graphs",
        config=cfg_to_dict(cfg),
    )

    # Process configuration (may add/modify config fields)
    cfg = process_cfg(cfg)

    # Load the dataset according to configuration
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )

    # Initialize object and edge containers for the map
    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    # If visualization rendering is enabled, set up the renderer
    if cfg.vis_render:
        view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
        obj_renderer = OnlineObjectRenderer(
            view_param=view_param,
            base_objects=None,
            gray_map=False,
        )
        frames = []

    # Prepare object classes and detection configuration
    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg["classes_file"],
        bg_classes=detections_exp_cfg["bg_classes"],
        skip_bg=detections_exp_cfg["skip_bg"],
    )

    room_data_list = []

    # Decide whether to run detections or load from disk
    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)

    prev_adjusted_pose = None

    # =========================
    # Detection Model Initialization
    # =========================
    manager = AdaptiveInferenceManager(
        output_dir=det_exp_path,
        save_frame_outputs=True,
        resource_log_interval=0.05,
        configuration=cfg.preffix,
    )
    if not new_inference_system:
        detection_model = measure_time(YOLO)("yolov8l-world.pt")
        sam_predictor = SAM("sam_l.pt")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        detection_model.set_classes(obj_classes.get_classes_arr())
    if run_detections:
        print("Running detections...")
        det_exp_path.mkdir(parents=True, exist_ok=True)
    else:
        print("NOT Running detections...")

    # # Initialize OpenAI client for VLM (Vision-Language Model) captions/edges
    if not new_inference_system:
        openai_client = get_vlm_openai_like_client(
            model="openai/gpt-4o-mini",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_API_BASE_URL"),
        )

    # Save configuration files for reproducibility
    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    # Prepare output directory for saving objects for all frames, if enabled
    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = (
            exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        )
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    # =========================
    # Main Processing Loop Over Frames
    # =========================

    exit_early_flag = False  # Used to break out of the loop early if needed
    counter = 0  # Frame counter
    stride = cfg.stride

    try:
        stride = int(stride)
    except ValueError:
        stride = 1

    manager.start_resource_logging()
    for frame_idx in trange(len(dataset)):
        tracker.curr_frame_idx = frame_idx
        counter += 1
        orr.set_time_sequence("frame", frame_idx)

        # Check for early exit signal (e.g., for debugging or interruption)
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True

        # If early exit is set and not at the last frame, skip this frame
        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        # =========================
        # Load Frame Data
        # =========================

        # Load color image path and original PIL image
        color_path = Path(dataset.color_paths[frame_idx])
        image_original_pil = Image.open(color_path)
        # Load color/depth tensors and camera intrinsics
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        if new_inference_system:
            manager.prepare_results(image_path=color_path, frame_idx=frame_idx * stride)

        # Convert depth and color tensors to numpy arrays for processing
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy()  # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8)  # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Prepare variables for detections and grounded observations
        raw_gobs = None
        gobs = None  # "gobs" = grounded observations

        # Prepare paths for VLM-annotated images
        vis_save_path_for_vlm = get_vlm_annotated_image_path(
            det_exp_vis_path, color_path
        )
        vis_save_path_for_vlm_edges = get_vlm_annotated_image_path(
            det_exp_vis_path, color_path, w_edges=True
        )

        # =========================
        # Pose and Camera Logging
        # =========================

        # Get the (untransformed) camera pose for this frame
        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()

        # No transformation applied to pose in this code
        adjusted_pose = unt_pose

        # Log camera pose to rerun (if enabled)
        prev_adjusted_pose = orr_log_camera(
            intrinsics,
            adjusted_pose,
            prev_adjusted_pose,
            cfg.image_width,
            cfg.image_height,
            frame_idx,
        )

        # =========================
        # Detection and Segmentation
        # =========================
        if run_detections:
            results = None
            # OpenCV cannot read Path objects, so convert to str
            image = cv2.imread(str(color_path))  # BGR color space
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if new_inference_system:
                try:
                    results = manager.detect_objects(color_path, image_rgb)
                except RuntimeError:
                    print(
                        f"Detection failed for frame {frame_idx * stride}, skipping frame."
                    )
                    continue

                if len(results) == 0:
                    print(f"No detections found, skipping frame: {frame_idx * stride}")
                    continue
                try:
                    masks = manager.segment_objects(color_path, image_rgb, results)
                    curr_det = manager.associate_masks_to_detections(results, masks)
                except (AssertionError, RuntimeError):
                    print(
                        f"Segmentation failed for frame {frame_idx * stride}, skipping frame."
                    )
                    continue

                curr_det = curr_det[curr_det.confidence > 0.4]
                detection_class_labels = manager.get_detection_classes(curr_det)
                try:
                    labels, edges, edge_image, captions, room_data = (
                        manager.perform_vlm_inference(color_path, image_rgb, curr_det)
                    )
                except ValueError:
                    print(
                        f"VLM inference failed for frame {frame_idx * stride}, skipping frame."
                    )
                    continue
                image_crops, image_feats, text_feats = manager.extract_features(
                    image_rgb, curr_det
                )
            else:
                # Run YOLO object detection
                results = detection_model.predict(color_path, conf=0.5, verbose=False)
                confidences = results[0].boxes.conf.cpu().numpy()
                detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                detection_class_labels = [
                    f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
                    for class_idx, class_id in enumerate(detection_class_ids)
                ]
                xyxy_tensor = results[0].boxes.xyxy
                xyxy_np = xyxy_tensor.cpu().numpy()

                # If there are detections, run SAM for segmentation masks
                if xyxy_tensor.numel() != 0:
                    sam_out = sam_predictor.predict(
                        color_path, bboxes=xyxy_tensor, verbose=False
                    )
                    masks_tensor = sam_out[0].masks.data
                    masks_np = masks_tensor.cpu().numpy()
                else:
                    # No detections: create empty mask array
                    masks_np = np.empty((0, *color_tensor.shape[:2]), dtype=np.float64)

                # Create a Detections object for this frame
                curr_det = sv.Detections(
                    xyxy=xyxy_np,
                    confidence=confidences,
                    class_id=detection_class_ids,
                    mask=masks_np,
                )

                # Generate VLM-based edges and captions for detected objects
                labels, edges, edge_image, captions, room_data = (
                    make_vlm_edges_and_captions(
                        image,
                        curr_det,
                        obj_classes,
                        detection_class_labels,
                        det_exp_vis_path,
                        color_path,
                        cfg.make_edges,
                        room_data_list,
                        openai_client,
                    )
                )

                # Compute CLIP features for detected objects (image and text)
                image_crops, image_feats, text_feats = compute_clip_features_batched(
                    image_rgb,
                    curr_det,
                    clip_model,
                    clip_preprocess,
                    clip_tokenizer,
                    obj_classes.get_classes_arr(),
                    cfg.device,
                )

            converted_pose = matrix_to_translation_rotation(adjusted_pose)

            try:
                room_data["pose"] = str(converted_pose)
            except (KeyError, TypeError):
                room_data = {
                    "room_class": "None",
                    "room_description": "None",
                    "pose": str(converted_pose),
                }

            room_data_list.append(room_data)

            # Update tracker with number of detections in this frame
            tracker.increment_total_detections(len(curr_det.xyxy))

            # Save detection results as a dictionary (numpy arrays)
            results = {
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "detection_class_labels": detection_class_labels,
                "labels": labels,
                "edges": edges,
                "captions": captions,
                "room_data": room_data,
            }

            raw_gobs = results

            # Optionally save detection results and visualizations
            if cfg.save_detections:
                vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                # Visualize and save annotated RGB image
                annotated_image, labels = vis_result_fast(
                    image, curr_det, obj_classes.get_classes_arr()
                )
                cv2.imwrite(str(vis_save_path), annotated_image)

                # Visualize and save annotated depth image
                if len(depth_array.shape) == 3 and depth_array.shape[2] == 4:
                    depth_array = depth_array[:, :, 3]
                depth_image_rgb = cv2.normalize(
                    depth_array, None, 0, 255, cv2.NORM_MINMAX
                )
                depth_image_rgb = depth_image_rgb.astype(np.uint8)
                depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                annotated_depth_image, labels = vis_result_fast_on_depth(
                    depth_image_rgb, curr_det, obj_classes.get_classes_arr()
                )
                cv2.imwrite(
                    str(vis_save_path).replace(".jpg", "_depth.jpg"),
                    annotated_depth_image,
                )
                cv2.imwrite(
                    str(vis_save_path).replace(".jpg", "_depth_only.jpg"),
                    depth_image_rgb,
                )
                # Save detection results to disk
                save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)
                # Save room data to disk
                save_room_data(det_exp_pkl_path / vis_save_path.stem, room_data)
        else:
            # =========================
            # Load Detections from Disk (if not running detection)
            # =========================
            # Support both current and legacy file naming conventions
            numerical_part = re.search(r"\d+", color_path.stem)
            basename = color_path.stem
            if numerical_part:
                basename = numerical_part.group(0)
            if os.path.exists(det_exp_pkl_path / basename):
                raw_gobs = load_saved_detections(det_exp_pkl_path / basename)
            elif os.path.exists(det_exp_pkl_path / f"{int(basename):06}"):
                raw_gobs = load_saved_detections(
                    det_exp_pkl_path / f"{int(basename):06}"
                )
            else:
                # Raise error if no detections found for this frame
                raise FileNotFoundError(
                    f"No detections found for frame {frame_idx}at paths \n{det_exp_pkl_path / basename} or \n{det_exp_pkl_path / f'{int(basename):06}'}."
                )

        # Log images and visualizations to rerun (if enabled)
        orr_log_rgb_image(color_path)
        orr_log_annotated_image(color_path, det_exp_vis_path)
        orr_log_depth_image(depth_tensor)
        orr_log_vlm_image(vis_save_path_for_vlm)
        orr_log_vlm_image(vis_save_path_for_vlm_edges, label="w_edges")

        # =========================
        # Preprocessing and Filtering Detections
        # =========================

        # Resize grounded observations if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # Filter grounded observations (remove background, low-confidence, etc.)
        filtered_gobs = filter_gobs(
            resized_gobs,
            image_rgb,
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

        gobs = filtered_gobs

        # If no detections after filtering, skip this frame
        if len(gobs["mask"]) == 0:
            continue

        # Refine the masks to remove regions corresponding to objects that are spatially contained within other objects.
        # This helps to better separate overlapping or nested objects (e.g., pillows on couches) by subtracting the mask of contained objects from their containers.

        gobs["mask"] = mask_subtract_contained(gobs["xyxy"], gobs["mask"])
        # =========================
        # Convert Detections to 3D Point Clouds and Bounding Boxes
        # =========================

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array,
            masks=gobs["mask"],
            cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=adjusted_pose,
            min_points_threshold=cfg.min_points_threshold,
            spatial_sim_type=cfg.spatial_sim_type,
            obj_pcd_max_points=cfg.obj_pcd_max_points,
            device=cfg.device,
        )

        # Refine each object's point cloud and compute its 3D bounding box.
        # This step includes optional downsampling and noise removal for the point cloud,
        # followed by bounding box estimation based on the processed point cloud.
        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg["spatial_sim_type"],
                    pcd=obj["pcd"],
                )

        # Create detection list for this frame (with 3D info)
        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )

        # If no valid detections, skip this frame
        if len(detection_list) == 0:
            continue

        # =========================
        # Object Map Update: Add, Match, and Merge
        # =========================

        # If this is the first frame (no objects yet), add all detections as new objects
        if len(objects) == 0:
            objects.extend(detection_list)
            tracker.increment_total_objects(len(detection_list))
            owandb.log(
                {
                    "total_objects_so_far": tracker.get_total_objects(),
                    "objects_this_frame": len(detection_list),
                }
            )
            continue

        # Compute spatial and visual similarities between detections and existing objects
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg["spatial_sim_type"],
            detection_list=detection_list,
            objects=objects,
            downsample_voxel_size=cfg["downsample_voxel_size"],
        )
        visual_sim = compute_visual_similarities(detection_list, objects)

        # Aggregate similarities using configured method and bias
        agg_sim = aggregate_similarities(
            match_method=cfg["match_method"],
            phys_bias=cfg["phys_bias"],
            spatial_sim=spatial_sim,
            visual_sim=visual_sim,
        )

        # Match detections to existing objects using similarity threshold
        match_indices = match_detections_to_objects(
            agg_sim=agg_sim,
            detection_threshold=cfg["sim_threshold"],
        )

        # Merge matched detections into existing objects, or add as new objects
        objects = merge_obj_matches(
            detection_list=detection_list,
            objects=objects,
            match_indices=match_indices,
            downsample_voxel_size=cfg["downsample_voxel_size"],
            dbscan_remove_noise=cfg["dbscan_remove_noise"],
            dbscan_eps=cfg["dbscan_eps"],
            dbscan_min_points=cfg["dbscan_min_points"],
            spatial_sim_type=cfg["spatial_sim_type"],
            device=cfg["device"],
        )

        # =========================
        # Post-processing: Class Name Correction
        # =========================

        if new_inference_system:
            for idx, obj in enumerate(objects):
                if isinstance(obj["captions"], str):
                    obj["captions"] = [obj["captions"]]
                info_combined = list(
                    zip(
                        obj["image_idx"],
                        obj["mask_idx"],
                        obj["color_path"],
                        obj["class_id"],
                        obj["xyxy"],
                        obj["captions"],
                        obj["mask"],
                        obj["conf"],
                        obj["contain_number"],
                    )
                )

                try:
                    info_combined = [
                        item
                        for item in info_combined
                        if item is not None
                        and item[5]["caption"] is not None
                        and item[5]["caption"].strip() != ""
                        and item[5]["caption"] != "null"
                    ]
                except KeyError:
                    print(f"No caption in item 5: {info_combined}")
                    info_combined = []

                if len(info_combined) == 0:
                    obj["image_idx"] = []
                    obj["mask_idx"] = []
                    obj["color_path"] = []
                    obj["class_id"] = []
                    obj["xyxy"] = []
                    obj["captions"] = []
                    obj["mask"] = []
                    obj["conf"] = []
                    obj["contain_number"] = []
                    obj["num_detections"] = 0
                    continue

                obj["image_idx"] = [item[0] for item in info_combined]
                obj["mask_idx"] = [item[1] for item in info_combined]
                obj["color_path"] = [item[2] for item in info_combined]
                obj["class_id"] = [item[3] for item in info_combined]
                obj["xyxy"] = [item[4] for item in info_combined]
                obj["captions"] = [item[5] for item in info_combined]
                obj["mask"] = [item[6] for item in info_combined]
                obj["conf"] = [item[7] for item in info_combined]
                obj["contain_number"] = [item[8] for item in info_combined]
                obj["num_detections"] = len(obj["class_id"])

        # For each object, set its class name to the most common detected class
        for obj in objects:
            temp_class_name = obj["class_name"]
            curr_obj_class_id_counter = Counter(obj["class_id"])
            try:
                most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
                most_common_class_name = obj_classes.get_classes_arr()[
                    most_common_class_id
                ]
            except IndexError:
                print(
                    f"Object has no class IDs, skipping class name update. Object info: {obj}"
                )
                continue
            if temp_class_name != most_common_class_name:
                obj["class_name"] = most_common_class_name

        # =========================
        # Edge Processing and Final Frame Handling
        # =========================

        # Update map edges based on new matches
        map_edges = process_edges(
            match_indices, gobs, len(objects), objects, map_edges, frame_idx
        )
        is_final_frame = frame_idx == len(dataset) - 1
        if is_final_frame:
            print("Final frame detected. Performing final post-processing...")

        # Remove outlier edges (edges with too few detections and old)
        edges_to_delete = []
        for curr_map_edge in map_edges.edges_by_index.values():
            curr_obj1_idx = curr_map_edge.obj1_idx
            curr_obj2_idx = curr_map_edge.obj2_idx
            curr_first_detected = curr_map_edge.first_detected
            curr_num_det = curr_map_edge.num_detections
            if (frame_idx - curr_first_detected > 5) and curr_num_det < 2:
                edges_to_delete.append((curr_obj1_idx, curr_obj2_idx))
        for edge in edges_to_delete:
            map_edges.delete_edge(edge[0], edge[1])

        # =========================
        # Periodic Post-processing: Denoising, Filtering, Merging
        # =========================

        # Denoise objects periodically or at final frame
        if processing_needed(
            cfg["denoise_interval"],
            cfg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                objects=objects,
            )

        # Filter objects periodically or at final frame
        if processing_needed(
            cfg["filter_interval"],
            cfg["run_filter_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = filter_objects(
                obj_min_points=cfg["obj_min_points"],
                obj_min_detections=cfg["obj_min_detections"],
                objects=objects,
                map_edges=map_edges,
            )

        # Merge objects periodically or at final frame
        if processing_needed(
            cfg["merge_interval"],
            cfg["run_merge_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects, map_edges = measure_time(merge_objects)(
                merge_overlap_thresh=cfg["merge_overlap_thresh"],
                merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                objects=objects,
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                do_edges=cfg["make_edges"],
                map_edges=map_edges,
            )

        # =========================
        # Logging and Saving Intermediate Results
        # =========================

        # Log objects and edges to rerun (if enabled)
        orr_log_objs_pcd_and_bbox(objects, obj_classes)
        orr_log_edges(objects, map_edges, obj_classes)

        # Save objects for this frame if enabled
        if cfg.save_objects_all_frames:
            save_objects_for_frame(
                obj_all_frames_out_path,
                frame_idx,
                objects,
                cfg.obj_min_detections,
                adjusted_pose,
                color_path,
            )

        # Optionally render visualization for this frame
        if cfg.vis_render:
            vis_render_image(
                objects,
                obj_classes,
                obj_renderer,
                image_original_pil,
                adjusted_pose,
                frames,
                frame_idx,
                color_path,
                cfg.obj_min_detections,
                cfg.class_agnostic,
                cfg.debug_render,
                is_final_frame,
                cfg.exp_out_path,
                cfg.exp_suffix,
            )

        # Periodically save the point cloud to disk
        if cfg.periodically_save_pcd and (
            counter % cfg.periodically_save_pcd_interval == 0
        ):
            save_pointcloud(
                exp_suffix=cfg.exp_suffix,
                exp_out_path=exp_out_path,
                cfg=cfg,
                objects=objects,
                obj_classes=obj_classes,
                latest_pcd_filepath=cfg.latest_pcd_filepath,
                create_symlink=True,
            )

        # Log frame-level statistics to wandb (if enabled)
        owandb.log(
            {
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
            }
        )

        # Update tracker and log object/detection statistics
        tracker.increment_total_objects(len(objects))
        tracker.increment_total_detections(len(detection_list))
        owandb.log(
            {
                "total_objects": tracker.get_total_objects(),
                "objects_this_frame": len(objects),
                "total_detections": tracker.get_total_detections(),
                "detections_this_frame": len(detection_list),
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
            }
        )
    # End of main frame loop

    # =========================
    # Final Post-processing and Saving
    # =========================

    # Consolidate captions for each object using VLM
    stride = cfg.stride
    try:
        stride = int(stride)
    except ValueError:
        stride = 1
    if new_inference_system:
        manager.set_frame_output_dir(output_dir_name="objects")
        objs_to_delete = []

    for idx, obj in enumerate(objects):
        if new_inference_system:
            if len(obj["captions"]) < 2:
                objs_to_delete.append(idx)
                continue
            manager.consolidate_captions(obj)
        else:
            obj_captions = obj["captions"][:20]
            consolidated_caption = consolidate_captions(openai_client, obj_captions)
            obj["consolidated_caption"] = consolidated_caption
            output_path = det_exp_path / "detections" / "objects"
            os.makedirs(str(output_path), exist_ok=True)
            obj_id = obj.get("id", None)
            if obj_id is None:
                obj["id"] = str(uuid.uuid4())
                obj_id = obj["id"]
            with open(output_path / f"{obj_id}.json", "w", encoding="utf-8") as f:
                obj_data = obj.copy()
                del obj_data["pcd"]
                del obj_data["bbox"]
                del obj_data["contain_number"]
                del obj_data["mask"]
                del obj_data["xyxy"]
                json.dump(
                    obj_data, f, ensure_ascii=False, indent=2, default=to_serializable
                )
        # consolidated_caption = consolidate_captions(openai_client, obj_captions)

    if new_inference_system:
        for idx in sorted(objs_to_delete, reverse=True):
            del objects[idx]

    manager.stop_resource_logging()
    # Save rerun logs if enabled
    orr.disconnect()
    # handle_rerun_saving(cfg.use_rerun, cfg.save_rerun, cfg.exp_suffix, exp_out_path)

    # Save the final point cloud to disk if enabled
    if cfg.save_pcd:
        save_pointcloud(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            cfg=cfg,
            objects=objects,
            obj_classes=obj_classes,
            room_data_list=room_data_list,
            latest_pcd_filepath=cfg.latest_pcd_filepath,
            create_symlink=True,
            edges=map_edges,
        )

    # Save objects and edges as JSON if enabled
    if cfg.save_json:
        try:
            save_obj_json(
                exp_suffix=cfg.exp_suffix, exp_out_path=exp_out_path, objects=objects
            )
        except IndexError:
            print("No edges to save to JSON, skipping edge JSON saving.")
        try:
            save_edge_json(
                exp_suffix=cfg.exp_suffix,
                exp_out_path=exp_out_path,
                objects=objects,
                edges=map_edges,
            )
        except IndexError:
            print("No edges to save to JSON, skipping edge JSON saving.")

    # Save metadata for all frames if enabled
    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / "meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump(
                {
                    "cfg": cfg,
                    "class_names": obj_classes.get_classes_arr(),
                    "class_colors": obj_classes.get_class_color_dict_by_index(),
                },
                f,
            )

    # Save video of detections if enabled and detections were run
    if run_detections:
        if cfg.save_video:
            save_video_detections(det_exp_path)

    # Finish wandb logging session
    if new_inference_system:
        manager.unload_all_models()
    owandb.finish()


def clean_and_check_progress() -> dict[str, list[int]]:
    """
    Checks the integrity of experiment outputs and cleans up interrupted runs.

    Iterates through defined house datasets, checking if the expected output
    files exist. If a directory exists but the specific .pkl.gz file is missing,
    it is considered an interrupted run and the directories are deleted to allow
    a fresh restart.

    :return: A dictionary where keys are the experiment modes (prefixes) and values
             are lists of house indices that need to be processed (either pending or cleaned).
    :rtype: Dict[str, List[int]]
    """
    houses = {
        "offline": list(range(1, 31)),
        "online": list(range(1, 31)),
        "original": list(range(1, 31)),
        "improved": list(range(1, 31)),
    }

    base_dataset_path = (
        r"C:\Users\lab\Documents\Datasets\Robot@VirtualHomeLarge\outputs"
    )

    pending_tasks: dict[str, list[int]] = {k: [] for k in houses}

    print("Starting verification and cleanup of interrupted executions...\n")

    for prefix, house_list in houses.items():
        print(f"--- Verifying mode: {prefix} ---")

        for house_id in house_list:
            house_folder_name = f"Home{house_id:02d}"
            exp_path = os.path.join(
                base_dataset_path, house_folder_name, "Wandering", "exps"
            )

            map_dir_name = f"{prefix}_house_{house_id}_map"
            det_dir_name = f"{prefix}_house_{house_id}_det"

            full_map_path = os.path.join(exp_path, map_dir_name)
            full_det_path = os.path.join(exp_path, det_dir_name)

            if os.path.exists(full_map_path):
                pkl_files = glob.glob(os.path.join(full_map_path, "*.pkl.gz"))

                if pkl_files:
                    print(f"[OK] House {house_id}: Complete execution found.")
                else:
                    print(
                        f"[FAIL] House {house_id}: Folder found without .pkl.gz (Interrupted)."
                    )
                    print(f"   -> Deleting folders for restart...")

                    try:
                        shutil.rmtree(full_map_path)
                        print(f"      Removed: {map_dir_name}")
                    except Exception as e:
                        print(f"      Error removing _map: {e}")

                    if os.path.exists(full_det_path):
                        try:
                            shutil.rmtree(full_det_path)
                            print(f"      Removed: {det_dir_name}")
                        except Exception as e:
                            print(f"      Error removing _det: {e}")

                    pending_tasks[prefix].append(house_id)

            else:
                print(f"[PENDING] House {house_id}: Not yet started.")
                pending_tasks[prefix].append(house_id)

        print("\n")

    return pending_tasks


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    houses = clean_and_check_progress()

    with hydra.initialize(version_base=None, config_path="../hydra_configs"):
        for preffix in houses:
            for selected_house in houses[preffix]:
                while True:
                    try:
                        print("#" * 50)
                        print(
                            f"Starting rerun realtime mapping for house {selected_house} with preffix {preffix}..."
                        )
                        print("#" * 50)
                        cfg = hydra.compose(
                            config_name="rerun_realtime_mapping",
                            overrides=[
                                f"selected_house={selected_house}",
                                f"preffix={preffix}",
                                f"save_detections={preffix=='original'}",
                            ],
                        )
                        run_mapping_process(
                            cfg, selected_house=selected_house, preffix=preffix
                        )
                        print("#" * 50)
                        print(
                            f"Finished rerun realtime mapping for house {selected_house} with preffix {preffix}."
                        )
                        print("#" * 50)
                        break
                    except Exception as e:
                        traceback.print_exc()
                        with open("failed.txt", "a") as f:
                            f.write(("#" * 25) + "  ERROR  " + ("#" * 25))
                            f.write(
                                f"\n\nThe processing of the house Home{selected_house:02d} failed for the mode {preffix}\n\n"
                                + ("-" * 50)
                                + "Error:\n"
                            )
                            f.write(str(e))
                            f.write("\n\n" + ("-" * 50) + "\n")
                        if preffix == "offline":
                            dataset_path = "C:\\Users\\lab\\Documents\\Datasets\\Robot@VirtualHomeLarge\\outputs\\Home{selected_house:02d}\\Wandering\\exps"
                            det_path = os.path.join(
                                dataset_path, f"{preffix}_house_{selected_house}_det"
                            )
                            map_path = os.path.join(
                                dataset_path, f"{preffix}_house_{selected_house}_map"
                            )
                            os.rmdir(det_path)
                            os.rmdir(map_path)
                            continue
                        else:
                            break

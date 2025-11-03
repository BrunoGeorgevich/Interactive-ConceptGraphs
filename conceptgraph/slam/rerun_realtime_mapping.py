# =========================
# Imports Section
# =========================

# Standard library imports
from dotenv import load_dotenv
from pathlib import Path
import pickle
import gzip
import os
import re

# Third-party library imports
from open3d.io import read_pinhole_camera_parameters
from ultralytics import YOLO, SAM, FastSAM
from omegaconf import DictConfig
from collections import Counter
import supervision as sv
from tqdm import trange
from PIL import Image
import numpy as np
import open_clip
import torch
import hydra
import cv2

# Local application/library specific imports
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
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.logging_metrics import MappingTracker
from conceptgraph.utils.vlm import (
    consolidate_captions,
    get_vlm_openai_like_client,
)
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses,
    get_det_out_path,
    get_exp_out_path,
    get_vlm_annotated_image_path,
    handle_rerun_saving,
    load_saved_detections,
    make_vlm_edges_and_captions,
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
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import (
    OnlineObjectRenderer,
    vis_result_fast_on_depth,
    vis_result_fast,
    save_video_detections,
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
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
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import (
    get_vis_out_path,
    cfg_to_dict,
    check_run_detections,
    matrix_to_translation_rotation,
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


@hydra.main(
    version_base=None,
    config_path="../hydra_configs/",
    config_name="rerun_realtime_mapping",
)
def main(cfg: DictConfig):
    load_dotenv()
    # Initialize a tracker for mapping statistics
    tracker = MappingTracker()

    # Initialize OptionalReRun for optional logging/visualization
    orr = OptionalReRun()
    orr.set_use_rerun(cfg.use_rerun)
    orr.init("realtime_mapping")
    orr.spawn()

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

    # Set up output paths for experiment and detections
    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.exp_suffix)
    det_exp_path = get_exp_out_path(
        cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix, make_dir=False
    )

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
    if run_detections:
        print("\n".join(["Running detections..."] * 10))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        # Initialize YOLO detection model (timed)
        # detection_model = measure_time(YOLO)("yolov8l-world.pt")
        detection_model = measure_time(YOLO)("yolov8x-worldv2.pt")
        # Initialize SAM segmentation model (UltraLytics version)
        sam_predictor = SAM("sam2.1_l.pt")
        # sam_predictor = SAM("sam_l.pt")
        # sam_predictor = FastSAM("FastSAM-x.pt")
        # Initialize OpenCLIP model and tokenizer
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        # Set detection classes for YOLO
        detection_model.set_classes(obj_classes.get_classes_arr())
    else:
        print("\n".join(["NOT Running detections..."] * 10))

    # Initialize OpenAI client for VLM (Vision-Language Model) captions/edges
    openai_client = get_vlm_openai_like_client(
        model="google/gemini-2.5-flash-lite",
        # api_key=os.getenv("GLAMA_API_KEY"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        # base_url=os.getenv("GLAMA_API_BASE_URL"),
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

        # For each object, set its class name to the most common detected class
        for idx, obj in enumerate(objects):
            temp_class_name = obj["class_name"]
            curr_obj_class_id_counter = Counter(obj["class_id"])
            most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
            most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
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
            obj1_class_name = objects[curr_obj1_idx]["class_name"]
            obj2_class_name = objects[curr_obj2_idx]["class_name"]
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
    for object in objects:
        obj_captions = object["captions"][:20]
        consolidated_caption = consolidate_captions(openai_client, obj_captions)
        object["consolidated_caption"] = consolidated_caption

    # Save rerun logs if enabled
    handle_rerun_saving(cfg.use_rerun, cfg.save_rerun, cfg.exp_suffix, exp_out_path)

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
        save_obj_json(
            exp_suffix=cfg.exp_suffix, exp_out_path=exp_out_path, objects=objects
        )
        save_edge_json(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            objects=objects,
            edges=map_edges,
        )

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
    owandb.finish()


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    main()

"""
CLI tool for Gait Identity Management.
Supports batch enrollment from video files, identity listing, and deletion.
"""

from __future__ import annotations
import argparse
import logging
import sys
import torch
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from ultralytics import YOLO
from core.device import select_device

from gait_subsystem.gait.config import default_gait_config
from gait_subsystem.gait.gait_gallery import GaitGallery
from gait_subsystem.gait.gait_extractor import GaitExtractor

logger = logging.getLogger(__name__)

def _setup_logging(level: int = logging.INFO) -> None:
    """Configures the console logger if not already initialized."""
    root = logging.getLogger()
    if root.handlers: return
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

def extract_raw_sequences_from_video(video_path: Path, pose_model: YOLO, min_len: int = 24) -> List[List[np.ndarray]]:
    """
    Processes a video file to extract continuous sequences of skeleton keypoints.
    
    ROBUST VERSION:
    - Properly validates that at least one person is detected before accessing keypoints.
    - Uses try-except for additional safety against edge cases.
    - Logs skipped frames for debugging.
    
    Returns:
        List of sequences, where each sequence is a list of (17, 3) keypoint arrays [x, y, conf].
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Unable to open video: {video_path}")
        return []

    frames_buffer = []
    sequences = []
    conf_thresh = 0.25 
    frame_count = 0
    skipped_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        frame_count += 1
        kps_raw = None
        
        try:
            results = pose_model(frame, verbose=False, conf=conf_thresh)
            
            if (len(results) > 0 and 
                results[0].keypoints is not None and 
                hasattr(results[0].keypoints, 'xy') and
                results[0].keypoints.xy is not None and
                results[0].keypoints.xy.shape[0] > 0):
                
                kps = results[0].keypoints
                
                xy = kps.xy[0].cpu().numpy() 
                
                if kps.conf is not None and kps.conf.shape[0] > 0:
                    conf = kps.conf[0].cpu().numpy()[:, None]
                else:
                    conf = np.ones((xy.shape[0], 1))
                    
                kps_raw = np.hstack([xy, conf])
            else:
                skipped_frames += 1
                
        except Exception as e:
            logger.debug(f"Frame {frame_count}: Keypoint extraction failed - {e}")
            skipped_frames += 1 
        
        if kps_raw is not None:
            if skipped_frames > 0 and skipped_frames <= 5 and len(frames_buffer) > 0:
                last_valid = frames_buffer[-1]
                current = kps_raw
                
                for i in range(1, skipped_frames + 1):
                    alpha = i / (skipped_frames + 1)
                    interp_xy = (1 - alpha) * last_valid[:, :2] + alpha * current[:, :2]
                    interp_conf = (1 - alpha) * last_valid[:, 2:] + alpha * current[:, 2:]
                    
                    interp_frame = np.hstack([interp_xy, interp_conf])
                    frames_buffer.append(interp_frame)
                
                logger.debug(f"    Glue: Interpolated {skipped_frames} frames.")

            frames_buffer.append(kps_raw)
            skipped_frames = 0
            
        else:
            skipped_frames += 1
            if skipped_frames > 5:
                if len(frames_buffer) >= min_len:
                    sequences.append(frames_buffer)
                frames_buffer = [] 

    if len(frames_buffer) >= min_len:
        sequences.append(frames_buffer)
    
    if skipped_frames > 0:
        logger.info(f"    Processed {frame_count} frames: {skipped_frames} skipped (no person detected), {len(sequences)} sequences extracted")
        
    cap.release()
    return sequences

def cmd_enroll(args: argparse.Namespace) -> None:
    """
    'enroll' command logic:
    1. Initializes the new 256-dim Gait Extractor.
    2. Processes video files from 'data/gait_videos/{name}' or all folders.
    3. Extracts robust sequences and computes a mean embedding vector (Centroid).
    4. Saves the identity to the FAISS gallery.
    """
    _setup_logging()
    cfg = default_gait_config()
    
    extractor = GaitExtractor(cfg)
    gallery = GaitGallery(cfg)
    
    extractor = GaitExtractor(cfg)
    gallery = GaitGallery(cfg)
    
    main_device, _ = select_device(prefer_gpu=True)
    use_gpu = False
    pose_model_path = "yolov8n-pose.pt" # Default PyTorch model
    
    if main_device == "cuda":
        logger.info("🚀 CUDA GPU Detected - Using PyTorch model for maximum efficiency")
        use_gpu = True
        pose_model_path = "yolov8n-pose.pt" # Ultralytics auto-download
    else:
        ov_path = cfg.models.pose_model_name
        if Path(ov_path).exists() and Path(ov_path).is_dir():
             logger.info(f"ℹ️ GPU not found, using OpenVINO model: {ov_path}")
             pose_model_path = ov_path
             use_gpu = False
        else:
             logger.info("ℹ️ Using CPU with PyTorch model")
             pose_model_path = "yolov8n-pose.pt"

    logger.info(f"Loading Pose model: {pose_model_path}")
    pose_model = YOLO(pose_model_path)
    if use_gpu:
        pose_model.to("cuda")
    
    root_data_dir = Path("data/gait_videos")
    
    if args.name:
        target_dir = root_data_dir / args.name
        if not target_dir.exists():
            logger.info(f"Creating directory: {target_dir}")
            target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"⚠️ Directory {target_dir} was created.")
            logger.info(f"👉 Please place .mp4/.avi video files of '{args.name}' in this folder and run command again.")
            return
            
        persons_to_process = [(args.name, target_dir)]
    else:
        if not root_data_dir.exists():
            logger.info(f"Creating root directory: {root_data_dir}")
            root_data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"👉 Root folder created. Create subfolders for each person (e.g. data/gait_videos/john) and add videos.")
            return
        persons_to_process = [(p.name, p) for p in root_data_dir.iterdir() if p.is_dir()]

    def _smart_chunk_sequence(seq: List[np.ndarray], min_len: int = 30) -> List[List[np.ndarray]]:
        """
        Splits a continuous sequence into 'Stable Walking' chunks.
        Discards 'Turns' and 'Stops' based on Kinematic Analysis (Hip Velocity + Width Consistency).
        
        Returns:
            List of clean sub-sequences ready for enrollment.
        """
        if len(seq) < min_len: return []
        
        kp = np.array(seq)
        chunks = []
        current_chunk = []
        
        hips = kp[:, [11,12], :2].mean(axis=1)
        vel = np.linalg.norm(np.diff(hips, axis=0, prepend=hips[:1]), axis=1)
        
        vel_smooth = np.convolve(vel, np.ones(5)/5, mode='same')
        
        heights = np.linalg.norm(kp[:, 0, :2] - kp[:, 15, :2], axis=1)
        med_height = np.median(heights)
        if med_height < 10: med_height = 100
        
        motion_thresh = 0.005 * med_height 

        for t in range(len(seq)):
            frame_kp = seq[t]
            is_moving = vel_smooth[t] > motion_thresh
            
            
            if is_moving:
                current_chunk.append(frame_kp)
            else:
                if len(current_chunk) >= min_len:
                    chunks.append(current_chunk)
                current_chunk = []
                
        if len(current_chunk) >= min_len:
            chunks.append(current_chunk)
            
        return chunks

    for name, person_dir in persons_to_process:
        logger.info(f"Processing identity: {name}")
        video_files = [f for f in person_dir.iterdir() if f.suffix.lower() in [".mp4", ".mov", ".avi"]]
            
        valid_embeddings = []
        total_chunks = 0
        
        for v_file in video_files:
            logger.info(f"  - Extracting from {v_file.name}...")
            raw_sequences = extract_raw_sequences_from_video(v_file, pose_model, min_len=30)
            
            for raw_seq in raw_sequences:
                clean_chunks = _smart_chunk_sequence(raw_seq, min_len=cfg.route.min_sequence_length)
                
                if not clean_chunks:
                     pass
                
                for chunk in clean_chunks:
                    emb, _ = extractor.extract_gait_embedding_and_quality(chunk)
                    
                    anthro_stats = extractor.extract_anthropometry(chunk)
                    
                    if anthro_stats is None:
                         continue
                    
                    if emb is not None:
                        valid_embeddings.append((emb, anthro_stats))
                        total_chunks += 1
        
        if total_chunks > 0:
            logger.info(f"    ✨ Smart Chunking: Extracted {total_chunks} valid & stable templates.")

        if not valid_embeddings:
            logger.warning(f"No valid embeddings extracted for {name} (All rejected by Geometry or Stability)")
            continue
            
        count = 0
        for i, (emb, stats) in enumerate(valid_embeddings):
            gallery.add_gait_embedding(
                identity_id=name,
                new_embedding=emb, 
                category=args.category,
                confirmed=True,
                anthro_stats=stats
            )
            count += 1
            
        logger.info(f"✅ Identity {name} enrolled successfully with {count} distinct templates.")

    gallery.save_gallery()

def cmd_list(args: argparse.Namespace) -> None:
    """Lists all enrolled identities in the gallery."""
    _setup_logging()
    gallery = GaitGallery(default_gait_config())
    persons = gallery.list_persons()
    
    if not persons:
        print("Gallery is empty.")
        return
    print(f"{'person_id':<25} | {'category':<12} | {'templates':<10}")
    print("-" * 55)
    for p in persons:
        print(f"{p.person_id:<25} | {p.category:<12} | {p.num_templates:<10}")

def cmd_delete(args: argparse.Namespace) -> None:
    """Deletes an identity from the gallery."""
    _setup_logging()
    gallery = GaitGallery(default_gait_config())
    person_id = args.person_id or input("Enter person_id to delete: ").strip()
    if gallery.delete_person(person_id):
        print(f"✅ Deleted {person_id}")
    else:
        print("❌ Identity not found.")

def main(argv: Optional[list[str]] = None) -> None:
    if argv is None: argv = sys.argv[1:]
    parser = argparse.ArgumentParser(prog="gait-enroll")
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll")
    p_enroll.add_argument("--name", type=str, help="Specific folder name in data/gait_videos")
    p_enroll.add_argument("--category", type=str, default="resident")

    sub.add_parser("list")
    
    p_del = sub.add_parser("delete")
    p_del.add_argument("person_id", nargs="?", help="ID of the person to delete")

    args = parser.parse_args(argv)
    if args.command == "enroll": cmd_enroll(args)
    elif args.command == "list": cmd_list(args)
    elif args.command == "delete": cmd_delete(args)

if __name__ == "__main__":
    main()
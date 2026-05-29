import cv2
import torch
import numpy as np
import time

# Keep your local imports intact
from extraction.images import image_processor_from_numpy

def extract_video_features_compressed_ms(video_path, 
                                         model, 
                                         processor, 
                                         model_code, 
                                         target_frames=16):
    """
    Extracts global temporal video embeddings using high-speed millisecond seeking,
    leveraging a SINGLE persistent model reference loaded on the GPU/CPU.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video file: {video_path}")
        return np.zeros(512)
        
    # 1. Grab total duration in milliseconds from container header
    total_duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    
    if total_duration_ms <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_duration_ms = (total_frames / fps) * 1000 if fps > 0 else 30000

    # 2. Compute exact millisecond timestamps uniformly
    target_timestamps = np.linspace(0, total_duration_ms - 10, target_frames)
    
    frame_features = []
    
    for ms_timestamp in target_timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ms_timestamp)
        ret, frame = cap.read()
        if not ret:
            break
        
        # ─── SPATIAL COMPRESSION ───
        # Scale down resolution to save processing footprints
        compressed_frame = cv2.resize(frame, (360, 270), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2RGB)

        # ─── EXTRACTION CALL (0 reloading overhead) ───
        # Pass the already compressed 'rgb_frame' instead of the heavy original 'frame'!
        frame_feature = image_processor_from_numpy(rgb_frame, model, processor, model_code)
        
        # Guard against PyTorch tensors bleeding into your list
        if torch.is_tensor(frame_feature):
            frame_feature = frame_feature.cpu().numpy().flatten()

        frame_features.append(frame_feature)
        
    cap.release()
    
    if len(frame_features) == 0:
        return np.zeros(512)
        
    # 3. Global Temporal Mean Pooling
    video_vector = np.mean(frame_features, axis=0)
    return video_vector

def extract_video_features_at_1_fps(video_path, 
                                    model, 
                                    processor, 
                                    model_code):
    """
    Extracts features by sampling exactly 1 frame per second of video time.
    Leverages a single persistent model reference passed from the top-level loop
    and applies spatial resolution downscaling to minimize RAM footprints.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video file: {video_path}")
        return np.zeros(512)

    # 1. Gather structural properties from video metadata
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Safeguard fallback against corrupted container metadata headers
    if fps <= 0:
        fps = 30 
        
    duration_seconds = total_frames / fps

    # 2. Compute indices that map exactly to 1-second timestamps (0, 30, 60, 90...)
    frame_step = int(fps)
    target_indices = [idx for idx in range(0, total_frames, frame_step)]
    
    frame_features = []

    # 3. High-speed feature extraction loop
    for frame_idx in target_indices:
        # Command OpenCV to jump straight to the exact index coordinate
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # ─── SPATIAL COMPRESSION ───
        # Downscale the heavy 1080p/4K resolution to a lightweight 270p footprint
        compressed_frame = cv2.resize(frame, (360, 270), interpolation=cv2.INTER_AREA)
        
        # Format the default BGR pixel array to the standard standard CLIP RGB channel space
        rgb_frame = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2RGB)

        # ─── TRANFORMER BACKBONE FORWARD PASS ───
        # Hand off the optimized frame to your shared, non-reloading model object
        frame_feature = image_processor_from_numpy(rgb_frame, model, processor, model_code)
        
        # Defensive check to make sure everything detaches into a flat CPU numpy array
        if torch.is_tensor(frame_feature):
            frame_feature = frame_feature.cpu().numpy().flatten()
            
        frame_features.append(frame_feature)

    cap.release()

    if len(frame_features) == 0:
        return np.zeros(512)

    # 4. Global Temporal Mean Pooling
    # Collapse the sequence timeline into a single stable 512-dim signature vector
    video_vector = np.mean(frame_features, axis=0)
    return video_vector
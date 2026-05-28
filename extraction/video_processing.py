import cv2
import torch
import numpy as np
import time

from extraction.images import model_loader, image_processor_from_numpy

def extract_features_at_1_fps(video_path, 
                              model_code, 
                              device):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        fps = 30 
        
    duration_seconds = total_frames / fps
    print(f"🎬 Video Profile: {duration_seconds:.2f}s total duration at {fps} FPS")

    frame_step = int(fps)
    target_indices = [idx for idx in range(0, total_frames, frame_step)]  
    frame_features = []

    print(f"📸 Extracting 1 frame per second. Total frames to process: {len(target_indices)}")

    for frame_idx in target_indices:
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        model, processor, model_code = model_loader(model_code)
        frame_feature = image_processor_from_numpy(frame, model, 
                                                    processor, model_code)
        frame_feature = frame_feature.detach().numpy().flatten()

        frame_features.append(frame_feature)

    cap.release()

    if len(frame_features) == 0:
        return np.zeros(512)

    video_vector = np.mean(frame_features, axis=0)
    
    return video_vector

def extract_video_features_fixed_stride(video_path,
                                        model_code,
                                        device, 
                                        target_frames=16):

    start_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file {video_path}")
        return np.zeros(512), 0.0
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    if total_frames > target_frames:
        frame_indices = np.linspace(0, 
                                    total_frames - 1, 
                                    target_frames, 
                                    dtype=int)
        
    else:
        frame_indices = np.arange(total_frames)
        
    frame_features = []

    for frame_idx in frame_indices:
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        model, processor, model_code = model_loader(model_code)
        frame_feature = image_processor_from_numpy(frame, model, 
                                                    processor, model_code)
        frame_feature = frame_feature.detach().numpy().flatten()

        frame_features.append(frame_feature)
        
    cap.release()

    if len(frame_features) == 0:
        elapsed_time = time.time() - start_time
        return np.zeros(512), elapsed_time
        
    video_vector = np.mean(frame_features, axis=0)
    
    elapsed_time = time.time() - start_time
    return video_vector, elapsed_time
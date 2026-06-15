import cv2
import torch
import numpy as np
import subprocess
from extraction.images import image_processor_from_numpy
from extraction.audios import audio_processor_from_numpy


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

def extract_video_features_mps_16_fps(video_path, model, processor):
    # Dynamically grab your Mac's GPU device reference
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video file: {video_path}")
        return np.zeros((1, 768))
        
    segment_features = []
    current_batch_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Fast CPU conversion
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_batch_frames.append(rgb_frame)
        
        # 2. Once we hit 16 frames, push them as a SINGLE batch to MPS
        if len(current_batch_frames) == 16:
            # Tokenize/preprocess all 16 frames at once
            inputs = processor(images=current_batch_frames, return_tensors="pt")
            
            # 🔥 CRITICAL: Transfer the input dictionary directly to MPS
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Forward pass the whole 16-frame chunk concurrently on Apple Silicon
                chunk_features = model.get_image_features(**inputs)
                
                # Mean pool temporally over the batch dimension directly on the GPU
                segment_vector = chunk_features.mean(dim=0)
                
                # Move only the single resulting vector back to CPU storage
                segment_features.append(segment_vector.cpu().numpy())
            
            # Clear our frame bucket
            current_batch_frames = []
            
    cap.release()
    
    # Handle remainder frames at the very end of the video
    if len(current_batch_frames) > 0:
        inputs = processor(images=current_batch_frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            segment_vector = model.get_image_features(**inputs).mean(dim=0)
            segment_features.append(segment_vector.cpu().numpy())
            
    if len(segment_features) == 0:
        return np.zeros((1, 768))
        
    return np.array(segment_features)

def extract_video_features_compressed_ms(video_path, model, processor, model_code, target_frames=16):
    """
    Extracts global temporal video embeddings using uniform frame seeking.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video file: {video_path}")
        return np.zeros(512)
        
    total_duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if total_duration_ms <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_duration_ms = (total_frames / fps) * 1000 if fps > 0 else 30000

    target_timestamps = np.linspace(0, max(0, total_duration_ms - 10), target_frames)
    frame_features = []
    
    for ms_timestamp in target_timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ms_timestamp)
        ret, frame = cap.read()
        if not ret:
            break
        
        compressed_frame = cv2.resize(frame, (360, 270), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(compressed_frame, cv2.COLOR_BGR2RGB)

        frame_feature = image_processor_from_numpy(rgb_frame, model, processor, model_code)
        if torch.is_tensor(frame_feature):
            frame_feature = frame_feature.cpu().numpy().flatten()

        frame_features.append(frame_feature)
        
    cap.release()
    return np.mean(frame_features, axis=0) if frame_features else np.zeros(512)


def extract_audio_waveform_from_video(video_path, target_sr=16000):
    """
    Extracts the audio track from a video container and outputs a 1D NumPy array.
    """
    command = [
        'ffmpeg', '-i', video_path,
        '-vn', '-ac', '1', '-ar', str(target_sr),
        '-f', 'f32le', '-acodec', 'pcm_f32le',
        '-loglevel', 'quiet', '-'
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, _ = process.communicate()
        if process.returncode != 0:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(stdout_data, dtype=np.float32)
    except Exception:
        return np.zeros(0, dtype=np.float32)

# ─── NEW UNIFIED MULTIMODAL EXTRACTION FEATURE ───
def extract_multimodal_features(video_path, 
                                image_model, image_processor, image_code, 
                                audio_model, audio_processor, audio_code,
                                target_frames=18):
    """
    Extracts both aggregated visual and auditory embeddings from a video container simultaneously.
    Dynamically aligns sample rates depending on the backend audio model selection.
    
    Returns:
    - tuple: (visual_embedding, audio_embedding)
    """
    # 1. Visual Feature Extraction Pass
    visual_vector = extract_video_features_compressed_ms(
        video_path=video_path,
        model=image_model,
        processor=image_processor,
        model_code=image_code,
        target_frames=target_frames
    )
    
    # 2. Dynamic Audio Target Sampling Alignment
    target_sr = 48000 if audio_code == 'clap' else 16000
    
    # 3. Extract Waveform and Generate Auditory Embedding
    audio_array = extract_audio_waveform_from_video(video_path, target_sr=target_sr)
    audio_vector = audio_processor_from_numpy(
        audio_array=audio_array,
        model=audio_model,
        processor=audio_processor,
        model_code=audio_code
    )
    
    return visual_vector, audio_vector
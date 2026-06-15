import os
import time
import torch
import cv2
import numpy as np

from extraction.images import model_loader as image_model_loader
from extraction.audios import model_loader as audio_model_loader
from extraction.video_processing import extract_video_features_compressed_ms, extract_audio_waveform_from_video
from extraction.audios import audio_processor_from_numpy

class XDViolenceExtractor:

    def __init__(self):
        # 1. Hardware Detection Guard
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        print(f"⏳ Loading transformer weights into active hardware space ({self.device.upper()})...")

        # 2. Warm up backbones once at instantiation level
        self.image_model, self.image_processor, self.image_active_code = image_model_loader(
            model_code='clip', device=self.device
        )
        self.audio_model, self.audio_processor, self.audio_active_code = audio_model_loader(
            model_code='clap', device=self.device
        )
        print(f"✅ Transformers successfully cached.\n")

        # Fixed: Resolved local scope resolution path naming bugs
        self.extract_path = "data/images/xd_violence"
        self.violence_dirs = [
            os.path.join(self.extract_path, 'Fighting'),
            os.path.join(self.extract_path, 'Shooting')
        ]
        self.normal_dirs = [os.path.join(self.extract_path, 'Normal')]

    def _process_directory_pool(self, target_directories, limit_count, category_label):
        """
        Reusable helper method that standardizes video and audio modality ingestion.
        Eliminates duplicate code loops completely.
        """
        visual_dataset, audio_dataset = [], []
        sub_limit = limit_count // len(target_directories)

        for target_dir in target_directories:
            if not os.path.exists(target_dir):
                print(f"⚠️ Directory pathway missing: {target_dir}. Skipping.")
                continue

            file_count = 0
            video_times, audio_times = [], []
            
            print(f"📂 Processing category [{category_label}] out of: {os.path.basename(target_dir)}")

            for file_name in os.listdir(target_dir):
                if file_count >= sub_limit:
                    break
                    
                video_path = os.path.join(target_dir, file_name)
                
                # Check video health check metrics
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0:
                    print(f"⚠️ Skipping corrupted container: {file_name}")
                    cap.release()
                    continue
                cap.release()

                try:
                    # ─── VISUAL PIPE ───
                    start_time = time.time()
                    visual_vector = extract_video_features_compressed_ms(
                        video_path=video_path,
                        model=self.image_model,        # Fixed missing self pointer references
                        processor=self.image_processor,
                        model_code=self.image_active_code,
                        target_frames=18
                    )
                    if np.all(visual_vector == 0):
                        continue
                    video_times.append(time.time() - start_time)

                    # ─── AUDIO PIPE ───
                    start_audio_time = time.time()
                    audio_array = extract_audio_waveform_from_video(video_path=video_path, target_sr=16000)
                    
                    if audio_array is None or len(audio_array) == 0:
                        print(f"⚠️ Missing sound track matrix array stream in: {file_name}")
                        continue

                    audio_vector = audio_processor_from_numpy(
                        audio_array=audio_array,
                        model=self.audio_model,
                        processor=self.audio_processor,
                        model_code=self.audio_active_code
                    )
                    audio_times.append(time.time() - start_audio_time)

                    # Commit to batch arrays
                    visual_dataset.append(visual_vector)
                    audio_dataset.append(audio_vector)
                    file_count += 1

                except Exception as e:
                    print(f"❌ Pipeline failure decoding file {file_name}: {e}")
                    continue

            if video_times and audio_times:
                print(f"⏱️ [{os.path.basename(target_dir)}] Avg Video Speed: {np.mean(video_times):.4f}s | Avg Audio Speed: {np.mean(audio_times):.4f}s")

        return np.array(visual_dataset), np.array(audio_dataset)

    def extracting_violence_data(self, violence_count=600, normal_count=100):
        # Route logic through the new directory processing engine cleanly
        v_vis, v_aud = self._process_directory_pool(self.violence_dirs, violence_count, "VIOLENCE")
        n_vis, n_aud = self._process_directory_pool(self.normal_dirs, normal_count, "NORMAL")
        return v_vis, v_aud, n_vis, n_aud

    def extract_and_save(self, violence_count=600, normal_count=100):
        v_vis, v_aud, n_vis, n_aud = self.extracting_violence_data(
            violence_count=violence_count, normal_count=normal_count
        )
        np.save('violence_visual_data.npy', v_vis)
        np.save('violence_audio_data.npy', v_aud)
        np.save('normal_visual_data.npy', n_vis)
        np.save('normal_audio_data.npy', n_aud)
        print("\n💾 Data arrays successfully compiled and saved to disk.")
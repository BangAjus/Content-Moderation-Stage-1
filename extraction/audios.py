import torch
import numpy as np
from transformers import (
    WhisperProcessor, WhisperModel, 
    Wav2Vec2Processor, Wav2Vec2Model,
    ClapProcessor, ClapModel
)

def model_loader(model_code='whisper', device="mps"):
    """
    Initializes and warms heavy audio transformer backbones.
    """
    if model_code == 'whisper':
        model = WhisperModel.from_pretrained("openai/whisper-base").to(device)
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        
    elif model_code == 'wav2vec2':
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
    elif model_code == 'clap':
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        
    else:
        raise ValueError(f"❌ Unknown audio model code: {model_code}")
        
    return model, processor, model_code

def audio_processor_from_numpy(audio_array, model, processor, model_code):
    """
    Extracts dense audio embeddings from a 1D NumPy float array.
    """
    assert isinstance(audio_array, np.ndarray), "❌ Input object is not a NumPy array"
    if len(audio_array) == 0:
        return np.zeros(768 if model_code != 'clap' else 512)
    
    current_device = next(model.parameters()).device
    
    if model_code == 'whisper':
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").to(current_device)
        with torch.no_grad():
            encoder_outputs = model.encoder(inputs.input_features)
            mean_pooled = encoder_outputs.last_hidden_state.mean(dim=1)
            audio_features = mean_pooled.squeeze(0).cpu().numpy().flatten()
            
    elif model_code == 'wav2vec2':
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").to(current_device)
        with torch.no_grad():
            outputs = model(**inputs)
            mean_pooled = outputs.last_hidden_state.mean(dim=1)
            audio_features = mean_pooled.squeeze(0).cpu().numpy().flatten()
            
    elif model_code == 'clap':
        inputs = processor(audios=audio_array, sampling_rate=48000, return_tensors="pt").to(current_device)
        with torch.no_grad():
            audio_features = model.get_audio_features(**inputs)
        audio_features = audio_features.cpu().numpy().flatten()
            
    return audio_features
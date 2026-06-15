import torch
import cv2
import numpy as np
# 🔥 FIXED: Added all missing architectural model and processor imports
from transformers import (
    CLIPProcessor, 
    CLIPModel, 
    AutoModel, 
    AutoProcessor, 
    Blip2Processor, 
    Blip2ForConditionalGeneration
)
    
def model_loader(model_code='clip', device="mps"):
    """
    Initializes and warms heavy vision transformer backbones into hardware memory.
    """
    if model_code == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    elif model_code == 'siglip':
        model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    elif model_code == 'blip2':
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16
        ).to(device)
    else:
        raise ValueError(f"❌ Unknown vision model code: {model_code}")
        
    return model, processor, model_code

def image_processor_from_path(image_path, model, processor, model_code):
    """
    Reads an image from a path and extracts its dense embedding vector.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Could not read image path: {image_path}")
        return None
    return image_processor_from_numpy(image, model, processor, model_code)

def image_processor_from_numpy(image, model, processor, model_code):
    """
    Processes raw BGR images from NumPy arrays, converts to RGB, and extracts model embeddings.
    """
    assert isinstance(image, np.ndarray), "❌ Input object is not a NumPy array"
    
    # Extract whatever hardware backend the model is currently inhabiting
    current_device = next(model.parameters()).device

    if model_code == 'blip2':
        inputs = processor(images=image, return_tensors="pt").to(current_device, torch.float16)
        with torch.no_grad():
            outputs = model(**inputs)
            q_former_outputs = outputs.qformer_outputs.last_hidden_state
            mean_pooled_vector = q_former_outputs.mean(dim=1)
            image_features = mean_pooled_vector.squeeze(0).cpu().numpy().flatten()
    else:
        # 🔥 FIXED: Processor outputs must be explicitly pushed to the model's device backend!
        inputs = processor(images=image, return_tensors="pt").to(current_device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        image_features = image_features.cpu().numpy().flatten()
    
    return image_features
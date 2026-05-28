import torch
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np

def model_loader(model_code='clip'):

    if model_code == 'clip':
        
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    elif model_code == 'siglip':

        model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    elif model_code == 'blip2':

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", 
                                                              torch_dtype=torch.float16)
    else:
        print(f"no model code such as {model_code}")
        
    return model, processor, model_code

def image_processor_from_path(image_path,
                                model,
                                processor,
                                model_code):

    image_features = None
    image = cv2.imread(image_path)
    
    if model_code == 'blip2':

        inputs = processor(images=image, return_tensors="pt").to(torch.float16)

        with torch.no_grad():
            
            outputs = model(**inputs)
            vision_outputs = outputs.vision_outputs.last_hidden_state
            q_former_outputs = outputs.qformer_outputs.last_hidden_state
        
            mean_pooled_vector = q_former_outputs.mean(dim=1)
            image_features = mean_pooled_vector.squeeze(0)
        
    else:

        inputs = processor(images=image, 
                           return_tensors="pt")
        image_features = model.get_image_features(**inputs)
    
    return image_features

def image_processor_from_numpy(image,
                                model,
                                processor,
                                model_code):

    image_features = None
    assert isinstance(image, np.ndarray), "Object is not a NumPy array"
    
    if model_code == 'blip2':

        inputs = processor(images=image, return_tensors="pt").to(torch.float16)

        with torch.no_grad():
            
            outputs = model(**inputs)
            vision_outputs = outputs.vision_outputs.last_hidden_state
            q_former_outputs = outputs.qformer_outputs.last_hidden_state
        
            mean_pooled_vector = q_former_outputs.mean(dim=1)
            image_features = mean_pooled_vector.squeeze(0)
        
    else:

        inputs = processor(images=image, 
                           return_tensors="pt")
        image_features = model.get_image_features(**inputs)
    
    return image_features
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def model_loader(device,
                 model_code='e5'):

    model = None
    
    if model_code == 'minilm':
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


    elif model_code == 'bge':
        model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)

    elif model_code == 'e5':
        model = SentenceTransformer('intfloat/e5-small-v2', device=device)
        
    else:
        print(f"no model code such as {model_code}")
        
    return model, model_code

def text_encoding(texts, 
                  model, 
                  model_code):
    
    assert isinstance(texts, np.ndarray) or isinstance(texts, list), "Object is not a NumPy array or list"
    
    if model_code == 'e5':
        
        e5_formatted_texts = [f"query: {text}" for text in texts]
        text_vectors = model.encode(e5_formatted_texts, 
                                    show_progress_bar=True, 
                                    convert_to_numpy=True)
        
    else:
        text_vectors = model.encode(texts, 
                                        show_progress_bar=True, 
                                        convert_to_numpy=True)

    return text_vectors
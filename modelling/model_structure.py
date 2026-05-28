import torch
import joblib
import os

# Assuming these are your custom local module imports
from extraction import texts, images
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class StageOneTextClassifier:

    def __init__(self, 
                 device, 
                 text_modality_model_code, 
                 hate_speech_classifier_path, 
                 suicide_classifier_path):
        
        """
        Initializes the Stage 1 text routing classifier.
        """
        self.device = device
        
        self.text_modality_model_code = text_modality_model_code
        self.hate_speech_classifier_path = hate_speech_classifier_path
        self.suicide_classifier_path = suicide_classifier_path

        # Lazy load models only when needed, or load them immediately during init
        self.text_model = None
        self.hate_speech_model = self._load_model(self.hate_speech_classifier_path)
        self.suicide_model = self._load_model(self.suicide_classifier_path)

    def _load_model(self, path):
        
        """
        Helper method to safely load 
        scikit-learn/XGBoost artifacts.
        """
        
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return joblib.load(f)
                
        else:
            print(f"⚠️ Warning: Model artifact not found at {path}")
            return None

    def text_extractor(self, text_input_object):
        
        """
        Loads the E5/BERT model text head 
        and extracts dense feature embeddings.
        """
        
        # Load the frozen model backbone if it hasn't been loaded yet
        if self.text_model == None:
            # Assumes texts.model_loader returns (model, tokenizer/processor)
            self.text_model, _ = texts.model_loader(
                self.device, 
                model_code=self.text_modality_model_code
            )
            
        # Extract features (Assumes text_encoding is imported from extraction)
        # We look up the text_encoding function dynamically
        text_features = texts.text_encoding(
            text_input_object, 
            self.text_model, 
            self.text_modality_model_code
        )
        
        return text_features

    def text_probability(self, text_input_object):
        
        """
        Processes text input, extracts features, and returns multi-binary 
        probabilities for policy violation categories.
        """
        
        if self.hate_speech_model is None or self.suicide_model is None:
            print("❌ Cannot predict: Classifier models are not fully loaded.")
            return None

        # 1. Extract features dynamically from the text object
        features = self.text_extractor(text_input_object)
        
        # Ensure features are shaped as a 2D array for sklearn/XGBoost (1, num_features)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        try:
            # 2. Extract raw calibrated probabilities for Class 1 (Violation)
            hate_speech_proba = self.hate_speech_model.predict_proba(features)[0][1]
            suicide_proba = self.suicide_model.predict_proba(features)[0][1]
            
            # Return as a clean, interpretable dictionary or array
            return {
                "hate_speech_probability": hate_speech_proba,
                "suicide_probability": suicide_proba
            }

        except Exception as e:
            print(f"❌ Prediction failed. Mapping error: {e}")
            print("💡 Check if your input embedding features match the dimensions of the trained head.")
            return None
import torch
import joblib
import os

# Ensure your local module imports map cleanly
from extraction import texts, images
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class StageOneClassifier:

    def __init__(self, 
                 device, 
                 text_modality_model_code="e5",
                 image_modality_model_code="clip",
                 hate_speech_classifier_path="model/hate_speech.joblib", 
                 suicide_classifier_path="model/suicide.joblib",
                 cyberbullying_classifier_path="model/cyberbullying.joblib",
                 violence_classifier_path="model/violence_new.joblib",
                 adult_classifier_path="model/adult.joblib",
                 harmful_classifier_path="model/harmful.joblib"):
        
        """
        Initializes the Stage 1 Content Moderation Classifier.
        Loads all classification heads and heavy transformer backbones 
        immediately during initialization to ensure zero cold-start latencies.
        """
        self.device = device
        self.text_modality_model_code = text_modality_model_code
        self.image_modality_model_code = image_modality_model_code
        
        self.hate_speech_classifier_path = hate_speech_classifier_path
        self.suicide_classifier_path = suicide_classifier_path
        self.cyberbullying_classifier_path = cyberbullying_classifier_path
        self.violence_classifier_path = violence_classifier_path
        self.adult_classifier_path = adult_classifier_path
        self.harmful_classifier_path = harmful_classifier_path

        print("🧬 [BOOT] Initializing production model layers...")
        
        # ─── 1. LOAD CLASSIFICATION HEADS ───
        self.hate_speech_model = self._load_model(self.hate_speech_classifier_path)
        self.suicide_model = self._load_model(self.suicide_classifier_path)
        self.cyberbullying_model = self._load_model(self.cyberbullying_classifier_path)
        self.violence_model = self._load_model(self.violence_classifier_path)
        self.adult_model = self._load_model(self.adult_classifier_path)
        self.harmful_model = self._load_model(self.harmful_classifier_path)
        
        # ─── 2. LOAD HEAVY TRANSFORMER BACKBONES (IMMEDIATE) ───
        print(f"📦 Loading Text Backbone [{self.text_modality_model_code}]...")
        self.text_model, _ = texts.model_loader(
            self.device, 
            model_code=self.text_modality_model_code
        )
        
        print(f"📦 Loading Image/Video Backbone [{self.image_modality_model_code}]...")
        self.image_model, self.image_processor, _ = images.model_loader(
            model_code=self.image_modality_model_code,
            device=self.device
        )
        
        print("🚀 [READY] Production inference pipeline fully active.")
        
    def _load_model(self, path):
        """
        Helper method to safely load scikit-learn/XGBoost artifacts.
        """
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return joblib.load(f)
        else:
            print(f"⚠️ Warning: Model artifact not found at {path}")
            return None

    def text_extractor(self, text_input_object):
        """
        Extracts dense text embeddings using the warm memory backbone.
        """
        return texts.text_encoding(
            text_input_object, 
            self.text_model, 
            self.text_modality_model_code
        )

    def image_extractor(self, image_input_object):
        """
        Extracts visual embeddings using the warm memory backbone.
        """
        return images.image_processor_from_numpy(
            image_input_object, 
            self.image_model, 
            self.image_processor,
            self.image_modality_model_code
        )

    def text_probability(self, text_input_object):
        """
        Processes text input, extracts features, and returns multi-binary 
        probabilities for policy violation categories.
        """
        if self.hate_speech_model is None or self.suicide_model is None or self.cyberbullying_model is None:
            print("❌ Cannot predict: Text classifier models are not fully loaded.")
            return None

        features = self.text_extractor(text_input_object)
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        try:
            hate_speech_proba = self.hate_speech_model.predict_proba(features)[0][1]
            suicide_proba = self.suicide_model.predict_proba(features)[0][1]
            cyberbullying_proba = self.cyberbullying_model.predict_proba(features)[0][1]
            
            return {
                "hate_speech_probability": hate_speech_proba,
                "suicide_probability": suicide_proba,
                "cyberbullying_probability": cyberbullying_proba
            }

        except Exception as e:
            print(f"❌ Prediction failed. Mapping error: {e}")
            return None

    def image_probability(self, image_input_object):
        """
        Processes image/video input matrices, extracts spatial/temporal 
        embeddings, and returns the binary probability for visual violations (violence).
        """
        # Fixed trailing comma bug inside conditional statement
        if self.violence_model is None or self.adult_model is None or self.harmful_model is None:
            print("❌ Cannot predict: Vision classifier heads are not fully loaded.")
            return None

        features = self.image_extractor(image_input_object)
        
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        try:
            violence_proba = self.violence_model.predict_proba(features)[0][1]
            adult_proba = self.adult_model.predict_proba(features)[0][1]
            # Fixed 'seld' typo variable reference assignment
            harmful_proba = self.harmful_model.predict_proba(features)[0][1]
            
            return {
                "violence_probability": violence_proba,
                "adult_probability": adult_proba,
                "harmful_probability": harmful_proba
            }

        except Exception as e:
            print(f"❌ Prediction failed. Visual mapping error: {e}")
            return None
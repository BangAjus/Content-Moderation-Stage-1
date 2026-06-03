# gatekeeper.py
import os
import torch
import json
from model_structure import StageOneClassifier

class StageOneGatekeeper:
    def __init__(self, 
                 device=None, 
                 lower_gate=0.21,  # ◄ Tuned threshold default
                 upper_gate=0.68): # ◄ Tuned threshold default
        
        """
        Coordinates Stage 1 Content Moderation routing.
        Accepts dynamic threshold boundaries to easily calibrate the 
        3-Bucket corporate footprint targets based on validation data audits.
        """
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🖥️  Routing execution target assigned to: {self.device.upper()}")

        # Instantiate your production-hardened model heads
        # Paths point directly to your 'model/' directory structure
        self.classifier = StageOneClassifier(
            device=self.device,
            text_modality_model_code="e5",
            image_modality_model_code="clip",
            hate_speech_classifier_path="model/hate_speech.joblib",
            suicide_classifier_path="model/suicide.joblib",
            cyberbullying_classifier_path="model/cyberbullying.joblib",
            violence_classifier_path="model/violence_new.joblib",
            adult_classifier_path="model/adult.joblib",
            harmful_classifier_path="model/harmful.joblib"
        )
        
        # Dynamic Tuning Gates
        self.LOWER_GATE = lower_gate
        self.UPPER_GATE = upper_gate
        
        print(f"🎛️  [GATEKEEPER ACTIVE] Lower Limit: {self.LOWER_GATE} | Upper Limit: {self.UPPER_GATE}")

    def _evaluate_probability_buckets(self, proba_dict):
        """
        Evaluates extracted prediction dictionaries against the 3-Bucket filters.
        """
        if not proba_dict:
            return {"status": "ERROR", "reason": "Inference layer execution failure"}

        # Extract the highest violation risk across all active categories
        max_risk = max(proba_dict.values())
        trigger_category = max(proba_dict, key=proba_dict.get)

        # ─── THE 3-BUCKET ROUTING FILTERS ───
        if max_risk < self.LOWER_GATE:
            return {
                "decision": "🟢 ALLOW",
                "confidence": round(max_risk, 4),
                "trigger_category": "None",
                "action": "Pass asset straight to the active user feed.",
                "all_probabilities": {k: round(v, 4) for k, v in proba_dict.items()}
            }
        elif max_risk > self.UPPER_GATE:
            return {
                "decision": "🔴 BAN",
                "confidence": round(max_risk, 4),
                "trigger_category": trigger_category,
                "action": "Immediate content removal and logging flag generated.",
                "all_probabilities": {k: round(v, 4) for k, v in proba_dict.items()}
            }
        else:
            return {
                "decision": "🟡 BUFFER (STAGE 2)",
                "confidence": round(max_risk, 4),
                "trigger_category": trigger_category,
                "action": "Route asset to Stage 2 Uncertain Queue for human audit.",
                "all_probabilities": {k: round(v, 4) for k, v in proba_dict.items()}
            }

    def moderate_text(self, text_input):
        """
        Public endpoint to moderate text strings (Hate Speech, Suicide, Cyberbullying).
        """
        probabilities = self.classifier.text_probability(text_input)
        return self._evaluate_probability_buckets(probabilities)

# 🧪 LOCAL WORKSTATION TESTING SECTION
if __name__ == "__main__":
    print("🚀 Initializing Gatekeeper System Testing Run...")
    moderator = StageOneGatekeeper(lower_gate=0.21, 
                                   upper_gate=0.68)
    
    # Simulate a toxic mock payload text string
    sample_text = "I am going to bully you until you quit this app."
    
    print(f"\n📝 Testing Text Payload: '{sample_text}'")
    routing_decision = moderator.moderate_text(sample_text)
    
    print("\n🏁 FINAL GATEKEEPER ROUTING DECISION:")
    print(json.dumps(routing_decision, indent=4))
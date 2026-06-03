# gatekeeper.py
import os
import torch
from model_structure import StageOneClassifier

class StageOneGatekeeper:
    def __init__(self, 
                 device=None, 
                 lower_gate=0.35,  # ◄ Default baseline from PDF
                 upper_gate=0.75): # ◄ Default baseline from PDF
        
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        # Instantiate your production classification heads
        self.classifier = StageOneClassifier(
            device=self.device,
            text_modality_model_code="e5",
            image_modality_model_code="clip"
        )
        
        # 🎯 DYNAMIC TUNING GATES: Assigned at boot time via arguments
        self.LOWER_GATE = lower_gate
        self.UPPER_GATE = upper_gate
        
        print(f"🎛️ [GATEKEEPER INITIALIZED] Lower Gate: {self.LOWER_GATE} | Upper Gate: {self.UPPER_GATE}")

    def _evaluate_probability_buckets(self, proba_dict):
        if not proba_dict:
            return {"status": "ERROR", "reason": "Execution failure"}

        max_risk = max(proba_dict.values())
        trigger_category = max(proba_dict, key=proba_dict.get)

        # The 3-Bucket Routing Filters run against your dynamically set gates
        if max_risk < self.LOWER_GATE:
            return {"decision": "🟢 ALLOW", "confidence": round(max_risk, 4), "trigger": "None"}
        elif max_risk > self.UPPER_GATE:
            return {"decision": "🔴 BAN", "confidence": round(max_risk, 4), "trigger": trigger_category}
        else:
            return {"decision": "🟡 BUFFER (STAGE 2)", "confidence": round(max_risk, 4), "trigger": trigger_category}

    def moderate_text(self, text_input):
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
    import json
    print(json.dumps(routing_decision, indent=4))
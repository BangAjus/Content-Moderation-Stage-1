# download_models.py
import os
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor, AutoModel, AutoTokenizer

def pre_install_models():
    print("🚚 [BUILD STEP] Pre-downloading Hugging Face models into image cache...")

    # 1. Text Backbone (E5 Base)
    text_model_name = "intfloat/e5-base-v2"
    print(f"📥 Downloading Text Model: {text_model_name}")
    AutoModel.from_pretrained(text_model_name)
    AutoTokenizer.from_pretrained(text_model_name)

    # 2. Vision Backbone (CLIP Base)
    vision_model_name = "openai/clip-vit-base-patch32"
    print(f"📥 Downloading Vision Model: {vision_model_name}")
    CLIPModel.from_pretrained(vision_model_name)
    CLIPProcessor.from_pretrained(vision_model_name)

    # 3. Audio Backbone (CLAP)
    audio_model_name = "laion/clap-htsat-unfused"
    print(f"📥 Downloading Audio Model: {audio_model_name}")
    ClapModel.from_pretrained(audio_model_name)
    ClapProcessor.from_pretrained(audio_model_name)

    print("✅ All tri-modal transformer models successfully cached inside image layers.")

if __name__ == "__main__":
    pre_install_models()
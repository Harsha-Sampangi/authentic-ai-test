import torch
import timm
from pathlib import Path

def download_deepfake_model():
    """
    Download EfficientNet-B4 model
    We'll use transfer learning weights from ImageNet,
    then fine-tune structure for deepfake detection
    """
    print("📥 Downloading EfficientNet-B4 model...")
    
    # Create models folder
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download pre-trained EfficientNet-B4
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1)
    
    # Save model
    model_path = models_dir / "efficientnet_b4_deepfake.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"✅ Model saved to: {model_path}")
    print("✅ Download complete!")
    
    return model_path

if __name__ == "__main__":
    download_deepfake_model()
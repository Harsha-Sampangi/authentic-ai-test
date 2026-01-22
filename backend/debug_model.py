
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import glob
import os

def check_model():
    print("Loading model...")
    model_name = "dima806/deepfake_vs_real_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    # Check config
    print(f"ID2LABEL: {model.config.id2label}")
    
    # Get all .jpeg images from uploads
    images = glob.glob("uploads/*.jpeg")
    images.sort()
    
    # Take first 3
    if not images:
        print("No images found in uploads to test.")
        return
        
    print(f"Testing {len(images)} images...")
    
    for img_path in images[:5]:
        try:
            image = Image.open(img_path).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            print(f"\nImage: {os.path.basename(img_path)}")
            print(f"Raw Logits: {outputs.logits.tolist()}")
            print(f"Probs: {probs.tolist()}")
            print(f"Class 0 ({model.config.id2label[0]}): {probs[0][0].item():.4f}")
            print(f"Class 1 ({model.config.id2label[1]}): {probs[0][1].item():.4f}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    check_model()

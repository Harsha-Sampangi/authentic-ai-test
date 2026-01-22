
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import librosa
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioDeepfakeDetector:
    """
    Audio deepfake detector using pre-trained model from Hugging Face
    """
    
    def __init__(self):
        print("🎙️ Loading Audio Deepfake Detection Model...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {self.device}")
        
        try:
            # MelodyMachine/Deepfake-audio-detection-V2
            # Labels: {0: 'fake', 1: 'real'}
            model_name = "MelodyMachine/Deepfake-audio-detection-V2"
            
            print(f"   Loading model: {model_name}")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("   ✅ Audio deepfake model loaded!")
            
        except Exception as e:
            print(f"   ⚠️ Could not load Audio model: {e}")
            self.model = None
            self.feature_extractor = None
    
    def predict(self, audio_path):
        """
        Analyze audio file for deepfake detection
        """
        try:
            print(f"   🔊 Analyzing audio file...")
            
            # Load audio file
            # Resample to 16kHz as typically required by Wav2Vec2/Audio models
            audio, sample_rate = librosa.load(audio_path, sr=16000)
            
            if self.model is not None and self.feature_extractor is not None:
                # Preprocess audio
                inputs = self.feature_extractor(
                    audio, 
                    sampling_rate=16000, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=16000 * 10 # Limit to 10 seconds for initial analysis to prevent OOM
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Model output: [fake_prob, real_prob] (id2label: {0: 'fake', 1: 'real'})
                fake_prob = probs[0][0].item()
                real_prob = probs[0][1].item()
                
                # Convert to authenticity score (0-100)
                authenticity_score = real_prob * 100
                
                print(f"   📊 Fake probability: {fake_prob:.4f}")
                print(f"   📊 Real probability: {real_prob:.4f}")
                print(f"   📊 Authenticity score: {authenticity_score:.1f}%")
                
                # Determine if deepfake
                is_deepfake = authenticity_score < 50
                
                # Calculate confidence
                distance_from_threshold = abs(authenticity_score - 50)
                if distance_from_threshold > 30:
                    confidence = "high"
                elif distance_from_threshold > 15:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                # Generate alerts
                alerts = self.generate_alerts(authenticity_score, is_deepfake)
                
                result = {
                    'authenticity_score': round(authenticity_score, 1),
                    'is_deepfake': is_deepfake,
                    'confidence': confidence,
                    'alerts': alerts,
                    'model_version': 'MelodyMachine V2'
                }
                
                print(f"   ✅ Analysis complete: {'FAKE' if is_deepfake else 'REAL'} ({confidence} confidence)\n")
                
                return result
                
        except Exception as e:
            print(f"   ❌ Error during audio prediction: {str(e)}")
            # Return error state
            raise e

    def generate_alerts(self, score, is_deepfake):
        """Generate contextual alerts for audio"""
        alerts = []
        
        if is_deepfake:
            alerts.append({
                "severity": "High",
                "title": "Synthetic Audio Detected",
                "icon": "🎙️",
                "description": "Audio characteristics match known deepfake patterns"
            })
        else:
            alerts.append({
                "severity": "Low",
                "title": "Natural Voice Patterns",
                "icon": "✅",
                "description": "Audio features consistent with human speech"
            })
            
        return alerts

# Helper for direct testing
if __name__ == "__main__":
    detector = AudioDeepfakeDetector()
    # Add test code here if needed

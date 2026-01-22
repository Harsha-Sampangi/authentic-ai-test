import cv2
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import json
import subprocess
import os

class VideoDeepfakeAnalyzer:
    """
    Analyze videos for deepfakes by extracting and analyzing frames
    """
    
    def __init__(self, image_detector):
        """
        Args:
            image_detector: The image deepfake detector instance
        """
        self.image_detector = image_detector
        print("🎬 Video Analyzer initialized")
    
    def extract_audio(self, video_path):
        """
        Extract audio track from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file (temp .wav) or None if no audio
        """
        try:
            print(f"   🎙️ Extracting audio from video...")
            
            # Create temp file for audio
            fd, audio_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            # Use ffmpeg to extract audio
            # -y: overwrite output
            # -i: input
            # -vn: disable video recording
            # -acodec pcm_s16le: encoded as 16-bit PCM (wav compatible)
            # -ar 16000: resample to 16kHz for model compatibility
            # -ac 1: mono
            command = [
                'ffmpeg', 
                '-y', 
                '-i', str(video_path), 
                '-vn', 
                '-acodec', 'pcm_s16le', 
                '-ar', '16000', 
                '-ac', '1', 
                str(audio_path)
            ]
            
            # Run command silently
            result = subprocess.run(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            if result.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                print(f"   ✅ Audio extracted successfully")
                return audio_path
            else:
                print(f"   ⚠️ No audio track found or extraction failed")
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                return None
                
        except Exception as e:
            print(f"   ❌ Audio extraction error: {e}")
            return None

    def extract_frames(self, video_path, max_frames=30, method='uniform'):
        """
        Extract frames from video for analysis
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            method: 'uniform' (evenly spaced) or 'keyframes' (scene changes)
        
        Returns:
            List of (frame_number, frame_image) tuples
        """
        print(f"   📹 Extracting frames from video...")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"   Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        
        frames = []
        
        if method == 'uniform':
            # Extract evenly spaced frames
            frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp = idx / fps if fps > 0 else 0
                    frames.append({
                        'frame_number': int(idx),
                        'timestamp': timestamp,
                        'image': frame_rgb
                    })
        
        cap.release()
        
        print(f"   ✅ Extracted {len(frames)} frames")
        return frames
    
    def analyze_video(self, video_path, max_frames=30):
        """
        Analyze entire video for deepfakes
        
        Returns:
            dict with overall score, frame-by-frame results, and timeline
        """
        print(f"\n🎬 Starting video analysis...")
        
        # Extract frames
        frames = self.extract_frames(video_path, max_frames)
        
        if not frames:
            raise ValueError("No frames could be extracted from video")
        
        # Analyze each frame
        print(f"   🧠 Analyzing {len(frames)} frames...")
        
        frame_results = []
        temp_dir = Path(tempfile.mkdtemp())
        
        for i, frame_data in enumerate(frames):
            print(f"   Frame {i+1}/{len(frames)}...", end=' ')
            
            # Save frame temporarily
            frame_path = temp_dir / f"frame_{i}.jpg"
            frame_image = Image.fromarray(frame_data['image'])
            frame_image.save(frame_path)
            
            # Analyze frame
            try:
                result = self.image_detector.predict(str(frame_path))
                
                frame_results.append({
                    'frame_number': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'authenticity_score': result['authenticity_score'],
                    'is_deepfake': result['is_deepfake'],
                    'confidence': result['confidence']
                })
                
                print(f"{result['authenticity_score']:.1f}%")
                
            except Exception as e:
                print(f"Error: {e}")
                frame_results.append({
                    'frame_number': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'authenticity_score': 50.0,
                    'is_deepfake': False,
                    'confidence': 'low'
                })
            
            # Cleanup temp frame
            frame_path.unlink()
        
        # Clean up temp directory
        temp_dir.rmdir()
        
        # Calculate overall statistics
        scores = [r['authenticity_score'] for r in frame_results]
        
        # Cast numpy types to python types for JSON serialization
        overall_score = float(np.mean(scores))
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        std_score = float(np.std(scores))
        
        # Count suspicious frames
        suspicious_frames = sum(1 for r in frame_results if r['is_deepfake'])
        suspicious_percentage = (suspicious_frames / len(frame_results)) * 100
        
        # Determine overall verdict
        is_deepfake = bool(overall_score < 50 or suspicious_percentage > 30)
        
        # Confidence based on consistency
        if std_score < 10:  # Consistent scores
            confidence = 'high'
        elif std_score < 20:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Find most suspicious segments (consecutive suspicious frames)
        suspicious_segments = self._find_suspicious_segments(frame_results)
        
        # Generate initial alerts (visual only)
        alerts = self._generate_video_alerts(
            overall_score, 
            suspicious_percentage, 
            len(suspicious_segments),
            std_score
        )
        
        result = {
            'overall_score': round(overall_score, 1),
            'is_deepfake': is_deepfake,
            'confidence': confidence,
            'frame_count': len(frame_results),
            'suspicious_frames': suspicious_frames,
            'suspicious_percentage': round(suspicious_percentage, 1),
            'score_range': {
                'min': round(min_score, 1),
                'max': round(max_score, 1),
                'std': round(std_score, 1)
            },
            'suspicious_segments': suspicious_segments,
            'frame_results': frame_results,
            'alerts': alerts,
            'timeline': self._create_timeline(frame_results)
        }
        
        print(f"\n   ✅ Video visual analysis complete!")
        print(f"   Overall score: {overall_score:.1f}%")
        print(f"   Suspicious frames: {suspicious_frames}/{len(frame_results)} ({suspicious_percentage:.1f}%)")
        print(f"   Verdict: {'LIKELY FAKE' if is_deepfake else 'LIKELY AUTHENTIC'}\n")
        
        return result
    
    def _find_suspicious_segments(self, frame_results, min_length=2):
        """Find consecutive suspicious frames"""
        segments = []
        current_segment = []
        
        for frame in frame_results:
            if frame['is_deepfake']:
                current_segment.append(frame)
            else:
                if len(current_segment) >= min_length:
                    segments.append({
                        'start_time': current_segment[0]['timestamp'],
                        'end_time': current_segment[-1]['timestamp'],
                        'frame_count': len(current_segment),
                        'avg_score': float(np.mean([f['authenticity_score'] for f in current_segment]))
                    })
                current_segment = []
        
        # Check last segment
        if len(current_segment) >= min_length:
            segments.append({
                'start_time': current_segment[0]['timestamp'],
                'end_time': current_segment[-1]['timestamp'],
                'frame_count': len(current_segment),
                'avg_score': float(np.mean([f['authenticity_score'] for f in current_segment]))
            })
        
        return segments
    
    def _generate_video_alerts(self, overall_score, suspicious_pct, segment_count, score_std, audio_score=None):
        """Generate alerts specific to video analysis"""
        alerts = []
        
        if overall_score < 30:
            alerts.append({
                "severity": "High",
                "title": "Severe Video Manipulation",
                "icon": "🎬",
                "description": f"Visual authenticity score of {overall_score:.1f}% indicates heavy manipulation"
            })
        elif overall_score < 50:
            alerts.append({
                "severity": "Medium",
                "title": "Video Manipulation Detected",
                "icon": "⚠️",
                "description": "Multiple frames show signs of deepfake technology"
            })
        
        if suspicious_pct > 50:
            alerts.append({
                "severity": "High",
                "title": "Majority of Frames Suspicious",
                "icon": "📊",
                "description": f"{suspicious_pct:.1f}% of analyzed frames flagged as manipulated"
            })
        elif suspicious_pct > 30:
            alerts.append({
                "severity": "Medium",
                "title": "Significant Suspicious Content",
                "icon": "📊",
                "description": f"{suspicious_pct:.1f}% of frames show manipulation indicators"
            })
        
        if segment_count > 0:
            alerts.append({
                "severity": "Medium",
                "title": f"{segment_count} Suspicious Segment(s)",
                "icon": "⏱️",
                "description": "Continuous sections of video show manipulation patterns"
            })
        
        if score_std > 25:
            alerts.append({
                "severity": "Medium",
                "title": "Inconsistent Frame Quality",
                "icon": "📉",
                "description": "Large variations between frames suggest selective editing"
            })
            
        # Mixed Media Alerts
        if audio_score is not None:
            visual_is_fake = overall_score < 50
            audio_is_fake = audio_score < 50
            
            if visual_is_fake and not audio_is_fake:
                alerts.append({
                    "severity": "High",
                    "title": "Mixed Media: Fake Video / Real Audio",
                    "icon": "🎭",
                    "description": "Visuals appear manipulated but audio seems authentic. Likely face-swap or re-enactment."
                })
            elif not visual_is_fake and audio_is_fake:
                 alerts.append({
                    "severity": "High",
                    "title": "Mixed Media: Real Video / Fake Audio",
                    "icon": "🗣️",
                    "description": "Visuals appear authentic but audio is synthetic. Potential voice cloning or dubbing."
                })
            elif visual_is_fake and audio_is_fake:
                alerts.append({
                    "severity": "High",
                    "title": "Full Deepfake Detected",
                    "icon": "🤖",
                    "description": "Both auditory and visual channels show signs of manipulation."
                })
        
        return alerts
    
    def _create_timeline(self, frame_results):
        """Create timeline visualization data"""
        return [
            {
                'timestamp': r['timestamp'],
                'score': r['authenticity_score'],
                'suspicious': r['is_deepfake']
            }
            for r in frame_results
        ]
    
    def generate_video_thumbnail(self, video_path, output_path, timestamp=1.0):
        """Extract a thumbnail from video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                cv2.imwrite(str(output_path), frame)
                return output_path
            
        except Exception as e:
            print(f"Thumbnail generation error: {e}")
        
        return None
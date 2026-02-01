"""
Authentic.AI - Fake News Detector
AI-powered misinformation detection using BERT-based models
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available for fake news detection")

# ==========================================
# SOURCE CREDIBILITY DATABASE
# ==========================================

# High credibility sources (score 80-100)
CREDIBLE_SOURCES = {
    "reuters.com": {"score": 95, "bias": "center", "type": "news_agency"},
    "apnews.com": {"score": 94, "bias": "center", "type": "news_agency"},
    "bbc.com": {"score": 90, "bias": "center-left", "type": "public_broadcaster"},
    "bbc.co.uk": {"score": 90, "bias": "center-left", "type": "public_broadcaster"},
    "npr.org": {"score": 88, "bias": "center-left", "type": "public_broadcaster"},
    "pbs.org": {"score": 88, "bias": "center", "type": "public_broadcaster"},
    "economist.com": {"score": 87, "bias": "center-right", "type": "magazine"},
    "nytimes.com": {"score": 85, "bias": "center-left", "type": "newspaper"},
    "washingtonpost.com": {"score": 84, "bias": "center-left", "type": "newspaper"},
    "wsj.com": {"score": 85, "bias": "center-right", "type": "newspaper"},
    "theguardian.com": {"score": 82, "bias": "left", "type": "newspaper"},
    "thehindu.com": {"score": 85, "bias": "center-left", "type": "newspaper"},
    "indianexpress.com": {"score": 82, "bias": "center", "type": "newspaper"},
    "hindustantimes.com": {"score": 80, "bias": "center", "type": "newspaper"},
    "nature.com": {"score": 95, "bias": "center", "type": "scientific"},
    "science.org": {"score": 95, "bias": "center", "type": "scientific"},
    "who.int": {"score": 92, "bias": "center", "type": "organization"},
    "cdc.gov": {"score": 90, "bias": "center", "type": "government"},
}

# Low credibility / flagged sources (score 0-40)
UNRELIABLE_SOURCES = {
    "beforeitsnews.com": {"score": 10, "type": "conspiracy", "warning": "Known for publishing conspiracy theories"},
    "naturalnews.com": {"score": 15, "type": "pseudoscience", "warning": "Promotes health misinformation"},
    "infowars.com": {"score": 5, "type": "conspiracy", "warning": "Extreme conspiracy content"},
    "theonion.com": {"score": 50, "type": "satire", "warning": "Satirical content - not real news"},
    "babylonbee.com": {"score": 50, "type": "satire", "warning": "Satirical content - not real news"},
    "clickhole.com": {"score": 50, "type": "satire", "warning": "Satirical content - not real news"},
}

# Clickbait patterns
CLICKBAIT_PATTERNS = [
    r"you won't believe",
    r"this one trick",
    r"doctors hate",
    r"scientists hate",
    r"what happens next will shock",
    r"shocking truth",
    r"they don't want you to know",
    r"exposed!",
    r"breaking:",
    r"miracle cure",
    r"secret revealed",
    r"\d+ reasons why",
    r"number \d+ will shock",
]


class FakeNewsDetector:
    """
    Multi-modal fake news detection system
    """
    
    def __init__(self):
        self.text_classifier = None
        self.sentiment_analyzer = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_models()
        else:
            logger.warning("Running in fallback mode without ML models")
    
    def _load_models(self):
        """Load AI models for text classification"""
        try:
            # Load fake news classifier
            logger.info("📰 Loading Fake News Detection models...")
            
            # Use a reliable model for fake news classification
            model_name = "mrm8488/bert-mini-finetuned-age_news-classification"
            
            try:
                self.text_classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    device=-1  # CPU
                )
                logger.info(f"✅ Loaded text classifier: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {e}")
                # Fallback to sentiment analysis as proxy
                self.text_classifier = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1
                )
                logger.info("✅ Using sentiment analysis as fallback")
            
            # Load sentiment analyzer for bias detection
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
            logger.info("✅ Loaded sentiment analyzer")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.text_classifier = None
            self.sentiment_analyzer = None
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text for fake news indicators
        
        Args:
            text: Article text or headline
            
        Returns:
            Dict with analysis results
        """
        results = {
            "text_score": 50,  # Default neutral
            "clickbait_score": 0,
            "emotional_score": 50,
            "clickbait_detected": False,
            "emotional_manipulation": False,
            "analysis_details": []
        }
        
        if not text or len(text.strip()) < 10:
            results["analysis_details"].append({
                "type": "warning",
                "message": "Text too short for reliable analysis"
            })
            return results
        
        # 1. Clickbait Detection (rule-based)
        clickbait_matches = []
        text_lower = text.lower()
        for pattern in CLICKBAIT_PATTERNS:
            if re.search(pattern, text_lower):
                clickbait_matches.append(pattern)
        
        if clickbait_matches:
            results["clickbait_detected"] = True
            results["clickbait_score"] = min(100, len(clickbait_matches) * 25)
            results["analysis_details"].append({
                "type": "warning",
                "message": f"Clickbait patterns detected: {len(clickbait_matches)} matches"
            })
        
        # 2. ML-based text classification
        if self.text_classifier:
            try:
                # Truncate text for model
                truncated = text[:512]
                prediction = self.text_classifier(truncated)[0]
                
                # Map prediction to score
                label = prediction.get('label', '').lower()
                confidence = prediction.get('score', 0.5)
                
                if 'negative' in label or 'fake' in label:
                    results["text_score"] = max(10, (1 - confidence) * 100)
                elif 'positive' in label or 'real' in label:
                    results["text_score"] = min(90, confidence * 100)
                else:
                    results["text_score"] = 50  # Neutral
                    
            except Exception as e:
                logger.error(f"Text classification error: {e}")
        
        # 3. Sentiment/Emotional analysis
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(text[:512])[0]
                confidence = sentiment.get('score', 0.5)
                
                # Extreme sentiment can indicate manipulation
                if confidence > 0.9:
                    results["emotional_manipulation"] = True
                    results["emotional_score"] = int((1 - confidence) * 100)
                    results["analysis_details"].append({
                        "type": "info",
                        "message": f"High emotional language detected ({confidence*100:.0f}% intensity)"
                    })
                else:
                    results["emotional_score"] = 70
                    
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
        
        # 4. Additional heuristics
        # Check for excessive caps
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.3:
            results["analysis_details"].append({
                "type": "warning",
                "message": "Excessive use of capital letters (shouting)"
            })
            results["text_score"] = max(20, results["text_score"] - 15)
        
        # Check for excessive exclamation marks
        exclamation_count = text.count('!')
        if exclamation_count > 3:
            results["analysis_details"].append({
                "type": "warning",
                "message": f"Excessive exclamation marks ({exclamation_count})"
            })
            results["text_score"] = max(20, results["text_score"] - 10)
        
        return results
    
    def analyze_source(self, url: str) -> Dict:
        """
        Analyze source credibility based on domain
        
        Args:
            url: Article URL
            
        Returns:
            Dict with source analysis
        """
        results = {
            "source_score": 50,  # Default unknown
            "domain": None,
            "source_type": "unknown",
            "bias": "unknown",
            "is_known_credible": False,
            "is_known_unreliable": False,
            "warnings": []
        }
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www prefix
            if domain.startswith("www."):
                domain = domain[4:]
            
            results["domain"] = domain
            
            # Check credible sources
            if domain in CREDIBLE_SOURCES:
                source_info = CREDIBLE_SOURCES[domain]
                results["source_score"] = source_info["score"]
                results["bias"] = source_info.get("bias", "unknown")
                results["source_type"] = source_info.get("type", "news")
                results["is_known_credible"] = True
            
            # Check unreliable sources
            elif domain in UNRELIABLE_SOURCES:
                source_info = UNRELIABLE_SOURCES[domain]
                results["source_score"] = source_info["score"]
                results["source_type"] = source_info.get("type", "unknown")
                results["is_known_unreliable"] = True
                results["warnings"].append(source_info.get("warning", "Flagged source"))
            
            # Check for suspicious TLDs
            suspicious_tlds = ['.xyz', '.tk', '.ml', '.ga', '.cf']
            if any(domain.endswith(tld) for tld in suspicious_tlds):
                results["source_score"] = max(20, results["source_score"] - 30)
                results["warnings"].append("Suspicious domain extension")
            
            # Check for news-like domains that might be fake
            fake_patterns = ['news', 'daily', 'times', 'post', 'gazette']
            if any(p in domain for p in fake_patterns) and results["source_score"] == 50:
                results["warnings"].append("Unverified news-like domain")
                
        except Exception as e:
            logger.error(f"Source analysis error: {e}")
            results["warnings"].append("Could not analyze source URL")
        
        return results
    
    def predict(self, text: str = None, url: str = None, title: str = None) -> Dict:
        """
        Main prediction method - analyze content for fake news
        
        Args:
            text: Article body text
            url: Article URL
            title: Article headline/title
            
        Returns:
            Complete analysis results
        """
        # Combine title and text for analysis
        full_text = ""
        if title:
            full_text = title + "\n\n"
        if text:
            full_text += text
        
        # Analyze text content
        text_analysis = self.analyze_text(full_text) if full_text else {
            "text_score": 50,
            "clickbait_score": 0,
            "clickbait_detected": False,
            "emotional_manipulation": False,
            "analysis_details": []
        }
        
        # Analyze source
        source_analysis = self.analyze_source(url) if url else {
            "source_score": 50,
            "domain": None,
            "is_known_credible": False,
            "is_known_unreliable": False,
            "warnings": []
        }
        
        # Calculate weighted overall score
        # Text: 40%, Source: 40%, Clickbait penalty: 20%
        text_weight = 0.4
        source_weight = 0.4
        clickbait_penalty = 0.2
        
        base_score = (
            text_analysis["text_score"] * text_weight +
            source_analysis["source_score"] * source_weight
        )
        
        # Apply clickbait penalty
        if text_analysis["clickbait_detected"]:
            base_score -= text_analysis["clickbait_score"] * clickbait_penalty
        
        overall_score = max(0, min(100, base_score))
        
        # Determine verdict
        if overall_score >= 70:
            verdict = "LIKELY CREDIBLE"
            recommendation = "safe_to_share"
        elif overall_score >= 40:
            verdict = "UNCERTAIN"
            recommendation = "verify_first"
        else:
            verdict = "LIKELY MISINFORMATION"
            recommendation = "do_not_share"
        
        # Generate alerts
        alerts = self._generate_alerts(text_analysis, source_analysis, overall_score)
        
        return {
            "credibility_score": round(overall_score, 1),
            "verdict": verdict,
            "recommendation": recommendation,
            "is_likely_fake": overall_score < 40,
            "confidence": "high" if abs(overall_score - 50) > 30 else "medium" if abs(overall_score - 50) > 15 else "low",
            "text_analysis": {
                "score": round(text_analysis["text_score"], 1),
                "clickbait_detected": text_analysis["clickbait_detected"],
                "clickbait_score": text_analysis["clickbait_score"],
                "emotional_manipulation": text_analysis["emotional_manipulation"],
                "details": text_analysis["analysis_details"]
            },
            "source_analysis": {
                "score": round(source_analysis["source_score"], 1),
                "domain": source_analysis["domain"],
                "bias": source_analysis.get("bias", "unknown"),
                "is_credible": source_analysis["is_known_credible"],
                "is_unreliable": source_analysis["is_known_unreliable"],
                "warnings": source_analysis["warnings"]
            },
            "alerts": alerts
        }
    
    def _generate_alerts(self, text_analysis: Dict, source_analysis: Dict, overall_score: float) -> List[Dict]:
        """Generate user-friendly alerts"""
        alerts = []
        
        # Critical alerts
        if source_analysis["is_known_unreliable"]:
            alerts.append({
                "severity": "High",
                "title": "Unreliable Source",
                "icon": "🚨",
                "description": f"Domain {source_analysis['domain']} is flagged as unreliable"
            })
        
        if overall_score < 30:
            alerts.append({
                "severity": "High",
                "title": "High Misinformation Risk",
                "icon": "⚠️",
                "description": "Multiple indicators suggest this content may be false"
            })
        
        # Medium alerts
        if text_analysis["clickbait_detected"]:
            alerts.append({
                "severity": "Medium",
                "title": "Clickbait Detected",
                "icon": "🎣",
                "description": "Headline uses manipulative clickbait patterns"
            })
        
        if text_analysis["emotional_manipulation"]:
            alerts.append({
                "severity": "Medium",
                "title": "Emotional Manipulation",
                "icon": "😠",
                "description": "Content uses highly emotional language"
            })
        
        # Positive indicators
        if source_analysis["is_known_credible"]:
            alerts.append({
                "severity": "Low",
                "title": "Credible Source",
                "icon": "✅",
                "description": f"Published by {source_analysis['domain']} (verified credible)"
            })
        
        return alerts


# Initialize detector
try:
    fake_news_detector = FakeNewsDetector()
    logger.info("✅ Fake News Detector initialized")
except Exception as e:
    logger.error(f"Failed to initialize Fake News Detector: {e}")
    fake_news_detector = None

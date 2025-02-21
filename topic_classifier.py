from transformers import AutoTokenizer, pipeline
import json
from collections import defaultdict
import logging

# ✅ Load BART-Large Model for Efficient Classification
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1  # Run on CPU efficiently
)

# ✅ Load Keywords from `generated_keywords.json`
def load_keywords(file_path="generated_keywords.json"):
    """Load category-specific keywords from JSON file with UTF-8 encoding."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"❌ Keywords file not found: {file_path}")
        return {}

# ✅ Truncate Long Text to 512 Words
def truncate_text(text, max_words=512):
    """Ensure the text is not too long for processing."""
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

# ✅ Keyword Matching (Enhanced)
def match_keywords(text, keyword_dict, confidence_threshold=0.8):
    """Match extracted text with category-specific keywords."""
    matched_topics = defaultdict(float)
    text = truncate_text(text)  # Ensure text is not too long
    
    for category, keywords in keyword_dict.items():
        total_keywords = len(keywords.get("core", [])) + len(keywords.get("products", [])) + len(keywords.get("services", [])) + len(keywords.get("audience", [])) + len(keywords.get("context", []))
        
        if total_keywords == 0:
            continue
        
        match_count = sum(1 for kw in (keywords.get("core", []) + keywords.get("products", []) + keywords.get("services", []) + keywords.get("audience", []) + keywords.get("context", [])) if kw.lower() in text.lower())
        
        confidence = match_count / total_keywords
        if confidence >= confidence_threshold:
            matched_topics[category] = confidence
    
    return sorted(matched_topics.items(), key=lambda x: x[1], reverse=True)

# ✅ Zero-Shot Classification with Text Truncation
def zero_shot_classification(text, candidate_labels, confidence_threshold=0.7):
    """Classifies text using Zero-Shot Learning (BART model)."""
    text = truncate_text(text)  # Prevent memory overload
    if not text.strip() or not candidate_labels:
        return []
    
    result = classifier(text, candidate_labels, multi_label=True)
    
    return [(label, score) for label, score in zip(result['labels'], result['scores']) if score >= confidence_threshold]

# ✅ Combine Keyword Matching & Zero-Shot ML
def ensemble_classification(text, candidate_labels, keyword_dict, confidence_threshold=0.7):
    """Combines rule-based keyword matching and AI-based classification."""
    text = truncate_text(text)  # Prevent memory overload
    keyword_results = match_keywords(text, keyword_dict, confidence_threshold)
    ml_results = zero_shot_classification(text, candidate_labels, confidence_threshold)
    
    combined_results = defaultdict(float)
    
    for label, score in keyword_results + ml_results:
        combined_results[label] = max(combined_results[label], score)
    
    return sorted(combined_results.items(), key=lambda x: x[1], reverse=True)

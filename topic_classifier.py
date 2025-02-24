from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from collections import defaultdict
import logging

# ✅ Load RoBERTa Model for Efficient Classification
MODEL_NAME = "roberta-base"  # Switching from BERT to RoBERTa

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)  # Adjust based on number of categories
model.eval()

def split_text_into_chunks(text, max_tokens=512):
    """Splits long text into manageable chunks within RoBERTa's token limit."""
    words = text.split()
    return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]

def classify_text_roberta(text, candidate_labels, confidence_threshold=0.7):
    """Classifies text using fine-tuned RoBERTa, handling text chunks if necessary."""
    text_chunks = split_text_into_chunks(text)
    category_scores = defaultdict(float)
    num_labels = model.config.num_labels  # Ensure we don't exceed model's expected output size
    candidate_labels = candidate_labels[:num_labels]  # Limit candidate labels to avoid index errors
    
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        for i in range(min(len(scores[0]), len(candidate_labels))):  # Prevent index error
            category_scores[candidate_labels[i]] += scores[0][i].item()
    
    return sorted([(label, score) for label, score in category_scores.items() if score >= confidence_threshold], key=lambda x: x[1], reverse=True)

# ✅ Load Keywords from `generated_keywords.json`
def load_keywords(file_path="generated_keywords.json"):
    """Load category-specific keywords from JSON file with UTF-8 encoding."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"❌ Keywords file not found: {file_path}")
        return {}

# ✅ Keyword Matching (Enhanced)
def match_keywords(text, keyword_dict, confidence_threshold=0.8):
    """Match extracted text with category-specific keywords."""
    matched_topics = defaultdict(float)
    
    for category, keywords in keyword_dict.items():
        total_keywords = sum(len(keywords.get(k, [])) for k in ["core", "products", "services", "audience", "context"])
        
        if total_keywords == 0:
            continue
        
        match_count = sum(1 for kw in sum((keywords.get(k, []) for k in ["core", "products", "services", "audience", "context"]), []) if kw.lower() in text.lower())
        
        confidence = match_count / total_keywords
        if confidence >= confidence_threshold:
            matched_topics[category] = confidence
    
    return sorted(matched_topics.items(), key=lambda x: x[1], reverse=True)

# ✅ Combine Keyword Matching & RoBERTa Classification
def ensemble_classification(text, candidate_labels, keyword_dict, confidence_threshold=0.7):
    """Combines rule-based keyword matching and AI-based classification, handling long text."""
    keyword_results = match_keywords(text, keyword_dict, confidence_threshold)
    ml_results = classify_text_roberta(text, candidate_labels, confidence_threshold)
    
    combined_results = defaultdict(float)
    
    for label, score in keyword_results + ml_results:
        combined_results[label] = max(combined_results[label], score)
    
    return sorted(combined_results.items(), key=lambda x: x[1], reverse=True)

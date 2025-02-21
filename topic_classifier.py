from transformers import AutoTokenizer, pipeline
import json
from collections import defaultdict

# ✅ Load SentencePiece Tokenizer for DeBERTa
deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=False)

# ✅ Load Zero-Shot Classification Model
classifier = pipeline(
    "zero-shot-classification",
    model="microsoft/deberta-v3-large",
    device=0  # Change to -1 if running on CPU
)

# Ensure entailment mapping
classifier.model.config.label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

# ✅ Load Keywords from `generated_keywords.json`
def load_keywords(file_path="generated_keywords.json"):
    """Load category-specific keywords from JSON file with UTF-8 encoding."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# ✅ Keyword Matching (More Accurate Now)
def match_keywords(text, keyword_dict):
    matched_topics = []
    for category, keywords in keyword_dict.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            matched_topics.append((category, 1.0))  # High confidence for rule matches
    return matched_topics

# ✅ Zero-Shot Classification with Updated Tokenizer
def zero_shot_classification(text, candidate_labels):
    inputs = deberta_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    result = classifier(text, candidate_labels, multi_label=True)
    return list(zip(result['labels'], result['scores']))

# ✅ Combine Keyword Matching & Zero-Shot ML
def ensemble_classification(text, candidate_labels, keyword_dict, confidence_threshold=0.7):
    keyword_results = match_keywords(text, keyword_dict)
    ml_results = zero_shot_classification(text, candidate_labels)

    combined_results = defaultdict(float)
    for label, score in keyword_results + ml_results:
        combined_results[label] = max(combined_results[label], score)

    # ✅ Filter by Confidence Threshold
    return [(label, score) for label, score in combined_results.items() if score >= confidence_threshold]

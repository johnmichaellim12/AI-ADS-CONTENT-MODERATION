import re
from difflib import SequenceMatcher
from textblob import TextBlob
from transformers import pipeline, logging
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np

logging.set_verbosity_error()

# Global variables for caching
_model = None
_nn = None
_categories = None
_category_embeddings = None

def setup_fast_classifier(categories):
    # Load a smaller, faster model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Pre-compute embeddings for all categories
    category_embeddings = model.encode(categories)
    
    # Initialize nearest neighbor search
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn.fit(category_embeddings)
    
    return model, nn, category_embeddings

def get_classifier(categories):
    global _model, _nn, _categories, _category_embeddings
    
    # Only initialize if categories have changed or first run
    if _model is None or _categories != categories:
        _model, _nn, _category_embeddings = setup_fast_classifier(categories)
        _categories = categories.copy()
    
    return _model, _nn, _categories, _category_embeddings

def fast_classify_topic(text, model, nn, categories, category_embeddings):
    if not text.strip() or not categories:
        return "Unknown"
    
    # Get embedding for the input text
    text_embedding = model.encode([text])[0].reshape(1, -1)
    
    # Find nearest neighbor
    distances, indices = nn.kneighbors(text_embedding)
    
    # Get the closest category
    closest_idx = indices[0][0]
    
    return categories[closest_idx]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'\b(?:https?|ftp):\/\/\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)  # 🔥 Keep hyphens for words like "semi-automatic"
    
    print(f"🧼 Fixed Cleaned Text: {text}")  # 🔍 Debugging Output
    return text.strip()




def deduplicate_text(text):
    sentences = re.split(r'(?<=\.|\?|!)\s+', text)
    filtered_sentences = []
    seen_sentences = set()
    for sentence in sentences:
        sentence_cleaned = " ".join(sentence.split())
        if sentence_cleaned and sentence_cleaned not in seen_sentences:
            filtered_sentences.append(sentence_cleaned)
            seen_sentences.add(sentence_cleaned)
    return ". ".join(filtered_sentences)

def filter_prohibited_words(text, prohibited_words):
    text_tokens = set(re.findall(r'\b\w+\b', text.lower()))
    return [word for word in prohibited_words if word.lower() in text_tokens]

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def classify_topic(text, candidate_labels):
    if not candidate_labels:
        return "Unknown"
    if len(text.strip()) < 10:  # Avoid classifying very short texts
        return "Text too short for classification"
    
    try:
        model, nn, categories, category_embeddings = get_classifier(candidate_labels)
        return fast_classify_topic(text, model, nn, categories, category_embeddings)
    except Exception as e:
        print(f"Error during topic classification: {e}")
        import traceback
        traceback.print_exc()
        return "Classification Error"

def nlp_pipeline(text, prohibited_words, candidate_labels):
    if not text.strip():
        raise ValueError("Input text is empty.")
    
    print("Cleaning text...")
    cleaned_text = clean_text(text)
    print("Deduplicating text...")
    cleaned_text = deduplicate_text(cleaned_text)
    print("Filtering prohibited words...")
    flagged_words = filter_prohibited_words(cleaned_text, prohibited_words)
    print("Analyzing sentiment...")
    sentiment = analyze_sentiment(cleaned_text)
    
    print("Initializing fast classifier...")
    model, nn, category_embeddings = setup_fast_classifier(candidate_labels)
    
    print("Classifying topic...")
    topic = fast_classify_topic(cleaned_text, model, nn, candidate_labels, category_embeddings)
    
    print("NLP pipeline completed.")
    return {
        'cleaned_text': cleaned_text,
        'flagged_words': flagged_words,
        'sentiment': sentiment,
        'topic': topic
    }

import re  # Import regex module
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob  # For sentiment analysis
from transformers import pipeline  # For topic classification

def load_prohibited_words(file_path="prohibited_words.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        words = [line.strip().lower() for line in file.readlines()]
    return words

def clean_text(text):
    """Cleans text by removing special characters, punctuation, and excessive spaces."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(?:https?|ftp):\/\/\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def deduplicate_text(text):
    """Removes duplicate sentences while keeping original structure."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s+', text)
    filtered_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        sentence_cleaned = " ".join(sentence.split())
        if not sentence_cleaned:
            continue
        is_duplicate = any(
            SequenceMatcher(None, sentence_cleaned, seen).ratio() > 0.85 for seen in seen_sentences
        )
        if not is_duplicate:
            filtered_sentences.append(sentence_cleaned)
            seen_sentences.add(sentence_cleaned)
    return ". ".join(filtered_sentences)

def filter_prohibited_words(text, prohibited_words):
    """Detects if the extracted text contains prohibited words, ignoring case and punctuation."""
    # Clean and tokenize text
    text_tokens = set(re.findall(r'\b\w+\b', text.lower()))
    flagged_words = [word for word in prohibited_words if word.lower() in text_tokens]
    return flagged_words if flagged_words else []

def text_similarity(text1, text2):
    """Computes similarity between two texts using TF-IDF + Cosine Similarity."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_score[0][0]

def analyze_sentiment(text):
    """Performs sentiment analysis on the text using TextBlob."""
    return TextBlob(text).sentiment.polarity

def classify_topic(text, candidate_labels):
    """Classifies the text topic using HuggingFace Transformers zero-shot classification."""
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(text, candidate_labels)
    return result['labels'][0]

def nlp_pipeline(text, prohibited_words, candidate_labels):
    """Full NLP pipeline that cleans text, checks for prohibited words, analyzes sentiment, and classifies topic."""
    cleaned_text = clean_text(text)
    cleaned_text = deduplicate_text(cleaned_text)
    flagged_words = filter_prohibited_words(cleaned_text, prohibited_words)
    sentiment = analyze_sentiment(cleaned_text)
    topic = classify_topic(cleaned_text, candidate_labels)
    return {
        'cleaned_text': cleaned_text,
        'flagged_words': flagged_words,
        'sentiment': sentiment,
        'topic': topic
    }

import os
import logging
from ocr_module import extract_text_from_image, extract_text_from_video_parallel
from topic_classifier import load_keywords, ensemble_classification
from text_filter import nlp_pipeline
from utils import load_categories, load_prohibited_words, match_category

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    categories = load_categories()
    prohibited_words = load_prohibited_words()
    keyword_dict = load_keywords("generated_keywords.json")  # Load category-specific keywords
except FileNotFoundError as e:
    logging.error(f"Error loading files: {e}")
    exit(1)

def run_moderation():
    store_category = input("Enter the store category: ")

    # Static file paths for image and video
    image_path = "sample_data/483-Aco-Panhandle/Crosshairs Marianna Llc - Gun Shop/ZKlbmFWTwOtvWy7SC3xG_483ACO-PAN_36HCrosshairsMarianna_VT_Safe_1.jpg"
    video_path = "sample_data/483-Aco-Panhandle/Crosshairs Marianna Llc - Gun Shop/N-Compass TV Development IN.webm"

    if not categories:
        logging.error("No categories found. Please check categories.txt")
        return

    image_text = extract_text_from_image(image_path) if os.path.exists(image_path) else ""
    video_text = extract_text_from_video_parallel(video_path) if os.path.exists(video_path) else ""

    if not image_text and not video_text:
        logging.error("No text extracted from image or video.")
        return

    combined_text = image_text + " " + video_text

    try:
        # **Fix Topic Classification**
        classification_results = ensemble_classification(combined_text, categories, keyword_dict)
        
        # Extract the highest confidence category
        if classification_results:
            top_category, confidence = classification_results[0]
        else:
            top_category, confidence = "Unknown", 0.0

        nlp_result = nlp_pipeline(combined_text, prohibited_words, categories)

        # Print detailed NLP results
        print("\n=== NLP Analysis Results ===")
        print(f"Topic Classification: {top_category} (Confidence: {confidence:.2f})")
        print(f"Sentiment Score: {nlp_result['sentiment']:.2f}")
        print(f"Flagged Words: {', '.join(nlp_result['flagged_words']) if nlp_result['flagged_words'] else 'None'}")
        print(f"Text Length After Cleaning: {len(nlp_result['cleaned_text'])} characters")

        category_match = match_category(store_category, nlp_result['cleaned_text'], categories)
        print(f"\n=== Category Matching Result ===")
        print(category_match)

    except ValueError as e:
        logging.error(f"Error in NLP pipeline: {e}")

if __name__ == "__main__":
    run_moderation()

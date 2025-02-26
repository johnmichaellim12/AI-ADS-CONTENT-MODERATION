import os
import logging
from ocr_module import process_image, process_video
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

    image_data = process_image(image_path) if os.path.exists(image_path) else {"text": "", "objects": []}
    video_data = process_video(video_path) if os.path.exists(video_path) else {"text": "", "objects": []}

    combined_text = image_data["text"] + " " + video_data["text"]
    detected_objects = image_data["objects"] + video_data["objects"]



    if not combined_text and not detected_objects:
        logging.error("No text or objects detected in image or video.")
        return

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

        print("\n=== 🚦 Moderation Decision ===")

        # Example moderation rules (you can customize these)
        restricted_objects = ["Weapon", "Handgun", "Violence", "Nudity"]
        flagged_objects = [obj for obj in detected_objects if obj["label"] in restricted_objects and obj["confidence"] > 0.60]


        if flagged_objects:
            print("❌ Ad Rejected: Contains Restricted Content!")
            print("🚫 Flagged Objects:")
            for obj in flagged_objects:
                print(f"- {obj['label']} (Confidence: {obj['confidence']:.2f})")
        else:
            print("✅ Ad Approved: No restricted objects detected.")

        print(f"\n=== Category Matching Result ===")
        print(category_match)

        print("\n=== 🖼️ Object Detection Results ===")
        if detected_objects:
            for obj in detected_objects:
                print(f"- {obj['label']} (Confidence: {obj['confidence']:.2f}) - BBox: {obj['bbox']}")
        else:
            print("No objects detected.")


    except ValueError as e:
        logging.error(f"Error in NLP pipeline: {e}")

if __name__ == "__main__":
    run_moderation()

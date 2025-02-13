import os
from ocr_module import extract_text_from_image, extract_text_from_video_parallel
from text_filter import clean_text, deduplicate_text, filter_prohibited_words, load_prohibited_words, text_similarity
from utils import load_categories, match_category
import logging

# ✅ Load categories & prohibited words
categories = load_categories()
prohibited_words = load_prohibited_words()


# Run OCR
if __name__ == "__main__":

    # Ask user to input the store category
    store_category = input("🔹 Enter the store category: ")

    image_path = "sample_data/1664358838030.png"  # Replace with your test image
    video_path = "sample_data/483-Aco-Panhandle/Crosshairs Marianna Llc - Gun Shop/N-Compass TV Development IN.webm"  # Replace with your test video

    image_text, video_text = "", ""

    # **Ensure `image_text` is always defined**
    if os.path.exists(image_path):
        # print("\n🔹 Extracting text from image...")
        image_text = extract_text_from_image(image_path)
        # print("Extracted Text:", image_text)
    else:
        print(f"🚨 Error: Image file '{image_path}' not found.")

    # **Ensure `video_text` is always defined**
    if os.path.exists(video_path):
        # print("\n🔹 Extracting text from video...")
        video_text = extract_text_from_video_parallel(video_path)
        # print("Extracted Text:", video_text)
    else:
        print(f"🚨 Error: Video file '{video_path}' not found.")

    # 🔥 Merge extracted text from both sources
    merged_text = clean_text(image_text + " " + video_text)
    deduplicated_text = deduplicate_text(merged_text)
    print("\n🔥 Final Merged Extracted Text:", merged_text)

    # ✅ Check for NSFW & Restricted Content
    moderation_result = filter_prohibited_words(deduplicated_text, prohibited_words)
    print("\n🔹 Content Moderation Check:", moderation_result)

    # Optionally use text_similarity for advanced comparisons
    similarity_score = text_similarity(deduplicated_text, store_category)
    print("🔹 Text Similarity Score:", similarity_score)

    # ✅ Check Category Matching
    print("\n🔹 Checking category match...")
    result = match_category(store_category, deduplicated_text, categories)
    print(result)

logging.getLogger('ppocr').setLevel(logging.ERROR)
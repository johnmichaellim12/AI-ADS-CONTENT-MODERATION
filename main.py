import cv2
from paddleocr import PaddleOCR
import os
import multiprocessing
import re  # Import regex module
import time
from difflib import SequenceMatcher
import numpy as np
from collections import OrderedDict


# Load categories from file
def load_categories(file_path="categories.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines()]

# Match detected text with category
def match_category(store_category, ad_text, categories):
    """Check if the store category is valid and matches ad content."""
    # Validate store category
    if store_category not in categories:
        # Suggest similar categories if input is incorrect
        suggestions = [cat for cat in categories if store_category.lower() in cat.lower()]
        if suggestions:
            return f"🚨 '{store_category}' not found. Did you mean: {', '.join(suggestions)}?"
        return f"🚨 '{store_category}' not recognized. Please select a valid category."

    # Use regex to ensure category matches whole words
    for category in categories:
        if re.search(rf'\b{re.escape(category)}\b', ad_text, re.IGNORECASE):
            return f"✅ Ad content matches the store category: '{store_category}'."

    return f"⚠️ Potential mismatch: Store category '{store_category}', but ad content doesn't seem related. Do you want to proceed?"

# Initialize both models
ocr_fast = PaddleOCR(det_db_box_thresh=0.5, use_angle_cls=False, use_mp=True)  # Lightweight Model
ocr_full = PaddleOCR(use_angle_cls=True)  # Full Model

from difflib import SequenceMatcher
import re

def clean_text(text):
    """Smart text deduplication: Remove duplicate sentences while keeping original structure."""
    
    # Step 1: Split into sentences properly (handles cases like "Dr. Smith.")
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Step 2: Remove duplicate or nearly identical sentences
    filtered_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        sentence_cleaned = " ".join(sentence.split())  # Remove extra spaces
        if not sentence_cleaned:
            continue
        
        # Use a similarity threshold to prevent near-duplicate sentences
        is_duplicate = any(
            SequenceMatcher(None, sentence_cleaned, seen).ratio() > 0.85
            for seen in seen_sentences
        )

        if not is_duplicate:
            filtered_sentences.append(sentence_cleaned)
            seen_sentences.add(sentence_cleaned)

    return ". ".join(filtered_sentences)  # ✅ Keeps original sentence structure



def preprocess_image(image_path):
    """Enhances the image for better OCR accuracy."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        return None  # Return None if image couldn't be loaded

    img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)  # Scale up the image
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Reduce noise
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Improve contrast
    return img

def preprocess_frame(frame):
    """Enhances a video frame for better OCR accuracy."""
    if frame is None:
        return None  # Skip empty frames

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.resize(gray, (0, 0), fx=1.3, fy=1.3)  # Resize slightly instead of 1.5x
    _, processed_frame = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Improve contrast
    return processed_frame

def extract_text_from_image(image_path):
    """Uses Lightweight OCR first, then Full Model if text is missing, and tracks execution time."""
    start_time = time.time()  # Start timing
    
    # Try Lightweight Model First
    result = ocr_fast.ocr(image_path, cls=True)
    extracted_text = " ".join(word[1][0] for line in result if line for word in line) if result else ""

    if extracted_text:
        end_time = time.time()
        print(f"✅ Image OCR completed in {end_time - start_time:.2f} seconds")
        return extracted_text

    # If Lightweight Model fails, use Full Model
    print("⚠️ Low text detected, switching to Full Model for better accuracy...")
    result = ocr_full.ocr(image_path, cls=True)

    extracted_text = " ".join(word[1][0] for line in result if line for word in line) if result else "No text detected"

    end_time = time.time()
    print(f"✅ Image OCR completed in {end_time - start_time:.2f} seconds")
    
    return extracted_text

last_frame = None

#Helper function to detect scene changes:
def detect_scene_change(frame, last_frame, threshold=150):
    """Detects scene changes based on pixel differences."""
    if last_frame is None:
        return False, frame  # First frame, no comparison

    diff = cv2.absdiff(last_frame, frame)
    non_zero_count = cv2.countNonZero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))
    
    return non_zero_count > threshold, frame  # ✅ Correct placement

def process_frame_ocr(frame):
    """Applies Hybrid OCR on a single video frame, ensuring a valid return value."""
    processed_frame = preprocess_frame(frame)
    if processed_frame is None:
        return "No text detected"  # Ensure multiprocessing always receives valid data

    # Try Lightweight OCR First
    result = ocr_fast.ocr(processed_frame, cls=True)

    # Ensure OCR result is valid
    if not result or result == [[]] or result is None:
        return "No text detected"

    text = " ".join(word[1][0] for line in result if line for word in line)

    if text:  # If Lightweight Model finds text, return it
        return text

    # If Lightweight OCR fails, use Full Model
    print("⚠️ Low text detected in frame, switching to Full Model...")
    result = ocr_full.ocr(processed_frame, cls=True)

    # Ensure Full Model OCR result is valid
    if not result or result == [[]] or result is None:
        return "No text detected"

    return " ".join(word[1][0] for line in result if line for word in line)



def extract_text_from_video_parallel(video_path):
    """Extracts text from video frames using multiprocessing and tracks execution time."""
    start_time = time.time()  # Start timing
    
    cap = cv2.VideoCapture(video_path)
    extracted_texts = []
    last_frame = None
    frames_to_process = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        scene_changed, last_frame = detect_scene_change(frame, last_frame)
        if scene_changed and frame is not None:  # Ensure valid frame
            frames_to_process.append(frame)

    cap.release()

    if not frames_to_process:
        print("🚨 No valid frames detected for OCR.")
        return "No text detected"

    # ✅ Keep original order while removing duplicates
    unique_frames = list(OrderedDict.fromkeys(cv2.imencode('.jpg', frame)[1].tobytes() for frame in frames_to_process))
    frames_to_process = [cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR) for f in unique_frames]

    # ✅ Use multiprocessing to process OCR in parallel
    with multiprocessing.Pool(processes=4) as pool:  # Adjust pool size if needed
        results = pool.map(process_frame_ocr, frames_to_process)

    # ✅ Filter out empty or "No text detected" results
    extracted_texts = [text for text in results if text and text != "No text detected"]
    
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Calculate total execution time

    print(f"✅ Video OCR completed in {elapsed_time:.2f} seconds")  # Show execution time

    return " ".join(extracted_texts) if extracted_texts else "No text detected"



# Run OCR
if __name__ == "__main__":
    categories = load_categories()

    # Ask user to input the store category
    store_category = input("🔹 Enter the store category: ")

    image_path = "sample_data/483-Aco-Panhandle/Crosshairs Marianna Llc - Gun Shop/uzCzz3ryTTaVKnwiEblH_483ACO-PAN_36HCrosshairsMarianna_VT_Expert_1.jpg"  # Replace with your test image
    video_path = "sample_data/483-Aco-Panhandle/Crosshairs Marianna Llc - Gun Shop/N-Compass TV Development IN.webm"  # Replace with your test video

    # **Ensure `image_text` is always defined**
    if os.path.exists(image_path):
        print("\n🔹 Extracting text from image...")
        image_text = extract_text_from_image(image_path)
        print("Extracted Text:", image_text)
    else:
        print(f"🚨 Error: Image file '{image_path}' not found.")
        image_text = ""  # Set to empty string instead of `None`

    # **Ensure `video_text` is always defined**
    if os.path.exists(video_path):
        print("\n🔹 Extracting text from video...")
        video_text = extract_text_from_video_parallel(video_path)
        print("Extracted Text:", video_text)
    else:
        print(f"🚨 Error: Video file '{video_path}' not found.")
        video_text = ""  # Set to empty string instead of `None`


    # 🔥 Merge extracted text from both sources
    merged_text = clean_text(image_text + " " + video_text)
    print("\n🔥 Final Merged Extracted Text:", merged_text)

    print("\n🔹 Checking category match...")
    result = match_category(store_category, merged_text, categories)
    print(result)
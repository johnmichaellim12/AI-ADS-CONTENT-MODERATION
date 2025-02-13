import cv2
from paddleocr import PaddleOCR
import time
import numpy as np
from collections import OrderedDict
import multiprocessing


# Initialize both models
ocr_fast = PaddleOCR(det_db_box_thresh=0.5, use_angle_cls=False, use_mp=True)  # Lightweight Model
ocr_full = PaddleOCR(use_angle_cls=True)  # Full Model


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

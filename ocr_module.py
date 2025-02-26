import cv2
from paddleocr import PaddleOCR
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import OrderedDict
import multiprocessing
from ultralytics import YOLO
import easyocr
import pytesseract



# Initialize both models
ocr_model = PaddleOCR(det_db_box_thresh=0.5, use_angle_cls=True, use_mp=True)  # Lightweight Model
ocr_full = PaddleOCR(use_angle_cls=True)  # Full Model
easyocr_reader = easyocr.Reader(['en'])  # EasyOCR for artistic/stylized fonts

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ✅ Initialize YOLOv8 for Object Detection
import os
from ultralytics import YOLO

# Get the script's current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the full path dynamically
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train2", "weights", "best.pt")

# Load YOLO model
yolo_model = YOLO(MODEL_PATH)


def detect_objects(image):
    """Detect objects in an image using trained YOLO model."""
    results = yolo_model(image)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            detected_objects.append({
                "label": result.names[int(box.cls)],  # Object name
                "confidence": float(box.conf[0]),  # Confidence score
                "bbox": box.xyxy[0].tolist()  # Bounding box coordinates [x1, y1, x2, y2]
            })
    return detected_objects


def preprocess_image(image_path):
    """Enhances the image for better OCR accuracy."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    processed = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    return processed


def extract_text_from_image(image):
    """Uses multiple OCR models (PaddleOCR, EasyOCR, Tesseract) to extract text from file or image array."""
    
    # Check if the input is a file path (string) or a NumPy image (array)
    if isinstance(image, str):  # If it's a file path
        processed_image = preprocess_image(image)
    else:  # If it's an image array (video frame)
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    if processed_image is None:
        return "No text detected"
    
    # PaddleOCR
    result_paddle = ocr_model.ocr(image if isinstance(image, str) else processed_image, cls=True)
    text_paddle = " ".join(word[1][0] for line in result_paddle if line for word in line) if result_paddle else ""
    
    # EasyOCR
    result_easyocr = easyocr_reader.readtext(processed_image, detail=0)
    text_easyocr = " ".join(result_easyocr)
    
    # Tesseract
    text_tesseract = pytesseract.image_to_string(processed_image, lang='eng+spa+fra')
    
    # Combine results
    combined_text = " ".join(set([text_paddle, text_easyocr, text_tesseract])).strip()
    
    return combined_text if combined_text else "No text detected"


# last_frame = None

def process_image(image_path):
    """Process an image for OCR text extraction and trained YOLO object detection."""
    image = cv2.imread(image_path)
    text = extract_text_from_image(image_path)
    detected_objects = detect_objects(image)

    print(f"✅ Processed Image: {image_path}")
    print(f"📜 Extracted Text: {text}")
    print(f"🎯 Detected Objects: {detected_objects}")

    return {"text": text, "objects": detected_objects}



#Helper function to detect scene changes:
def detect_scene_change(frame1, frame2, threshold=0.5):
    """Detects scene changes using Structural Similarity Index (SSIM)."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    similarity = ssim(gray1, gray2)
    return similarity < threshold

def process_frame_ocr(frame):
    """Applies Hybrid OCR on a single video frame."""
    text = extract_text_from_image(frame)  # Now works with NumPy arrays!
    objects = detect_objects(frame)  # YOLO works with NumPy arrays
    return {"text": text, "objects": objects}



def extract_text_from_video_parallel(video_path):
    """Extracts text from video frames using multiprocessing and integrates YOLO detection."""
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    frames_to_process = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if not frames_to_process or detect_scene_change(frames_to_process[-1], frame):
            frames_to_process.append(frame)
    
    cap.release()

    if not frames_to_process:
        print("🚨 No valid frames detected for OCR.")
        return {"text": "No text detected", "objects": []}
    
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(process_frame_ocr, frames_to_process)
    
    extracted_texts = [res["text"] for res in results if res["text"] and res["text"] != "No text detected"]
    detected_objects = list(set(obj for res in results for obj in res["objects"]))
    
    end_time = time.time()
    print(f"✅ Video OCR & Object Detection completed in {end_time - start_time:.2f} seconds")
    
    return {"text": " ".join(extracted_texts), "objects": detected_objects}

def process_video(video_path):
    """Process a video for OCR text extraction and YOLO object detection."""
    ocr_results = extract_text_from_video_parallel(video_path)
    
    print(f"✅ Processed Video: {video_path}")
    print(f"📜 Extracted Text: {ocr_results['text']}")
    print(f"🎯 Detected Objects: {ocr_results['objects']}")

    return ocr_results


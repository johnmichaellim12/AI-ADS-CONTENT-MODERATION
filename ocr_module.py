import cv2
from paddleocr import PaddleOCR
import time
import numpy as np
from collections import OrderedDict
import multiprocessing
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim



# Initialize both models
ocr_model = PaddleOCR(det_db_box_thresh=0.5, use_angle_cls=True, use_mp=True)  # Lightweight Model
ocr_full = PaddleOCR(use_angle_cls=True)  # Full Model


# ✅ Initialize YOLOv8 for Object Detection
yolo_model = YOLO("yolov8.pt")  # Load pre-trained YOLOv8 model

def detect_objects(image_path):
    """Detect objects in an image using YOLOv8."""
    results = yolo_model(image_path)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            detected_objects.append(result.names[int(box.cls)])
    return detected_objects

# def preprocess_image(image_path):
#     """Enhances the image for better OCR accuracy."""
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
#     if img is None:
#         return None  # Return None if image couldn't be loaded

#     img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)  # Scale up the image
#     img = cv2.GaussianBlur(img, (5, 5), 0)  # Reduce noise
#     _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Improve contrast
#     return img

# def preprocess_frame(frame):
#     """Enhances a video frame for better OCR accuracy."""
#     if frame is None:
#         return None  # Skip empty frames

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     gray = cv2.resize(gray, (0, 0), fx=1.3, fy=1.3)  # Resize slightly instead of 1.5x
#     _, processed_frame = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Improve contrast
#     return processed_frame

def extract_text_from_image(image):
    """Uses Lightweight OCR first, then Full Model if text is missing."""
    result = ocr_model.ocr(image, cls=True)
    extracted_text = " ".join(word[1][0] for line in result if line for word in line) if result else ""
    
    if not extracted_text:
        result = ocr_full.ocr(image, cls=True)
        extracted_text = " ".join(word[1][0] for line in result if line for word in line) if result else "No text detected"
    
    return extracted_text

# last_frame = None

def process_image(image_path):
    """Process an image for both OCR text extraction and object detection."""
    image = cv2.imread(image_path)
    text = extract_text_from_image(image)
    objects = detect_objects(image)
    return {"text": text, "objects": objects}


#Helper function to detect scene changes:
def detect_scene_change(frame1, frame2, threshold=0.5):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    similarity = ssim(gray1, gray2)
    return similarity < threshold

def process_frame_ocr(frame):
    """Applies Hybrid OCR on a single video frame."""
    text = extract_text_from_image(frame)
    objects = detect_objects(frame)
    return {"text": text, "objects": objects}



def extract_text_from_video_parallel(video_path):
    """Extracts text from video frames using multiprocessing and integrates YOLO detection."""
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    extracted_data = []
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
    """Process a video for both OCR text extraction and object detection."""
    return extract_text_from_video_parallel(video_path)

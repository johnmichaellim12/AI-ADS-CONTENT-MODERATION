from ultralytics import YOLO
import sys
import json

# Load trained model
model = YOLO("dataset/runs/detect/train2/weights/best.pt")

def run_detection(image_path):
    results = model(image_path)
    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "class_id": int(box.cls[0]),  # Detected class ID
                "confidence": float(box.conf[0]),  # Confidence score
                "bbox": box.xyxy[0].tolist()  # Bounding box [x1, y1, x2, y2]
            })

    return json.dumps({"detections": detections})

if __name__ == "__main__":
    image_path = sys.argv[1]  # Get image path from Node.js
    print(run_detection(image_path))

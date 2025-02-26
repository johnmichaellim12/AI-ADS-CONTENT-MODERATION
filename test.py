import pandas as pd
import os
import shutil
from tqdm import tqdm

# Define paths
DATASET_PATH = "OpenImages"  # Change this to your dataset root folder
OUTPUT_PATH = "YOLO_Dataset"  # Output folder for YOLO-formatted dataset

# Create output directories
for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_PATH, "images", folder), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "labels", folder), exist_ok=True)

# Load class descriptions (Maps OpenImages ID → Class Name)
class_descriptions = pd.read_csv(os.path.join(DATASET_PATH, "class-descriptions-boxable.csv"), header=None)
class_descriptions.columns = ["LabelName", "ClassName"]
class_mapping = {row["LabelName"]: idx for idx, row in class_descriptions.iterrows()}  # YOLO class mapping

# Function to process each dataset (train, validation, test)
def process_annotations(csv_file, image_folder, output_folder):
    df = pd.read_csv(os.path.join(DATASET_PATH, csv_file))

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_file}"):
        image_id = row["ImageID"]
        class_id = class_mapping.get(row["LabelName"], -1)
        if class_id == -1:
            continue  # Skip unknown classes

        # Normalize bounding box values (OID uses absolute values)
        x_min, x_max = row["XMin"], row["XMax"]
        y_min, y_max = row["YMin"], row["YMax"]
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        # Save annotation in YOLO format
        label_path = os.path.join(OUTPUT_PATH, "labels", output_folder, f"{image_id}.txt")
        with open(label_path, "a") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        # Move image to the corresponding directory
        img_src = os.path.join(DATASET_PATH, image_folder, f"{image_id}.jpg")
        img_dest = os.path.join(OUTPUT_PATH, "images", output_folder, f"{image_id}.jpg")
        if os.path.exists(img_src):
            shutil.copy(img_src, img_dest)

# Convert training, validation, and test annotations
process_annotations("oidv6-train-annotations-bbox.csv", "train", "train")
process_annotations("validation-annotations-bbox.csv", "validation", "val")
process_annotations("test-annotations-bbox.csv", "test", "test")

print("✅ Open Images Dataset converted to YOLO format successfully!")

import os
import shutil
import random

# Define paths
image_dir = "dataset/images/train"
label_dir = "dataset/labels/train"

val_image_dir = "dataset/images/val"
val_label_dir = "dataset/labels/val"

os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# List all images
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
random.shuffle(image_files)

# Split (80% train, 20% val)
split_index = int(len(image_files) * 0.8)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Move validation files
for img_file in val_files:
    shutil.move(os.path.join(image_dir, img_file), val_image_dir)
    txt_file = os.path.splitext(img_file)[0] + ".txt"
    shutil.move(os.path.join(label_dir, txt_file), val_label_dir)

print("✅ Dataset split into train and val successfully!")

import os

# Define class mapping
class_mapping = {"Weapon": "0", "Handgun": "1"}  # Update with correct class names

# Paths to label folders
label_dirs = ["dataset/labels/train", "dataset/labels/val"]

for label_dir in label_dirs:
    for label_file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, label_file)

        # Read and fix the label file
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts[0] in class_mapping:  # Convert class name to index
                parts[0] = class_mapping[parts[0]]
                new_lines.append(" ".join(parts))

        # Overwrite the file with corrected data
        if new_lines:
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))

print("✅ Label files have been converted to YOLO format!")

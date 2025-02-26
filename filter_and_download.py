import os
import pandas as pd
from openimages.download import download_dataset

# 📌 Load Class Descriptions
class_desc_file = "dataset/class-descriptions-boxable.csv"
annotations_file = "dataset/oidv6-train-annotations-bbox.csv"  # Change if using validation/test

# 🟢 Define target categories (Modify as needed)
target_categories = ["Gun", "Firearm", "Store", "Logo", "Identity Document", "Weapon", "Retail"]

# ✅ Load Class Descriptions
class_descriptions = pd.read_csv(class_desc_file, header=None, names=["LabelName", "Category"])

# ✅ Extract LabelName IDs for selected categories
filtered_labels = class_descriptions[class_descriptions["Category"].isin(target_categories)]
label_ids = filtered_labels["LabelName"].tolist()

if not label_ids:
    print("\n❌ No matching categories found. Please check your category names.")
    exit()

print("\n✅ Extracted LabelName IDs:", label_ids)

# 📌 Filter Annotations
annotations = pd.read_csv(annotations_file)

# ✅ Keep only images containing selected categories
filtered_annotations = annotations[annotations["LabelName"].isin(label_ids)]
filtered_annotations.to_csv("dataset/filtered_annotations.csv", index=False)
print("\n✅ Filtered annotations saved.")

# 📌 Download Only Relevant Images
output_folder = "images/train"
os.makedirs(output_folder, exist_ok=True)

# ✅ Use OpenImages API to download images
print("\n⏳ Downloading images using OpenImages API...")
download_dataset(output_folder, classes=target_categories, image_limit=100)

print("\n🎉 Image Download Completed Successfully!")

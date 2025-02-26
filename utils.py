import re  # Import regex module
import logging

def load_prohibited_words(file_path="prohibited_words.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        words = [line.strip().lower() for line in file.readlines()]
    return words

# Load categories from file
def load_categories(file_path="categories.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            categories = [line.strip() for line in file.readlines() if line.strip()]
        if not categories:
            categories = ["General", "Retail", "Technology", "Other"]
            logging.warning("Using default categories as categories.txt was empty")
        print(f"📂 Loaded Categories: {categories}")  # Debugging Output
        return categories
    except FileNotFoundError:
        logging.error(f"Categories file not found: {file_path}")
        return ["General", "Retail", "Technology", "Other"]


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




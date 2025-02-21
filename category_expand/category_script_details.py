import json
import os
import sys
import traceback
import logging
import time
import re
import random
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keyword_generation.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables
load_dotenv()

# Validate API key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logging.error("❌ No OpenAI API key found. Please set OPENAI_API_KEY in your .env file.")
    sys.exit(1)

# Initialize OpenAI client
try:
    client = OpenAI(api_key=API_KEY)
except Exception as e:
    logging.error(f"❌ Failed to initialize OpenAI client: {e}")
    sys.exit(1)

# Configuration
OUTPUT_FILE = "generated_keywords.json"
REVISION_FILE = "revised_keywords.json"

def save_keywords(keywords, filename):
    """Save keywords to a file."""
    try:
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(keywords, f, indent=4, ensure_ascii=False)
        logging.info(f"Keywords saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save keywords: {e}")

def load_keywords():
    """Load existing keywords."""
    try:
        with open(OUTPUT_FILE, "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("❌ No existing keywords file found.")
        return {}
    except json.JSONDecodeError:
        logging.error("❌ Keywords file is corrupted.")
        return {}

def parse_json_safely(text):
    """Attempt to parse JSON with multiple fallback strategies."""
    # Remove any code block markers
    text = text.replace('```json', '').replace('```', '').strip()
    
    # Try direct parsing
    try:
        parsed = json.loads(text)
        # Validate structure
        if all(key in parsed for key in ["core", "products", "services", "audience", "context"]):
            return parsed
    except (json.JSONDecodeError, KeyError):
        pass
    
    # Try extracting JSON between first { and last }
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            if all(key in parsed for key in ["core", "products", "services", "audience", "context"]):
                return parsed
    except:
        pass
    
    # Fallback for malformed responses
    return {
        "core": [],
        "products": [],
        "services": [],
        "audience": [],
        "context": [],
        "error": "Failed to parse JSON",
        "raw_output": text
    }

def generate_keywords_revision(category, existing_keywords):
    """Generate revised and expanded keywords."""
    for attempt in range(3):
        try:
            # Random delay to be kind to the API
            time.sleep(random.uniform(2, 5))
            
            # Detailed prompt for keyword revision
            prompt = f"""
            Review and enhance the existing keywords for the category '{category}'.
            
            Existing Keywords:
            {json.dumps(existing_keywords, indent=2)}

            Revise the keywords by:
            1. Identifying any missing crucial keywords
            2. Adding more specific or niche subcategories
            3. Expanding the context and potential variations
            4. Ensuring comprehensive coverage of the category

            Provide an updated JSON with:
            - Additional keywords in each section
            - More depth and breadth
            - Unique insights not in the original list

            Return the JSON in the same format:
            {{
              "core": [],
              "products": [],
              "services": [],
              "audience": [],
              "context": []
            }}
            """
            
            # API call
            response = client.chat.completions.create(
                model="gpt-4o",  # Can switch to gpt-3.5-turbo if needed
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,  # Slightly higher temperature for creativity
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            # Parse and return the response
            output = response.choices[0].message.content.strip()
            revised_keywords = parse_json_safely(output)
            
            # Merge with existing keywords
            merged_keywords = merge_keywords(existing_keywords, revised_keywords)
            return merged_keywords
        
        except Exception as e:
            logging.warning(f"Revision attempt {attempt+1} failed for {category}: {e}")
            
            # Handle rate limiting
            if "rate limit" in str(e).lower():
                wait_time = random.uniform(30, 90)
                logging.info(f"Rate limited. Waiting for {wait_time:.1f}s")
                time.sleep(wait_time)
            
            # Last attempt handling
            if attempt == 2:
                logging.error(f"Failed to revise keywords for {category}")
                return existing_keywords

def merge_keywords(original, revised):
    """
    Merge original and revised keywords, removing duplicates.
    Prioritize adding new, unique keywords.
    """
    merged = {}
    for key in ["core", "products", "services", "audience", "context"]:
        # Combine lists and remove duplicates while preserving order
        combined = original.get(key, []) + revised.get(key, [])
        merged[key] = list(dict.fromkeys(combined))
    
    return merged

def main():
    """Keyword revision process."""
    # Load existing keywords
    existing_keywords = load_keywords()
    
    if not existing_keywords:
        logging.error("No keywords to revise. Generate keywords first.")
        return
    
    # Prepare for revision
    revised_keywords = {}
    total_categories = len(existing_keywords)
    
    logging.info(f"Starting revision for {total_categories} categories")
    
    # Revise keywords for each category
    for category, keywords in tqdm(existing_keywords.items(), desc="Revising Keywords"):
        try:
            # Skip error entries
            if isinstance(keywords, dict) and keywords.get('error'):
                logging.warning(f"Skipping {category} due to previous error")
                revised_keywords[category] = keywords
                continue
            
            # Generate revised keywords
            revised_entry = generate_keywords_revision(category, keywords)
            revised_keywords[category] = revised_entry
            
            # Save progress periodically
            if len(revised_keywords) % 10 == 0:
                save_keywords(revised_keywords, REVISION_FILE)
        
        except Exception as e:
            logging.error(f"Unexpected error revising {category}: {e}")
            traceback.print_exc()
            revised_keywords[category] = keywords
    
    # Final save
    save_keywords(revised_keywords, REVISION_FILE)
    logging.info("✅ Keyword revision complete!")

if __name__ == "__main__":
    main()
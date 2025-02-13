import os

def remove_duplicates_from_file(file_path, output_file=None):
    """Removes duplicate lines from a file while keeping the original order."""
    
    if not os.path.exists(file_path):
        print(f"🚨 Error: File '{file_path}' not found!")
        return
    
    seen = set()
    unique_lines = []

    # Read and filter out duplicates
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_cleaned = line.strip()  # Remove extra spaces/newlines
            if line_cleaned and line_cleaned not in seen:
                seen.add(line_cleaned)
                unique_lines.append(line_cleaned + "\n")  # Keep proper newlines

    # If no output file is specified, overwrite the original file
    output_file = output_file if output_file else file_path

    # Write unique lines back to the file
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.writelines(unique_lines)

    print(f"✅ Removed duplicates in '{file_path}'. Saved to '{output_file}'. Total unique lines: {len(unique_lines)}")

# Example Usage:
remove_duplicates_from_file("prohibited_words.txt")  # Overwrites the original file
# remove_duplicates_from_file("prohibited_words.txt", "cleaned_prohibited_words.txt")  # Saves to a new file

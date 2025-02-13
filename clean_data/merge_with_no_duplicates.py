import os

def merge_and_sort_text_files(output_file, *input_files):
    """Merge multiple text files into one and sort the content while keeping duplicates."""
    merged_lines = []

    # Read all lines from input files
    for file in input_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                merged_lines.extend(f.readlines())  # Append all lines
            print(f"✅ Merged '{file}'.")
        else:
            print(f"🚨 Warning: File '{file}' not found!")

    # Sort the lines alphabetically while keeping duplicates
    merged_lines.sort()

    # Write sorted data to output file
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.writelines(merged_lines)

    print(f"\n✅ Finished merging and sorting {len(input_files)} files into '{output_file}'.")

# Example Usage:
merge_and_sort_text_files("prohibited_words.txt", "cleaned_data_fb.csv", "cleaned_data_yt.csv", "cleaned_data_wp.csv", "en.txt", "nsfw_words.txt", "british-swear-words-list_text-file.txt")

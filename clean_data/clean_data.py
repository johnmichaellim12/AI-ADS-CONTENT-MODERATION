import pandas as pd

# Specify the correct file path
file_path = r"C:\Users\NCTV_User_002\Desktop\pagsulay\clean_data\wordpress-comment-blacklist-words-list_text-file_2021-01-19.txt"  

# Read the file and handle errors
try:
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()  # Read all lines

    # Process each line, split by commas, and clean data
    words = [word.strip() for line in lines for word in line.split(",")]

    # Create DataFrame
    df = pd.DataFrame(words, columns=["Bad_Words"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Save cleaned data
    df.to_csv("cleaned_data_wp.csv", index=False, header=False, encoding="utf-8")

    print("Data cleaned and saved to 'cleaned_data_wp.csv'")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")

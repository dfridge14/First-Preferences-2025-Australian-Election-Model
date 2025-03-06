import zipfile
import os

# Set your folder paths
zip_folder = "C:\\Dania\\ABS downloads"  # Replace with your actual folder
extract_to = "C:\\Dania\\ABS downloads"  # Replace with your desired output folder

# Ensure the output folder exists
os.makedirs(extract_to, exist_ok=True)

# Loop through all .zip files in the folder
for file in os.listdir(zip_folder):
    if file.endswith(".zip"):  # Only process zip files
        zip_path = os.path.join(zip_folder, file)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)  # Extract all files into the target folder

print("All ZIP files extracted successfully!")
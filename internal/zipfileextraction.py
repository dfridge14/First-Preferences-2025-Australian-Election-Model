import zipfile
import os
from pathlib import Path


# Set your folder paths


base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)

zip_folder = f"{base_dir}\\ABS downloads for Similarity\\Electorates" if os.name == "nt" else f"{base_dir}/ABS downloads for Similarity/Electorates"  # zip folder
extract_to = f"{base_dir}\\ABS downloads for Similarity\\Electorates" if os.name == "nt" else f"{base_dir}/ABS downloads for Similarity/Electorates"  # output folder

import pdb;pdb.set_trace()



# Ensure the output folder exists
os.makedirs(extract_to, exist_ok=True)

# Loop through all .zip files in the folder
for file in os.listdir(zip_folder):
    import pdb;pdb.set_trace()
    if file.endswith(".zip"):  # Only process zip files
        zip_path = os.path.join(zip_folder, file)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)  # Extract all files into the target folder

print("All ZIP files extracted successfully!")
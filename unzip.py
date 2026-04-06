import zipfile
from tqdm import tqdm
import os

def unzip_with_progress(zip_path, extract_path='.'):
    """
    Unzip a file with a progress bar showing extraction progress.
    
    Args:
        zip_path: Path to the zip file
        extract_path: Directory to extract files to (default: current directory)
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        
        # Create progress bar
        for file in tqdm(file_list, desc="Extracting", unit="file"):
            zip_ref.extract(file, extract_path)

if __name__ == "__main__":
    zip_file = "/home/locth/omni2rect/dataset_fisheye.zip"  # Replace with your zip file path
    output_dir = "/home/locth/omni2rect/"  # Replace with desired output directory
    
    os.makedirs(output_dir, exist_ok=True)
    unzip_with_progress(zip_file, output_dir)
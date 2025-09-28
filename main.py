#Import datasets
import kagglehub
import os

from matplotlib import pyplot as plt
from engine import LicensePlateOCR
from processor import DatasetProcessor
import cv2
from tester import test_ocr_engine

# Dataset
path = kagglehub.dataset_download("topkek69/vietnamese-license-plate-ocr")
print("Path to dataset files:", path)
print(os.listdir(path))

# Initialize dataset processor
processor = DatasetProcessor(path)

# Initialize OCR
ocr = LicensePlateOCR(ocr_engine='easyocr')

new_path = '/kaggle/input/vietnamese-license-plate-ocr/generated'
# Test with the first image from the Kaggle dataset
test_files = [f for f in os.listdir(new_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if test_files:
    # Load first image
    first_image = test_files[1]
    image_path = os.path.join(new_path, first_image)
    plate_image = cv2.imread(image_path)

    # Display original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Show preprocessed image
    processed = ocr.preprocess_plate(plate_image)
    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title('Preprocessed Image')
    plt.axis('off')
    plt.show()

    # Recognize text
    result = ocr.recognize_plate(plate_image)
    print(f"Recognized text: '{result}'")
else:
    print("No images found in the dataset.")
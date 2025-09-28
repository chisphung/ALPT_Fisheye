import os
import cv2
import kagglehub as kh
from kagglehub import utils
from engine import LicensePlateOCR

#Dataset
path = kh.dataset_download("topkek69/vietnamese-license-plate-ocr")

def test_ocr_engine():
    # Initialize OCR engine
    ocr = LicensePlateOCR(ocr_engine='easyocr')
    path = kh.dataset_download("topkek69/vietnamese-license-plate-ocr")

    # Test with sample images
    test_images_path = path

    for image_file in os.listdir(test_images_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load image
            image_path = os.path.join(test_images_path, image_file)
            plate_image = cv2.imread(image_path)

            if plate_image is not None:
                # Recognize text
                result = ocr.recognize_plate(plate_image)
                print(f"Image: {image_file} -> Text: '{result}'")

if __name__ == "__main__":
    test_ocr_engine()


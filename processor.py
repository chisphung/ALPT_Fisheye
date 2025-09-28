import os
import json
from typing import List, Dict, Optional
from engine import LicensePlateOCR
from processor import DatasetProcessor
import cv2

class DatasetProcessor:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.ocr = LicensePlateOCR(ocr_engine='easyocr')

    def process_dataset(self) -> List[Dict]:
        """Process entire dataset and return results"""
        results = []

        for image_file in os.listdir(self.dataset_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.dataset_path, image_file)
                plate_image = cv2.imread(image_path)

                if plate_image is not None:
                    recognized_text = self.ocr.recognize_plate(plate_image)

                    result = {
                        'image_filename': image_file,
                        'recognized_text': recognized_text,
                        'characters': list(recognized_text.replace(' ', ''))
                    }
                    results.append(result)
                    print(f"Processed: {image_file} -> {recognized_text}")

        return results

# Usage
if __name__ == "__main__":
    processor = DatasetProcessor(path)
    results = processor.process_dataset()
    print(f"Processed {len(results)} images")
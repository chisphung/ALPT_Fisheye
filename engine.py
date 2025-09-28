"""e
Vietnamese License Plate OCR Engine
Integrates with existing YOLO detection pipeline
"""
from PIL import Image
import easyocr 
import pytesseract
import cv2
import numpy as np
from typing import List, Dict, Optional

class LicensePlateOCR:
    def __init__(self, ocr_engine = 'easyocr'):
        self.ocr_engine = ocr_engine
        if ocr_engine == 'easyocr':
            self.reader = easyocr.Reader(['vi', 'en'])

    def preprocess_license_plate(image_path):
        # Read image
        img = cv2.imread(image_path)
        
        # Resize if too small
        height, width = img.shape[:2]
        if height < 100 or width < 200:
            scale = max(100/height, 200/width)
            img = cv2.resize(img, None, fx=scale, fy=scale, 
                            interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Threshold
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        
        # Clean up
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_easyOCR(self, plate_image):
        """Extract text using EasyOCR"""
        results = self.reader.readtext(plate_image)
        # Check confidence score, filter, and combine results
        text_results = []
        for (bbox, text, confidence) in results:
            if confidence > 0.6: #Set threshold to 0.6
                text_results.append(text)
        return ' '.join(text_results)
    
    def extract_text_tesseract(self, plate_image):
        """Extract text using Tesseract"""
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
        
        # Custom config for license plates
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        text = pytesseract.image_to_string(pil_image, config=custom_config)
        return text.strip()
    
    def recognize_plate(self, plate_image):
        """Main method to recognize license plate text"""
        # Preprocess the image
        processed_image = self.preprocess_plate(plate_image)
        
        if self.ocr_engine == 'easyocr':
            return self.extract_text_easyOCR(processed_image)
        else:
            return self.extract_text_tesseract(processed_image)
        pass
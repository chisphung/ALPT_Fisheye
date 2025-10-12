import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from model.yolo.ultralytics.models.yolo.model import YOLO
from utils.rectify import rectify_all_plates
import cv2


class PlateDetection:
    def __init__(self, weights_path=None, image_folder_path=None, output_folder="../output", use_obb=True):
        """
        Initialize plate detection model.
        
        Args:
            weights_path: Path to YOLO weights
            image_folder_path: Path to input images
            output_folder: Path to save cropped plates
            use_obb: Whether model is OBB (oriented bounding box) or regular detection
        """
        if weights_path is None:
            weights_path = "weight/recognition_v11m.pt"
        if image_folder_path is None:
            image_folder_path = "dataset/test"
        self.output_folder = output_folder
        self.detection_model = YOLO(weights_path)
        self.image_folder_path = image_folder_path
        self.use_obb = use_obb

    def detect_plates(self, image: np.ndarray, conf_threshold=0.25, iou_threshold=0.45):
        """
        Detect plates in image.
        
        Returns:
            YOLO result object (not a list of tuples)
        """
        results = self.detection_model(image, conf=conf_threshold, iou=iou_threshold)
        return results[0]  # Return first result object

    def crop_plate(self, image: np.ndarray, box: tuple) -> np.ndarray:
        """Crop plate using bounding box (for non-OBB models)."""
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]

    def extract_plates(self, image: np.ndarray, result, pad: int = 5):
        """
        Extract plates from detection result.
        
        Args:
            image: Source image
            result: YOLO result object
            pad: Padding for OBB rectification
            
        Returns:
            List of cropped/rectified plate images
        """
        plates = []
        
        if self.use_obb and result.obb is not None and len(result.obb) > 0:
            # OBB detection - rectify using 4 corners from rectify.py
            plates = rectify_all_plates(image, result, pad=pad)
        elif result.boxes is not None and len(result.boxes) > 0:
            # Regular detection - simple crop
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plates.append(self.crop_plate(image, (x1, y1, x2, y2)))
        
        return plates

    def detect_from_folder(self, conf_threshold=0.25, iou_threshold=0.45, pad: int = 5) -> None:
        """
        Detect and extract plates from all images in folder.
        
        Args:
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold for NMS
            pad: Padding for OBB rectification
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
        for img_name in os.listdir(self.image_folder_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(self.image_folder_path, img_name)
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"Failed to read: {img_path}")
                    continue
                
                # Detect plates
                result = self.detect_plates(image, conf_threshold, iou_threshold)
                
                # Extract plates (OBB rectification or regular crop)
                plates = self.extract_plates(image, result, pad=pad)
                
                # Save plates
                for idx, plate in enumerate(plates):
                    plate_filename = f"{os.path.splitext(img_name)[0]}_plate_{idx}.png"
                    output_path = os.path.join(self.output_folder, plate_filename)
                    cv2.imwrite(output_path, plate)
                    
                print(f"Processed {img_name}: found {len(plates)} plate(s)")
    
    
if __name__ == "__main__":
    detector = PlateDetection(weights_path="/home/chisphung/ALPR_Fisheye/weight/obb_transfer.pt", image_folder_path="/home/chisphung/ALPR_Fisheye/dataset/test/images")
    detector.detect_from_folder(conf_threshold=0.25, iou_threshold=0.45)
    print("Detection completed. Check the output folder for results.")
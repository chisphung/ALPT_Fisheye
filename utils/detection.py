import numpy as np
import os
from typing import List
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from model.yolo.ultralytics.models.yolo.model import YOLO
import cv2


class PlateDetection:
    def __init__(self, weights_path=None, image_folder_path=None, output_folder="output"):
        if weights_path is None:
            weights_path = "weight/recognition_v11m.pt"
        if image_folder_path is None:
            image_folder_path = "dataset/test"
        self.output_folder = output_folder
        self.detection_model = YOLO(weights_path)
        self.image_folder_path = image_folder_path

    def detect_plates(self, image: np.ndarray, conf_threshold=0.25, iou_threshold=0.45) -> List:
        results = self.detection_model(image)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] >= conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    detections.append((cls_id, conf, (x1, y1, x2, y2)))
        return detections
    
    def crop_plate(self, image: np.ndarray, box: tuple) -> np.ndarray:
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]

    def detect_from_folder(self, conf_threshold=0.25, iou_threshold=0.45) -> None:
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        for img_name in os.listdir(self.image_folder_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(self.image_folder_path, img_name)
                image = cv2.imread(img_path)
                detections = self.detect_plates(image, conf_threshold, iou_threshold)
                cropped_plates = [self.crop_plate(image, box) for _, _, box in detections]
                for idx, plate in enumerate(cropped_plates):
                    plate_filename = f"{os.path.splitext(img_name)[0]}_plate_{idx}.png"
                    cv2.imwrite(os.path.join(self.output_folder, plate_filename), plate)
    
    
if __name__ == "__main__":
    detector = PlateDetection(weights_path="weight/recognition_v11m.pt", image_folder_path="dataset/test")
    detector.detect_from_folder(conf_threshold=0.25, iou_threshold=0.45)
    print("Detection completed. Check the output folder for results.")
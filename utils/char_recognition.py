import numpy as np
import cv2
import os
from typing import List
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from model.yolo.ultralytics.models.yolo.model import YOLO
import math


class CharRecognition:
    def __init__(self, weights_path=None, image_folder_path=None):
        if weights_path is None:
            weights_path = "weight/Charcter-LP.pt"
        if image_folder_path is None:
            image_folder_path = "output"
        self.detection_model = YOLO(weights_path)
        self.image_folder_path = image_folder_path

    def linear_equation(self, x1, y1, x2, y2):
        b = y1 - (y2 - y1) * x1 / (x2 - x1)
        a = (y1 - b) / x1
        return a, b

    def check_point_linear(self, x, y, x1, y1, x2, y2):
        a, b = self.linear_equation(x1, y1, x2, y2)
        y_pred = a * x + b
        return math.isclose(y_pred, y, abs_tol=3)

    # main plate decoding
    def decode_plate(self, yolo_model, im):
        results = yolo_model(im)
        r = results[0]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            return "unknown"

        # class-id → label map
        names = getattr(r, "names", None)
        if not names:
            names = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
                    10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',
                    19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',
                    28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z'}

        # collect centers + class ids
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        center_list = []
        y_sum = 0
        for bb, cid in zip(xyxy, cls_ids):
            x_c = (bb[0] + bb[2]) / 2
            y_c = (bb[1] + bb[3]) / 2
            y_sum += y_c
            center_list.append([x_c, y_c, names.get(cid, "?")])
        # print("Centers:", center_list)
        # if len(center_list) < 7 or len(center_list) > 10:
        #     return "unknown"
        
        # check if 2-line plate
        LP_type = "1"
        l_point = min(center_list, key=lambda c: c[0])
        r_point = max(center_list, key=lambda c: c[0])
        for ct in center_list:
            if l_point[0] != r_point[0]:
                if not self.check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]):
                    LP_type = "2"

        y_mean = y_sum / len(center_list)

        # build plate string
        license_plate = ""
        if LP_type == "2":
            line_1 = [c for c in center_list if c[1] <= y_mean]
            line_2 = [c for c in center_list if c[1] > y_mean]

            for l1 in sorted(line_1, key=lambda x: x[0]):
                license_plate += l1[2]
            license_plate += "-"
            for l2 in sorted(line_2, key=lambda x: x[0]):
                license_plate += l2[2]
        else:
            for l in sorted(center_list, key=lambda x: x[0]):
                license_plate += l[2]

        return license_plate

    def recognize_from_folder(self, input_folder: str) -> List[str]:
        recognized_plates = []
        for img_name in os.listdir(input_folder):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(input_folder, img_name)
                image = cv2.imread(img_path)
                plate_text = self.decode_plate(self.detection_model, image)
                recognized_plates.append(plate_text)
        return recognized_plates

if __name__ == "__main__":
    char_recognition = CharRecognition()
    recognized_plates = char_recognition.recognize_from_folder("output")
    for plate in recognized_plates:
        print(f"Recognized Plate: {plate}")

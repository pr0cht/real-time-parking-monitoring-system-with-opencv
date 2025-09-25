import cv2
import xml.etree.ElementTree as ET
import os
import numpy as np
import easyocr

class LicensePlateDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.plates_data = self.load_training_data()
        
    def load_training_data(self):
        plates_data = []
        annotations_path = "licenseplates/annotations"
        images_path = "licenseplates/images"
        
        for xml_file in os.listdir(annotations_path):
            if xml_file.endswith('.xml'):
                tree = ET.parse(os.path.join(annotations_path, xml_file))
                root = tree.getroot()
                
                # Get image path
                image_file = root.find('filename').text
                image_path = os.path.join(images_path, image_file)
                
                # Get bounding box
                for obj in root.findall('.//object'):
                    if obj.find('name').text == 'licence':
                        bbox = obj.find('bndbox')
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)
                        
                        plates_data.append({
                            'image': cv2.imread(image_path),
                            'bbox': (xmin, ymin, xmax, ymax)
                        })
        
        return plates_data
    
    def detect_plate(self, frame):
        best_match = None
        best_score = float('inf')
        plate_text = None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try to match with training data
        for plate_data in self.plates_data:
            template = cv2.cvtColor(plate_data['image'], cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            
            if score < best_score:
                best_score = score
                best_match = plate_data['bbox']
        
        if best_score < 0.8:  # Threshold for matching
            x1, y1, x2, y2 = best_match
            plate_roi = frame[y1:y2, x1:x2]
            
            # Use EasyOCR to read the plate
            results = self.reader.readtext(plate_roi)
            if results:
                plate_text = results[0][1]  # Get the text from first result
                
            return True, (x1, y1, x2, y2), plate_text
            
        return False, None, None

    def process_frame(self, frame):
        detected, bbox, plate_text = self.detect_plate(frame)
        
        if detected:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if plate_text:
                cv2.putText(frame, plate_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
        return frame, plate_text
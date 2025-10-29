# license_plate_detector.py
import cv2
import easyocr
import numpy as np
import re

class LicensePlateDetector:
    def __init__(self, cascade_path='haarcascade_russian_plate_number.xml'):
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise IOError(f"Failed to load Haar Cascade from {cascade_path}. Make sure the file is in the correct directory.")
            
        self.reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have a compatible NVIDIA GPU

        # --- NEW: Regex to keep only uppercase letters and numbers ---
        self.char_filter = re.compile("[^A-Z0-9]")

    def __preprocess_roi(self, roi):
        """
        Applies image processing to the cropped plate to improve OCR accuracy.
        A low-quality camera feed needs this boost.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to create a clean, high-contrast binary image
        # This is excellent for handling shadows and uneven lighting.
        thresh = cv2.adaptiveThreshold(gray, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, # Use INV to get black text on white background
                                       11, # Size of the pixel neighborhood
                                       2)  # Constant subtracted from the mean
        
        # You could also add blurring here if the image is very noisy
        # e.g., median = cv2.medianBlur(thresh, 3)
        # return median
        
        return thresh

    def __clean_text(self, text):
        """
        Cleans the raw OCR output to a standardized format.
        """
        if not text:
            return None
            
        # Remove all non-alphanumeric characters
        cleaned_text = self.char_filter.sub('', text)
        
        # --- NEW: Add a length filter ---
        # This filters out random noise like 'I' or 'T' that OCR might see.
        # Adjust 4 and 8 based on Philippine plate formats (e.g., LLL NNN, LL NNNN)
        if 4 <= len(cleaned_text) <= 8:
            return cleaned_text
        
        return None

def detect_and_read(self, frame):
        plate_text = None
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- MODIFIED: Increased minNeighbors to reduce false positives ---
        # Try 10. If it still finds text, try 15.
        # If it stops finding the real plate, you may need a value in between.
        plates = self.cascade.detectMultiScale(gray_frame, 
                                               scaleFactor=1.1, 
                                               minNeighbors=10, # Was 4
                                               minSize=(40, 40))
        
        if len(plates) > 0:
            x, y, w, h = plates[0]
            
            x1 = max(0, x - 5)
            y1 = max(0, y - 5)
            x2 = min(frame.shape[1], x + w + 5)
            y2 = min(frame.shape[0], y + h + 5)
            
            plate_roi = frame[y1:y2, x1:x2]

            if plate_roi.size > 0:
                processed_roi = self.__preprocess_roi(plate_roi)
                
                ocr_result = self.reader.readtext(processed_roi, 
                                                  detail=0, 
                                                  allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                
                if ocr_result:
                    raw_text = "".join(ocr_result).strip().upper()
                    plate_text = self.__clean_text(raw_text)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            label = plate_text if plate_text else "Detecting..."
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        return frame, plate_text
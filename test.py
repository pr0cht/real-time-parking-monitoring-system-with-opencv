import cv2
import time

def test_camera():
    # Try to release any existing camera connections first
    for i in range(5):
        temp_cap = cv2.VideoCapture(0)
        temp_cap.release()
    
    # Wait a moment
    time.sleep(1)
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    print("Camera opened successfully")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        cap.release()
        return
    
    print("Successfully read frame")
    
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Test', frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Properly release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released")

if __name__ == "__main__":
    test_camera()
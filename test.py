import cv2
import pygame
import numpy as np 
import dearpygui.dearpygui as dpg
import time  
import datetime
import easyocr
from license_plate_detector import LicensePlateDetector
import json

pygame.mixer.init()
obstructed_sound = pygame.mixer.Sound("beep_obs.wav") # Sound for obstruction
vacant_sound = pygame.mixer.Sound("beep_vac.wav")   # Sound for vacancy when a car leaves

car_cascade = cv2.CascadeClassifier("cars.xml") # Load Haar Cascade for car detection

# First create context
dpg.create_context()

# Only create the status/log window in DearPyGUI
with dpg.window(label="Parking Status", width=400, height=500, pos=[0, 0], no_close=True):
    dpg.add_text("Parking 01: Vacant", tag="parking1_status")
    dpg.add_text("Parking 02: Vacant", tag="parking2_status")
    dpg.add_text("Parking 03: Vacant", tag="parking3_status")
    dpg.add_text("Parking 04: Vacant", tag="parking4_status")
    dpg.add_text("Parking 05: Vacant", tag="parking5_status")
    dpg.add_separator()
    dpg.add_text("Parking Log:", tag="log_label")
    dpg.add_text("", tag="parking_log")
    dpg.add_separator()
    dpg.add_text("Detection Parameters")
    dpg.add_slider_float(label="Scale Factor", tag="scale_factor", default_value=1.010, min_value=1.01, max_value=1.5, format="%.3f")
    dpg.add_slider_int(label="Min Neighbors", tag="min_neighbors", default_value=3, min_value=1, max_value=10)
    dpg.add_slider_int(label="History", tag="history", default_value=1000, min_value=100, max_value=2000)
    dpg.add_slider_int(label="VarThreshold", tag="var_threshold", default_value=300, min_value=10, max_value=1000)
    dpg.add_slider_int(label="Obstruction %", tag="obstruction_percent", default_value=30, min_value=5, max_value=90)
    dpg.add_slider_int(label="Stabilization Frames", tag="stabilization_frames", default_value=15, min_value=1, max_value=60)
    dpg.add_separator()
    dpg.add_text("Camera Controls")
    dpg.add_slider_float(label="Brightness", tag="cam_brightness", default_value=0.5, min_value=0.0, max_value=1.0)
    dpg.add_slider_float(label="Contrast", tag="cam_contrast", default_value=0.5, min_value=0.0, max_value=1.0)
    dpg.add_slider_float(label="Saturation", tag="cam_saturation", default_value=0.5, min_value=0.0, max_value=1.0)
    dpg.add_separator()
    dpg.add_text("Vehicle Entry/Exit Log:", tag="vehicle_log_label")
    dpg.add_text("", tag="vehicle_log")
    dpg.add_separator()
    dpg.add_text("Keyboard Controls:")
    dpg.add_text("Press 'S' - Set/Update Reference Frame")
    dpg.add_text("Press 'Q' - Quit Application")

dpg.create_viewport(title="Parking System", width=420, height=520)
dpg.setup_dearpygui()
dpg.show_viewport()

# Camera setup as before
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get frame dimensions
ret, test_frame = cap.read()
if ret:
    height, width = test_frame.shape[:2]
    print(f"Camera frame dimensions: {width}x{height}")
else:
    print("Error: Could not read test frame")
    cap.release()
    dpg.destroy_context()
    exit()

# Replace the existing parking area definitions with this
def calculate_parking_areas(frame_width, frame_height):
    # Calculate dimensions that fit within the frame
    parking_width = int(frame_width * 0.15)  # 15% of frame width
    parking_height = int(frame_height * 0.3)  # 30% of frame height
    
    # Start position for first parking space
    start_x = int(frame_width * 0.05)  # 5% margin from left
    start_y = int(frame_height * 0.2)  # 20% from top
    
    # Space between parking areas
    space_between = int(parking_width * 0.1)  # 10% of parking width
    
    parking_areas = []
    for i in range(5):
        start = (start_x + (parking_width + space_between) * i, start_y)
        end = (start[0] + parking_width, start[1] + parking_height)
        status_tag = f"parking{i+1}_status"
        parking_areas.append((start, end, status_tag))
    
    return parking_areas

# Initialize frame dimensions and parking areas
ret, init_frame = cap.read()
if not ret:
    print("Error: Could not read initial frame from camera")
    exit()

frame_height, frame_width = init_frame.shape[:2]
print(f"Camera frame dimensions: {frame_width}x{frame_height}")

# Calculate parking areas based on frame size
parking_areas = calculate_parking_areas(frame_width, frame_height)

# Remove the old parking area definitions
# (Delete all the parking1_start through parking5_end variables)

# Initialize dictionaries
previous_status = {status_tag: "Vacant" for _, _, status_tag in parking_areas}
detection_counters = {status_tag: 0 for _, _, status_tag in parking_areas}
status_history = {status_tag: [] for _, _, status_tag in parking_areas}
history_length = 15  # Length of history to keep for each parking area

detection_threshold = 10  # Threshold for detecting a car in the parking area

# Initialize stabilization counters for each parking area
stabilization_counters = {status_tag: {"Occupied": 0, "Obstructed": 0, "Vacant": 0} for _, _, status_tag in parking_areas}
stabilization_threshold = 15  # Number of consecutive frames required to confirm a status change

print("Press 's' anytime to set/update the reference frame. Press 'q' to quit.")

# Reference frame for detecting changes
ref_frame = None

# Initialize dictionaries for parking duration tracking
parking_start_time = {status_tag: None for _, _, status_tag in parking_areas}
parking_end_time = {status_tag: None for _, _, status_tag in parking_areas}
delay_counters = {status_tag: 0 for _, _, status_tag in parking_areas}  # Delay mechanism   

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=300, detectShadows=False)

log_messages = []

# Initialize license plate detector
plate_detector = LicensePlateDetector()

# Dictionary to track vehicles
vehicles = {}

# First, modify the global variables section before the main loop
frame = None
entry_frame = None
exit_frame = None
ref_frame = None
frame_queue = []
last_key_pressed = None  # Add this to track key presses

# Add these functions after calculate_parking_areas()
def draw_parking_areas(frame, areas):
    for i, (start, end, _) in enumerate(areas):
        cv2.rectangle(frame, start, end, (0, 255, 0), 2)
        cv2.putText(frame, f"P{i+1}", (start[0], start[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def process_parking_areas(frame, ref_frame, areas):
    if ref_frame is None:
        return frame

    for start, end, status_tag in areas:
        # Draw ROI rectangle
        cv2.rectangle(frame, start, end, (0, 255, 0), 2)
        
        # Process parking area
        parking = frame[start[1]:end[1], start[0]:end[0]]
        ref_parking = ref_frame[start[1]:end[1], start[0]:end[0]]
        
        # Use background subtraction
        fg_mask = bg_subtractor.apply(parking)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresh)
        
        # Calculate obstruction percentage
        total_pixels = parking.shape[0] * parking.shape[1]
        obstruction_percent = (white_pixels / total_pixels) * 100
        
        # Update status based on obstruction
        if obstruction_percent > dpg.get_value("obstruction_percent"):
            status = "Occupied"
            color = (0, 0, 255)  # Red for occupied
        else:
            status = "Vacant"
            color = (0, 255, 0)  # Green for vacant
            
        # Update GUI status
        dpg.set_value(status_tag, f"Parking {status_tag[-2:]}: {status}")

        # Draw status text on the frame
        text_pos = (start[0], start[1] - 15)
        cv2.putText(frame, status, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame

def process_parking_lot(frame, ref_frame, parking_areas):
    statuses = []
    for i, (start, end, status_tag) in enumerate(parking_areas):
        # Draw ROI rectangle
        cv2.rectangle(frame, start, end, (0, 255, 0), 2)
        parking_roi = frame[start[1]:end[1], start[0]:end[0]]
        ref_roi = ref_frame[start[1]:end[1], start[0]:end[0]] if ref_frame is not None else None

        # Only process if reference frame is set
        if ref_roi is not None and parking_roi.shape == ref_roi.shape:
            fg_mask = bg_subtractor.apply(parking_roi)
            _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            white_pixels = cv2.countNonZero(thresh)
            total_pixels = parking_roi.shape[0] * parking_roi.shape[1]
            obstruction_percent = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0

            status = "Occupied" if obstruction_percent > dpg.get_value("obstruction_percent") else "Vacant"
            color = (0, 0, 255) if status == "Occupied" else (0, 255, 0)
        else:
            status = "Unknown"
            color = (0, 255, 255)

        # Draw status text on the frame
        text_pos = (start[0], start[1] - 10)
        cv2.putText(frame, f"P{i+1}: {status}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        statuses.append((status_tag, status))
    return frame, statuses

# Key handler for reference frame
with dpg.handler_registry():
    def key_handler(sender, app_data):
        global ref_frame, frame
        if app_data == 83 or app_data == 115:  # S or s
            if frame is not None:
                ref_frame = frame.copy()
                print("Reference frame updated.")
                log_messages.append("Reference frame updated at " + 
                    datetime.datetime.now().strftime("%H:%M:%S"))
                dpg.set_value("parking_log", "\n".join(log_messages[-10:]))
        elif app_data == 81 or app_data == 113:  # Q or q
            dpg.stop_dearpygui()
    dpg.add_key_press_handler(callback=key_handler)

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read camera frame")
        time.sleep(0.033)
        continue

    # Update reference frame on 's' key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        ref_frame = frame.copy()
        print("Reference frame updated.")
        log_messages.append("Reference frame updated at " + datetime.datetime.now().strftime("%H:%M:%S"))
        dpg.set_value("parking_log", "\n".join(log_messages[-10:]))

    # Process parking lot and update statuses
    processed_frame, statuses = process_parking_lot(frame.copy(), ref_frame, parking_areas)
    for status_tag, status in statuses:
        dpg.set_value(status_tag, f"{status_tag.replace('_status','').replace('parking','Parking ')}: {status}")

    # Show parking camera feed in its own OS window
    cv2.imshow("Parking Camera Feed", processed_frame)

    # For entry/exit feeds, just show the raw frame (or process as needed)
    cv2.imshow("Entry Camera Feed", frame)
    cv2.imshow("Exit Camera Feed", frame)

    dpg.render_dearpygui_frame()
    time.sleep(0.033)

cap.release()
cv2.destroyAllWindows()
dpg.destroy_context()
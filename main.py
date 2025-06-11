#backup purposes

import cv2
import pygame
import numpy as np
import dearpygui.dearpygui as dpg
import time  # Import time for tracking parking duration

# Initialize pygame for sounds
pygame.mixer.init()
obstructed_sound = pygame.mixer.Sound("beep_obs.wav")
vacant_sound = pygame.mixer.Sound("beep_vac.wav")

# Load Haar Cascade for car detection
car_cascade = cv2.CascadeClassifier("cars.xml")

# Initialize Dear PyGui
dpg.create_context()

with dpg.window(label="Parking Status", width=400, height=200):
    dpg.add_text("Parking 01: Vacant", tag="parking1_status")
    dpg.add_text("Parking 02: Vacant", tag="parking2_status")
    dpg.add_text("Parking 03: Vacant", tag="parking3_status")
    dpg.add_text("Parking 04: Vacant", tag="parking4_status")
    dpg.add_text("Parking 05: Vacant", tag="parking5_status")

dpg.create_viewport(title="Parking Scanner UI", width=600, height=300)
dpg.setup_dearpygui()
dpg.show_viewport()

# Initialize camera
cap = cv2.VideoCapture(0) # Camera Index
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1680)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Define parking1 dimensions first
parking1_start = (50, 100)
parking1_end = (250, 400)  # Example size: 200x200

# Define parking area size based on parking1
parking_width = parking1_end[0] - parking1_start[0]
parking_height = parking1_end[1] - parking1_start[1]

# Define parking areas aligned from left to right
parking2_start = (parking1_end[0] + 20, parking1_start[1])
parking2_end = (parking2_start[0] + parking_width, parking2_start[1] + parking_height)

parking3_start = (parking2_end[0] + 20, parking2_start[1])
parking3_end = (parking3_start[0] + parking_width, parking3_start[1] + parking_height)

parking4_start = (parking3_end[0] + 20, parking3_start[1])
parking4_end = (parking4_start[0] + parking_width, parking4_start[1] + parking_height)

parking5_start = (parking4_end[0] + 20, parking4_start[1])
parking5_end = (parking5_start[0] + parking_width, parking5_start[1] + parking_height)

# Define parking areas
parking_areas = [
    (parking1_start, parking1_end, "parking1_status"),
    (parking2_start, parking2_end, "parking2_status"),
    (parking3_start, parking3_end, "parking3_status"),
    (parking4_start, parking4_end, "parking4_status"),
    (parking5_start, parking5_end, "parking5_status"),
]

# Initialize dictionaries
previous_status = {status_tag: "Vacant" for _, _, status_tag in parking_areas}
detection_counters = {status_tag: 0 for _, _, status_tag in parking_areas}
status_history = {status_tag: [] for _, _, status_tag in parking_areas}
history_length = 15  # Length of history to keep for each parking area

detection_threshold = 10  # Threshold for detecting a car in the parking area

# Initialize stabilization counters for each parking area
stabilization_counters = {status_tag: {"Occupied": 0, "Obstructed": 0, "Vacant": 0} for _, _, status_tag in parking_areas}
stabilization_threshold = 12  # Number of consecutive frames required to confirm a status change

print("Press 's' anytime to set/update the reference frame. Press 'q' to quit.")

# Reference frame for detecting changes
ref_frame = None

# Initialize dictionaries for parking duration tracking
parking_start_time = {status_tag: None for _, _, status_tag in parking_areas}
parking_end_time = {status_tag: None for _, _, status_tag in parking_areas}
delay_counters = {status_tag: 0 for _, _, status_tag in parking_areas}  # Delay mechanism   

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=250, detectShadows=False)

while dpg.is_dearpygui_running():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from the camera.")
        break

    key = cv2.waitKey(1) & 0xFF

    # Set or update the reference frame
    if key == ord('s'):
        ref_frame = frame.copy()
        print("Reference frame updated.")
        continue

    # Skip processing if reference frame is not set
    if ref_frame is None:
        cv2.imshow("Parking Scanner", frame)
        if key == ord('q'):
            break
        dpg.render_dearpygui_frame()
        continue

    # Process each parking area
    for i, (start, end, status_tag) in enumerate(parking_areas, start=1):
        parking = frame[start[1]:end[1], start[0]:end[0]]

        # Detect cars in the parking area
        gray_parking = cv2.cvtColor(parking, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray_parking, scaleFactor=1.015, minNeighbors=5, minSize=(50, 50))

        # Use background subtraction for obstruction detection
        fg_mask = bg_subtractor.apply(parking)

        # Display the foreground mask for testing
        cv2.imshow(f"Foreground Mask - Parking {i:02}", fg_mask)

        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresh)

        total_pixels = thresh.shape[0] * thresh.shape[1]
        percent_change = (white_pixels / total_pixels) * 100

        # Determine the current status
        current_status = "Vacant"
        color = (0, 255, 0)

        if len(cars) > 0:
            current_status = "Occupied"
            color = (0, 0, 255)
        elif percent_change > 30:  # Increase sensitivity threshold
            current_status = "Obstructed"
            color = (0, 165, 255)

        # Update stabilization counters
        for status in stabilization_counters[status_tag]:
            if status == current_status:
                stabilization_counters[status_tag][status] += 1
            else:
                stabilization_counters[status_tag][status] = max(0, stabilization_counters[status_tag][status] - 1)

        # Confirm the status change only if it persists for the stabilization threshold
        if stabilization_counters[status_tag][current_status] >= stabilization_threshold:
            if previous_status[status_tag] != current_status:
                # Log parking duration when switching from "Occupied" to "Vacant"
                if previous_status[status_tag] == "Occupied" and current_status == "Vacant":
                    parking_end_time[status_tag] = time.time()
                    duration = parking_end_time[status_tag] - parking_start_time[status_tag]
                    print(f"Parking {i:02} was occupied for {duration:.2f} seconds.")

                # Start tracking time when switching to "Occupied"
                if current_status == "Occupied":
                    parking_start_time[status_tag] = time.time()

                previous_status[status_tag] = current_status

                # Play sound only when the status changes
                pygame.mixer.stop()
                if current_status == "Occupied":
                    pass  # No sound for "Occupied"
                elif current_status == "Obstructed":
                    obstructed_sound.play()
                elif current_status == "Vacant":
                    vacant_sound.play()

        # Update Dear PyGui UI
        dpg.set_value(status_tag, f"Parking {i:02}: {previous_status[status_tag]}")

        # Draw rectangles and text on the frame
        cv2.rectangle(frame, start, end, color, 2)
        cv2.putText(frame, previous_status[status_tag], (start[0], start[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display the original frame
    cv2.imshow("Parking Scanner", frame)

    # Exit on 'q' key press
    if key == ord('q'):
        break

    dpg.render_dearpygui_frame()

cap.release()
cv2.destroyAllWindows()
dpg.destroy_context()
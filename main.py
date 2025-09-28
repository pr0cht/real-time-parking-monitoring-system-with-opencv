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

dpg.create_context() # Initialize UI window
with dpg.window(label="Parking Status", width=400, height=500):
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
dpg.create_viewport(title="Parking Scanner UI", width=600, height=300)
dpg.setup_dearpygui()
dpg.show_viewport()

# Initialize single camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Get initial frame to check dimensions
ret, test_frame = cap.read()
if ret:
    height, width = test_frame.shape[:2]
    print(f"Camera frame dimensions: {width}x{height}")
else:
    print("Error: Could not read initial frame")
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

while dpg.is_dearpygui_running():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from the camera.")
        break

    # Create copies for different views
    parking_frame = frame.copy()
    entry_frame = frame.copy()
    exit_frame = frame.copy()

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Set or update the reference frame
        ref_frame = parking_frame.copy()
        print("Reference frame updated.")
        continue

    if ref_frame is None:     # Skip processing if reference frame is not set
        cv2.imshow("Parking Scanner", parking_frame)
        if key == ord('q'):
            break
        dpg.render_dearpygui_frame()
        continue

    # Get current parameter values from GUI
    scale_factor = dpg.get_value("scale_factor")
    min_neighbors = dpg.get_value("min_neighbors")
    history = dpg.get_value("history")
    var_threshold = dpg.get_value("var_threshold")
    obstruction_percent = dpg.get_value("obstruction_percent")
    stabilization_threshold = dpg.get_value("stabilization_frames")

    # Get camera control values from GUI
    cam_brightness = dpg.get_value("cam_brightness")
    cam_contrast = dpg.get_value("cam_contrast")
    cam_saturation = dpg.get_value("cam_saturation")

    # Set camera properties
    cap.set(cv2.CAP_PROP_BRIGHTNESS, cam_brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, cam_contrast)
    cap.set(cv2.CAP_PROP_SATURATION, cam_saturation)

    # Update background subtractor if history or varThreshold changed
    if (bg_subtractor.getHistory() != history) or (bg_subtractor.getVarThreshold() != var_threshold):
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False
        )

    # Modify the parking area processing loop
    for i, (start, end, status_tag) in enumerate(parking_areas, start=1):
        # Add bounds checking
        start_x = max(0, min(start[0], frame.shape[1]-1))
        start_y = max(0, min(start[1], frame.shape[0]-1))
        end_x = max(0, min(end[0], frame.shape[1]-1))
        end_y = max(0, min(end[1], frame.shape[0]-1))
        
        # Skip if invalid dimensions
        if start_x >= end_x or start_y >= end_y:
            print(f"Warning: Invalid dimensions for parking area {i}")
            continue
        
        try:
            parking = frame[start_y:end_y, start_x:end_x]
            if parking.size == 0:
                print(f"Warning: Empty parking area {i}")
                continue
                
            gray_parking = cv2.cvtColor(parking, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(
                gray_parking,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )

            # Use background subtraction for obstruction detection
            fg_mask = bg_subtractor.apply(parking)

            _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            white_pixels = cv2.countNonZero(thresh)

            total_pixels = thresh.shape[0] * thresh.shape[1]
            percent_change = (white_pixels / total_pixels) * 100

            # Use adjustable obstruction threshold
            current_status = "Vacant"
            color = (0, 255, 0)
            if len(cars) > 0:
                current_status = "Occupied"
                color = (0, 0, 255)
            elif percent_change > obstruction_percent:
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
                        if duration >= 10:  # Only log if occupied for 10 seconds or more for guaranteed parking 
                            # Convert duration to h:m:s
                            hours = int(duration // 3600)
                            minutes = int((duration % 3600) // 60)
                            seconds = int(duration % 60)
                            log_entry = f"Parking {i:02} was occupied for {hours:02}:{minutes:02}:{seconds:02}."
                            print(log_entry)
                            log_messages.append(log_entry)
                            # Keep only the last 10 log entries
                            if len(log_messages) > 10:
                                log_messages.pop(0)
                            dpg.set_value("parking_log", "\n".join(log_messages))
                            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                            with open(f"parking_log_{date_str}.txt", "a") as logfile:
                                logfile.write(log_entry + "\n")
                        else:
                            pass

                    # Start tracking time when switching to "Occupied"
                    if current_status == "Occupied":
                        parking_start_time[status_tag] = time.time()

                    previous_status[status_tag] = current_status

                    # Play sound only when the status changes
                    pygame.mixer.stop()
                    if current_status == "Occupied":
                        pass  
                    elif current_status == "Obstructed":
                        obstructed_sound.play()
                    elif current_status == "Vacant":
                        vacant_sound.play()

            display_text = f"Parking {i:02}: {previous_status[status_tag]}"
            if previous_status[status_tag] == "Occupied" and parking_start_time[status_tag] is not None:
                elapsed = int(time.time() - parking_start_time[status_tag])
                hours = elapsed // 3600
                minutes = (elapsed % 3600) // 60
                seconds = elapsed % 60
                display_text += f" [{hours:02}:{minutes:02}:{seconds:02}]"
            dpg.set_value(status_tag, display_text)

            cv2.rectangle(frame, start, end, color, 2)
            cv2.putText(frame, previous_status[status_tag], (start[0], start[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except cv2.error as e:
            print(f"Error processing parking area {i}: {e}")
            continue

    cv2.imshow("Parking Scanner", frame)

    # Process entry view (using the same frame)
    if entry_frame is not None:
        entry_frame, plate_text = plate_detector.process_frame(entry_frame)
        if plate_text and plate_text not in vehicles:
            vehicles[plate_text] = {
                "entry_time": datetime.datetime.now(),
                "exit_time": None
            }
            log_entry = f"Vehicle {plate_text} entered at {vehicles[plate_text]['entry_time'].strftime('%H:%M:%S')}"
            log_messages.append(log_entry)
        
        cv2.imshow("Entry Camera", entry_frame)

    # Process exit view (using the same frame)
    if exit_frame is not None:
        exit_frame, plate_text = plate_detector.process_frame(exit_frame)
        if plate_text and plate_text in vehicles and vehicles[plate_text]["exit_time"] is None:
            vehicles[plate_text]["exit_time"] = datetime.datetime.now()
            duration = vehicles[plate_text]["exit_time"] - vehicles[plate_text]["entry_time"]
            log_entry = f"Vehicle {plate_text} exited at {vehicles[plate_text]['exit_time'].strftime('%H:%M:%S')} (Duration: {duration})"
            log_messages.append(log_entry)
        
        cv2.imshow("Exit Camera", exit_frame)

    # Update vehicle log display
    if log_messages:
        dpg.set_value("vehicle_log", "\n".join(log_messages[-10:]))  # Show last 10 logs

    # Save vehicle data to JSON file
    with open("vehicle_log.json", "w") as f:
        json_data = {k: {
            "entry_time": v["entry_time"].strftime("%Y-%m-%d %H:%M:%S") if v["entry_time"] else None,
            "exit_time": v["exit_time"].strftime("%Y-%m-%d %H:%M:%S") if v["exit_time"] else None
        } for k, v in vehicles.items()}
        json.dump(json_data, f, indent=4)

    if key == ord('q'):
        break

    dpg.render_dearpygui_frame()

cap.release()
cv2.destroyAllWindows()
dpg.destroy_context()
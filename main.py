# main.py
import cv2
import dearpygui.dearpygui as dpg
import time
import datetime
from threading import Thread, Event
from queue import Queue
from ultralytics import YOLO
import easyocr
import re
import torch
import os
import atexit # For saving logs on exit

# --- Pygame for Sound ---
import pygame
try:
    pygame.mixer.init()
    entry_sound = pygame.mixer.Sound("entry_beep.wav")
    exit_sound = pygame.mixer.Sound("exit_beep.wav")
    sound_enabled = True
except Exception as e:
    print(f"[Warning] Failed to initialize sound or load WAV files: {e}. Sound disabled.")
    sound_enabled = False

# --- Global Configuration & State ---
stop_event = Event()
MIN_SCAN_DURATION = 2.0  # seconds
RESET_COOLDOWN = 1.0
FRAME_SKIP_INTERVAL = 5

# --- YOLO and OCR Initialization ---
try:
    yolo_model = YOLO('best.pt') # Your custom license plate model
    device_to_use = 0 if torch.cuda.is_available() else 'cpu' # Force GPU if available
    print(f"Using device: {device_to_use}")
except Exception as e:
    print(f"Error initializing YOLO: {e}. Ensure 'best.pt' is present.")
    exit() # Exit if model can't load

try:
     ocr_reader = easyocr.Reader(['en'], gpu=(device_to_use == 0)) # Use GPU if available
except Exception as e:
     print(f"Error initializing EasyOCR on GPU: {e}. Falling back to CPU.")
     ocr_reader = easyocr.Reader(['en'], gpu=False)

# --- Plate Scanning State ---
scan_state = {
    "entry": {"plate": None, "first_seen": 0, "last_seen": 0, "logged": False, "frame_count": 0, "last_detected_plate": None},
    "exit": {"plate": None, "first_seen": 0, "last_seen": 0, "logged": False, "frame_count": 0, "last_detected_plate": None}
}

# --- Logging Data Structures ---
entry_log_messages = []
exit_log_messages = []
duration_log_messages = []
vehicles_inside = {} # Store plate -> entry_datetime

# --- Parking State ---
bg_subtractor = None
ref_frame = None
parking_areas = [] # List of tuples: (start_coords, end_coords, status_tag, occupied_flag)
latest_parking_frame = None
# --- NEW: Car Cascade for Parking ---
try:
    car_cascade = cv2.CascadeClassifier('cars.xml')
    if car_cascade.empty():
        raise IOError("Could not load cars.xml")
    print("Car cascade loaded successfully for parking validation.")
except Exception as e:
    print(f"[ERROR] Failed to load 'cars.xml': {e}. Car validation in parking will be disabled.")
    car_cascade = None

# --- Utility Functions ---
def format_duration(seconds):
    """Formats seconds into H hours M minutes S seconds"""
    if seconds < 0: return "Negative duration?"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

# --- Function to save logs on exit ---
log_file_name = f"parking_session_log_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
def save_logs():
    print(f"\nSaving logs to {log_file_name}...")
    try:
        with open(log_file_name, 'w') as f:
            f.write("--- Session Log ---\n")
            f.write(f"End Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

            f.write("--- Entry Logs ---\n")
            if entry_log_messages:
                f.write("\n".join(entry_log_messages) + "\n")
            else:
                f.write("No entries recorded.\n")
            f.write("\n")

            f.write("--- Exit Logs ---\n")
            if exit_log_messages:
                f.write("\n".join(exit_log_messages) + "\n")
            else:
                f.write("No exits recorded.\n")
            f.write("\n")

            f.write("--- Duration Logs (Completed Stays) ---\n")
            if duration_log_messages:
                f.write("\n".join(duration_log_messages) + "\n")
            else:
                f.write("No completed stays recorded.\n")
            f.write("\n")

            # Log vehicles still inside at shutdown
            f.write("--- Vehicles Still Inside at Shutdown ---\n")
            if vehicles_inside:
                for plate, entry_time in vehicles_inside.items():
                    f.write(f"{plate} (Entered: {entry_time:%Y-%m-%d %H:%M:%S})\n")
            else:
                f.write("No vehicles inside at shutdown.\n")

        print("Logs saved successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to save logs: {e}")

# Register the save function to run on exit
atexit.register(save_logs)


# --- Camera Worker (Unchanged) ---
def camera_worker(camera_source, output_queue, processing_function, camera_name):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_source}")
        return

    print(f"[{camera_name}] Thread started.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Set a consistent resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_name}] Warning: Failed to grab frame.")
            time.sleep(0.5) # Wait longer if failing
            # Attempt to reconnect (optional, can be complex)
            # cap.release()
            # cap = cv2.VideoCapture(camera_source)
            # if not cap.isOpened(): print(f"[{camera_name}] Reconnect failed."); break
            continue

        if camera_name == "parking":
            global latest_parking_frame
            latest_parking_frame = frame.copy()

        processed_frame, data = processing_function(frame, camera_name)

        if not output_queue.full():
            output_queue.put((camera_name, processed_frame, data))
        else:
            time.sleep(0.005) # Prevent busy-waiting if queue is full

    cap.release()
    print(f"[{camera_name}] Thread stopped.")


# --- License Plate Processing Function (Added Sound) ---
YOLO_CONF_THRESHOLD = 0.40 # Confidence threshold

def process_license_plate_frame(frame, camera_name):
    now = time.time()
    state = scan_state[camera_name]
    state["frame_count"] += 1
    plate_text = None
    processed_frame = frame.copy()
    plate_bbox = None

    if state["frame_count"] % FRAME_SKIP_INTERVAL == 0:
        results = yolo_model.predict(
            processed_frame, imgsz=320, verbose=False, device=device_to_use,
            classes=[0], conf=YOLO_CONF_THRESHOLD
        )
        best_conf = 0
        plate_roi = None
        for result in results:
            boxes = result.boxes
            if boxes:
                for box in boxes:
                    conf = box.conf[0].item()
                    if conf >= YOLO_CONF_THRESHOLD and conf > best_conf:
                         best_conf = conf
                         xyxy = box.xyxy[0].int().tolist()
                         plate_bbox = xyxy
                         x1, y1, x2, y2 = xyxy
                         if x1 < x2 and y1 < y2 and x1 >=0 and y1 >=0 and x2 <= processed_frame.shape[1] and y2 <= processed_frame.shape[0]:
                             plate_roi = processed_frame[y1:y2, x1:x2]
                         else:
                             plate_roi = None; plate_bbox = None

        if plate_roi is not None and plate_roi.size > 0:
            gray_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
            ocr_results = ocr_reader.readtext(thresh_roi, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

            if ocr_results:
                raw_text = "".join(ocr_results).strip().upper().replace(" ", "")
                raw_text = re.sub(r'[^A-Z0-9]', '', raw_text) # Keep only A-Z, 0-9
                raw_text = raw_text.replace('O', '0').replace('I', '1').replace('S', '5').replace('B', '8').replace('Z', '2')
                if 5 <= len(raw_text) <= 7:
                    plate_text = raw_text
                    state["last_detected_plate"] = plate_text
                else: # Log failure only once per detection change for clarity
                    if state.get("last_raw_text") != raw_text:
                        print(f"[{camera_name}] OCR text '{raw_text}' (len {len(raw_text)}) failed length filter (5-8). Raw: {ocr_results}")
                        state["last_raw_text"] = raw_text

            if plate_bbox: # Only draw if YOLO detected something this frame
                cv2.rectangle(processed_frame, (plate_bbox[0], plate_bbox[1]), (plate_bbox[2], plate_bbox[3]), (0, 255, 0), 2)
                label = plate_text if plate_text else ("Reading..." if plate_bbox else "No Plate")
                cv2.putText(processed_frame, label, (plate_bbox[0], plate_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
             state["last_detected_plate"] = None
             plate_text = None
    else:
        plate_text = state.get("last_detected_plate", None)

    current_plate_in_state = state["plate"]
    if plate_text:
        state["last_seen"] = now
        if plate_text == current_plate_in_state:
            scan_duration = now - state["first_seen"]
            if scan_duration >= MIN_SCAN_DURATION and not state["logged"]:
                event_type = "ENTRY" if camera_name == "entry" else "EXIT"
                log_this_event = False
                current_time = datetime.datetime.now()

                if event_type == "ENTRY":
                    if plate_text not in vehicles_inside:
                        vehicles_inside[plate_text] = current_time # Store entry time
                        log_this_event = True
                        print(f"Vehicle {plate_text} entered. Currently inside: {list(vehicles_inside.keys())}")
                        if sound_enabled: entry_sound.play() # Play entry sound
                elif event_type == "EXIT":
                    if plate_text in vehicles_inside:
                        entry_time = vehicles_inside.pop(plate_text) # Remove and get entry time
                        duration = current_time - entry_time
                        formatted_duration = format_duration(duration.total_seconds())
                        duration_msg = f"{plate_text} - Parked for: {formatted_duration}"
                        duration_log_messages.append(duration_msg)
                        log_this_event = True
                        print(f"Vehicle {plate_text} exited. Duration: {formatted_duration}. Currently inside: {list(vehicles_inside.keys())}")
                        if sound_enabled: exit_sound.play() # Play exit sound

                state["logged"] = True
                if log_this_event:
                    timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"{timestamp_str} - {plate_text}" # Simplified log format
                    state["last_detected_plate"] = None
                    # Return event type along with log entry for GUI sorting
                    return processed_frame, {"log": log_entry, "type": event_type}
        else:
            state["plate"] = plate_text
            state["first_seen"] = now
            state["logged"] = False
    else:
        if current_plate_in_state is not None and (now - state["last_seen"]) > RESET_COOLDOWN:
            state["plate"] = None; state["first_seen"] = 0; state["logged"] = False; state["last_detected_plate"] = None

    return processed_frame, None

# --- Parking Lot Processing Function (Added Car Detection) ---
def process_parking_frame(frame, camera_name):
    global ref_frame, bg_subtractor, parking_areas
    statuses = [] # Store (status_tag, status_text)
    processed_frame = frame.copy() # Work on a copy
    dpg_values_ok = False # Flag to check if DPG values were read

    # --- Check 1: Is ref_frame actually set? ---
    if ref_frame is not None:
        # --- Check 2: Is bg_subtractor valid? ---
        if bg_subtractor is None:
             print("[Parking Process] Error: ref_frame is set, but bg_subtractor is None!")
             # Draw yellow boxes and return if subtractor invalid
             for i, (start, end, status_tag, _) in enumerate(parking_areas):
                 cv2.rectangle(processed_frame, start, end, (0, 255, 255), 2)
                 cv2.putText(processed_frame, f"P{i+1}", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
             cv2.putText(processed_frame, "Subtractor Error!", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
             return processed_frame, {"statuses": [ (tag, "Error") for _,_,tag,_ in parking_areas ]}

        # --- Check 3: Can we get DPG values? ---
        try:
            current_history = dpg.get_value("history")
            current_var_threshold = dpg.get_value("var_threshold")
            obstruction_threshold = dpg.get_value("obstruction_percent")
            bg_subtractor.setHistory(current_history)
            bg_subtractor.setVarThreshold(current_var_threshold)
            dpg_values_ok = True # Mark success
        except Exception as e:
            # Don't print constantly, maybe just once? Or use a flag.
            # print(f"[Parking Process] DPG error getting values: {e}")
            pass # Continue, but drawing might be affected if thresholds invalid

        gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)

        for i, (start, end, status_tag, _) in enumerate(parking_areas):
            parking_roi = processed_frame[start[1]:end[1], start[0]:end[0]]
            status_text = "Vacant" # Default
            color = (0, 255, 0) # Default green

            if parking_roi.size == 0: continue

            if dpg_values_ok: # Only process if we got DPG values
                try:
                    fg_mask = bg_subtractor.apply(parking_roi, learningRate=0) # Use learningRate=0 for detection phase
                    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                    white_pixels = cv2.countNonZero(thresh)
                    total_pixels = parking_roi.shape[0] * parking_roi.shape[1]
                    obstruction_percent = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                    is_obstructed = obstruction_percent > obstruction_threshold

                    is_car_present = False
                    if is_obstructed and car_cascade is not None:
                        roi_gray = gray_frame[start[1]:end[1], start[0]:end[0]]
                        cars = car_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        if len(cars) > 0:
                            is_car_present = True
                            # Optional: Draw car bounding box for debugging
                            # for (x, y, w, h) in cars:
                            #    cv2.rectangle(parking_roi, (x, y), (x+w, y+h), (255, 0, 0), 1)

                    if is_car_present:
                        status_text = "Occupied"
                        color = (0, 0, 255)
                    elif is_obstructed:
                        status_text = "Obstructed"
                        color = (0, 165, 255)

                    statuses.append((status_tag, status_text))
                    # --- Drawing happens here ---
                    cv2.rectangle(processed_frame, start, end, color, 2)
                    cv2.putText(processed_frame, f"P{i+1}: {status_text}", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                except Exception as e:
                    print(f"Error processing parking ROI {i+1}: {e}")
                    cv2.rectangle(processed_frame, start, end, (0, 255, 255), 2)
                    cv2.putText(processed_frame, f"P{i+1}: Error", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    statuses.append((status_tag, "Error"))
            else:
                 # Draw yellow if DPG values failed
                 cv2.rectangle(processed_frame, start, end, (0, 255, 255), 2)
                 cv2.putText(processed_frame, f"P{i+1}: DPG?", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                 statuses.append((status_tag, "DPG?"))


    else: # Draw initial yellow boxes if no ref frame
        for i, (start, end, status_tag, _) in enumerate(parking_areas):
             cv2.rectangle(processed_frame, start, end, (0, 255, 255), 2)
             cv2.putText(processed_frame, f"P{i+1}", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if ref_frame is None: # Explicit check just for the text
             cv2.putText(processed_frame, "Press 'S' to set reference frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if ref_frame is not None and bg_subtractor is not None:
        try:
            bg_subtractor.setHistory(dpg.get_value("history"))
            bg_subtractor.setVarThreshold(dpg.get_value("var_threshold"))
            obstruction_threshold = dpg.get_value("obstruction_percent")
        except Exception as e:
            print(f"DPG error getting values: {e}")
            return processed_frame, None # Return early if GUI isn't ready

        gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY) # Grayscale once for car detection

        for i, (start, end, status_tag, _) in enumerate(parking_areas): # Unpack coords and tag
            parking_roi = processed_frame[start[1]:end[1], start[0]:end[0]]
            status_text = "Vacant" # Default
            color = (0, 255, 0) # Default green

            if parking_roi.size == 0: continue

            try:
                fg_mask = bg_subtractor.apply(parking_roi)
                _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                white_pixels = cv2.countNonZero(thresh)
                total_pixels = parking_roi.shape[0] * parking_roi.shape[1]
                obstruction_percent = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0

                is_obstructed = obstruction_percent > obstruction_threshold

                # --- NEW: Check for car only if obstructed ---
                is_car_present = False
                if is_obstructed and car_cascade is not None:
                    roi_gray = gray_frame[start[1]:end[1], start[0]:end[0]] # Use pre-grayed frame
                    # Adjust minSize/maxSize based on expected car size in ROI
                    cars = car_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(cars) > 0:
                        is_car_present = True

                # --- Update Status based on Car Detection ---
                if is_car_present:
                    status_text = "Occupied"
                    color = (0, 0, 255) # Red
                elif is_obstructed:
                    status_text = "Obstructed" # Something is there, but not a car
                    color = (0, 165, 255) # Orange
                # else: status remains Vacant, color remains Green

                statuses.append((status_tag, status_text)) # Store status for GUI update
                cv2.rectangle(processed_frame, start, end, color, 2)
                cv2.putText(processed_frame, f"P{i+1}: {status_text}", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            except Exception as e:
                print(f"Error processing parking ROI {i+1}: {e}")
                cv2.rectangle(processed_frame, start, end, (0, 255, 255), 2)
                cv2.putText(processed_frame, f"P{i+1}: Error", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                statuses.append((status_tag, "Error"))

    else: # Draw boxes if no ref frame
        for i, (start, end, status_tag, _) in enumerate(parking_areas):
             cv2.rectangle(processed_frame, start, end, (0, 255, 255), 2)
             cv2.putText(processed_frame, f"P{i+1}", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if ref_frame is None:
             cv2.putText(processed_frame, "Press 'S' to set reference frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return processed_frame, {"statuses": statuses} # Pass statuses back

# --- GUI Setup (Updated with Tabs and Duration Log) ---
def setup_gui():
    dpg.create_context()
    parking_status_items = {} # Store dynamically created tags here

    with dpg.window(label="System Control Panel", width=550, height=700, no_close=True):
        with dpg.tab_bar():
            # --- Entry Log Tab ---
            # ... (no changes here) ...
            with dpg.tab(label="Entry Log"):
                 dpg.add_text("Vehicle Entries:")
                 dpg.add_text("", tag="vehicle_log_entry", wrap=500)

            # --- Exit Log Tab ---
            # ... (no changes here) ...
            with dpg.tab(label="Exit Log"):
                 dpg.add_text("Vehicle Exits:")
                 dpg.add_text("", tag="vehicle_log_exit", wrap=500)

            # --- Duration Log Tab ---
            # ... (no changes here) ...
            with dpg.tab(label="Parking Duration"):
                 dpg.add_text("Completed Stays:")
                 dpg.add_text("", tag="duration_log", wrap=500)

            # --- Parking Status Tab ---
            # --- ADD TAG HERE ---
            with dpg.tab(label="Parking Status", tag="parking_status_tab"): # Added tag
                dpg.add_text("Parking Spot Status:")
                # Status items will be added dynamically later using the tag as parent
                dpg.add_separator()
                dpg.add_text("Parking Detection Parameters")
                dpg.add_slider_int(label="History", tag="history", default_value=500, min_value=100, max_value=2000)
                dpg.add_slider_int(label="VarThreshold", tag="var_threshold", default_value=16, min_value=10, max_value=500)
                dpg.add_slider_int(label="Obstruction %", tag="obstruction_percent", default_value=25, min_value=5, max_value=90)

        # ... (Rest of setup_gui is unchanged) ...
        dpg.add_separator()
        dpg.add_text("Keyboard Controls")
        dpg.add_text("Press 'S' (on any window) - Set Reference Frame")
        dpg.add_text("Press 'Q' (on any window) - Quit")

    dpg.create_viewport(title="Parking System", width=550, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    return parking_status_items # Return the dictionary to store tags

# --- Main Application Logic ---
# --- Main Application Logic ---
if __name__ == "__main__":
    parking_status_display_tags = setup_gui() # Setup GUI first

    # ... (camera_configs setup unchanged) ...
    camera_configs = {
        "entry": {"source": 1, "processor": process_license_plate_frame},
        "exit": {"source": 2, "processor": process_license_plate_frame},
        "parking": {"source": 0, "processor": process_parking_frame}
    }


    print("Initializing parking area... waiting for camera frame.")
    # ... (getting test_frame and calculating dimensions unchanged) ...
    parking_cam_index = camera_configs["parking"]["source"]
    temp_cap = cv2.VideoCapture(parking_cam_index)
    ret, test_frame = temp_cap.read()
    if ret:
        frame_h, frame_w = test_frame.shape[:2]
        print(f"Parking camera resolution: {frame_w}x{frame_h}")
        parking_width = int(frame_w * 0.15)
        parking_height = int(frame_h * 0.3)
        start_x = int(frame_w * 0.05)
        start_y = int(frame_h * 0.2)
        space_between = int(parking_width * 0.1)

        # --- MODIFIED: Use parent tag for adding status items ---
        # No need to search for the tab anymore
        parking_tab_exists = dpg.does_item_exist("parking_status_tab")

        if parking_tab_exists:
            for i in range(5): # Assuming 5 spots
                start = (start_x + (parking_width + space_between) * i, start_y)
                end = (start[0] + parking_width, start[1] + parking_height)
                status_tag = f"parking_status_{i+1}"
                # Add status text using the specific tab tag as parent
                parking_status_display_tags[status_tag] = dpg.add_text(
                    f"Parking {i+1:02d}: Initializing...",
                    tag=status_tag,
                    parent="parking_status_tab" # Use the tag here
                )
                parking_areas.append((start, end, status_tag, False))
            print("Parking areas and GUI elements initialized.")
        else:
            print("[ERROR] Could not find item with tag 'parking_status_tab' to add status text items.")
        # --- END MODIFIED SECTION ---

        # ... (bg_subtractor initialization unchanged) ...
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
             history=dpg.get_value("history"),
             varThreshold=dpg.get_value("var_threshold"),
             detectShadows=False
        )
    else:
        # ... (error handling unchanged) ...
        print(f"[FATAL] Could not get a test frame from the parking camera (index {parking_cam_index}). Exiting.")
        if dpg.does_context_exist(): dpg.destroy_context()
        exit()
    temp_cap.release()

    threads = []
    output_queues = {}
    for name, config in camera_configs.items():
        output_queues[name] = Queue(maxsize=5) # Slightly larger queue
        thread = Thread(target=camera_worker, args=(config["source"], output_queues[name], config["processor"], name), daemon=True)
        threads.append(thread)
        thread.start()

    # --- Main Loop ---
    try:
        while dpg.is_dearpygui_running():
            time.sleep(0.01) # Reduce CPU usage slightly more

            all_data = {} # Store data from all cameras this cycle
            # Process queues non-blockingly
            for name, q in output_queues.items():
                try:
                    while not q.empty(): # Process all items in queue this cycle
                        cam_name, frame, data = q.get_nowait()
                        all_data[cam_name] = (frame, data) # Store latest frame & data
                except Exception: # Handle queue empty exception
                    pass

            # --- Update GUI and State ---
            log_updated = False
            for cam_name, (frame, data) in all_data.items():
                try:
                    cv2.imshow(f"{cam_name.title()} Camera", frame)
                except Exception as e:
                    print(f"Error displaying frame for {cam_name}: {e}")

                if data:
                    # Handle Log data
                    if "log" in data and "type" in data:
                        log_updated = True
                        log_msg = data["log"]
                        log_type = data["type"]
                        if log_type == "ENTRY":
                            entry_log_messages.append(log_msg)
                        elif log_type == "EXIT":
                            exit_log_messages.append(log_msg)
                            # Update duration log immediately when exit happens
                            if dpg.does_item_exist("duration_log"):
                                dpg.set_value("duration_log", "\n".join(duration_log_messages[-20:]))


                    # Handle Parking Status data
                    if "statuses" in data:
                         for status_tag, status_text in data["statuses"]:
                              if dpg.does_item_exist(status_tag):
                                   dpg.set_value(status_tag, f"Parking {status_tag.split('_')[-1]}: {status_text}")


            # Update log displays if needed (outside the camera loop)
            if log_updated:
                 if dpg.does_item_exist("vehicle_log_entry"):
                      dpg.set_value("vehicle_log_entry", "\n".join(entry_log_messages[-20:])) # Show last 20
                 if dpg.does_item_exist("vehicle_log_exit"):
                      dpg.set_value("vehicle_log_exit", "\n".join(exit_log_messages[-20:]))
                 if dpg.does_item_exist("duration_log"): # Ensure duration updates even if no exit this cycle
                      dpg.set_value("duration_log", "\n".join(duration_log_messages[-20:]))


            if dpg.is_dearpygui_running():
                dpg.render_dearpygui_frame()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                dpg.stop_dearpygui(); break
            elif key == ord('s'):
                if latest_parking_frame is not None:
                    print("Setting new reference frame...")
                    ref_frame = latest_parking_frame.copy()
                    bg_subtractor = cv2.createBackgroundSubtractorMOG2( # Recreate to reset history
                        history=dpg.get_value("history"),
                        varThreshold=dpg.get_value("var_threshold"),
                        detectShadows=False
                    )
                    print("Reference frame for parking lot has been set!")
                else:
                    print("Could not set reference frame: No frame available from parking camera.")

    finally:
        print("Shutting down...")
        stop_event.set() # Signal threads to stop
        # Give threads a moment to finish
        time.sleep(1)
        for t in threads:
            if t.is_alive():
                 print(f"Waiting for thread {t.name} to join...")
                 t.join(timeout=2) # Wait max 2 seconds per thread
                 if t.is_alive(): print(f"Warning: Thread {t.name} did not terminate gracefully.")

        cv2.destroyAllWindows()
        if dpg.is_dearpygui_running(): # Check before destroying
             dpg.stop_dearpygui()
        if dpg.does_context_exist():
            dpg.destroy_context()

        # Log saving is handled by atexit
        print("Shutdown sequence complete.") # Keep this AFTER DPG destroy
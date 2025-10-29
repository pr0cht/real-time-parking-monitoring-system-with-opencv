# main.py
import cv2
import dearpygui.dearpygui as dpg
import time
import datetime
from threading import Thread, Event # No Lock needed now
from queue import Queue, Empty
from ultralytics import YOLO
import easyocr
import re
import torch
import os
import atexit

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
MIN_SCAN_DURATION = 2.0
RESET_COOLDOWN = 1.0
FRAME_SKIP_INTERVAL = 5
# MAX_CAM_INDEX = 5 # No longer needed

# --- YOLO and OCR Initialization ---
try:
    yolo_model = YOLO('best.pt')
    device_to_use = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_to_use}")
except Exception as e:
    print(f"Error initializing YOLO: {e}. Ensure 'best.pt' is present.")
    exit()

try:
     ocr_reader = easyocr.Reader(['en'], gpu=(device_to_use == 0))
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
vehicles_inside = {}

# --- Parking State ---
bg_subtractor = None
ref_frame = None
parking_areas = []
latest_parking_frame = None
try:
    car_cascade = cv2.CascadeClassifier('cars.xml')
    if car_cascade.empty(): raise IOError("Could not load cars.xml")
    print("Car cascade loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load 'cars.xml': {e}. Car validation disabled.")
    car_cascade = None

# --- Utility Functions (unchanged) ---
def format_duration(seconds):
    if seconds < 0: return "Negative duration?"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

# --- Log Saving (unchanged) ---
log_file_name = f"parking_session_log_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
def save_logs():
    print(f"\nSaving logs to {log_file_name}...")
    # ...(rest of save_logs function is the same)...
    try:
        with open(log_file_name, 'w') as f:
            f.write("--- Session Log ---\n")
            f.write(f"End Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
            f.write("--- Entry Logs ---\n")
            f.write("\n".join(entry_log_messages) + "\n\n" if entry_log_messages else "No entries recorded.\n\n")
            f.write("--- Exit Logs ---\n")
            f.write("\n".join(exit_log_messages) + "\n\n" if exit_log_messages else "No exits recorded.\n\n")
            f.write("--- Duration Logs (Completed Stays) ---\n")
            f.write("\n".join(duration_log_messages) + "\n\n" if duration_log_messages else "No completed stays recorded.\n\n")
            f.write("--- Vehicles Still Inside at Shutdown ---\n")
            if vehicles_inside:
                for plate, entry_time in vehicles_inside.items():
                    f.write(f"{plate} (Entered: {entry_time:%Y-%m-%d %H:%M:%S})\n")
            else: f.write("No vehicles inside at shutdown.\n")
        print("Logs saved successfully.")
    except Exception as e: print(f"[ERROR] Failed to save logs: {e}")
atexit.register(save_logs)


# --- Camera Worker (REVERTED to simpler version) ---
def camera_worker(camera_source, output_queue, processing_function, camera_name):
    """ Reads frames from a FIXED camera source and processes them. """
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"[ERROR][{camera_name}] Cannot open camera {camera_source}")
        return

    print(f"[{camera_name}] Thread started on camera {camera_source}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"[{camera_name}] Warning: Failed to grab frame from camera {camera_source}.")
            time.sleep(0.5)
            continue # Keep trying on the same source

        if camera_name == "parking":
            global latest_parking_frame
            latest_parking_frame = frame.copy()

        try:
            processed_frame, data = processing_function(frame, camera_name)
            if not output_queue.full():
                output_queue.put((camera_name, processed_frame, data))
            else:
                time.sleep(0.005)
        except Exception as e:
             print(f"[ERROR][{camera_name}] Exception during processing: {e}")
             raw_frame_data = (camera_name, frame, None)
             try: output_queue.put_nowait(raw_frame_data)
             except Full: pass
             time.sleep(0.1)

    cap.release()
    print(f"[{camera_name}] Thread stopped.")


# --- License Plate Processing Function (Reads Confidence from GUI) ---
# (Unchanged from previous version)
def process_license_plate_frame(frame, camera_name):
    now = time.time()
    state = scan_state[camera_name]
    state["frame_count"] += 1
    plate_text = None
    processed_frame = frame.copy()
    plate_bbox = None

    try:
        current_yolo_conf = dpg.get_value("yolo_confidence")
    except Exception:
        current_yolo_conf = 0.40 # Default if GUI fails

    if state["frame_count"] % FRAME_SKIP_INTERVAL == 0:
        results = yolo_model.predict(
            processed_frame, imgsz=320, verbose=False, device=device_to_use,
            classes=[0], conf=current_yolo_conf
        )
        best_conf = 0; plate_roi = None
        for result in results:
            boxes = result.boxes
            if boxes:
                for box in boxes:
                    conf = box.conf[0].item()
                    if conf >= current_yolo_conf and conf > best_conf:
                         best_conf = conf; xyxy = box.xyxy[0].int().tolist(); plate_bbox = xyxy
                         x1, y1, x2, y2 = xyxy
                         if x1<x2 and y1<y2 and x1>=0 and y1>=0 and x2<=processed_frame.shape[1] and y2<=processed_frame.shape[0]:
                             plate_roi = processed_frame[y1:y2, x1:x2]
                         else: plate_roi = None; plate_bbox = None
        if plate_roi is not None and plate_roi.size > 0:
            gray_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
            thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
            ocr_results = ocr_reader.readtext(thresh_roi, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            if ocr_results:
                raw_text = "".join(ocr_results).strip().upper().replace(" ", "")
                raw_text = re.sub(r'[^A-Z0-9]', '', raw_text)
                raw_text = raw_text.replace('O', '0').replace('I', '1').replace('S', '5').replace('B', '8').replace('Z', '2')
                if 5 <= len(raw_text) <= 8:
                    plate_text = raw_text; state["last_detected_plate"] = plate_text
                # else: (optional logging)
            if plate_bbox:
                cv2.rectangle(processed_frame, (plate_bbox[0], plate_bbox[1]), (plate_bbox[2], plate_bbox[3]), (0, 255, 0), 2)
                label = plate_text if plate_text else ("Reading..." if plate_bbox else "No Plate")
                cv2.putText(processed_frame, label, (plate_bbox[0], plate_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else: state["last_detected_plate"] = None; plate_text = None
    else: plate_text = state.get("last_detected_plate", None)

    current_plate_in_state = state["plate"]
    if plate_text:
        state["last_seen"] = now
        if plate_text == current_plate_in_state:
            scan_duration = now - state["first_seen"]
            if scan_duration >= MIN_SCAN_DURATION and not state["logged"]:
                event_type = "ENTRY" if camera_name == "entry" else "EXIT"
                log_this_event = False; current_time = datetime.datetime.now()
                if event_type == "ENTRY":
                    if plate_text not in vehicles_inside:
                        vehicles_inside[plate_text] = current_time; log_this_event = True; print(f"Vehicle {plate_text} entered.")
                        if sound_enabled: entry_sound.play()
                elif event_type == "EXIT":
                    if plate_text in vehicles_inside:
                        entry_time = vehicles_inside.pop(plate_text); duration = current_time - entry_time
                        formatted_duration = format_duration(duration.total_seconds()); duration_msg = f"{plate_text} - Parked for: {formatted_duration}"
                        duration_log_messages.append(duration_msg); log_this_event = True; print(f"Vehicle {plate_text} exited.")
                        if sound_enabled: exit_sound.play()
                state["logged"] = True
                if log_this_event:
                    timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S"); log_entry = f"{timestamp_str} - {plate_text}"
                    state["last_detected_plate"] = None
                    return processed_frame, {"log": log_entry, "type": event_type}
        else: state["plate"] = plate_text; state["first_seen"] = now; state["logged"] = False
    else:
        if current_plate_in_state is not None and (now - state["last_seen"]) > RESET_COOLDOWN:
            state["plate"] = None; state["first_seen"] = 0; state["logged"] = False; state["last_detected_plate"] = None
    return processed_frame, None


# --- Parking Lot Processing Function (Unchanged) ---
def process_parking_frame(frame, camera_name):
    global ref_frame, bg_subtractor, parking_areas
    # ...(Same as the previous version)...
    statuses = []; processed_frame = frame.copy(); dpg_values_ok = False
    if ref_frame is not None:
        if bg_subtractor is None:
             print("[Parking Process] Error: ref_frame set, but bg_subtractor None!"); # Simplified error message
             # Draw yellow boxes and return if subtractor invalid
             for i, (start, end, status_tag, _) in enumerate(parking_areas): cv2.rectangle(processed_frame, start, end, (0, 255, 255), 2); cv2.putText(processed_frame, f"P{i+1}", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
             cv2.putText(processed_frame, "Subtractor Error!", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
             return processed_frame, {"statuses": [ (tag, "Error") for _,_,tag,_ in parking_areas ]}
        try:
            current_history=dpg.get_value("history"); current_var_threshold=dpg.get_value("var_threshold"); obstruction_threshold=dpg.get_value("obstruction_percent")
            bg_subtractor.setHistory(current_history); bg_subtractor.setVarThreshold(current_var_threshold); dpg_values_ok = True
        except Exception: pass
        gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        for i, (start, end, status_tag, _) in enumerate(parking_areas):
            parking_roi = processed_frame[start[1]:end[1], start[0]:end[0]]; status_text = "Vacant"; color = (0, 255, 0)
            if parking_roi.size == 0: continue
            if dpg_values_ok:
                try:
                    fg_mask = bg_subtractor.apply(parking_roi, learningRate=0); _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                    white_pixels = cv2.countNonZero(thresh); total_pixels = parking_roi.shape[0] * parking_roi.shape[1]
                    obstruction_percent = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                    is_obstructed = obstruction_percent > obstruction_threshold; is_car_present = False
                    if is_obstructed and car_cascade is not None:
                        roi_gray = gray_frame[start[1]:end[1], start[0]:end[0]]
                        cars = car_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        if len(cars) > 0: is_car_present = True
                    if is_car_present: status_text = "Occupied"; color = (0, 0, 255)
                    elif is_obstructed: status_text = "Obstructed"; color = (0, 165, 255)
                    statuses.append((status_tag, status_text)); cv2.rectangle(processed_frame, start, end, color, 2)
                    cv2.putText(processed_frame, f"P{i+1}: {status_text}", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception as e:
                    print(f"Error ROI {i+1}: {e}"); cv2.rectangle(processed_frame, start, end, (0, 255, 255), 2); cv2.putText(processed_frame, f"P{i+1}: Error", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    statuses.append((status_tag, "Error"))
            else: cv2.rectangle(processed_frame, start, end, (0, 255, 255), 2); cv2.putText(processed_frame, f"P{i+1}: DPG?", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); statuses.append((status_tag, "DPG?"))
    else:
        for i, (start, end, status_tag, _) in enumerate(parking_areas): cv2.rectangle(processed_frame, start, end, (0, 255, 255), 2); cv2.putText(processed_frame, f"P{i+1}", (start[0], start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if ref_frame is None: cv2.putText(processed_frame, "Press 'S' to set reference frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return processed_frame, {"statuses": statuses}

# --- Camera Selection Callback (REMOVED) ---
# def update_camera_source(sender, app_data): ... (Removed)


# --- GUI Setup (REMOVED Camera Selectors) ---
def setup_gui(): # No longer needs initial_indices
    dpg.create_context()
    parking_status_items = {}
    with dpg.window(label="System Control Panel", width=600, height=750, no_close=True):
        with dpg.tab_bar():
            with dpg.tab(label="Entry Log"): dpg.add_text("Vehicle Entries:"); dpg.add_text("", tag="vehicle_log_entry", wrap=550)
            with dpg.tab(label="Exit Log"): dpg.add_text("Vehicle Exits:"); dpg.add_text("", tag="vehicle_log_exit", wrap=550)
            with dpg.tab(label="Parking Duration"): dpg.add_text("Completed Stays:"); dpg.add_text("", tag="duration_log", wrap=550)
            with dpg.tab(label="Parking Status", tag="parking_status_tab"):
                dpg.add_text("Parking Spot Status:")
                dpg.add_separator()
                dpg.add_text("Parking Detection Parameters")
                dpg.add_slider_int(label="History", tag="history", default_value=500, min_value=100, max_value=2000)
                dpg.add_slider_int(label="VarThreshold", tag="var_threshold", default_value=16, min_value=10, max_value=500)
                dpg.add_slider_int(label="Obstruction %", tag="obstruction_percent", default_value=25, min_value=5, max_value=90)

            # --- Settings Tab (REMOVED Camera Selectors) ---
            with dpg.tab(label="Settings"):
                dpg.add_text("License Plate Detection")
                dpg.add_slider_float( label="YOLO Confidence", tag="yolo_confidence", default_value=0.40, min_value=0.1, max_value=0.95, format="%.2f", width=200 )
                # dpg.add_separator() # Removed
                # dpg.add_text("Camera Selection") # Removed
                # dpg.add_combo(...) # Removed Entry Select
                # dpg.add_combo(...) # Removed Exit Select
                # dpg.add_combo(...) # Removed Parking Select

        dpg.add_separator()
        dpg.add_text("Keyboard Controls")
        dpg.add_text("Press 'S' (on any window) - Set Reference Frame")
        dpg.add_text("Press 'Q' (on any window) - Quit")

    dpg.create_viewport(title="Parking System", width=600, height=750)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    return parking_status_items


# --- Main Application Logic ---
if __name__ == "__main__":
    # --- Define FIXED camera sources ---
    camera_configs = {
        "entry": {"source": 1}, # Example: Use Cam 1 for Entry
        "exit": {"source": 2},  # Example: Use Cam 2 for Exit
        "parking": {"source": 0} # Example: Use Cam 0 for Parking
    }
    # (No need to update global state here anymore)

    # Pass dummy dict or None to setup_gui if it expects arg
    parking_status_display_tags = setup_gui() # No indices needed

    # --- Initialize Parking ---
    print("Initializing parking area...")
    parking_cam_index = camera_configs["parking"]["source"] # Get fixed index
    temp_cap = cv2.VideoCapture(parking_cam_index)
    # ...(rest of parking initialization, same as before)...
    ret, test_frame = temp_cap.read()
    if ret:
        frame_h, frame_w = test_frame.shape[:2]; print(f"Parking res: {frame_w}x{frame_h}")
        parking_width=int(frame_w*0.15); parking_height=int(frame_h*0.3)
        start_x=int(frame_w*0.05); start_y=int(frame_h*0.2); space_between=int(parking_width*0.1)
        parking_tab_exists = dpg.does_item_exist("parking_status_tab")
        if parking_tab_exists:
            for i in range(5):
                start=(start_x + (parking_width+space_between)*i, start_y)
                end=(start[0]+parking_width, start[1]+parking_height)
                status_tag=f"parking_status_{i+1}"
                parking_status_display_tags[status_tag] = dpg.add_text(f"P{i+1}: Init...", tag=status_tag, parent="parking_status_tab")
                parking_areas.append((start, end, status_tag, False))
            print("Parking GUI elements initialized.")
        else: print("[ERROR] Could not find 'parking_status_tab'.")
        if dpg.does_item_exist("history"):
             bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=dpg.get_value("history"), varThreshold=dpg.get_value("var_threshold"), detectShadows=False)
             print("Background subtractor initialized.")
        else: print("[ERROR] DPG sliders not found, cannot init bg_subtractor."); bg_subtractor = None
    else:
        print(f"[FATAL] Failed test frame from parking cam {parking_cam_index}.");
        if dpg.does_context_exist(): dpg.destroy_context(); exit()
    temp_cap.release()

    # --- Start Threads (Using fixed sources) ---
    threads = []
    output_queues = {}
    processors = { "entry": process_license_plate_frame, "exit": process_license_plate_frame, "parking": process_parking_frame }

    for name, config in camera_configs.items(): # Iterate through fixed config
        fixed_source_index = config["source"]
        output_queues[name] = Queue(maxsize=5)
        processor_func = processors[name]
        thread = Thread(
            target=camera_worker,
            args=(fixed_source_index, output_queues[name], processor_func, name), # Pass fixed index
            daemon=True, name=f"{name}_worker"
        )
        threads.append(thread)
        thread.start()

    # --- Main Loop (Unchanged conceptually) ---
    # ...(Same as previous version)...
    try:
        while dpg.is_dearpygui_running():
            time.sleep(0.01); all_data = {}
            for name, q in output_queues.items():
                try:
                    while not q.empty(): cam_name, frame, data = q.get_nowait(); all_data[cam_name] = (frame, data)
                except Empty: pass
            log_updated = False; parking_updated = False
            for cam_name, (frame, data) in all_data.items():
                try: cv2.imshow(f"{cam_name.title()} Camera", frame)
                except Exception as e: print(f"imshow error {cam_name}: {e}")
                if data:
                    if "log" in data and "type" in data:
                        log_updated = True; log_msg, log_type = data["log"], data["type"]
                        if log_type == "ENTRY": entry_log_messages.append(log_msg)
                        elif log_type == "EXIT":
                            exit_log_messages.append(log_msg)
                            if dpg.does_item_exist("duration_log"): dpg.set_value("duration_log", "\n".join(duration_log_messages[-30:]))
                    if "statuses" in data:
                         parking_updated = True
                         for status_tag, status_text in data["statuses"]:
                              if dpg.does_item_exist(status_tag): dpg.set_value(status_tag, f"Parking {status_tag.split('_')[-1]}: {status_text}")
            if log_updated:
                 if dpg.does_item_exist("vehicle_log_entry"): dpg.set_value("vehicle_log_entry", "\n".join(entry_log_messages[-30:]))
                 if dpg.does_item_exist("vehicle_log_exit"): dpg.set_value("vehicle_log_exit", "\n".join(exit_log_messages[-30:]))
            if dpg.is_dearpygui_running(): dpg.render_dearpygui_frame()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): dpg.stop_dearpygui(); break
            elif key == ord('s'):
                if latest_parking_frame is not None:
                    print("Setting new reference frame...")
                    ref_frame = latest_parking_frame.copy()
                    if dpg.does_item_exist("history"):
                         bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=dpg.get_value("history"), varThreshold=dpg.get_value("var_threshold"), detectShadows=False)
                         print("Ref frame set & subtractor reset.")
                    else: print("Ref frame set, cannot reset subtractor.")
                else: print("Could not set ref frame: No frame available.")
    finally: # Unchanged shutdown logic
        print("Shutting down...")
        stop_event.set(); time.sleep(1)
        for t in threads:
            if t.is_alive(): print(f"Wait {t.name}..."); t.join(timeout=2);
            if t.is_alive(): print(f"Warn: {t.name} timeout.")
        cv2.destroyAllWindows()
        if dpg.is_dearpygui_running(): dpg.stop_dearpygui()
        if dpg.does_context_exist(): dpg.destroy_context()
        print("Shutdown sequence complete.")
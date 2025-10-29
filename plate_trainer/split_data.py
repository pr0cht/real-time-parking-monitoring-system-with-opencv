import os
import shutil
import random
import glob

# --- Configuration ---

# 1. Directory containing your .jpg images and .txt labels
SOURCE_DIR = 'positive_raw'

# 2. Base directory where the 'datasets' folder with train/val splits will be created
DEST_DIR = '.' # Current directory (plate_trainer)

# 3. Ratio of data to put in the training set (e.g., 0.85 means 85% train, 15% val)
TRAIN_RATIO = 0.85

# --- End of Configuration ---

# --- Create Directory Structure ---
datasets_base = os.path.join(DEST_DIR, 'datasets')
train_img_dir = os.path.join(datasets_base, 'images', 'train')
val_img_dir = os.path.join(datasets_base, 'images', 'val')
train_lbl_dir = os.path.join(datasets_base, 'labels', 'train')
val_lbl_dir = os.path.join(datasets_base, 'labels', 'val')

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

print("Created directory structure under 'datasets'.")

# --- Find Image Files ---
# Find all .jpg files (adjust extension if needed, e.g., '*.jpeg')
image_files = glob.glob(os.path.join(SOURCE_DIR, '*.jpg'))

if not image_files:
    print(f"Error: No .jpg files found in '{SOURCE_DIR}'. Check the path and file extensions.")
    exit()

print(f"Found {len(image_files)} image files.")

# --- Shuffle and Split ---
random.shuffle(image_files) # Shuffle randomly
split_index = int(len(image_files) * TRAIN_RATIO)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

print(f"Splitting into {len(train_files)} training and {len(val_files)} validation samples.")

# --- Function to Move Files ---
def move_files(file_list, img_dest_folder, lbl_dest_folder, set_name):
    moved_count = 0
    skipped_txt_count = 0
    for img_path in file_list:
        try:
            # Construct corresponding label file path
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            lbl_filename = f"{base_filename}.txt"
            lbl_path = os.path.join(SOURCE_DIR, lbl_filename)

            # Define destination paths
            img_dest = os.path.join(img_dest_folder, os.path.basename(img_path))
            lbl_dest = os.path.join(lbl_dest_folder, lbl_filename)

            # Move the image file
            shutil.move(img_path, img_dest)

            # Check if the label file exists and move it
            if os.path.exists(lbl_path):
                shutil.move(lbl_path, lbl_dest)
            else:
                print(f"[Warning] Label file not found for {os.path.basename(img_path)}, image moved but label skipped.")
                skipped_txt_count +=1

            moved_count += 1
            if moved_count % 500 == 0: # Print progress
                 print(f"Moved {moved_count} files to {set_name} set...")

        except Exception as e:
            print(f"[Error] Failed to move {os.path.basename(img_path)} or its label: {e}")
    print(f"Finished moving {moved_count} images to {set_name} set.")
    if skipped_txt_count > 0:
        print(f"[Warning] Skipped moving {skipped_txt_count} non-existent label files for the {set_name} set.")


# --- Move Training Files ---
print("\nMoving training files...")
move_files(train_files, train_img_dir, train_lbl_dir, "training")

# --- Move Validation Files ---
print("\nMoving validation files...")
move_files(val_files, val_img_dir, val_lbl_dir, "validation")

print("\n---")
print("File splitting and moving complete!")
print(f"Check the '{datasets_base}' directory.")
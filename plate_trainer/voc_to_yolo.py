import xml.etree.ElementTree as ET
import os
import glob # Used for finding files

# --- Configuration ---

# 1. Path to the folder containing both your .xml annotations and .jpg images
INPUT_DIR = 'positive_raw'

# 2. Path to the folder where you want to save the YOLO .txt files
#    It's usually best practice to keep images and labels together,
#    so we'll save them in the same input directory.
OUTPUT_DIR = 'positive_raw'

# 3. Define your class names and assign them an ID (starting from 0)
#    Since you likely only have license plates, this is simple.
#    IMPORTANT: The name 'license_plate' MUST EXACTLY MATCH the <name> tag
#    inside your XML files. Check a few XMLs to be sure.
CLASS_MAP = {
    'license_plate': 0,
    'licence': 0,
    'License Plate': 0, # <-- ADD THIS LINE
}

# --- End of Configuration ---

def convert_coordinates(size, box):
    """Converts Pascal VOC (xmin, ymin, xmax, ymax) to YOLO format."""
    dw = 1.0 / size[0] # image width
    dh = 1.0 / size[1] # image height
    x = (box[0] + box[1]) / 2.0 # center x
    y = (box[2] + box[3]) / 2.0 # center y
    w = box[1] - box[0] # width
    h = box[3] - box[2] # height
    
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_xml_to_yolo(xml_file_path, output_dir):
    """Parses a single XML file and saves the corresponding YOLO txt file."""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        size = root.find('size')
        if size is None:
            print(f"[Warning] Skipping {xml_file_path}: Missing <size> tag.")
            return

        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        if img_width <= 0 or img_height <= 0:
            print(f"[Warning] Skipping {xml_file_path}: Invalid image dimensions ({img_width}x{img_height}).")
            return

        yolo_lines = []
        objects_found = 0
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in CLASS_MAP:
                print(f"[Warning] Skipping object in {xml_file_path}: Class '{class_name}' not in CLASS_MAP.")
                continue

            class_id = CLASS_MAP[class_name]

            bndbox = obj.find('bndbox')
            if bndbox is None:
                print(f"[Warning] Skipping object in {xml_file_path}: Missing <bndbox> tag.")
                continue
                
            # Extract coordinates safely
            try:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
            except AttributeError:
                print(f"[Warning] Skipping object in {xml_file_path}: Missing coordinate tags in <bndbox>.")
                continue
            except ValueError:
                 print(f"[Warning] Skipping object in {xml_file_path}: Non-numeric coordinate values in <bndbox>.")
                 continue


            # Validate coordinates
            if xmin >= xmax or ymin >= ymax:
                 print(f"[Warning] Skipping object in {xml_file_path}: Invalid initial coordinates (min >= max). xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
                 continue

            # Clamp coordinates to image boundaries
            original_coords = (xmin, ymin, xmax, ymax) # Keep for logging if needed
            xmin = max(0.0, xmin)
            ymin = max(0.0, ymin)
            xmax = min(float(img_width), xmax)
            ymax = min(float(img_height), ymax)

            # Check if clamping made the box invalid (width or height became zero or negative)
            if xmin >= xmax or ymin >= ymax:
                print(f"[Warning] Skipping object in {xml_file_path}: Clamping resulted in invalid box. Original: {original_coords}, Clamped: ({xmin}, {ymin}, {xmax}, {ymax}), Image Size: ({img_width}, {img_height})")
                continue
            # --- END OF MODIFIED SECTION ---


            voc_box = (xmin, xmax, ymin, ymax) # Use clamped values
            yolo_box = convert_coordinates((img_width, img_height), voc_box)

            # --- ADDED: Check if calculated YOLO coordinates are valid ---
            # Sometimes clamping can still lead to issues if width/height becomes tiny
            if not (0 <= yolo_box[0] <= 1 and 0 <= yolo_box[1] <= 1 and 0 <= yolo_box[2] <= 1 and 0 <= yolo_box[3] <= 1):
                 print(f"[Warning] Skipping object in {xml_file_path}: Calculated YOLO coordinates invalid after clamping. YOLO box: {yolo_box}")
                 continue
            # --- END OF ADDED CHECK ---


            voc_box = (xmin, xmax, ymin, ymax)
            yolo_box = convert_coordinates((img_width, img_height), voc_box)

            yolo_lines.append(f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")
            objects_found += 1

        # Save the YOLO file only if objects were found and converted
        if objects_found > 0:
            # Create the output filename by replacing .xml with .txt
            base_filename = os.path.splitext(os.path.basename(xml_file_path))[0]
            output_filepath = os.path.join(output_dir, f"{base_filename}.txt")

            with open(output_filepath, 'w') as f:
                f.write("\n".join(yolo_lines))
        # else:
            # print(f"No valid objects found or converted in {xml_file_path}, no .txt file generated.")

    except ET.ParseError:
        print(f"[Error] Failed to parse XML file: {xml_file_path}")
    except Exception as e:
        print(f"[Error] An unexpected error occurred processing {xml_file_path}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Create output directory if it doesn't exist (though it should be the same as input)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all .xml files in the input directory
    xml_files = glob.glob(os.path.join(INPUT_DIR, '*.xml'))
    
    if not xml_files:
        print(f"Error: No .xml files found in '{INPUT_DIR}'. Please check the INPUT_DIR path.")
        exit()

    print(f"Found {len(xml_files)} XML files in '{INPUT_DIR}'. Starting conversion...")

    processed_count = 0
    for xml_file in xml_files:
        convert_xml_to_yolo(xml_file, OUTPUT_DIR)
        processed_count += 1
        if processed_count % 500 == 0: # Print progress every 500 files
            print(f"Processed {processed_count}/{len(xml_files)} files...")

    print(f"\nConversion complete. YOLO .txt files saved in '{OUTPUT_DIR}'.")
    print("Please check for any [Warning] or [Error] messages above.")
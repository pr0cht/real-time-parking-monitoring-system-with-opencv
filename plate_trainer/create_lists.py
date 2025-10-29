# create_lists.py
import os
import xml.etree.ElementTree as ET

def generate_positive_list():
    print("Generating positives.txt...")
    count = 0
    with open('positives.txt', 'w') as f:
        for filename in os.listdir('positive_raw'):
            if not filename.endswith('.xml'):
                continue

            xml_path = os.path.join('positive_raw', filename)

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                img_filename = root.find('filename').text
                img_path = os.path.join('positive_raw', img_filename)

                if not os.path.exists(img_path):
                    print(f"[Warning] Skipping {xml_path}, image file {img_filename} not found.")
                    continue

                objects_on_line = []
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    w = xmax - xmin
                    h = ymax - ymin
                    objects_on_line.append(f"{xmin} {ymin} {w} {h}")

                if objects_on_line:
                    line = f"{img_path} {len(objects_on_line)} {' '.join(objects_on_line)}\n"
                    f.write(line)
                    count += 1
            except Exception as e:
                print(f"[Warning] Failed to parse {xml_path}: {e}")

    print(f"Done. Wrote {count} lines to positives.txt.")

def generate_negative_list():
    print("Generating negatives.txt...")
    count = 0
    with open('negatives.txt', 'w') as f:
        for filename in os.listdir('negative_images'):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                f.write(f"negative_images/{filename}\n")
                count += 1
    print(f"Done. Wrote {count} lines to negatives.txt.")

# --- Run the functions ---
generate_positive_list()
generate_negative_list()
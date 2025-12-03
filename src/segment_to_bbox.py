import os
import glob
from config import COMBINED_DATASET_ROOT


def convert_segment_to_bbox(values):
    class_id = int(values[0])
    coords = list(map(float, values[1:]))

    xs = coords[0::2]   
    ys = coords[1::2]   
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin

    # return in the YOLO format
    return class_id, xc, yc, w, h


def process_label_file(path):
    new_rows = []
    changed = False

    with open(path, "r") as file:
        for row in file:
            row_entries = row.strip().split()

            # We will assume if there are more than 5 entries, then must be a segment
            if len(row_entries) > 5:
                # print(f"[DEBUG] {path}"")
                class_id, xc, yc, w, h = convert_segment_to_bbox(row_entries)
                new_rows.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" + "\n")

                # if never comes here, no need to overwrite the file
                changed = True
            else:
                # just re-add it
                new_rows.append(row)

    if changed:
        with open(path, "w") as file:
            file.writelines(new_rows)


def convert_dataset_labels(dataset_root):
    labels_dir_paths = glob.glob(os.path.join(dataset_root, "**", "*.txt"), recursive=True)

    print(f"[INFO] Found {len(labels_dir_paths)} label files.")
    for path in labels_dir_paths:
        process_label_file(path)

    print()
    print("[INFO] Conversion complete!")


if __name__ == "__main__":
    # Change path in config.py if needed
    convert_dataset_labels(COMBINED_DATASET_ROOT)

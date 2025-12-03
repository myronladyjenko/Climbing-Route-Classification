# preprocessing.py

import os
import cv2
from config import PREPROCESSED_IMAGES_DIR


def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)


def apply_clahe_and_gaussian_blur(img_bgr, clip_limit = 2.0, tile_grid_size = (8, 8), blur_ksize = (3, 3), blur_sigma = 0.9):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # apply the CLAHE 
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    lab = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # apply the Gaussian blur 
    img_blurred_and_clahe = cv2.GaussianBlur(img_clahe, blur_ksize, blur_sigma)
    return img_blurred_and_clahe


def preprocess_image_file(input_path, output_path = None):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Image not found at {input_path}")

    processed_image = apply_clahe_and_gaussian_blur(img)
    # resize in order to be the same as for when training the YOLO model
    RESIZE_SHAPE = (640, 640) 
    processed_image = cv2.resize(processed_image, RESIZE_SHAPE, interpolation=cv2.INTER_AREA)

    if output_path is None:
        filename = os.path.basename(input_path)
        output_path = os.path.join(PREPROCESSED_IMAGES_DIR, filename)
    ensure_directory_exists(os.path.dirname(output_path))
    cv2.imwrite(output_path, processed_image)

    return output_path


if __name__ == "__main__":
    print(f"[WARNING] This file is not meant to be run")
    exit(0)

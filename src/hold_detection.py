import os
import hdbscan  
from typing import List, Dict
import cv2
import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from config import ROOT_DIRECTORY
from sklearn.cluster import KMeans
import cv2


# Create a dictionary with yolo bboxes detections
def load_yolo_detections(txt_file_path):
    detections = []
    with open(txt_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # since we outputted confidence in the file when saved, we need to get rid of it 
            parts = parts[:-1]

            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            detections.append({
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            })
    return detections


# Get Region of interest from the image, i.e. the bounding box for the hold
def extract_roi(image, detection, trim = 2):
    # Used trim, to remove the blue outline of the box (border)
    img_h, img_w, _ = image.shape

    xc = int(detection["x_center"] * img_w)
    yc = int(detection["y_center"] * img_h)
    w = int(detection["width"] * img_w)
    h = int(detection["height"] * img_h)

    x1 = max(0, xc - w // 2 + trim)
    y1 = max(0, yc - h // 2 + trim)
    x2 = min(img_w, xc + w // 2 - trim)
    y2 = min(img_h, yc + h // 2 - trim)

    if x1 >= x2 or y1 >= y2:
        return None
    return image[y1:y2, x1:x2].copy()


def select_furthest_colour(centres, background_colour):
    # Get distances to background for each centre of the bounding box
    distances = [np.linalg.norm(c - background_colour) for c in centres]
    max_idx = int(np.argmax(distances))
    return centres[max_idx]


def kmeans(roi, background_colour, k = 2):
    # Mask out blue pixels - important because some boundingx boxes may overlap 
    # which means there can be a blue line in the middle 
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    non_blue_mask = cv2.bitwise_not(mask)

    masked_roi = cv2.bitwise_and(roi, roi, mask=non_blue_mask)
    pixels = masked_roi.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

    if pixels.size == 0:
        return np.array([0, 0, 0], dtype=np.uint8), np.full_like(roi, 0)

    # perform K-Means
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)
    centres = kmeans.cluster_centers_.astype(int)

    dominant = centres[0] if centres.shape[0] == 1 else select_furthest_colour(centres, background_colour)
    labels = kmeans.labels_
    quantised = np.full_like(roi, 0)
    quantised_flat = quantised.reshape(-1, 3)
    nonzero_mask = np.any(masked_roi.reshape(-1, 3) != [0, 0, 0], axis=1)
    quantised_flat[nonzero_mask] = centres[labels].astype(np.uint8)

    return dominant.astype(np.uint8), quantised


# perform mean-shift, ensure no blue pixels from bounding box are present 
# and background is accounted for
def meanshift(roi, background_colour, quantile = 0.2):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Mask out blue pixels, similar to above
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    non_blue_mask = cv2.bitwise_not(mask)

    masked_roi = cv2.bitwise_and(roi, roi, mask = non_blue_mask)
    pixels = masked_roi.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis = 1)]

    if pixels.size == 0:
        return np.array([0, 0, 0], dtype=np.uint8), np.full_like(roi, 0)

    bandwidth = estimate_bandwidth(pixels, quantile=quantile, n_samples=200)
    if bandwidth <= 0:
        bandwidth = 1.0

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(pixels)
    centres = ms.cluster_centers_.astype(int)
    labels = ms.labels_

    if centres.shape[0] == 1:
        dominant = centres[0]
    else:
        dominant = select_furthest_colour(centres, background_colour)

    quantised = np.full_like(roi, 0)
    nonzero_mask = np.any(masked_roi.reshape(-1, 3) != [0, 0, 0], axis=1)
    quantised_flat = quantised.reshape(-1, 3)
    quantised_pixels = centres[labels].astype(np.uint8)
    quantised_flat[nonzero_mask] = quantised_pixels

    return dominant.astype(np.uint8), quantised


# perform Gaussian-Mixture-Models so that colours get soft assignment and maybe achieve better accuracy
def gmm(roi, background_colour, n_components=2):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    non_blue_mask = cv2.bitwise_not(mask)

    masked_roi = cv2.bitwise_and(roi, roi, mask=non_blue_mask)
    pixels = masked_roi.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

    if pixels.size == 0:
        return np.array([0, 0, 0], dtype=np.uint8), np.full_like(roi, 0)

    pixels = pixels.astype(np.float64)
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
    gmm.fit(pixels)
    centres = gmm.means_.astype(int)

    if centres.shape[0] == 1:
        dominant = centres[0]
    else:
        dominant = select_furthest_colour(centres, background_colour)

    labels = gmm.predict(pixels)
    quantised = np.full_like(roi, 0)
    nonzero_mask = np.any(masked_roi.reshape(-1, 3) != [0, 0, 0], axis=1)
    quantised_flat = quantised.reshape(-1, 3)
    quantised_pixels = centres[labels].astype(np.uint8)
    quantised_flat[nonzero_mask] = quantised_pixels

    return dominant.astype(np.uint8), quantised


# Try this, as was done in one of the literature readings
def hsv_hue(roi):
    if roi is None or roi.size == 0:
        return (0, 0, 0), None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue_values = hsv[:, :, 0].flatten()
    hue_values = hue_values.astype(np.int32)
    hue_values = hue_values[hue_values >= 0]
    hue_values = hue_values[hue_values < 180]

    if hue_values.size == 0:
        return (0, 0, 0), None

    hist = np.bincount(hue_values, minlength=180)
    bin_centers = np.arange(180)

    dominant_hue = bin_centers[np.argmax(hist)]
    mean_s = int(np.mean(hsv[:, :, 1]))
    mean_v = int(np.mean(hsv[:, :, 2]))
    color_bgr = cv2.cvtColor(np.uint8([[[dominant_hue, mean_s, mean_v]]]), cv2.COLOR_HSV2BGR)[0][0]

    return tuple(map(int, color_bgr)), hist


# Note that I have just selected some 'what seem good' values for the parameters
def cluster_colours(colours, method = "hdbscan", eps = 25.0, min_samples = 2, hdbscan_min_cluster_size = 2, distance_threshold = 35.0):
    if len(colours) == 0:
        return np.array([], dtype=int)
    X = np.array(colours).reshape(-1, 3)

    labels = []
    method = method.lower()
    if method == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(X)
    elif method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
        labels = clusterer.fit_predict(X)
    elif method == "agglomerative":
        clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage='ward')
        labels = clusterer.fit_predict(X)
    else:
        raise ValueError(f"[ERROR] Unsupported clustering method: {method}")
    return labels


def draw_clustered_holds(image, detections, labels, out_path):
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (0, 128, 128), (128, 0, 128)
    ]
    while len(palette) < len(set(labels)):
        palette.append(tuple(np.random.randint(0, 256, size=3).tolist()))

    drawn = image.copy()
    h_img, w_img, _ = drawn.shape
    for det, lbl in zip(detections, labels):
        xc = int(det["x_center"] * w_img)
        yc = int(det["y_center"] * h_img)
        bw = int(det["width"] * w_img)
        bh = int(det["height"] * h_img)

        x1 = xc - bw // 2
        y1 = yc - bh // 2
        x2 = xc + bw // 2
        y2 = yc + bh // 2

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        colour = (160, 160, 160) if lbl == -1 else palette[int(lbl) % len(palette)]
        cv2.rectangle(drawn, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(drawn, str(lbl), (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_PLAIN, 0.5, colour, 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, drawn)

# Group 'similar' colours into same categories for better route detection
def group_similar_colours(colours, n_groups = 6):    
    colours_lab = cv2.cvtColor(np.array(colours, dtype=np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_groups, n_init=10, random_state=42)
    labels = kmeans.fit_predict(colours_lab)
    centres_lab = kmeans.cluster_centers_

    centres_rgb = cv2.cvtColor(centres_lab.astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    grouped_colours = [centres_rgb[label].astype(np.uint8) for label in labels]
    return grouped_colours


def process_image(image_path, detections_path, output_base_dir, background_colour = (145, 157, 170), 
                  colour_methods = ("kmeans", "meanshift", "gmm", "hsv"), 
                  route_cluster_methods = ("dbscan", "hdbscan", "agglomerative")):
    image = cv2.imread(image_path)
    detections = load_yolo_detections(detections_path)
    background_colour_np = np.array(background_colour)

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    for method in colour_methods:
        method_dir = os.path.join(output_base_dir, img_name, method.lower())
        os.makedirs(method_dir, exist_ok=True)

    method_colours: Dict[str, List[np.ndarray]] = {m: [] for m in colour_methods}
    for i, det in enumerate(detections):
        roi = extract_roi(image, det)
        if roi is None or roi.size == 0:
            for m in colour_methods:
                method_colours[m].append(np.array([0, 0, 0], dtype=np.uint8))
            continue
        
        # Colour detection
        for m in colour_methods:
            out_dir = os.path.join(output_base_dir, img_name, m.lower())
            if m.lower() == "kmeans":
                colour, quant = kmeans(roi, background_colour_np, k=2)
            elif m.lower() == "meanshift":
                colour, quant = meanshift(roi, background_colour_np, quantile=0.2)
            elif m.lower() == "gmm":
                colour, quant = gmm(roi, background_colour_np, n_components=2)
            elif m.lower() == "hsv":
                colour, quant = hsv_hue(roi)
            else:
                raise ValueError(f"[ERROR] Unknown colour method: {m}")
            method_colours[m].append(colour)

            roi_out = os.path.join(out_dir, f"{img_name}_hold{i}_roi.png")
            quant_out = os.path.join(out_dir, f"{img_name}_hold{i}_quant.png")
            colour_swatch_out = os.path.join(out_dir, f"{img_name}_hold{i}_colour.png")
            cv2.imwrite(roi_out, roi)
            cv2.imwrite(quant_out, quant)
            cv2.imwrite(colour_swatch_out, np.full((50, 50, 3), colour, dtype=np.uint8))

    # Route clustering
    for m in colour_methods:
        # Group similar colour for less variability
        colours = group_similar_colours(method_colours[m], n_groups = 6)

        for route_clustering_method in route_cluster_methods:
            labels = cluster_colours(colours, method = route_clustering_method.lower())

            out_dir = os.path.join(output_base_dir, img_name, m.lower())
            cluster_img_out = os.path.join(out_dir, f"{img_name}_clusters_{route_clustering_method.lower()}.png")
            draw_clustered_holds(image, detections, labels, cluster_img_out)


if __name__ == "__main__":
    # Adjust all the paths to do hold detection on other images
    base_dir = "detection_outputs_w_preprocessing/inference_n_90"
    image_file = "2_bbox.png" 
    detections_file = "2.txt"
    image_path = os.path.join(ROOT_DIRECTORY, base_dir, "images", image_file)
    detections_path = os.path.join(ROOT_DIRECTORY, base_dir, "labels", detections_file)
    output_dir = "clustering_results"

    if os.path.isfile(image_path) and os.path.isfile(detections_path):
        process_image(
            image_path=image_path,
            detections_path=detections_path,
            output_base_dir=output_dir,
            # In the future this needs to be autodetection or, to begin, at least an array
            background_colour=(145, 157, 170),
            colour_methods=("kmeans", "meanshift", "gmm", "hsv"),
            route_cluster_methods=("dbscan", "hdbscan", "agglomerative")
        )
    else:
        print("[ERROR] YOLO detection files were not found. Please specify correct paths.")